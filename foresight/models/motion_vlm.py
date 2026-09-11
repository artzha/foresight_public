# foresight/models/motion_vlm.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List, Optional, Iterable

from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from foresight.core.printer import rank0_print
from foresight.core.constants import IGNORE_INDEX
from foresight.models.base_vlm import BaseVLM
from foresight.models.monkey_patch_forward import replace_qwen3_with_mixed_modality_forward

import foresight.utils.train_utils as tu
from foresight.utils.log import LogManager
from foresight.utils.metric import MetricManager
from foresight.prompts.interface import parse_and_unify, OutputFormat

# ---- helper functions adapted from train_sft.py ----

def set_requires_grad(parameters, requires_grad: bool) -> None:
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(
    model: nn.Module,
    training_cfg: Dict[str, Any],
    compute_dtype,
    device: Optional[str] = None,
) -> None:
    """
    Mirror of configure_vision_tower() from train_sft.py, but with
    a plain dict `training_cfg`.
    """
    if not hasattr(model, "visual"):
        return

    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype)

    freeze_vision_tower = training_cfg.get("freeze_vision_tower", False)
    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not freeze_vision_tower)

    # Handle merger specifically
    freeze_merger = training_cfg.get("freeze_merger", False)
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not freeze_merger)

    if hasattr(model.visual, "deepstack_merger_list"):
        deepstack_merger_list_params = model.visual.deepstack_merger_list.parameters()
        set_requires_grad(deepstack_merger_list_params, not freeze_merger)

def configure_llm(model: nn.Module, training_cfg: Dict[str, Any]) -> None:
    freeze_llm = bool(training_cfg.get("freeze_llm", False))
    lora_enable = bool(training_cfg.get("lora_enable", False))

    # If not using LoRA, old behavior is fine
    if not lora_enable:
        if hasattr(model, "lm_head"):
            set_requires_grad(model.lm_head.parameters(), not freeze_llm)
        if hasattr(model, "language_model"): 
            set_requires_grad(model.language_model.parameters(), not freeze_llm)
        return

def unfreeze_topk_layers(model: nn.Module, k_llm: int = 0, k_vis: int = 0) -> None:
    if k_llm and hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        for layer in model.language_model.layers[-k_llm:]:
            for p in layer.parameters():
                p.requires_grad = True

    if k_vis and hasattr(model, "visual") and hasattr(model.visual, "blocks"):
        for blk in model.visual.blocks[-k_vis:]:
            for p in blk.parameters():
                p.requires_grad = True

def find_lora_targets_llm_only(model, exclude=(), num_lora_modules=-1, verbose=True):
    names = []
    for name, module in model.named_modules():
        if ".language_model." not in name:   # <-- key change
            continue
        if any(ex in name for ex in exclude):
            continue
        if isinstance(module, torch.nn.Linear):
            names.append(name)

    if num_lora_modules and num_lora_modules > 0:
        names = names[-num_lora_modules:]

    assert len(names) > 0, (
        "LoRA target_modules resolved to 0. "
        "This will cause PEFT to use defaults and may LoRA the vision tower."
    )
    if verbose:
        rank0_print(f"[lora] LLM-only targets={len(names)} (first 25): {names[:25]}")
    return names

# ------------------------------------------------------------------
#   MotionVLM class
# ------------------------------------------------------------------

class MotionVLM(BaseVLM):
    """
    Motion-generation VLM wrapper for Qwen3-VL.

    For now, this class only:
      - initializes Qwen3-VL backbone
      - applies mixed-modality monkey patches
      - configures quantization / vision tower / LLM / gradient checkpointing
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        model_id: str,
    ):
        super().__init__(cfg, model_family="qwen", model_id=model_id)

        # Split config into model vs training sections if present
        self.model_cfg: Dict[str, Any] = cfg['model']
        self.train_cfg: Dict[str, Any] = cfg['trainer']

        # Processor mirrors train_sft.py
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Build the HF Qwen3 backbone w/ all train_sft.py-style config
        self.backbone = self._init_qwen3_backbone(model_id)

        # Generation config
        self.generation_config = self.backbone.generation_config

        # Build hf required attributes
        # self.backbone.config.hidden_size = self.backbone.config.vision_config.hidden_size

    # ------------------------------------------------------------------
    #   Qwen3 init mirroring train_sft.py
    # ------------------------------------------------------------------

    def _init_qwen3_backbone(self, model_id: str) -> nn.Module:
        """
        Equivalent in spirit to the model-loading + configuration logic
        in train_sft.py, but adapted for an Accelerate-based stack.
        """

        # ----- monkey patch mixed-modality forward -----
        replace_qwen3_with_mixed_modality_forward()

        training_args: Dict[str, Any] = self.train_cfg
        bits = training_args.get("bits", 16)
        fp16 = training_args.get("fp16", False)
        bf16 = training_args.get("bf16", False)
        disable_flash = training_args.get("disable_flash_attn2", False)
        device = training_args.get("device", None)  # kept for future if you want device_map

        compute_dtype = (
            torch.float16 if fp16
            else torch.bfloat16 if bf16
            else torch.float32
        )

        # -----------------------
        # BitsAndBytes quantization
        # -----------------------
        bnb_kwargs: Dict[str, Any] = {}
        if bits in [4, 8]:
            bnb_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=(bits == 4),
                load_in_8bit=(bits == 8),
                llm_int8_skip_modules=["visual", "lm_head"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.get("double_quant", True),
                bnb_4bit_quant_type=training_args.get("quant_type", "nf4"),
            )
            # With Accelerate we usually let it handle device placement (no explicit device_map here).

        # -----------------------
        # Load Qwen3 model
        # -----------------------
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not disable_flash else "sdpa",
            **bnb_kwargs,
        )

        # Do *not* rely on cache during training
        model.config.use_cache = False

        lora_enable = bool(training_args.get("lora_enable", False))
        vision_lora = bool(training_args.get("vision_lora", False))

        freeze_llm = bool(training_args.get("freeze_llm", False))
        freeze_vision_tower = bool(training_args.get("freeze_vision_tower", False))
        freeze_merger = bool(training_args.get("freeze_merger", False))

        gradient_checkpointing = bool(training_args.get("gradient_checkpointing", False))

        # Initialize model

        if lora_enable and not freeze_llm:
            raise ValueError("If training_cfg['lora_enable'] is True, training_cfg['freeze_llm'] must also be True.")

        if (not lora_enable) and vision_lora:
            raise ValueError("training_cfg['vision_lora']=True requires training_cfg['lora_enable']=True")

        if vision_lora and not freeze_vision_tower:
            raise ValueError("If training_cfg['vision_lora'] is True, training_cfg['freeze_vision_tower'] must also be True.")

        lora_namespan_exclude = list(training_args.get("lora_namespan_exclude", []))

        # -----------------------
        # Configure which parts are trainable
        # -----------------------
        configure_llm(model, training_args)
        configure_vision_tower(model, training_args, compute_dtype, device=device)

        unfreeze_topk_layers(
            model,
            k_llm=training_args.get("unfreeze_topk_llm", 0),
            k_vis=training_args.get("unfreeze_topk_vision", 0),
        )

        # Gradient checkpointing kwargs (compat across transformers)
        if gradient_checkpointing:
            if training_args['vision_lora']:
                training_args['gradient_checkpointing_kwargs'] = {"use_reentrant": False}
            else:
                training_args['gradient_checkpointing_kwargs'] = {"use_reentrant": True}
            model.enable_input_require_grads()

        # For 4/8-bit, prep model for k-bit training *before* applying LoRA
        if bits in [4, 8]:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=gradient_checkpointing,
                gradient_checkpointing_kwargs=training_args['gradient_checkpointing_kwargs'],
            )

        if lora_enable:
            peft_config = LoraConfig(
                r=training_args['lora_rank'],
                lora_alpha=training_args['lora_alpha'],
                target_modules=find_lora_targets_llm_only(
                    model,
                    exclude=lora_namespan_exclude,
                    num_lora_modules=training_args['num_lora_modules'],
                ),
                lora_dropout=training_args['lora_dropout'],
                bias=training_args.get("lora_bias", "none"),
            )
            rank0_print(f"[lora] Adding LoRA to the model")
            model = get_peft_model(model, peft_config)

            if not training_args['freeze_vision_tower']:
                for n, p in model.named_parameters():
                    if "visual" in n:
                        p.requires_grad = True
            if not training_args['freeze_merger']:
                for n, p in model.named_parameters():
                    if "merger" in n:
                        p.requires_grad = True

        processor = AutoProcessor.from_pretrained(model_id)

        if bits in [4, 8]:
            from peft.tuners.lora import LoraLayer
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer) and bf16:
                    module.to(torch.bfloat16)
                if "norm" in name:
                    module.to(torch.float32)
                if ("lm_head" in name or "embed_token" in name) and hasattr(module, "weight"):
                    if bf16 and module.weight.dtype == torch.float32:
                        module.to(torch.bfloat16)

        # Dump all model paramters
        with open("model_params.txt","w") as f: f.write("\n".join([f"{n} | {tuple(p.shape)} | requires_grad={p.requires_grad}" for n,p in model.named_parameters()]))
        # Dump trainable model paramters
        # with open("trainable_params.txt","w") as f: f.write("\n".join([f"{n} | {tuple(p.shape)}" for n,p in model.named_parameters() if p.requires_grad]))
        # Dump only LoRA paramters
        # with open("lora_params.txt","w") as f: f.write("\n".join([f"{n} | {tuple(p.shape)} | grad={p.requires_grad}" for n,p in model.named_parameters() if "lora_" in n.lower()]))
        # Dump parameter group summary
        # with open("optimizer_groups.txt","w") as f: f.write("\n\n".join([f"group {i}: lr={g.get('lr')} wd={g.get('weight_decay')} params={sum(p.numel() for p in g['params'])}" for i,g in enumerate(optimizer.param_groups)]))
        return model

    # ------------------------------------------------------------------
    #   BaseVLM required API
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: str,
        **kwargs,
    ) -> "MotionVLM":
        """
        Simple wrapper: re-instantiate with cfg+model_id and then load weights.
        """
        cfg = kwargs.get("cfg", {})
        obj = cls(cfg=cfg, model_id=pretrained_checkpoint)
        # optionally load checkpoint weights here if different from HF hub
        return obj

    def freeze_backbones(self, stage: str) -> None:
        """
        Basic freezing control:
          stage ∈ {"all", "llm", "vision", "none"}
        """
        if stage in ("all", "llm") and hasattr(self.backbone, "language_model"):
            self.set_trainability(self.backbone.language_model, False)
        if stage in ("all", "vision") and hasattr(self.backbone, "visual"):
            self.set_trainability(self.backbone.visual, False)
        if stage == "none":
            self.set_trainability(self.backbone, True)

    def load_from_checkpoint(
        self,
        stage: str,
        run_dir: str,
        pretrained_checkpoint: Optional[str] = None,
    ) -> None:
        """
        Placeholder for your future Accelerate/FSDP checkpoint loading logic.
        """
        ckpt_file = pretrained_checkpoint or f"{run_dir}/pytorch_model.bin"
        state = torch.load(ckpt_file, map_location="cpu")
        missing, unexpected = self.load_state_dict(state, strict=False)
        rank0_print("[MotionVLM] Loaded checkpoint:", ckpt_file)
        rank0_print("  missing:", missing)
        rank0_print("  unexpected:", unexpected)

    def get_fsdp_wrapping_policy(self):
        """
        Stub – let your training stack decide how to shard.
        """
        return lambda module: False

    # ------------------------------------------------------------------
    #   Forward: just delegate into Qwen3 backbone
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pixel_values=None,
        image_grid_thw=None,
        **kwargs,
    ) -> Dict[str, Any]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    

# ------------------------------------------------------------------
#   Post-hoc vLLM inference runner for MotionVLM
# ------------------------------------------------------------------

class MotionVLLMInference:
    """
    Local vLLM inference wrapper for a MotionVLM HF-export folder.

    Expects hf_ckpt_dir like:
      <...>/hf_checkpoints/epoch=...-step=.../
        - model.safetensors
        - config.json
        - processor files
    """

    def __init__(
        self,
        *,
        cfg: Dict[str, Any],
        hf_ckpt_dir: str | Path,
        model_id_for_parsing: str,
        logger: Any = None,  # e.g. WandbLogger from Lightning, or anything LogManager accepts
    ):
        self.cfg = cfg
        self.hf_ckpt_dir = Path(hf_ckpt_dir)
        self.model_id_for_parsing = model_id_for_parsing
        self.logger = logger

        self.metrics = MetricManager(cfg["metrics"], logger)
        self.log_manager = LogManager(cfg["visualizations"])

        # vLLM is optional for training; import only here
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(str(self.hf_ckpt_dir))

        from vllm import LLM
        self.llm = LLM(model=str(self.hf_ckpt_dir), trust_remote_code=True)

    def _prepare_inputs_for_vllm(self, messages: list[dict]) -> dict:
        from qwen_vl_utils import process_vision_info

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # qwen_vl_utils 0.0.14+ behavior (as in Qwen3-VL vLLM snippet)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

    def _decode_motion_trajectories(self, raw_texts: List[str]) -> Dict[str, Any]:
        trajectories: List[list] = []
        cleaned: List[str] = []

        for t in raw_texts:
            text = (t or "").strip()
            cleaned.append(text)

            if not text:
                trajectories.append([])
                continue

            try:
                env = parse_and_unify(
                    text,
                    OutputFormat.TRAJECTORY_V1,
                    meta=None,
                    model_name=self.model_id_for_parsing,
                    usage={},
                )
                trajectories.append(env.unified.get("trajectory", []) or [])
            except Exception:
                trajectories.append([])

        return {"trajectory": trajectories, "raw_text": cleaned}

    def _generate_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        from vllm import SamplingParams

        inf_cfg = self.cfg.get("inference", {}) or {}
        sampling_params = SamplingParams(
            temperature=float(inf_cfg.get("temperature", 0.0)),
            top_p=float(inf_cfg.get("top_p", 1.0)),
            max_tokens=int(inf_cfg.get("max_new_tokens", 1024)),
            # IMPORTANT: Qwen3-VL example uses empty stop_token_ids
            stop_token_ids=list(inf_cfg.get("stop_token_ids", [])),
        )

        # ---- Preferred path: multimodal messages -> prompt string + mm_data ----
        if "vllm_messages" in batch:
            # collator keeps as list-of-messages per sample; if batch_size==1 may be just messages
            msgs = batch["vllm_messages"]
            if isinstance(msgs, list) and len(msgs) > 0 and isinstance(msgs[0], dict):
                msgs = [msgs]  # single sample -> wrap
    
            inputs = [self._prepare_inputs_for_vllm(m) for m in msgs]
            outs = self.llm.generate(inputs, sampling_params=sampling_params)
            raw_texts = [(o.outputs[0].text if o.outputs else "") for o in outs]
            return self._decode_motion_trajectories(raw_texts)

        # ---- Fallback: text-only, allow token prompts ----
        prompt_token_ids = self._prompt_token_ids_from_batch(batch)
        inputs = [{"prompt_token_ids": toks} for toks in prompt_token_ids]
        outs = self.llm.generate(inputs, sampling_params=sampling_params)
        raw_texts = [(o.outputs[0].text if o.outputs else "") for o in outs]
        return self._decode_motion_trajectories(raw_texts)

    def validate(
        self,
        *,
        val_dataloader: Iterable[Dict[str, Any]],
        stage: str = "val",
        limit_batches: Optional[int] = None,
        step_offset: int = 0,
    ) -> int:
        """
        Iterate val_dataloader, generate trajectories with vLLM, and log:
          - trajectory drawn on image (via LogManager)
          - metrics (via MetricManager)

        Returns: final step counter (step_offset + num_batches_logged)
        """
        step = int(step_offset)

        # If user didn't pass limit_batches, honor cfg["inference"].get("limit_batches")
        if limit_batches is None:
            limit_batches = self.cfg.get("inference", {}).get("limit_batches", None)

        for bidx, batch in enumerate(val_dataloader):
            if limit_batches is not None and bidx >= int(limit_batches):
                break

            response = self._generate_batch(batch)
            import pdb; pdb.set_trace()
            outputs = {"loss": None, "logits": None, "response": response}
            merged = tu.merge_dict(("inputs", batch), ("outputs", outputs))

            # mirror your MotionTrainer-style logging hooks
            self.log_manager(merged, logger=self.logger, step=step, stage=stage)
            self.metrics.update(merged, step=step, stage=stage)
            step += 1

        # epoch-level cleanup (matches MotionTrainer.on_validation_epoch_end intent)
        self.metrics.reset()
        return step