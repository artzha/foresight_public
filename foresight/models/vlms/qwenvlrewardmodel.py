from __future__ import annotations

from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import AutoModelForTokenClassification, AutoProcessor

from foresight.prompts.interface import ChatQuery, ContentType, OutputFormat, UnifiedEnvelope, parse_and_unify
from foresight.utils.log import logging


def _resolve_torch_dtype(dtype_name: str):
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(str(dtype_name).lower(), dtype_name)


class QwenVLRewardModel:
    """
    Adapter for Qwen3-VL token-classification reward checkpoints.

    This class matches the VLM provider interface used across the repository:
      - compile_prompt(prompts) -> compiled messages
      - generate_response(...) -> UnifiedEnvelope
      - generate_batch_response(...) -> list[UnifiedEnvelope | None]
    """

    def __init__(
        self,
        *,
        model_name: str,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        timeout: float = 60.0,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ):
        self.timeout = float(timeout)
        self._model_name = model_name

        # Trigger AutoModelForTokenClassification registration side-effect. verl is an
        # optional dependency; only the reward model role needs it.
        import verl.models.transformers.qwen3_vl_reward  # noqa: F401

        processor_kwargs = dict(kwargs.pop("processor_kwargs", {}) or {})
        model_kwargs = dict(kwargs.pop("model_kwargs", {}) or {})
        self.apply_chat_template_kwargs = dict(kwargs.pop("apply_chat_template_kwargs", {}) or {})

        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            torch_dtype=_resolve_torch_dtype(torch_dtype),
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
        self.model.eval()
        self.input_device = self._resolve_input_device()
        logging.info(f"QwenVLRewardModel loaded successfully on device: {self.input_device}")
        logging.info("Reward apply_chat_template kwargs: %s", self.apply_chat_template_kwargs)

    def _resolve_input_device(self) -> torch.device:
        for p in self.model.parameters():
            if p.device.type != "meta":
                return p.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_model_name(self) -> str:
        return self._model_name

    def format_content(self, q: ChatQuery) -> Dict[str, Any]:
        qtype = str(q.type)
        if qtype == ContentType.TEXT.value:
            return {"type": "text", "text": q.content}
        if qtype == ContentType.IMAGE.value:
            if isinstance(q.content, Image.Image):
                return {"type": "image", "image": q.content}
            return {"type": "image", "image": str(q.content)}
        if qtype == ContentType.VIDEO.value:
            content = q.content
            if isinstance(content, dict):
                if "video" not in content:
                    raise ValueError("video content dict must include 'video' field")
                payload = {"type": "video", "video": content["video"]}
                for key in ("sample_fps", "fps", "num_frames", "timestamps", "distances"):
                    if key in content:
                        payload[key] = content[key]
                return payload
            if not isinstance(content, (list, tuple)) or not content:
                raise ValueError("video content must be a non-empty list/tuple of PIL images or path strings")
            video_items: List[Any] = []
            for idx, item in enumerate(content):
                if isinstance(item, (Image.Image, str)):
                    video_items.append(item)
                else:
                    raise ValueError(
                        f"video content[{idx}] must be PIL.Image or str path/URI, got {type(item)!r}"
                    )
            return {"type": "video", "video": video_items}
        raise ValueError(f"Unsupported content type: {q.type}")

    def compile_prompt(self, prompts: List[ChatQuery]) -> Dict[str, Any]:
        msgs = []
        cur_role, cur_content = None, []
        for prompt in prompts:
            if prompt.role != cur_role:
                if cur_role is not None:
                    msgs.append({"role": cur_role, "content": cur_content})
                cur_role, cur_content = prompt.role, []
            cur_content.append(self.format_content(prompt))
        if cur_role is not None:
            msgs.append({"role": cur_role, "content": cur_content})
        return {"contents": msgs}

    def _coerce_messages(self, inputs: Any) -> List[Dict[str, Any]]:
        if isinstance(inputs, dict) and "contents" in inputs:
            return list(inputs["contents"])
        if isinstance(inputs, list) and inputs:
            if isinstance(inputs[0], ChatQuery):
                return list(self.compile_prompt(inputs)["contents"])
            if isinstance(inputs[0], dict):
                return list(inputs)
        raise ValueError("Unsupported input format for QwenVLRewardModel.generate_response")

    @staticmethod
    def _extract_reward(logits: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.dim() == 3 and logits.shape[-1] == 1:
            token_scores = logits.squeeze(-1)
        elif logits.dim() == 3:
            # Keep compatibility with multi-label token heads, but log this because
            # reward checkpoints are expected to use num_labels=1.
            logging.warning(
                "Reward head emitted %s labels per token; using label-0 logits for reward extraction.",
                logits.shape[-1],
            )
            token_scores = logits[..., 0]
        else:
            token_scores = logits

        if token_scores.dim() != 2:
            raise ValueError(f"Expected token_scores to be rank-2 [bs, seq], got shape {tuple(token_scores.shape)}")
        if attention_mask is None:
            raise ValueError("attention_mask is required for last-valid-token reward extraction")
        if attention_mask.dim() != 2:
            raise ValueError(f"Expected attention_mask rank-2 [bs, seq], got shape {tuple(attention_mask.shape)}")

        valid_lengths = attention_mask.to(torch.int64).sum(dim=-1)
        last_valid_token_idxs = torch.clamp(valid_lengths, min=1) - 1
        rewards = token_scores[torch.arange(token_scores.shape[0], device=token_scores.device), last_valid_token_idxs]
        return rewards, last_valid_token_idxs

    def _prepare_model_inputs(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        apply_chat_template_kwargs = {"do_sample_frames": False}
        apply_chat_template_kwargs.update(self.apply_chat_template_kwargs)
        model_inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )
        model_inputs = dict(model_inputs)
        tensor_inputs = {}
        for key, value in model_inputs.items():
            tensor_inputs[key] = value.to(self.input_device) if torch.is_tensor(value) else value
        return tensor_inputs

    def generate_batch_response(
        self,
        instructions: str,
        inputs: List[Any],
        output_format: OutputFormat,
        meta: List[Dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> List[UnifiedEnvelope | None]:
        outputs: List[UnifiedEnvelope | None] = []
        for idx, input_item in enumerate(inputs):
            meta_item = meta[idx] if meta is not None and idx < len(meta) else {}
            try:
                outputs.append(
                    self.generate_response(
                        instructions=instructions,
                        inputs=input_item,
                        output_format=output_format,
                        meta=meta_item,
                        **kwargs,
                    )
                )
            except Exception as exc:
                logging.warning(f"Reward batch item {idx} failed: {exc}")
                outputs.append(None)
        return outputs

    def generate_response(
        self,
        instructions: str,
        inputs: Any,
        output_format: OutputFormat,
        meta: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> UnifiedEnvelope:
        if output_format != OutputFormat.REWARD_V1:
            raise ValueError(f"QwenVLRewardModel supports only {OutputFormat.REWARD_V1.value}")

        messages = self._coerce_messages(inputs)
        if instructions and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": [{"type": "text", "text": instructions}]}] + messages

        model_inputs = self._prepare_model_inputs(messages)
        try:
            with torch.no_grad():
                outputs = self.model(use_cache=False, **model_inputs, **kwargs)
                reward_scores, last_valid_token_idxs = self._extract_reward(
                    logits=outputs.logits,
                    attention_mask=model_inputs["attention_mask"],
                )
                # Apply sigmoid to the reward scores
                # reward_scores = torch.sigmoid(reward_scores)
                # reward_scores = 2.0 * reward_scores - 1.0
                reward_val = float(reward_scores[0].detach().cpu().item())
        except Exception:
            attn = model_inputs.get("attention_mask")
            attn_shape = tuple(attn.shape) if torch.is_tensor(attn) else None
            attn_sums = attn.to(torch.int64).sum(dim=-1).tolist() if torch.is_tensor(attn) else None
            logging.exception(
                "Reward forward/extraction failed. model=%s logits_shape=%s attention_mask_shape=%s attention_sums=%s",
                self._model_name,
                tuple(outputs.logits.shape) if "outputs" in locals() and hasattr(outputs, "logits") else None,
                attn_shape,
                attn_sums,
            )
            raise

        logging.info(
            "Reward extracted. model=%s logits_shape=%s last_valid_token_idxs=%s reward=%.6f",
            self._model_name,
            tuple(outputs.logits.shape),
            last_valid_token_idxs.detach().cpu().tolist(),
            reward_val,
        )
        if reward_val < -1.0 or reward_val > 1.0:
            logging.warning(
                "Reward %.6f is outside RewardPayload range [-1, 1]; parse_and_unify may fail.",
                reward_val,
            )

        usage = {
            "input_tokens": int(model_inputs["attention_mask"].sum().item()),
            "output_tokens": 0,
            "cached_tokens": 0,
        }
        try:
            return parse_and_unify(
                {"reward": reward_val},
                output_format,
                meta=meta or {},
                model_name=self._model_name,
                usage=usage,
            )
        except Exception:
            logging.exception(
                "Reward parse failed. model=%s reward=%s usage=%s output_format=%s",
                self._model_name,
                reward_val,
                usage,
                output_format,
            )
            raise
