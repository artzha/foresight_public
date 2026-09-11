# qwenvlmodel.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import os
import time
import random
import io
import base64
import pprint

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from pydantic import BaseModel, Field, conint, constr

from transformers import AutoTokenizer, AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# from vllm import LLM, EngineArgs, SamplingParams

from foresight.prompts.interface import (
    ContentType, Role, ChatQuery, 
    OutputFormat, parse_and_unify, UnifiedEnvelope
)
from foresight.utils.log import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class QwenResponse:
    """Response object matching expected interface."""
    output_parsed: ReasoningTrace
    usage: Any
    
    def __init__(self, output_parsed, usage=None):
        self.output_parsed = output_parsed
        if usage is None:
            # Create default empty usage
            @dataclass
            class InputTokensDetails:
                cached_tokens: int = 0
            
            @dataclass
            class Usage:
                input_tokens: int = 0
                output_tokens: int = 0
                input_tokens_details: InputTokensDetails = InputTokensDetails()
            
            usage = Usage()
        self.usage = usage


@dataclass
class QwenConfig:
    """Default configuration for Qwen VL usage."""
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    default_role: str = "user"
    torch_dtype: str = "auto"

    # generation defaults
    max_new_tokens: int = 512
    temperature: float = 1.3
    top_p: float = 0.95
    top_k: int = -1

    # model loading / runtime defaults
    tensor_parallel_size: int = 1
    max_model_length: int = 4096
    gpu_memory_utilization: float = 0.85
    enforce_eager: bool = True
    swap_space_gb: int = 4
    seed: int = 0

    # tokenizer / remote code
    trust_remote_code: bool = True
    tokenizer: Optional[str] = None

    # catch-all for extra model kwargs
    extra: Dict[str, Any] = field(default_factory=dict)

    def model_kwargs(self) -> Dict[str, Any]:
        """Return a mapping suitable for model loader kwargs."""
        return {
            "tokenizer": self.tokenizer or self.model_name,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.torch_dtype,
            "tensor_parallel_size": int(self.tensor_parallel_size),
            "max_model_len": int(self.max_model_length),
            "gpu_memory_utilization": float(self.gpu_memory_utilization),
            "enforce_eager": bool(self.enforce_eager),
            "swap_space": int(self.swap_space_gb),
            "seed": int(self.seed),
            **self.extra,
        }

    def sampling_kwargs(self) -> Dict[str, Any]:
        """Return sampling params compatible with the sampler constructor."""
        return {
            "temperature": float(self.temperature),
            "max_tokens": int(self.max_new_tokens),
            "top_p": float(self.top_p),
            "top_k": int(self.top_k),
        }

class QwenVLModel:
    """
    Wrapper for Qwen VL model using local vllm implementation.
    Matches the OpenAI provider interface so the two are interchangeable.
    """

    def __init__(
        self,
        *,
        default_role: str = "user",
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        torch_dtype: str = "auto",
        sampling_params: Optional[Dict[str, Any]] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize Qwen VL model.
        
        Args:
            api_key: Not used for local model (kept for interface compatibility)
            base_url: Not used for local model (kept for interface compatibility)
            timeout: Request timeout in seconds
            service_tier: Not used for local model (kept for interface compatibility)
            default_model_args: Default model generation arguments
            default_role: Default role for messages
            model_name: Hugging Face model name or local path
            torch_dtype: Torch dtype for model ("auto", "float16", "bfloat16")
            device_map: Device map for model loading
            attn_implementation: Attention implementation ("flash_attention_2" or None)
            min_pixels: Minimum pixels for vision processing
            max_pixels: Maximum pixels for vision processing
        """
        self.timeout = kwargs.pop("timeout", 60.0)

        self.config = QwenConfig(
            model_name=model_name,
            torch_dtype=torch_dtype,
            default_role=default_role,
        )

        self.default_model_args = kwargs.pop("default_model_args", {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
        })
        self._default_role = default_role
        self._model_name = model_name
        
        logging.info(f"Loading Qwen VL model: {model_name}")
                    
        trust_remote_code = kwargs.get("trust_remote_code", True)
        tokenizer_name = kwargs.get("tokenizer", model_name)
        self.tokenizer_kwargs = {"trust_remote_code": trust_remote_code}

        # Load processor
        self.processor_kwargs = {
            # "tokenize": kwargs.pop("tokenize", False),
            # "add_generation_prompt": kwargs.pop("add_generation_prompt", True),
            # "enable_thinking": kwargs.pop("enable_thinking", False)
        }

        engine_defaults = {
            "tensor_parallel_size": 1,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": True,
            "disable_log_stats": True,
            "swap_space": 4,
            "seed": 0,
        }
        engine_cfg = dict(engine_kwargs or {})
        if "vllm" in engine_cfg:
            engine_cfg = dict(engine_cfg.get("vllm") or {})
        engine_cfg = {**engine_defaults, **engine_cfg}

        self.enable_sleep_mode = bool(engine_cfg.get("enable_sleep_mode", False))

        self.model_kwargs = {
            "tokenizer": tokenizer_name,
            "trust_remote_code": trust_remote_code,
            "dtype": torch_dtype,
            **engine_cfg,
        }

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **self.tokenizer_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name, **self.processor_kwargs)
        self.llm = LLM(model=model_name, **self.model_kwargs)   
    
        # Sampling parameters
        sampling_defaults = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_tokens": 512,
        }
        sampling_cfg = dict(sampling_params or {})
        # Always remove token-count aliases; use them as fallback for max_tokens only
        # if max_tokens is not already present. This handles the case where both
        # max_new_tokens (from a base config) and max_tokens (from an override) are
        # present after a deep-merge.
        for alias in ("max_new_tokens", "response_length"):
            if alias in sampling_cfg:
                val = sampling_cfg.pop(alias)
                sampling_cfg.setdefault("max_tokens", val)
        sampling_cfg.pop("prompt_length", None)
        sampling_cfg = {**sampling_defaults, **sampling_cfg}
        self._print_init_config(
            model_name=model_name,
            torch_dtype=torch_dtype,
            tokenizer_name=tokenizer_name,
            trust_remote_code=trust_remote_code,
            engine_kwargs=engine_cfg,
            sampling_params=sampling_cfg,
        )
        self.sampling_params = SamplingParams(**sampling_cfg)

        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"QwenVLModel loaded successfully on device: {self.device}")
        logging.info(f"Model dtype: {torch_dtype}")

    @staticmethod
    def _print_init_config(
        *,
        model_name: str,
        torch_dtype: str,
        tokenizer_name: str,
        trust_remote_code: bool,
        engine_kwargs: Dict[str, Any],
        sampling_params: Dict[str, Any],
    ) -> None:
        print("QwenVLModel init configuration:")
        print(f"  model_name: {model_name}")
        print(f"  torch_dtype: {torch_dtype}")
        print(f"  tokenizer: {tokenizer_name}")
        print(f"  trust_remote_code: {trust_remote_code}")
        print("  engine_kwargs:")
        print(pprint.pformat(engine_kwargs, indent=4, width=120))
        print("  sampling_params:")
        print(pprint.pformat(sampling_params, indent=4, width=120))

    # -------------- Required interface methods --------------

    def get_model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def set_default_model_args(self, **updates: Any) -> None:
        """Update default model arguments."""
        self.default_model_args.update({k: v for k, v in updates.items() if v is not None})

    def sleep(self, level: int = 1) -> bool:
        """
        Put the vLLM engine into sleep mode if enabled at init.
        Returns True if a sleep attempt was made.
        """
        if not hasattr(self, "llm") or not hasattr(self.llm, "sleep"):
            return False
        try:
            self.llm.sleep(level=level)
        except TypeError:
            self.llm.sleep()
        return True

    def wake_up(self) -> bool:
        """
        Wake the vLLM engine if sleep mode was enabled at init.
        Returns True if a wake attempt was made.
        """
        if not hasattr(self, "llm") or not hasattr(self.llm, "wake_up"):
            return False
        self.llm.wake_up()
        return True

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
                # Keep optional metadata passthrough; do not require constant sample_fps.
                for key in ("sample_fps", "fps", "num_frames", "timestamps", "distances"):
                    if key in content:
                        payload[key] = content[key]
                return payload
            if not isinstance(content, (list, tuple)) or not content:
                raise ValueError("video content must be a non-empty list/tuple of PIL images or path strings")
            video_items: List[Any] = []
            for idx, item in enumerate(content):
                if isinstance(item, Image.Image):
                    video_items.append(item)
                elif isinstance(item, str):
                    video_items.append(item)
                else:
                    raise ValueError(
                        f"video content[{idx}] must be PIL.Image or str path/URI, got {type(item)!r}"
                    )
            return {"type": "video", "video": video_items}
        raise ValueError(f"Unsupported content type: {q.type}")

    def compile_prompt(self, prompts: List[ChatQuery]) -> List[Dict[str, Any]]:
        msgs = []
        cur_role, cur_content = None, []
        for p in prompts:
            if p.role != cur_role:
                if cur_role is not None:
                    msgs.append({"role": cur_role, "content": cur_content})
                cur_role, cur_content = p.role, []
            cur_content.append(self.format_content(p))
        if cur_role is not None:
            msgs.append({"role": cur_role, "content": cur_content})
        return { 'contents': msgs }
    
    @staticmethod
    def prepare_inputs_for_vllm(messages, processor):
        tmpl_kw = {"tokenize": False, "add_generation_prompt": True}
        try:
            text = processor.apply_chat_template(
                messages,
                **tmpl_kw,
                enable_thinking=False,
            )
        except TypeError:
            text = processor.apply_chat_template(messages, **tmpl_kw)
        # qwen_vl_utils 0.0.14+ reqired
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs

        return {
            'prompt': text,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs
        }

    def generate_batch_response(
        self,
        instructions: str,
        inputs: List[Any],
        output_format: OutputFormat,
        meta: List[Dict[str, Any]] | None = None,
        **kwargs: Any
    ) -> List[UnifiedEnvelope | None]:
        """
        Generate responses in a single batch using Qwen VL model.
        """
        return self.generate_responses(
            instructions,
            inputs,
            output_format,
            meta=meta,
            batch_size=len(inputs),
            **kwargs,
        )

    def generate_responses(
        self,
        instructions: str,
        inputs: List[Any],
        output_format: OutputFormat,
        meta: List[Dict[str, Any]] | None = None,
        batch_size: int | None = None,
        **kwargs: Any
    ) -> List[UnifiedEnvelope | None]:
        """
        Generate responses in minibatches using Qwen VL model.
        """
        max_retries = int(kwargs.pop("max_retries", 3))
        _ = kwargs.pop("timeout", self.timeout)

        if isinstance(inputs[0], dict):
            inputs = [item['contents'] for item in inputs]

        compiled_inputs = []
        for item in inputs:
            if item and isinstance(item[0], ChatQuery):
                compiled_inputs.append(self.compile_prompt(item))
            else:
                compiled_inputs.append(item)

        messages_list = []
        for messages in compiled_inputs:
            msgs = list(messages)
            if instructions and msgs[0]['role'] != 'system':
                msgs.insert(0, {
                    "role": "system",
                    "content": instructions
                })
            messages_list.append(msgs)

        if batch_size is None or batch_size <= 0:
            batch_size = len(messages_list)

        for attempt in range(max_retries):
            try:
                all_results: List[UnifiedEnvelope | None] = []
                parse_failures: List[str] = []
                for start in range(0, len(messages_list), batch_size):
                    batch_msgs = messages_list[start:start + batch_size]
                    vllm_inputs = [
                        self.prepare_inputs_for_vllm(message, self.processor)
                        for message in batch_msgs
                    ]

                    outputs = self.llm.generate(
                        vllm_inputs, sampling_params=self.sampling_params, use_tqdm=False
                    )

                    raw_texts = [
                        output.outputs[0].text if output.outputs else ""
                        for output in outputs
                    ]

                    for idx, output_text in enumerate(raw_texts):
                        usage = {
                            "input_tokens": 0,
                            "output_tokens": len(output_text.split()),
                            "cached_tokens": 0,
                        }

                        meta_item = (
                            meta[start + idx]
                            if meta is not None and start + idx < len(meta)
                            else {}
                        )
                        try:
                            all_results.append(parse_and_unify(
                                output_text,
                                output_format,
                                meta=meta_item,
                                model_name=self._model_name,
                                usage=usage,
                            ))
                        except Exception as parse_exc:
                            preview = output_text[:600].replace("\n", "\\n")
                            parse_failures.append(
                                f"batch_item={start + idx} parse_error={parse_exc}; raw_preview={preview}"
                            )
                            all_results.append(None)

                    if len(outputs) < len(batch_msgs):
                        all_results.extend([None] * (len(batch_msgs) - len(outputs)))

                if all(result is None for result in all_results):
                    detail = parse_failures[-1] if parse_failures else "model returned no parseable outputs"
                    raise RuntimeError(
                        f"All responses failed to parse for {output_format.value}. {detail}"
                    )

                return all_results
            except Exception as e:
                logging.warning(
                    f"Error in generate_responses attempt {attempt+1}/{max_retries}: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        raise Exception(f"Failed to generate responses after {max_retries} retries")

    def generate_response(
        self, 
        instructions: str, 
        inputs: List[Dict[str, Any]], 
        output_format: OutputFormat,
        meta: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> UnifiedEnvelope:
        """
        Generate a single response using Qwen VL model.
        """
        responses = self.generate_responses(
            instructions,
            [inputs],
            output_format,
            meta=[meta or {}],
            batch_size=1,
            **kwargs,
        )
        response = responses[0] if responses else None
        if response is None:
            raise RuntimeError(
                f"Failed to generate response for output_format={output_format.value}. "
                "Check preceding parse_error/raw_preview logs."
            )
        return response
    
    @staticmethod
    def _create_usage(input_tokens: int, output_tokens: int, cached_tokens: int = 0):
        """Create a usage object."""
        @dataclass
        class InputTokensDetails:
            cached_tokens: int = 0
        
        @dataclass
        class Usage:
            input_tokens: int
            output_tokens: int
            input_tokens_details: InputTokensDetails
        
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens)
        )

    @staticmethod
    def get_cost(
        model_name: str,
        input_tokens: int = 0,
        cached_tokens: int = 0,
        output_tokens: int = 0
    ) -> tuple[float, Dict[str, float]]:
        """
        Calculate cost for Qwen model usage.
        For local models, cost is 0.
        
        Args:
            model_name: Name of the model
            input_tokens: Number of input tokens
            cached_tokens: Number of cached tokens
            output_tokens: Number of output tokens
            
        Returns:
            Tuple of (total_cost, cost_breakdown)
        """
        # Local model has no API cost
        return 0.0, {
            "input_cost": 0.0,
            "cached_cost": 0.0,
            "output_cost": 0.0
        }
