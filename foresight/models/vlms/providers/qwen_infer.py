# foresight/models/vlms/providers/qwen_infer.py
from __future__ import annotations
from typing import Any, Dict, Optional
from foresight.models.vlms.qwenvlmodel import QwenVLModel

def create(
    *,
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    default_model_args: Optional[Dict[str, Any]] = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    attn_implementation: Optional[str] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    timeout: float = 60.0,
    **kwargs
) -> QwenVLModel:
    """
    Build QwenVLModel with sensible defaults. The instance exposes:
      - compile_prompt(prompts) -> messages
      - generate_response(instructions, messages, **kwargs) -> QwenResponse
      - get_model_name() -> str
      - get_cost(...) -> (cost, breakdown)
    """
    dma = default_model_args or {
        "max_new_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.9,
    }

    return QwenVLModel(
        model_name=model,
        default_model_args=dma,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        timeout=timeout,
        **kwargs
    )
