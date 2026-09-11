from __future__ import annotations

from typing import Any, Dict, Optional

from foresight.models.vlms.qwenvlrewardmodel import QwenVLRewardModel


def create(
    *,
    model: str = "Qwen/Qwen3-VL-2B-Instruct",
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    timeout: float = 60.0,
    trust_remote_code: bool = True,
    default_model_args: Optional[Dict[str, Any]] = None,
    sampling_params: Optional[Dict[str, Any]] = None,
    engine_kwargs: Optional[Dict[str, Any]] = None,
    apply_chat_template_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> QwenVLRewardModel:
    """
    Build token-classification Qwen reward adapter.

    `default_model_args`, `sampling_params`, and `engine_kwargs` are accepted for
    config compatibility with other providers, but are not used by this
    discriminative reward-model path.
    """
    _ = default_model_args
    _ = sampling_params
    _ = engine_kwargs
    return QwenVLRewardModel(
        model_name=model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        timeout=timeout,
        trust_remote_code=trust_remote_code,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
        **kwargs,
    )
