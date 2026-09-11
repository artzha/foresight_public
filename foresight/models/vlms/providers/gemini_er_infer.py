from __future__ import annotations

from typing import Any, Dict, Optional

from foresight.models.vlms.gemini_ermodel import GeminiERModel


def create(
    *,
    model: str = "gemini-robotics-er-1.6-preview",
    sampling_params: Optional[Dict[str, Any]] = None,
    gemini_api_key: str = "",
    timeout: float = 120.0,
    thinking_budget: int = 0,
    **kwargs: Any,
) -> GeminiERModel:
    return GeminiERModel(
        model=model,
        sampling_params=sampling_params,
        gemini_api_key=gemini_api_key,
        timeout=timeout,
        thinking_budget=thinking_budget,
        **kwargs,
    )
