# foresight/models/vlms/providers/gemini_infer.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from google import genai
from google.genai import types
from foresight.models.vlms.geminimodel import GeminiModel

def create(
    *,
    model: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    **kwargs
):
    """
    Build an Gemini's OpenAI-compatible endpoint.
    The instance exposes:
      - preprocess(text/images/file_ids/...) -> messages
      - generate(messages, **responses_kwargs) -> str
      - upload_file(path) -> file_id
      - to_messages(prompt) -> messages
      - set_default_model_args(...)
    """
    resolved_api_key = (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )

    return GeminiModel(
        model=model,
        sampling_params=kwargs.get('sampling_params', {}),
    )


def create_gemini_pro(**kwargs: Any):
    """Convenience alias: same as create(model='gemini-1.5-pro', ...)."""
    model = kwargs.pop("model",  "gemini-3-flash-preview")
    return create(model=model, **kwargs)
