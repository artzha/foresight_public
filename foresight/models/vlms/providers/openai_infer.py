# foresight/models/vlms/providers/openai_infer.py
from __future__ import annotations
from typing import Any, Dict, Optional
from foresight.models.vlms.openaimodel import OpenAIModel

def create(
    *,
    model: str = "gpt-5",
    default_model_args: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 60.0,
    service_tier: Optional[str] = "flex",
    default_role: str = "user",
    **kwargs
) -> OpenAIModel:
    """
    Build your OpenAIModel with sensible defaults. The instance exposes:
      - preprocess(text/images/file_ids/...) -> messages
      - generate(messages, **responses_kwargs) -> str
      - upload_file(path) -> file_id
      - to_messages(prompt) -> messages
      - set_default_model_args(...)
    """
    dma = {"model": model}
    if default_model_args:
        dma.update(default_model_args)

    return OpenAIModel(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        service_tier=service_tier,
        default_model_args=dma,
        default_role=default_role,
        **kwargs
    )

def create_gpt5(**kwargs: Any) -> OpenAIModel:
    """Convenience alias: same as create(model='gpt-5', ...)."""
    return create(model="gpt-5", **kwargs)