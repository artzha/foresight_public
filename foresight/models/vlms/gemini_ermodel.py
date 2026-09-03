from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from joblib import Parallel, delayed
from PIL import Image

from cotnav.prompts.interface import (
    ChatQuery,
    ContentType,
    OutputFormat,
    UnifiedEnvelope,
    parse_and_unify,
    schema_for,
)
from cotnav.utils.log import logging


@dataclass
class GeminiERConfig:
    model_name: str = "gemini-robotics-er-1.6-preview"
    max_output_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    thinking_budget: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def sampling_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "max_output_tokens": int(self.max_output_tokens),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
        }
        if int(self.top_k) >= 0:
            kwargs["top_k"] = int(self.top_k)
        kwargs.update(self.extra)
        return kwargs


class GeminiERModel:
    def __init__(
        self,
        *,
        model: str = "gemini-robotics-er-1.6-preview",
        sampling_params: Optional[Dict[str, Any]] = None,
        gemini_api_key: str = "",
        timeout: float = 120.0,
        thinking_budget: int = 0,
        max_img_h: int = 224,
        max_img_w: int = 392,
        **kwargs: Any,
    ) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required for Gemini ER. Install with `pip install google-genai`."
            ) from exc

        self._types = types
        self.model = model
        self.timeout = float(timeout)
        self.max_img_h = int(max_img_h)
        self.max_img_w = int(max_img_w)
        self.config = GeminiERConfig(model_name=model, thinking_budget=int(thinking_budget))

        sampling_defaults = self.config.sampling_kwargs()
        sampling_cfg = dict(sampling_params or {})
        if "max_new_tokens" in sampling_cfg:
            legacy = int(sampling_cfg.pop("max_new_tokens"))
            sampling_cfg.setdefault("max_output_tokens", legacy)
        self.sampling_params = {**sampling_defaults, **sampling_cfg}

        api_key = self._resolve_api_key(gemini_api_key)
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=max(1, int(round(self.timeout * 1000.0)))
            ),
        )

        self.default_model_args: Dict[str, Any] = dict(kwargs.get("default_model_args") or {})

    @staticmethod
    def _resolve_api_key(api_key: str) -> str:
        explicit = str(api_key).strip() if api_key is not None else ""
        if explicit:
            return explicit
        env_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get(
            "GOOGLE_API_KEY", ""
        ).strip()
        if env_key:
            return env_key
        raise RuntimeError(
            "Gemini API key is required. Set `gemini_api_key`, `GEMINI_API_KEY`, or `GOOGLE_API_KEY`."
        )

    def get_model_name(self) -> str:
        return self.model

    def set_default_model_args(self, **updates: Any) -> None:
        self.default_model_args.update({k: v for k, v in updates.items() if v is not None})

    @staticmethod
    def _norm_role(role: Any) -> str:
        from cotnav.prompts.interface import Role
        value = role.value if isinstance(role, Role) else role
        if value in ("assistant", "model"):
            return "model"
        if value in ("user", "human"):
            return "user"
        if value in ("system", "developer"):
            return "system"
        return str(value)

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to fit within (max_img_h, max_img_w) preserving aspect ratio."""
        w, h = image.size
        if w <= self.max_img_w and h <= self.max_img_h:
            return image
        scale = min(self.max_img_w / w, self.max_img_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return image.resize((new_w, new_h), Image.LANCZOS)

    @staticmethod
    def _image_to_png_bytes(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()

    def _pil_to_part(self, image: Image.Image) -> Any:
        """Resize image and return a types.Part for PNG bytes."""
        img = self._resize_image(image)
        return self._types.Part.from_bytes(
            data=self._image_to_png_bytes(img), mime_type="image/png"
        )

    def compile_prompt(self, prompts: List[ChatQuery]) -> Dict[str, Any]:
        contents: List[Dict[str, Any]] = []
        prev_role: Optional[str] = None
        current_message: Optional[Dict[str, Any]] = None

        for q in prompts:
            role = self._norm_role(q.role)
            if role == "system":
                continue

            # Build a list of Parts for this ChatQuery (TEXT → 1, IMAGE → 1, VIDEO → N)
            new_parts: List[Any] = []

            if q.type == ContentType.TEXT:
                new_parts.append(self._types.Part.from_text(text=str(q.content)))

            elif q.type == ContentType.IMAGE:
                if not isinstance(q.content, Image.Image):
                    raise ValueError(
                        f"Expected PIL.Image for IMAGE content, got {type(q.content)!r}"
                    )
                new_parts.append(self._pil_to_part(q.content))

            elif q.type == ContentType.VIDEO:
                # Gemini has no native PIL-frames-as-video type; unpack each frame
                # as a separate image/png Part so the model sees the temporal sequence.
                frames = q.content
                if isinstance(frames, dict):
                    frames = frames.get("video", [])
                if not isinstance(frames, (list, tuple)) or not frames:
                    raise ValueError(
                        f"VIDEO content must be a non-empty list/tuple or dict with 'video' key, "
                        f"got {type(q.content)!r}"
                    )
                for idx, frame in enumerate(frames):
                    if not isinstance(frame, Image.Image):
                        raise ValueError(
                            f"VIDEO frame[{idx}] must be PIL.Image, got {type(frame)!r}"
                        )
                    new_parts.append(self._pil_to_part(frame))

            else:
                raise ValueError(f"Unsupported content type for Gemini ER: {q.type!r}")

            if current_message is None or role != prev_role:
                current_message = {"role": role, "parts": new_parts}
                contents.append(current_message)
                prev_role = role
            else:
                current_message["parts"].extend(new_parts)

        if not contents:
            raise ValueError("Prompt compilation produced empty contents.")
        return {"contents": contents}

    def generate_batch_response(
        self,
        instructions: str,
        inputs: List[Any],
        output_format: OutputFormat,
        meta: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[Optional[UnifiedEnvelope]]:
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
        meta: Optional[List[Dict[str, Any]]] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Optional[UnifiedEnvelope]]:
        """Threaded fan-out over `inputs`.

        Each item still runs sequentially through `max_retries` attempts, but
        items within a chunk are dispatched concurrently via joblib's threading
        backend (Gemini calls are I/O-bound, so threads release the GIL during
        HTTP). Results are returned in input order.
        """
        if not inputs:
            return []

        n_jobs = max(
            1,
            int(kwargs.pop("n_jobs", kwargs.pop("single_response_n_jobs", 4))),
        )
        max_retries = int(kwargs.pop("max_retries", 3))
        if batch_size is None or int(batch_size) <= 0:
            batch_size = len(inputs)
        batch_size = int(batch_size)

        ordered_results: List[Optional[UnifiedEnvelope]] = [None] * len(inputs)

        def _one_call(global_idx: int, input_item: Any) -> tuple[int, Optional[UnifiedEnvelope]]:
            meta_item = (
                meta[global_idx]
                if isinstance(meta, list) and global_idx < len(meta)
                else {}
            )
            # Per-call kwargs copy: `generate_response` reads `thinking_budget`
            # / `sampling_params` from kwargs, so each thread/attempt should see
            # an unmutated dict.
            base_kwargs = dict(kwargs)
            parse_failures: List[str] = []
            for attempt in range(max_retries):
                try:
                    parsed = self.generate_response(
                        instructions,
                        input_item,
                        output_format,
                        meta=meta_item,
                        **dict(base_kwargs),
                    )
                    if parsed is not None:
                        return global_idx, parsed
                    parse_failures.append(f"attempt={attempt} returned None")
                except Exception as exc:
                    parse_failures.append(f"attempt={attempt} exception={exc}")
                    logging.warning(
                        f"GeminiERModel.generate_responses error "
                        f"(idx={global_idx}, attempt {attempt + 1}/{max_retries}): {exc}"
                    )
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

            if parse_failures:
                logging.warning(
                    f"GeminiERModel: all {max_retries} attempts failed for item "
                    f"{global_idx}; failures={parse_failures}"
                )
            return global_idx, None

        for start in range(0, len(inputs), batch_size):
            chunk = inputs[start : start + batch_size]
            chunk_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_one_call)(start + local_idx, input_item)
                for local_idx, input_item in enumerate(chunk)
            )
            for global_idx, result in chunk_results:
                ordered_results[global_idx] = result

        return ordered_results

    def generate_response(
        self,
        instructions: str,
        inputs: Any,
        output_format: OutputFormat,
        meta: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[UnifiedEnvelope]:
        request = inputs
        if isinstance(request, list) and request and isinstance(request[0], ChatQuery):
            request = self.compile_prompt(request)

        if isinstance(request, dict):
            contents = request.get("contents", request)
        elif isinstance(request, list):
            contents = request
        else:
            raise ValueError("inputs must be a request dict, contents list, or ChatQuery list")

        schema_cls = schema_for(output_format, return_cls=True)

        thinking_budget = int(kwargs.get("thinking_budget", self.config.thinking_budget))
        config_kwargs: Dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": schema_cls,
            "thinking_config": self._types.ThinkingConfig(
                thinking_budget=thinking_budget
            ),
        }
        if instructions:
            config_kwargs["system_instruction"] = instructions

        call_sampling = dict(self.sampling_params)
        call_sampling.update(self.default_model_args)
        if "sampling_params" in kwargs and isinstance(kwargs["sampling_params"], dict):
            call_sampling.update(kwargs["sampling_params"])
        config_kwargs.update(call_sampling)

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self._types.GenerateContentConfig(**config_kwargs),
        )

        raw_text = getattr(response, "text", None)
        if raw_text is None:
            try:
                raw_text = response.candidates[0].content.parts[0].text
            except Exception:
                raw_text = ""

        usage = getattr(response, "usage_metadata", None)
        usage_obj = {
            "input_tokens": getattr(usage, "prompt_token_count", 0) if usage is not None else 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) if usage is not None else 0,
            "cached_tokens": getattr(usage, "cached_content_token_count", 0) if usage is not None else 0,
        }

        try:
            return parse_and_unify(
                raw_text,
                output_format,
                model_name=self.model,
                usage=usage_obj,
                meta=meta or {},
            )
        except Exception as exc:
            preview = (raw_text or "")[:300].replace("\n", "\\n")
            logging.warning(
                f"GeminiERModel.generate_response parse failure "
                f"(format={output_format.value}): {exc}; raw_preview={preview!r}"
            )
            return None

    @staticmethod
    def get_cost(
        model_name: str,
        input_tokens: int = 0,
        cached_tokens: int = 0,
        output_tokens: int = 0,
    ) -> tuple[float, Dict[str, float]]:
        _ = (model_name, input_tokens, cached_tokens, output_tokens)
        return 0.0, {
            "input_cost": 0.0,
            "cached_cost": 0.0,
            "output_cost": 0.0,
        }
