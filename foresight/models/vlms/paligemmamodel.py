# paligemmamodel.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable, Union
from pathlib import Path
import time
import re
import json
import io

import torch
import numpy as np
from PIL import Image

from pydantic import BaseModel, Field, conint, constr
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from cotnav.prompts.interface import ContentType, Role, ChatQuery
from cotnav.utils.log_utils import logging


# ---------- Structured output (matches your Qwen wrapper) ----------
class ChoiceReason(BaseModel):
    choice: conint(ge=0, le=9) = Field(description="Index of the path choice.")  # type: ignore
    reason: constr(min_length=3, max_length=512) = Field(description="Brief rationale for this choice.")  # type: ignore


class ReasoningTrace(BaseModel):
    decisions: List[ChoiceReason] = Field(
        description="A list of decisions, where each entry contains the chosen path index and its rationale."
    )


@dataclass
class PaliGemmaResponse:
    """Response object matching expected interface (parallel to QwenResponse)."""
    output_parsed: ReasoningTrace
    usage: Any

    def __init__(self, output_parsed, usage=None):
        self.output_parsed = output_parsed
        if usage is None:
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


class PaliGemmaModel:
    """
    Wrapper for PaliGemma (google/paligemma-3b-mix-224) using local transformers implementation.
    Intentionally mirrors the Qwen VL wrapper interface so pivot_wrapper can treat them the same.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,          # kept for interface compatibility
        base_url: Optional[str] = None,         # kept for interface compatibility
        timeout: float = 30.0,
        service_tier: Optional[str] = None,     # kept for interface compatibility
        default_model_args: Optional[Dict[str, Any]] = None,
        default_role: str = "user",
        model_name: str = "google/paligemma-3b-mix-224",
        torch_dtype: Union[str, torch.dtype] = "auto",
        device_map: Optional[str] = "auto",
        attn_implementation: Optional[str] = None,
    ):
        self.timeout = timeout
        self.service_tier = service_tier
        self.default_model_args = default_model_args or {
            "max_new_tokens": 256,
            "temperature": 0.0,     # deterministic by default for eval reproducibility
            "do_sample": False,
        }
        self._default_role = default_role
        self._model_name = model_name

        logging.info(f"Loading PaliGemma model: {model_name}")

        # Load model + processor
        model_kwargs: Dict[str, Any] = {}
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = torch_dtype
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_name)

        # Select device (for non-accelerate, device_map may still place on cuda:0)
        self.device = next(self.model.parameters()).device
        logging.info(f"PaliGemma loaded successfully on device: {self.device}")
        logging.info(f"Model dtype: {next(self.model.parameters()).dtype}")

    # -------------- Minimal interface parity helpers --------------

    def get_model_name(self) -> str:
        return self._model_name

    def set_default_model_args(self, **updates: Any) -> None:
        self.default_model_args.update({k: v for k, v in updates.items() if v is not None})

    # ---------- Content formatting & prompt compilation (mirrors Qwen) ----------

    def _to_pil(self, obj: Any) -> Image.Image:
        """Robustly coerce common inputs into a PIL.Image."""
        if isinstance(obj, Image.Image):
            return obj
        if isinstance(obj, (str, Path)):
            return Image.open(str(obj)).convert("RGB")
        if isinstance(obj, bytes):
            return Image.open(io.BytesIO(obj)).convert("RGB")
        raise ValueError(f"Unsupported image content type for PaliGemma: {type(obj)}")

    def format_content(self, prompt: ChatQuery) -> Dict[str, Any]:
        """
        Format a ChatQuery part into a normalized dict that compile_prompt can use.
        Intentionally mirrors the Qwen wrapper's shape: {"type": "text"/"image", <field>: ...}
        """
        if prompt.type == ContentType.TEXT:
            return {"type": "text", "text": prompt.content}
        elif prompt.type == ContentType.IMAGE:
            return {"type": "image", "image": self._to_pil(prompt.content)}
        else:
            raise ValueError(f"Unsupported content type: {prompt.type}")

    def compile_prompt(self, prompts: List[ChatQuery]) -> List[Dict[str, Any]]:
        """
        Compile a list of ChatQuery parts into role-grouped messages.
        Output format mirrors the Qwen wrapper so higher layers can reuse the same code:
        [
          {"role": "user", "content": [{"type":"text","text":"..."}, {"type":"image","image": PIL.Image}]},
          {"role": "assistant", "content": [...]},
          ...
        ]
        """
        messages: List[Dict[str, Any]] = []
        current_role: Optional[str] = None
        current_content: List[Dict[str, Any]] = []

        for part in prompts:
            if part.role != current_role:
                if current_role is not None:
                    messages.append({"role": current_role, "content": current_content})
                current_role = part.role
                current_content = []
            current_content.append(self.format_content(part))

        if current_role is not None:
            messages.append({"role": current_role, "content": current_content})

        return messages

    # ---------- Generation (API parity with Qwen wrapper) ----------

    def _messages_to_text_and_images(
        self, instructions: str, messages: List[Dict[str, Any]]
    ) -> tuple[str, List[Image.Image]]:
        """
        Convert role/content messages into a flat text prompt and a list of PIL images.
        PaliGemma does not use a chat template; we build a simple instruction-following prompt.
        """
        # Gather images in order of appearance; concatenate text with role headers
        images: List[Image.Image] = []
        lines: List[str] = []

        if instructions:
            lines.append(f"[SYSTEM]\n{instructions.strip()}\n")

        ROLE_HEADER = {
            "system": "[SYSTEM]",
            "user": "[USER]",
            "assistant": "[ASSISTANT]",
        }

        for msg in messages:
            role = msg.get("role", "user")
            header = ROLE_HEADER.get(role, f"[{role.upper()}]")
            text_chunks: List[str] = []
            for c in msg.get("content", []):
                if c.get("type") == "text":
                    txt = str(c.get("text", "")).strip()
                    if txt:
                        text_chunks.append(txt)
                elif c.get("type") == "image":
                    img = c.get("image", None)
                    if img is not None:
                        images.append(self._to_pil(img))
                        # Drop an explicit marker to hint the model there was an image
                        text_chunks.append("<image>")
            if text_chunks:
                lines.append(f"{header}\n" + "\n".join(text_chunks) + "\n")

        # Final instruction to elicit the desired JSON structure.
        # (Keeps behavior close to your Qwen wrapper which expects ReasoningTrace JSON.)
        lines.append(
            "[ASSISTANT]\n"
            "Respond ONLY with a JSON object of the form:\n"
            '{"decisions":[{"choice":0,"reason":"..."}]}\n'
        )

        full_text = "\n".join(lines).strip()
        return full_text, images

    def generate_response(
        self,
        instructions: str,
        inputs: List[Dict[str, Any]],
        **kwargs: Any
    ) -> PaliGemmaResponse:
        """
        Generate a response using PaliGemma, mirroring the Qwen wrapper flow:
          - Merge model args
          - Prepend system instructions
          - Convert messages -> (text, images)
          - Tokenize, generate, decode
          - Parse ReasoningTrace (JSON), fallback if needed
          - Return usage with input/output token counts
        """
        model_args = kwargs.pop("model_args", self.default_model_args.copy())
        timeout = kwargs.pop("timeout", self.timeout)
        max_retries = int(kwargs.pop("max_retries", 3))
        # Allow ad-hoc overrides
        model_args.update(kwargs)

        # Build prompt + collect images
        messages = inputs.copy()
        text_prompt, images = self._messages_to_text_and_images(instructions, messages)

        logging.info(f"Generating response with PaliGemma: {self._model_name}")
        logging.info(f"Messages: {len(messages)}, Images: {len(images)}")

        # Minimal safety: if no images were provided, PaliGemma still works with text-only.
        for attempt in range(max_retries):
            try:
                model_inputs = self.processor(
                    text=text_prompt,
                    images=images if len(images) > 0 else None,
                    return_tensors="pt",
                    padding=True,
                )
                model_inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in model_inputs.items()}

                input_len = model_inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    generated = self.model.generate(
                        **model_inputs,
                        **model_args
                    )

                # Keep only newly generated tokens (beyond the prompt)
                gen_ids = generated[0][input_len:]
                output_text = self.processor.decode(
                    gen_ids, skip_special_tokens=True
                )

                logging.info(f"Generated output (truncated): {output_text[:200]}...")

                # ---- Parse like in Qwen wrapper ----
                cleaned_text = output_text.strip()
                if cleaned_text.startswith("```"):
                    m = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned_text, re.DOTALL)
                    if m:
                        cleaned_text = m.group(1).strip()
                    else:
                        cleaned_text = re.sub(r'^```(?:json)?\s*', '', cleaned_text)
                        cleaned_text = re.sub(r'\s*```$', '', cleaned_text)

                try:
                    output_dict = json.loads(cleaned_text)
                    parsed_output = ReasoningTrace(**output_dict)
                    if not parsed_output.decisions:
                        logging.warning("Empty decisions; creating fallback.")
                        parsed_output = ReasoningTrace(
                            decisions=[ChoiceReason(choice=0, reason="Model returned empty decisions; fallback to 0")]
                        )
                except Exception as e:
                    logging.warning(f"Failed to parse ReasoningTrace: {e}")
                    logging.warning(f"Raw output: {output_text}")
                    logging.warning(f"Cleaned text: {cleaned_text}")
                    parsed_output = ReasoningTrace(
                        decisions=[ChoiceReason(choice=0, reason=output_text[:512])]
                    )

                # ---- Usage, mirroring Qwen wrapper shape ----
                input_tokens = int(model_inputs["input_ids"].shape[-1])
                output_tokens = int(gen_ids.shape[-1]) if hasattr(gen_ids, "shape") else len(gen_ids)

                usage = self._create_usage(input_tokens=input_tokens, output_tokens=output_tokens)
                return PaliGemmaResponse(output_parsed=parsed_output, usage=usage)

            except Exception as e:
                logging.warning(f"PaliGemma generate attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        raise RuntimeError(f"Failed to generate response after {max_retries} retries")

    # ---------- Cost + usage helpers (identical semantics to Qwen) ----------

    @staticmethod
    def _create_usage(input_tokens: int, output_tokens: int, cached_tokens: int = 0):
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
            input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens),
        )

    @staticmethod
    def get_cost(
        model_name: str,
        input_tokens: int = 0,
        cached_tokens: int = 0,
        output_tokens: int = 0
    ) -> tuple[float, Dict[str, float]]:
        """
        Local model; mirror Qwen wrapper and return zero-cost breakdown.
        """
        return 0.0, {
            "input_cost": 0.0,
            "cached_cost": 0.0,
            "output_cost": 0.0,
        }