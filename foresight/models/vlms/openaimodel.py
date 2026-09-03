# openaimodel.py
from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union
import os, time, random
import hashlib
from dotenv import load_dotenv
load_dotenv()

import io
import json
import PIL
import copy
from PIL import Image
from pathlib import Path
from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError
import tempfile, numpy as np, os

from cotnav.prompts.interface import ( 
    ContentType, Role, ChatQuery, 
    OutputFormat, parse_and_unify, schema_for, UnifiedEnvelope
)
from cotnav.utils.log import logging
from pydantic import BaseModel

class ResponsesMessage:
    def input_text_message(self, text):
        return {
            "type": "input_text",
            "text": text
        }
    def file_message(self, file_data, filename):
        return {
            "type": "input_file",
            "file_data": file_data,
            "filename": filename
        }
    def input_file_id(self, file_id):
        return {
            "type": "input_image",
            "file_id": file_id,
            "detail": "high"
        }
    def output_text_message(self, text):
        return {
            "type": "output_text",
            "text": text
        }

RM = ResponsesMessage()

@dataclass
class OpenAIResponse:
    """Response object matching expected interface."""
    parsed_output: Any
    usage: Any
    
    def __init__(self, parsed_output, usage=None):
        self.parsed_output = parsed_output
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

class OpenAIModel:
    """
    Minimal Responses-API wrapper.
    Now also owns:
      - default_model_args (persisted per instance)
      - preprocess(...) to build messages
      - generate(messages, **kwargs) to return output_text
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        service_tier: Optional[str] = None,  # e.g., "flex"
        default_model_args: Optional[Dict[str, Any]] = None,
        default_role: str = "user",
        **kwargs,
    ):
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )
        self.timeout = timeout
        self.service_tier = service_tier
        self.default_model_args = default_model_args or {}
        self._default_role = default_role
        self._image_file_cache: dict[str, str] = {}

        self.model_kwargs = kwargs

    # -------------- Owned conveniences --------------

    def get_model_name(self) -> str:
        return self.default_model_args.get("model", "unknown")

    def set_default_model_args(self, **updates: Any) -> None:
        self.default_model_args.update({k: v for k, v in updates.items() if v is not None})

    def preprocess_image(self, img):
        assert isinstance(img, Image.Image), "Only supports preprocess PIL.Image for now" 

        max_dim = self.model_kwargs.get("max_pixel_dim", 224)

        w, h = img.size
        largest = max(w, h)
        if largest > max_dim:
            scale = max_dim / float(largest)
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            img = img.resize(new_size, resample=Image.LANCZOS)
        return img

    def preprocess_video(self, video):
        # OpenAI vision files endpoint does not currently accept mp4 uploads.
        # Treat videos as frame sequences and send as multiple input_image items.
        if isinstance(video, (list, tuple)):
            frames = [_frame_to_uint8_rgb(frame) for frame in video]
            if len(frames) == 0:
                raise ValueError("Video frame list is empty")
            return [Image.fromarray(frame, mode="RGB") for frame in frames]
        raise ValueError(
            "OpenAI video input expects a list/tuple of frames (PIL/numpy) in this wrapper."
        )

    def format_content(self, prompt):
        if prompt.type == ContentType.TEXT:
            # Use output_text for assistant turns, input_text otherwise
            if str(prompt.role) == Role.ASSISTANT.value or str(prompt.role) == "assistant":
                return RM.output_text_message(prompt.content)
            else:
                return RM.input_text_message(prompt.content)

        elif prompt.type == ContentType.IMAGE:
            # images are user inputs; upload & reference by file_id
            image = self.preprocess_image(prompt.content)
            file_id = self.get_or_upload_image_file_id(image)
            return RM.input_file_id(file_id)

        elif prompt.type == ContentType.VIDEO:
            # Send videos as multiple image frames.
            frames = self.preprocess_video(prompt.content)
            parts = []
            for frame in frames:
                file_id = self.get_or_upload_image_file_id(frame)
                parts.append(RM.input_file_id(file_id))
            return parts

        else:
            return None

    def compile_prompt(self, prompts: List[ChatQuery]):
        valid_roles = {"assistant", "system", "developer", "user"}
        messages = []
        for prompt in prompts:
            part = self.format_content(prompt)
            assert part is not None, "Missing part in compile_prompt()"
            role = str(prompt.role.value if hasattr(prompt.role, "value") else prompt.role)
            if role not in valid_roles:
                raise ValueError(f"Unsupported role for OpenAI Responses API: {role}")
            if isinstance(part, list):
                content = part
            else:
                content = [part]
            messages.append({"role": role, "content": content})
        return {'contents': messages}

    def generate_response(
        self, 
        instructions: str, 
        inputs: Dict[str, Any],
        output_format: OutputFormat,
        meta: Dict[str, Any] | {} = {},
        **kwargs: Any
    ) -> UnifiedEnvelope:
        """
        One-call text generation using this instance's defaults.
        Per-call overrides:
          - instructions
          - model_args={...}  (merged into this instance's defaults for THIS CALL only)
          - timeout, service_tier, max_retries, auto_bump_tokens, max_bump_cap
          - any Responses params (e.g., max_output_tokens, stop_sequences, temperature/top_p if supported)
        """
        schema = schema_for(output_format, return_cls=True)
        model_args = kwargs.pop("model_args", None)
        if model_args is None:
            model_args = copy.deepcopy(self.default_model_args)
        assert model_args is not None, "Model args were not provided"

        timeout       = kwargs.pop("timeout", self.timeout)
        service_tier  = kwargs.pop("service_tier", None)
        max_retries   = int(kwargs.pop("max_retries", 3))
        auto_bump     = kwargs.pop("auto_bump_tokens", True)
        max_bump_cap  = int(kwargs.pop("max_bump_cap", 8192))

        # Strict contract: compile_prompt returns {"contents": [...]}
        assert 'contents' in inputs, "Expected dict input with key 'contents'"
    
        for i in range(max_retries):
            tmp_model_args = copy.deepcopy(model_args)
            try:
                response = self.client.responses.parse(
                    **tmp_model_args,
                    instructions=instructions,
                    input=inputs['contents'],
                    service_tier=service_tier,
                    text_format=schema
                )
                if response.output_parsed.model_dump() is None:
                    continue

                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cached_tokens": response.usage.input_tokens_details.cached_tokens
                }
                output_text = response.output_parsed.to_unified(meta)

                return parse_and_unify(
                    output_text,
                    output_format,
                    meta=meta,
                    model_name=self.get_model_name(),
                    usage=usage
                )
            except Exception as e:
                logging.warning(f"Error in response {i+1}/{max_retries}: {e}")
                sleep_time = 5 * (2**i) + random.uniform(1, 2)
                logging.info(f"Sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
                logging.info(f"Retrying...")
            
        raise Exception(f"Failed to get response after {max_retries} retries")

    def _img_to_bytes(self, img: Any) -> bytes:
        buf = _img_to_buf(img)      # has .name = "image.png"
        return buf.getvalue()

    def _hash_bytes(self, b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    def get_or_upload_image_file_id(self, img: Any, purpose: str = "vision") -> str:
        """
        Deduplicate image uploads by content. If an identical image has been uploaded
        during this process lifetime, reuse its file_id.
        """
        # Build the named buffer ONCE so we can both hash and upload it.
        buf = _img_to_buf(img)                # buf.name = "image.png" (important!)
        data = buf.getvalue()
        key = self._hash_bytes(data)

        if key in self._image_file_cache:
            return self._image_file_cache[key]

        # Upload the SAME named buffer so the API sees a real filename/extension
        buf.seek(0)
        f = self.client.files.create(file=buf, purpose=purpose)  # purpose="vision" is best for images
        self._image_file_cache[key] = f.id
        return f.id
    
    def _remove_cached_file_id(self, file_id: str) -> None:
        for key, cached_id in list(self._image_file_cache.items()):
            if cached_id == file_id:
                del self._image_file_cache[key]

    def delete_file(self, file_id: str) -> None:
        try:
            self.client.files.delete(file_id)
            self._remove_cached_file_id(file_id)
        except Exception as exc:
            print(f"[openai] failed to delete file {file_id}: {exc}")

    def cleanup(self) -> None:
        """Delete all cached uploaded files."""
        cached_ids = set(self._image_file_cache.values())
        for file_id in list(cached_ids):
            self.delete_file(file_id)
        self._image_file_cache.clear()

    def generate_batch_response(
        self, 
        instructions: str, inputs: List[str], **kwargs: Any) -> Dict[str, Any]:
        """
        Process multiple requests using OpenAI's Batch API.
        Per-call overrides:
          - endpoint: API endpoint for batch processing
          - completion_window: Batch completion window
          - metadata: Optional metadata for the batch
          - poll_interval: Seconds between status checks
          - model_args: Default model arguments to merge with request bodies
          - timeout, service_tier: Inherited from instance defaults
        
        Args:
            requests: List of request dicts with 'custom_id' and 'body' keys
            
        Returns:
            List of response dicts with results
        """
        # Extract parameters with defaults similar to generate_response
        model_args = kwargs.pop("model_args", self.default_model_args)
        endpoint = kwargs.pop("endpoint", "/v1/responses")
        completion_window = kwargs.pop("completion_window", "24h")
        metadata = kwargs.pop("metadata", None)
        poll_interval = float(kwargs.pop("poll_interval", 30.0))
        
        # Create batch input file
        batch_input_lines = []
        requests = []
        for inp in inputs:
            requests.append({
                "custom_id": str(random.randint(100000, 999999)),
                "method": "POST",
                "url": endpoint,
                "body": {
                    "instructions": instructions,
                    "input": inp,
                    **model_args
                }
            })

        for req in requests:
            batch_input_lines.append(json.dumps(req))
        
        batch_content = "\n".join(batch_input_lines)
        # print(batch_content)
        
        # Upload input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(batch_content)
            f.flush()
            temp_file_name = f.name
            
        with open(temp_file_name, 'rb') as upload_file:
            input_file = self.client.files.create(
                file=upload_file,
                purpose="batch"
            )
        
        os.unlink(temp_file_name)
        
        # Create batch
        batch_kwargs = {
            "input_file_id": input_file.id,
            "endpoint": endpoint,
            "completion_window": completion_window
        }
        if metadata:
            batch_kwargs["metadata"] = metadata
            
        batch = self.client.batches.create(**batch_kwargs)
        
        # Poll for completion
        while batch.status in ["validating", "in_progress", "finalizing"]:
            time.sleep(poll_interval)
            batch = self.client.batches.retrieve(batch.id)
            
        if batch.status != "completed":
            if hasattr(batch, 'error_file_id') and batch.error_file_id:
                try:
                    error_content = self.client.files.content(batch.error_file_id)
                    print(f"Batch failed. Error details:\n{error_content.text}")
                except Exception:
                    pass
            raise RuntimeError(f"Batch failed with status: {batch.status}")
            
        if not batch.output_file_id:
            raise RuntimeError(f"Batch completed but no output file available")
            
        # Download results
        output_content = self.client.files.content(batch.output_file_id)
        output_lines = output_content.text.strip().split('\n')
        
        results = []
        for line in output_lines:
            if line.strip():
                result = json.loads(line)
                results.append(result)
                
        # Cleanup files
        self.delete_file(input_file.id)
        if batch.output_file_id:
            self.delete_file(batch.output_file_id)
        
        return results

    @staticmethod
    def get_cost(
        model_name, input_tokens=0, cached_tokens=0, output_tokens=0
    ):
        """Get the cost of an OpenAI response."""
        # Price table: https://openai.com/api/pricing/
        # Prices are per 1M tokens, as of July 12, 2025.
        if model_name == "o3":
            price_input_tokens = 2.0
            price_cached_tokens = 0.5
            price_output_tokens = 8.0
        elif model_name == "o3-mini":
            price_input_tokens = 1.1
            price_cached_tokens = 0.55
            price_output_tokens = 4.4
        elif model_name == "o4-mini":
            price_input_tokens = 1.1
            price_cached_tokens = 0.275
            price_output_tokens = 4.4
        elif model_name == "gpt-5":
            price_input_tokens = 1.25
            price_cached_tokens = 0.125
            price_output_tokens = 10.0
        elif model_name == "gpt-5.1":
            price_input_tokens = 1.25
            price_cached_tokens = 0.125
            price_output_tokens = 10.0
        elif model_name == "gpt-4o":
            price_input_tokens = 2.50
            price_cached_tokens = 1.25
            price_output_tokens = 10.0
        elif model_name == "gpt-4o-mini":
            price_input_tokens = 0.15
            price_cached_tokens = 0.075
            price_output_tokens = 0.60
        elif model_name == "gpt-5-mini":
            price_input_tokens = 0.25
            price_cached_tokens = 0.03
            price_output_tokens = 2.0
        else:
            raise ValueError(f"Model {model_name} pricing not known")

        input_cost = (input_tokens / 1_000_000) * price_input_tokens
        cached_cost = (cached_tokens / 1_000_000) * price_cached_tokens
        output_cost = (output_tokens / 1_000_000) * price_output_tokens

        total_cost = input_cost + cached_cost + output_cost

        return total_cost, {
            "input_cost": input_cost,
            "cached_cost": cached_cost,
            "output_cost": output_cost
        }

def _img_to_buf(img):

    if isinstance(img, np.ndarray):
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(img, mode="RGB")

    image_format = "PNG"
    buf = io.BytesIO()
    if isinstance(img, Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
    elif isinstance(img, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img)).convert("RGB")
    else:
        try:
            img = Image.fromarray(np.asarray(img)).convert("RGB")
        except Exception:
            raise ValueError("Unsupported image type for _img_to_buf")
    img.save(buf, format=image_format)
    buf.seek(0)
    buf.name = f"image.{image_format.lower()}"
    return buf


def _frame_to_uint8_rgb(frame: Any) -> np.ndarray:
    if isinstance(frame, Image.Image):
        arr = np.asarray(frame.convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
    return arr