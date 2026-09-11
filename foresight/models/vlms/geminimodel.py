#geminimodel.py

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path

import io
import copy
import base64
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from PIL import Image
from joblib import Parallel, delayed
from ...prompts.interface import (
    schema_for,
    OutputFormat, 
    parse_and_unify, 
    ChatQuery, 
    ContentType, 
    Role
)

from google import genai
from google.genai import types

# Keys in JSON Schema that the Gemini structured-output API does not accept.
_GEMINI_UNSUPPORTED_SCHEMA_KEYS = frozenset({"$schema", "title"})


def _clean_schema_for_gemini(schema: Any) -> Any:
    """Recursively remove keys unsupported by the Gemini structured-output API.

    Strips ``$schema`` and ``title`` (top-level metadata Gemini rejects) and
    ``additionalProperties: false`` (the Pydantic closed-object marker).
    ``additionalProperties: {type: ...}`` is kept because it carries the
    value-type for Dict fields.
    """
    if isinstance(schema, dict):
        cleaned = {}
        for k, v in schema.items():
            if k in _GEMINI_UNSUPPORTED_SCHEMA_KEYS:
                continue
            if k == "additionalProperties" and v is False:
                continue
            cleaned[k] = _clean_schema_for_gemini(v)
        return cleaned
    if isinstance(schema, list):
        return [_clean_schema_for_gemini(v) for v in schema]
    return schema

@dataclass
class GeminiConfig:
    model_name: str
    
    # generate defaults
    max_output_tokens: int = 512
    temperature: float = 1.1
    top_p: float = 0.95
    top_k: int = -1

    extra: Dict[str, Any] = field(default_factory=dict)

    def sampling_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.top_k >= 0:
            kwargs["top_k"] = self.top_k
        kwargs.update(self.extra)
        return kwargs

class GeminiModel:
    """
    Wrapper class for Gemini VLMs.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-3-flash-preview",
        sampling_params: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.config = GeminiConfig(model_name=model)

        sampling_defaults = self.config.sampling_kwargs()
        sampling_cfg = sampling_params or {}
        sampling_cfg = {**sampling_defaults, **sampling_cfg}
        self.sampling_params = sampling_cfg

        self.client = genai.Client()

        self.total_cost = 0.0

    def _upload_file_with_retry(
        self,
        buf: io.BytesIO,
        *,
        max_retries: int = 6,
        backoff_base_s: float = 5.0,
        backoff_max_s: float = 120.0,
    ):
        from google.genai.errors import ServerError
        # Snapshot bytes upfront; the SDK terminates the upload session on the
        # original buffer object after a failed attempt, so we must create a
        # brand-new BytesIO for every try rather than just rewinding.
        buf.seek(0)
        buf_bytes = buf.read()
        buf_name = getattr(buf, "name", "image.jpg")

        attempt = 0
        while True:
            fresh_buf = io.BytesIO(buf_bytes)
            fresh_buf.name = buf_name
            try:
                return self.client.files.upload(
                    file=fresh_buf,
                    config=types.UploadFileConfig(
                        display_name=buf_name,
                        mime_type="image/jpeg",
                    ),
                )
            except Exception as err:
                msg = str(err).lower()
                transient = (
                    isinstance(err, ServerError)
                    or "503" in msg
                    or "service unavailable" in msg
                    or "429" in msg
                    or "too many requests" in msg
                    or "rate limit" in msg
                    or "resource exhausted" in msg
                )
                if not transient or attempt >= max_retries:
                    raise
                sleep_s = min(backoff_max_s, backoff_base_s * (2 ** attempt))
                print(
                    f"Gemini file upload transient error ({err.__class__.__name__}), "
                    f"sleeping {sleep_s:.1f}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(sleep_s)
                attempt += 1

    def to_bytes(self, img: Path | Image.Image) -> str:
        if isinstance(img, Path):
            img = Image.open(img)
        elif not isinstance(img, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(img)}")
        
        buffered = io.BytesIO()
        img.convert("RGB").save(buffered, format="JPEG")
        img_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_bytes

    def to_buf(self, img: Path | Image.Image, name: str="image.jpg") -> str:
        if isinstance(img, Path):
            img = Image.open(img)
        elif not isinstance(img, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(img)}")
        
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        buf.seek(0)
        buf.name = name
        return buf
        
    def compile_prompt(
            self, 
            prompts: List[ChatQuery],
            upload_images: bool = True,
            flatten_multiturn: Optional[bool] = None,
        ) -> Dict[str, Any]:
        """
        Convert a list of ChatQuery objects into a Gemini-compatible request payload.
        Returns {"contents": [...]} for a multi-turn conversation.
        System instructions are handled separately by generate_batch_response.
        """
        def _norm_role(q_role: Any) -> str:
            role = q_role.value if isinstance(q_role, Role) else q_role
            if role in ("assistant", "model"):
                return "assistant"
            if role in ("user", "human"):
                return "user"
            if role in ("system", "developer"):
                return "system"
            return str(role)

        if flatten_multiturn is None:
            flatten_multiturn = any(_norm_role(q.role) == "assistant" for q in prompts)

        if flatten_multiturn:
            parts: List[Dict[str, Any]] = []
            seen_user_text = False
            for q in prompts:
                role = _norm_role(q.role)
                if role == "system":
                    continue

                if q.type == ContentType.TEXT:
                    prefix = ""
                    if role == "assistant":
                        prefix = "Assistant: "
                    elif role == "user":
                        prefix = "" if not seen_user_text else "User: "
                    text = f"{prefix}{q.content}"
                    parts.append({"text": text})
                    if role == "user":
                        seen_user_text = True
                elif q.type == ContentType.IMAGE:
                    assert isinstance(q.content, Image.Image), f"Expected PIL Image, got {type(q.content)}"
                    if upload_images:
                        buf = self.to_buf(q.content)
                        file_obj = self._upload_file_with_retry(buf)
                        file_uri = getattr(file_obj, "uri", None) or getattr(file_obj, "name", None)
                        if not file_uri:
                            raise ValueError("Uploaded file did not return a uri or name")
                        parts.append({
                            "fileData": {
                                "fileUri": file_uri,
                                "mimeType": "image/jpeg",
                            }
                        })
                    else:
                        parts.append({
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": self.to_bytes(q.content),
                            }
                        })
                elif q.type == ContentType.VIDEO:
                    # Expand frame sequence into individual image parts.
                    frames = q.content
                    if not isinstance(frames, (list, tuple)) or len(frames) == 0:
                        raise ValueError("VIDEO content must be a non-empty list of PIL Images")
                    for frame in frames:
                        assert isinstance(frame, Image.Image), f"Expected PIL Image frame, got {type(frame)}"
                        if upload_images:
                            buf = self.to_buf(frame)
                            file_obj = self._upload_file_with_retry(buf)
                            file_uri = getattr(file_obj, "uri", None) or getattr(file_obj, "name", None)
                            if not file_uri:
                                raise ValueError("Uploaded file did not return a uri or name")
                            parts.append({
                                "fileData": {
                                    "fileUri": file_uri,
                                    "mimeType": "image/jpeg",
                                }
                            })
                        else:
                            parts.append({
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": self.to_bytes(frame),
                                }
                            })
                else:
                    raise ValueError(f"Unsupported content type: {q.type}")

            assert len(parts) > 0, "Payload contents should not be empty."
            return {"contents": [{"role": "user", "parts": parts}]}

        contents: List[Dict[str, Any]] = []
        prev_role = None
        current_content = None
        for q in prompts:
            role = _norm_role(q.role)
            if role == "system":
                continue
            if role == "assistant":
                role = "model"
            elif role == "user":
                role = "user"
            else:
                raise ValueError(f"Unsupported role for Gemini contents: {role}")

            if q.type == ContentType.TEXT:
                message = {"text": q.content}
                if current_content is None or role != prev_role:
                    current_content = {"parts": [message], "role": role}
                    contents.append(current_content)
                    prev_role = role
                else:
                    current_content["parts"].append(message)
            elif q.type == ContentType.IMAGE:
                assert isinstance(q.content, Image.Image), f"Expected PIL Image, got {type(q.content)}"
                if upload_images:
                    buf = self.to_buf(q.content)
                    file_obj = self._upload_file_with_retry(buf)
                    file_uri = getattr(file_obj, "uri", None) or getattr(file_obj, "name", None)
                    if not file_uri:
                        raise ValueError("Uploaded file did not return a uri or name")
                    message = {
                        "fileData": {
                            "fileUri": file_uri,
                            "mimeType": "image/jpeg",
                        }
                    }
                else:
                    message = {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": self.to_bytes(q.content),
                        }
                    }
                if current_content is None or role != prev_role:
                    current_content = {"parts": [message], "role": role}
                    contents.append(current_content)
                    prev_role = role
                else:
                    current_content["parts"].append(message)
            elif q.type == ContentType.VIDEO:
                # Expand frame sequence into individual image parts within the same turn.
                frames = q.content
                if not isinstance(frames, (list, tuple)) or len(frames) == 0:
                    raise ValueError("VIDEO content must be a non-empty list of PIL Images")
                for frame in frames:
                    assert isinstance(frame, Image.Image), f"Expected PIL Image frame, got {type(frame)}"
                    if upload_images:
                        buf = self.to_buf(frame)
                        file_obj = self._upload_file_with_retry(buf)
                        file_uri = getattr(file_obj, "uri", None) or getattr(file_obj, "name", None)
                        if not file_uri:
                            raise ValueError("Uploaded file did not return a uri or name")
                        frame_msg = {
                            "fileData": {
                                "fileUri": file_uri,
                                "mimeType": "image/jpeg",
                            }
                        }
                    else:
                        frame_msg = {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": self.to_bytes(frame),
                            }
                        }
                    if current_content is None or role != prev_role:
                        current_content = {"parts": [frame_msg], "role": role}
                        contents.append(current_content)
                        prev_role = role
                    else:
                        current_content["parts"].append(frame_msg)
            else:
                raise ValueError(f"Unsupported content type: {q.type}")

            assert len(contents) > 0, "Payload contents should not be empty."

        return {"contents": contents}

    def save_batch_request_file(
        self,
        requests: List[Dict[str, Any]],
        output_path: str | Path,
        *,
        system_instruction: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        key_prefix: str = "request-",
    ) -> Path:
        """
        Save Gemini batch requests to a JSONL file.
        Each line contains a request object matching the Gemini Batch API file format.
        """
        if not isinstance(requests, list) or not requests:
            raise ValueError("requests must be a non-empty list")

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = []
        for idx, req in enumerate(requests, start=1):
            if not isinstance(req, dict):
                raise ValueError("Each request must be a dict")
            request = dict(req)
            key = request.pop("key", f"{key_prefix}{idx}")
            
            if system_instruction is not None:
                request['systemInstruction'] = { 'parts': [ { 'text': system_instruction } ] }

            if generation_config is not None:
                request['generationConfig'] = generation_config

            content_req = {"key": key, "request": request}
            lines.append(json.dumps(content_req, ensure_ascii=True))

        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out_path

    def generate_responses(
        self,
        instructions: str,
        inputs: List[Any],
        output_format: OutputFormat,
        meta: Optional[List[Dict[str, Any]]] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Any]:
        """
        Parallel interactive generation for Gemini with stable input ordering.
        Uses joblib threading backend to avoid process/client pickling issues.
        """
        if not isinstance(inputs, list) or len(inputs) == 0:
            return []

        n_jobs = max(1, int(kwargs.pop("n_jobs", kwargs.pop("single_response_n_jobs", 4))))
        max_retries = int(kwargs.pop("max_retries", 2))
        if batch_size is None or int(batch_size) <= 0:
            batch_size = len(inputs)
        batch_size = int(batch_size)

        ordered_results: list[Any | None] = [None] * len(inputs)

        def _one_call(global_idx: int, input_item: Any) -> tuple[int, Any | None]:
            meta_item = meta[global_idx] if isinstance(meta, list) and global_idx < len(meta) else {}
            for attempt in range(max_retries + 1):
                try:
                    response = self.generate_response(
                        instructions=instructions,
                        input_item=input_item,
                        output_format=output_format,
                        meta=meta_item,
                        **kwargs,
                    )
                    return global_idx, response
                except Exception as exc:
                    if attempt >= max_retries:
                        print(
                            f"[GeminiModel] generate_responses failed for idx={global_idx} "
                            f"after {max_retries + 1} attempts: {exc}"
                        )
                        return global_idx, None
                    sleep_s = min(30.0, 2.0 ** attempt)
                    time.sleep(sleep_s)
            return global_idx, None

        for start in range(0, len(inputs), batch_size):
            chunk_inputs = inputs[start:start + batch_size]
            chunk_results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_one_call)(start + local_idx, input_item)
                for local_idx, input_item in enumerate(chunk_inputs)
            )
            for global_idx, result in chunk_results:
                ordered_results[global_idx] = result

        return ordered_results

    def generate_batch_response(
        self,
        instructions: str,
        inputs: List[Any],
        output_format: OutputFormat,
        **kwargs,
    ):
        """
        Process multiple requests using Gemini's Batch API.
        """
        max_retries = int(kwargs.pop("max_retries", 5))
        backoff_base_s = float(kwargs.pop("backoff_base_s", 5.0))
        backoff_max_s = float(kwargs.pop("backoff_max_s", 120.0))
        max_batch_size = kwargs.pop("max_batch_size", 1000)

        schema_cls = schema_for(output_format, return_cls=True)
        schema_json_dict = _clean_schema_for_gemini(schema_cls.model_json_schema())
        
        requests_file = kwargs.pop("requests_file", "batch_requests.jsonl")
        key_prefix = kwargs.pop("key_prefix", "request-")

        generation_config = types.GenerationConfig(
            responseMimeType='application/json',
            responseSchema=schema_json_dict,
        ).to_json_dict()

        batches = [inputs]
        if max_batch_size and len(inputs) > max_batch_size:
            batches = [
                inputs[i:i + max_batch_size]
                for i in range(0, len(inputs), max_batch_size)
            ]

        batch_jobs = []
        req_path = Path(requests_file)
        for batch_idx, batch_inputs in enumerate(batches, start=1):
            batch_requests_file = str(req_path)
            if len(batches) > 1:
                batch_requests_file = str(
                    req_path.with_name(
                        f"{req_path.stem}-part{batch_idx:03d}{req_path.suffix}"
                    )
                )

            self.save_batch_request_file(
                requests=batch_inputs,
                output_path=batch_requests_file,
                system_instruction=instructions,
                generation_config=generation_config,
                key_prefix=key_prefix,
            )

            upload_time = f'{time.time():.3f}'
            attempt = 0
            while True:
                try:
                    uploaded_file = self.client.files.upload(
                        file=str(batch_requests_file),
                        config=types.UploadFileConfig(
                            display_name=f"foresight-requests-{upload_time}",
                            mime_type="application/jsonl",
                        )
                    )
                    
                    model_name = self.model
                    if not model_name.startswith("models/"):
                        model_name = f"models/{model_name}"

                    batch_job = self.client.batches.create(
                        model=model_name,
                        src=uploaded_file.name,
                        config=types.CreateBatchJobConfig(
                            display_name=f"foresight-batch-{upload_time}",
                        )
                    )
                    break
                except Exception as err:
                    msg = str(err).lower()
                    rate_limited = (
                        "429" in msg
                        or "too many requests" in msg
                        or "rate limit" in msg
                        or "resource exhausted" in msg
                    )
                    if not rate_limited or attempt >= max_retries:
                        raise
                    sleep_s = min(backoff_max_s, backoff_base_s * (2 ** attempt))
                    print(
                        f"Gemini batch submit rate-limited, sleeping {sleep_s:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(sleep_s)
                    attempt += 1
 
            print(f"Submitted batch job {batch_job.name} for {len(batch_inputs)} requests.")
            batch_jobs.append(batch_job)

        if len(batch_jobs) == 1:
            return batch_jobs[0]
        return batch_jobs

    def generate_response(
        self,
        instructions: str,
        input_item: Any,
        output_format: OutputFormat,
        meta: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Generate a single response using generate_content with structured output.
        """
        schema_cls = schema_for(output_format, return_cls=True)
        schema_json_dict = _clean_schema_for_gemini(schema_cls.model_json_schema())

        upload_images = kwargs.pop("upload_images", True)
        if input_item and isinstance(input_item, list) and isinstance(input_item[0], ChatQuery):
            request = self.compile_prompt(input_item, upload_images=upload_images)
        else:
            request = input_item

        if isinstance(request, dict):
            contents = request.get("contents", request)
        elif isinstance(request, list):
            contents = request
        else:
            raise ValueError("input_item must be a request dict, contents list, or ChatQuery list")

        config_kwargs: Dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": schema_json_dict,
        }
        if instructions:
            config_kwargs["system_instruction"] = instructions

        config_kwargs.update(self.sampling_params)

        model_name = self.model
        response = self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        # Compute cost of the response
        usage = response.usage_metadata
        cost, breakdown = self.get_cost(
            model=self.model,
            input_tokens=getattr(usage, 'prompt_token_count') or 0,
            cached_tokens=getattr(usage, 'cached_content_token_count') or 0,
            output_tokens=getattr(usage, 'candidates_token_count') or 0,
            mode="interactive",
        )

        raw_text = getattr(response, "text", None)
        if raw_text is None:
            try:
                raw_text = response.candidates[0].content.parts[0].text
            except Exception:
                raw_text = ""

        finish_reason = response.candidates[0].finish_reason.name
        return parse_and_unify(
            raw_text,
            output_format,
            model_name=self.model,
            usage={
                "cost": cost,
                "breakdown": breakdown,
            },
            meta={
                'key': input_item.get("key", None),
                'finishReason': finish_reason
            },
            verbosity=1
        )

    def poll_batch_job(self, batch_job, output_format: OutputFormat, poll_interval: int = 30):
        """
        Poll the batch job until it is complete and return the results.

        Content Datatype: https://ai.google.dev/api/caching#Content
        """
        if isinstance(batch_job, (list, tuple)):
            results = []
            for job in batch_job:
                results.extend(self.poll_batch_job(job, output_format, poll_interval))
            return results

        while batch_job.state.name == 'JOB_STATE_PENDING':
            print(f"Waiting for batch job {batch_job.name} to start... Polling again in {poll_interval}s.")
            time.sleep(poll_interval)
            batch_job = self.client.batches.get(name=batch_job.name)
            self.total_cost = 0.0

        while batch_job.state.name == 'JOB_STATE_RUNNING': # Use .name to access enum value
            print(f"Waiting for batch job {batch_job.name} to complete... Polling again in {poll_interval}s.")
            time.sleep(poll_interval)
            # Re-fetch the job to get the latest state
            batch_job = self.client.batches.get(name=batch_job.name)

        if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
            print(f"Batch job {batch_job.name} finished with state: {batch_job.state.name}")
            # Handle failed/cancelled/expired jobs
            # You might want to retrieve error details if available
            return [None] * len(batch_job.requests) # Return list of None for failed jobs

        # Download and parse results
        file_name = batch_job.dest.file_name
        file_content_bytes = self.client.files.download(file=file_name)
        file_content_str = file_content_bytes.decode('utf-8')
        # Dump response string to a local file for debugging
        debug_output_path = "batch_job_results.jsonl"
        Path(debug_output_path).write_text(file_content_str, encoding="utf-8")

        parsed_results = []
        for line in file_content_str.splitlines():
            if not line:
                continue
            parsed_response = json.loads(line)

            try:
                assert all(k in parsed_response for k in ("key", "response")), f"Malformed response line: {line}"
                key = parsed_response['key']
                response = parsed_response['response']
                usage_metadata = response.get('usageMetadata', {})

                cost, breakdown = self.get_cost(
                    model=self.model,
                    input_tokens=usage_metadata.get('promptTokenCount', 0),
                    cached_tokens=usage_metadata.get('cachedContentTokenCount', 0),
                    output_tokens=usage_metadata.get('candidatesTokenCount', 0),
                    mode="batch",
                )
                self.total_cost += cost

                finish_reason = response['candidates'][0]['finishReason']
                assert isinstance(finish_reason, str), "finishReason should be a string"
                part = response['candidates'][0]['content'].get('parts', [None])[0]
                if part is None:
                    parsed_results.append(None)
                    continue

                thought_signature = part.get('thoughtSignature', None)
                raw_text = part.get('text', None)
                if raw_text is None:
                    parsed_results.append(None)
                    continue

                unified_response = parse_and_unify(
                    raw_text,
                    output_format,
                    model_name=self.model,
                    usage={
                        "cost": cost,
                        "breakdown": breakdown,
                    },
                    meta={
                        'key': key,
                        'finishReason': finish_reason,
                        'thoughtSignature': thought_signature,
                    }
                )
                parsed_results.append(unified_response)
            except Exception as e:
                print(f"Failed to parse a response for key {key} from batch job {batch_job.name}: {e}")
                parsed_results.append(None)

        non_null_count = sum(1 for r in parsed_results if r is not None)
        print(f"Batch job {batch_job.name} completed successfully with {non_null_count}/{len(parsed_results)} valid results.")
        print(f"Total cost for batch job {batch_job.name}: ${self.total_cost:.3f}")
        return parsed_results

    def cancel_batch_job(self, batch_job):
        """Cancels a running batch job."""
        # The genai library's batch_generate_contents returns an Operation object.
        # The Operation object itself might not have a direct `cancel` method.
        # Typically, you'd use `client.batches.cancel(name=batch_job.name)`.
        # The `google-generativeai` library's `Operation` object does not directly expose `cancel()`.
        # Instead, you would use `genai.cancel_batch_generate_contents(name=batch_job.name)`.
        try:
            # Re-fetch the job to ensure we have the latest state before attempting to cancel
            current_job_state = self.client.batches.get(name=batch_job.name)
            if current_job_state.state.name == 'JOB_STATE_RUNNING':
                self.client.batches.cancel(name=batch_job.name)
                print(f"Cancelled batch job: {batch_job.name}")
            else:
                print(f"Batch job {batch_job.name} is not running (state: {current_job_state.state.name}), cannot cancel.")
                self.client.batches.delete(name=batch_job.name)
                print(f"Deleted batch job: {batch_job.name}")
        except Exception as e:
            print(f"Failed to cancel batch job {batch_job.name}: {e}")

    @staticmethod
    def get_cost(
        model: str,
        input_tokens: int = 0,  # Note: Only supports text/image tokens for now
        cached_tokens: int = 0, # 
        output_tokens: int = 0,
        mode: str = "batch",
    ):
        """
        Return (cost, breakdown) for a given model and token counts.
        https://ai.google.dev/api/generate-content#UsageMetadata
        """
        
        if model == "gemini-3-flash-preview" and mode == "batch":
            price_input_tokens = 0.25
            price_cached_tokens = 0.05
            price_output_tokens = 1.50
        elif model == "gemini-3-flash-preview" and mode == "interactive":
            price_input_tokens = 0.50
            price_cached_tokens = 0.10
            price_output_tokens = 3.00
        elif model == "gemini-2.5-flash" and mode == "batch":
            price_input_tokens = 0.15
            price_cached_tokens = 0.03
            price_output_tokens = 1.25
        elif model == "gemini-2.5-flash" and mode == "interactive":
            price_input_tokens = 0.30
            price_cached_tokens = 0.03
            price_output_tokens = 2.50
        else:
            raise ValueError(f"Model {model} with mode {mode} not supported for cost calculation.")

        input_cost = (input_tokens / 1_000_000) * price_input_tokens
        cached_cost = (cached_tokens / 1_000_000) * price_cached_tokens
        output_cost = (output_tokens / 1_000_000) * price_output_tokens
        total_cost = input_cost + cached_cost + output_cost
        
        return total_cost, {
            "input_cost": input_cost,
            "cached_cost": cached_cost,
            "output_cost": output_cost,
        }
