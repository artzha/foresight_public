
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import os
import json
import time

import torch
import numpy as np
from PIL import Image

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from cotnav.prompts.interface import (
    ContentType, Role, ChatQuery, ReasoningTrace
)
from cotnav.utils.log_utils import logging

@dataclass
class MolmoResponse:
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

class MolmoModel:
    """
    Wrapper for Molmo model using local transformers implementation
    Matches OpenAI interface for comptability with pivot_wrapper.
    """
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        service_tier: Optional[str] = None,
        default_model_args: Optional[Dict[str, Any]] = None,
        default_role: str = "user",
        model_name: str = "allenai/Molmo-7B-D-0924",
        torch_dtype: str = "auto",
        device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        **kwargs
    ):
        """
        Initialize Molmo model.
        
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
        """
        self.timeout = timeout
        self.service_tier = service_tier
        self.default_model_args = default_model_args or {
        }
        self._default_role = default_role
        self._model_name = model_name

        logging.info(f"Loading Molmo model: {model_name}")

        # Load model
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if trust_remote_code:
            model_kwargs["trust_remote_code"] = trust_remote_code
        self.model_kwargs = model_kwargs

        processor_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto"
        }
        if trust_remote_code is not None:
            processor_kwargs["trust_remote_code"] = trust_remote_code

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name, **processor_kwargs
        )

        self.device = next(self.model.parameters()).device

        logging.info(f"MolmoModel loaded successfully on device: {self.device}")
        logging.info(f"Model dtype: {next(self.model.parameters()).dtype}")
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    def set_default_model_args(self, **updates: Any) -> None:
        """Update default model arguments."""
        self.default_model_args.update({k: v for k, v in updates.items() if v is not None})

    def format_content(self, prompt: ChatQuery) -> Dict[str, Any]:
        """
        Format a ChatQuery into Qwen API format.
        """
        if prompt.type == ContentType.TEXT:
            return {
                "type": "text",
                "text": prompt.content
            }
        elif prompt.type == ContentType.IMAGE:
            # Molmo expects PIL Image objects
            # Save image to temporary file or use file:// path
            if isinstance(prompt.content, Image.Image):
                # For now, we'll pass the PIL Image directly
                # Qwen's process_vision_info can handle PIL Images
                return {
                    "type": "image",
                    "image": prompt.content  # PIL Image
                }
            elif isinstance(prompt.content, (str, Path)):
                return {
                    "type": "image",
                    "image": str(prompt.content)
                }
            else:
                raise ValueError(f"Unsupported image type: {type(prompt.content)}")
        else:
            raise ValueError(f"Unsupported content type: {prompt.type}")

    def compile_prompt(self, prompts: List[ChatQuery]) -> List[Dict[str, Any]]:
        """
        Compile list of ChatQuery into Qwen message format.
        
        Returns:
            List of messages in format:
            [{"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image", "image": ...}]}]
        """
        # Group prompts by role to create messages
        # Qwen format expects messages with role and content list
        messages = []
        current_role = None
        current_content = []
        
        for prompt in prompts:
            if prompt.role != current_role:
                # Save previous message if exists
                if current_role is not None:
                    messages.append({
                        "role": current_role,
                        "content": current_content
                    })
                # Start new message
                current_role = prompt.role
                current_content = []
            
            # Add content to current message
            content_part = self.format_content(prompt)
            current_content.append(content_part)
        
        # Add final message
        if current_role is not None:
            messages.append({
                "role": current_role,
                "content": current_content
            })
        
        return messages

    def _messages_to_template_ready(self, messages):
        """
        Transform compile_prompt output into what Molmo's chat_template expects:
        - role alternates strictly user/assistant/...
        - content is a single string
        - images are replaced by <|image|> and also collected in order
        Returns (templ_messages, pil_images)
        """
        from PIL import Image
        templ_msgs = []
        pil_images = []

        # 1) fold system into first user (if present)
        if messages and messages[0]["role"] == "system":
            sys_text = " ".join(
                p.get("text","") for p in messages[0].get("content", []) if p.get("type")=="text"
            )
            # remove system
            messages = messages[1:]
            if messages and messages[0]["role"] == "user":
                # prepend to the first user turn
                messages[0]["content"].insert(0, {"type":"text","text":sys_text})
            else:
                # no user yet: create one
                messages.insert(0, {"role":"user","content":[{"type":"text","text":sys_text}]})

        # 2) flatten content and gather images
        def _ensure_rgb(img):
            return img if img.mode == "RGB" else img.convert("RGB")

        for m in messages:
            parts = []
            for c in m.get("content", []):
                if c.get("type") == "text":
                    parts.append(c.get("text",""))
                elif c.get("type") == "image":
                    parts.append("<|image|>")
                    im = c.get("image")
                    if isinstance(im, Image.Image):
                        pil_images.append(_ensure_rgb(im))
                    else:
                        pil_images.append(_ensure_rgb(Image.open(str(im))))
            templ_msgs.append({"role": m["role"], "content": " ".join(p for p in parts if p)})

        # 3) enforce alternation: user/assistant/user/assistant/...
        # (Molmo's template asserts this; if you generate only a user prompt, that's fine.)
        return templ_msgs, pil_images

    def generate_response(
        self, 
        instructions: str, 
        inputs: List[Dict[str, Any]], 
        **kwargs: Any
    ) -> MolmoResponse:
        model_args = kwargs.pop("model_args", self.default_model_args.copy())
        timeout = kwargs.pop("timeout", self.timeout)
        max_retries = int(kwargs.pop("max_retries", 3))
        
        # Merge any additional kwargs into model_args
        model_args.update(kwargs)
        
        # Prepend system message with instructions if provided
        messages = inputs.copy()
        if instructions:
            messages.insert(0, {
                "role": "system",
                "content": [{"type": "text", "text": instructions}]
            })
        
        logging.info(f"Generating response with Molmo model: {self._model_name}")
        logging.info(f"Number of messages: {len(messages)}")

        device = self.device
        dtype = self.model.dtype
        for attempt in range(max_retries):
            try:
                # Apply chat template
                templ_msgs, images = self._messages_to_template_ready(messages)
                chat_text = self.processor.tokenizer.apply_chat_template(
                    templ_msgs, tokenize=False, add_generation_prompt=True
                )

                # Process vision inputs
                processor_inputs = self.processor.process(
                    images=images, 
                    text=chat_text,
                    padding=True,
                    return_tensors="pt"    
                )

                processor_inputs = {k: v.to(device).unsqueeze(0) for k, v in processor_inputs.items()}
                output = self.model.generate_from_batch(
                    processor_inputs,
                    GenerationConfig(
                        **model_args,
                        stop_strings="<|endoftext|>"
                    ),
                    tokenizer=self.processor.tokenizer
                )
                generated_tokens = output[0,processor_inputs['input_ids'].size(1):]
                generated_text = self.processor.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                # print(generated_text)
                parsed_output = json.loads(generated_text)
                
                input_tokens = processor_inputs['input_ids'].shape[1]        
                output_tokens = len(generated_tokens)

                usage = self._create_usage(input_tokens, output_tokens)

                return MolmoResponse(parsed_output=parsed_output, usage=usage)
            except Exception as e:
                logging.warning(f"Error in generate_response attempt {attempt+1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    raise
        
        raise Exception(f"Failed to generate response after {max_retries} retries")

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
        