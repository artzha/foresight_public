# cotnav/models/experts/ventura.py
from tqdm import tqdm

import copy
import torch
from torch import nn
from torchvision.transforms.functional import resize
from torch.nn import Conv2d
from torch.nn.parameter import Parameter

from transformers import CLIPTokenizer, CLIPTextModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from diffusers import (
    AutoencoderKL,
    AutoencoderTiny,
    DDPMScheduler,
    DDIMScheduler,
    LCMScheduler,
    UNet2DConditionModel,
)
from typing import Dict, Optional, Union, Sequence
from cotnav.models.blocks.DiT import DiT_models
from cotnav.utils.log_utils import logging
from cotnav.utils.image_utils import (get_tv_resample_method, resize_max_res)
from cotnav.core.format import llava_to_openai
from hydra.utils import instantiate

class VenturaModel(nn.Module):
    def __init__(self, model_cfg: Dict):
        super().__init__()
        self.model_cfg = model_cfg
        self.pipeline_cfg = model_cfg['pipeline']
        self.sched_cfg = model_cfg['inference']['scheduler']
        self.encoder_cfg = model_cfg['encoder']
        self.backbone_cfg = model_cfg['backbone']
        self.vae_latent_size = model_cfg['backbone']['latent_dim']
        self.setup_inputs(model_cfg)
        self.unet_in_channels =  self.vae_latent_size * self.num_unet_inputs

        self.fwd_kwargs = {
            'denoising_steps': self.sched_cfg['kwargs'].get('denoising_steps', 4),
            'scale_invariant': self.sched_cfg['kwargs'].get('scale_invariant', True),
            'shift_invariant': self.sched_cfg['kwargs'].get('shift_invariant', True),
            'processing_res': self.sched_cfg['kwargs'].get('processing_res', 180),
        }

        #1 Initialize encoders
        self.vae = instantiate(self.encoder_cfg['rgb'])
        # self.tokenizer = instantiate(self.encoder_cfg['tokenizer'], torch_dtype=torch.float16)
        # self.text_encoder = instantiate(self.encoder_cfg['text_encoder'], torch_dtype=torch.float16)
        self.max_txt_len = self.encoder_cfg.get(
            'max_position_embeddings', 1024
        )

        #2 Initialize backbones
        self.backbone = instantiate(self.backbone_cfg)
        if 'stable-diffusion' in self.backbone_cfg.get('pretrained_model_name_or_path', ''):
            self._replace_unet_conv_in()

        # TODO: Initialize schedulers, etc.
        self.warned_once = False
        self.latent_scale_factor = self.encoder_cfg['rgb']['latent_scale_factor']

        self.scheduler = instantiate(self.sched_cfg)
        self._apply_freezing_from_pipeline()

    def setup_inputs(self, model_cfg):
        pipeline_flags = model_cfg['pipeline']
        comp_inputs = {}
        num_unet_inputs = 1 # One for the target (always there)
        
        for comp, comp_dict in pipeline_flags.items():
            input_keys = comp_dict.get('input_keys', [])
            if len(input_keys) == 0:
                continue

            comp_key = f'{comp}_inputs'
            if not comp_key in comp_inputs:
                comp_inputs[comp_key] = {}

            for input_dict in input_keys:
                name = input_dict['name']
                modality = input_dict['modality']
                comp_inputs[comp_key][name] = input_dict.copy()
                if modality == 'image' and comp == 'backbone':
                    num_unet_inputs += 1
            setattr(self, comp_key, comp_inputs[comp_key])
            logging.info(f"Input modality mapping for {comp}: {comp_inputs[comp_key]}")
        if model_cfg['backbone']['cond_method'] == "Marigold":
            setattr(self, 'num_unet_inputs', num_unet_inputs)
        else:
            setattr(self, 'num_unet_inputs', 2) # Always include path mask label

    @staticmethod
    def _freeze_module(module: nn.Module, name: str = ""):
        """
        Set requires_grad=False for all params and switch to eval() mode.
        """
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = False
        module.eval()
        if name:
            logging.info(f"[Ventura] Froze module '{name}' "
                         f"({sum(p.numel() for p in module.parameters())} params).")

    def _apply_freezing_from_pipeline(self):
        """
        Read self.pipeline_cfg and freeze any component with frozen=True.

        Expected keys in pipeline:
            backbone:
              frozen: bool
            vae:
              frozen: bool
            text_encoder:
              frozen: bool
        """
        pipe = getattr(self, "pipeline_cfg", {}) or {}

        # Map pipeline names -> VenturaModel attributes
        name_to_attr = {
            "backbone": "backbone",
            "vae": "vae",
            "text_encoder": "text_encoder",
            "text_processor": "text_processor",
        }
        for comp_name, comp_cfg in pipe.items():
            if not isinstance(comp_cfg, dict):
                continue

            frozen = comp_cfg.get("frozen", False)
            if not frozen:
                logging.info(f"[Ventura] Component '{comp_name}' is trainable (frozen=False).")
                continue

            attr_name = name_to_attr.get(comp_name, None)
            if attr_name is None:
                logging.warning(
                    f"[Ventura] pipeline component '{comp_name}' marked frozen, "
                    f"but no attribute mapping is defined."
                )
                continue

            if not hasattr(self, attr_name):
                logging.warning(
                    f"[Ventura] pipeline component '{comp_name}' marked frozen, "
                    f"but VenturaModel has no attribute '{attr_name}'."
                )
                continue

            module = getattr(self, attr_name)
            if not isinstance(module, nn.Module):
                logging.warning(
                    f"[Ventura] Attribute '{attr_name}' for component '{comp_name}' "
                    f"is not an nn.Module; skipping freeze."
                )
                continue

            self._freeze_module(module, name=comp_name)

    def _replace_unet_conv_in(self):
        # replace the first layer to accept 4* #in_channels (8 with RGB + depth)
        _weight = self.backbone.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = self.backbone.conv_in.bias.clone()  # [320]
        _weight = _weight.repeat((1, self.num_unet_inputs, 1, 1))  # Keep selected channel(s)
        _weight *= 1.0 / self.num_unet_inputs # halves activation magnitude if 2 modalities

        # new conv_in channel
        _n_convin_out_channel = self.backbone.conv_in.out_channels
        _new_conv_in = Conv2d(
            self.unet_in_channels, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        # TODO: Check if we need to register this explicitly with pl
        self.backbone.conv_in = _new_conv_in
        logging.info("Unet conv_in layer is replaced")
        # replace config
        self.backbone.config["in_channels"] = self.unet_in_channels
        logging.info("Unet config is updated")
        return

    def _check_inference_step(self, n_step: int) -> None:
        """Check if denoising step is reasonable"""
        assert n_step > 1, f"Number of denoising steps must be greater than 1, got {n_step}."

        if self.warned_once:
            return
        self.warned_once = True

        if isinstance(self.scheduler, DDIMScheduler):
            if "trailing" != self.scheduler.config.timestep_spacing:
                logging.warning(
                    f"The loaded `DDIMScheduler` is configured with `timestep_spacing="
                    f'"{self.scheduler.config.timestep_spacing}"`; the recommended setting is `"trailing"`. '
                    f"This change is backward-compatible and yields better results. "
                    f"Consider using `prs-eth/marigold-depth-v1-1` for the best experience."
                )
            else:
                if n_step > 10:
                    logging.warning(
                        f"Setting too many denoising steps ({n_step}) may degrade the prediction; consider relying on "
                        f"the default values."
                    )
            if not self.scheduler.config.rescale_betas_zero_snr:
                logging.warning(
                    f"The loaded `DDIMScheduler` is configured with `rescale_betas_zero_snr="
                    f"{self.scheduler.config.rescale_betas_zero_snr}`; the recommended setting is True. "
                    f"Consider using `prs-eth/marigold-depth-v1-1` for the best experience."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}. ")

    # def encode_text(
    #     self, 
    #     text
    # ) -> torch.Tensor:
    #     """
    #     Encodes one or more text prompts into CLIP embeddings in a single batch.

    #     Args:
    #         text:  either a single string, or a list/tuple of strings

    #     Returns:
    #         last_hidden_state: torch.Tensor of shape (B, L, D)
    #     """
    #     # ensure we have a list of strings
    #     texts = [text] if isinstance(text, str) else list(text)
    #     conversations = [
    #         [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "text",
    #                         "text": t,
    #                     }
    #                 ],
    #             }
    #         ]
    #         for t in texts
    #     ]
    #     # tokenize in batch
    #     inputs = self.text_processor.apply_chat_template(
    #         conversations,
    #         tokenize=True,
    #         add_generation_prompt=False,  # we only want the prompt, not the assistant stub
    #         return_dict=True,
    #         return_tensors="pt",
    #         padding="max_length",
    #         max_length=self.max_txt_len,
    #         truncation=True,
    #     )
    #     inputs.pop("token_type_ids", None)

    #     device = next(self.text_encoder.parameters()).device
    #     inputs = {k: v.to(device) for k, v in inputs.items()}

    #     # Forward through Qwen3-VL; we only need hidden states
    #     # For CausalLM models, hidden states are in `hidden_states`
    #     out = self.text_encoder(
    #         **inputs,
    #         output_hidden_states=True,
    #         return_dict=True,
    #     )

    #     # Use last layer hidden states as text embedding
    #     # shape: (B, L, D)
    #     text_tokens = out.hidden_states[-1].float()

    #     return text_tokens

    def encode_text(
        self, 
        text: Union[str, Sequence[str]]
    ) -> torch.Tensor:
        """
        Encodes one or more text prompts into CLIP embeddings in a single batch.

        Args:
            text:  either a single string, or a list/tuple of strings

        Returns:
            last_hidden_state: torch.Tensor of shape (B, L, D)
        """
        # ensure we have a list of strings
        texts = [text] if isinstance(text, str) else list(text)

        # tokenize in batch
        text_tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_token_ids = text_tokens.input_ids
        untruncated_ids = self.tokenizer(texts, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_token_ids.shape[-1] and not torch.equal(
            text_token_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logging.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        # move to same device as text_encoder
        text_tokens = {k: v.to(self.text_encoder.device) for k, v in text_tokens.items()}

        # forward once through CLIPTextModel
        out = self.text_encoder(**text_tokens)
        # last_hidden_state: (B, L, D)
        return out.last_hidden_state.float()

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        empty_embed = self.encode_text(prompt)  # this returns (1, seq_len, hidden_dim)
        
        # sanity‐check the dims
        assert empty_embed.ndim == 3 and empty_embed.shape[0] == 1, (
            f"empty_text_embed must be (1, L, D), got {tuple(empty_embed.shape)}"
        )
        
        self.empty_text_embed = empty_embed


    @staticmethod
    def encode_empty_image_like(latent: torch.Tensor) -> torch.Tensor:
        """
        Return an all-zero latent with the same shape & device as `latent`.
        Useful for classifier-free image guidance dropout.
        """
        return torch.zeros_like(latent)

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encodes RGB images to latent representations 

        Args:
            rgb_in (torch.Tensor): Input RGB image tensor.

        Returns:
            torch.Tensor: Encoded latent representation of the RGB image.
        """
        if not isinstance(rgb_in, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")

        # If time dim exists, ensure it's one
        # Add time dimension if not exists
        exp_dim = self.pipeline_cfg['vae']['in_dim']
        if rgb_in.ndim == 4 and exp_dim == 5:
            rgb_in = rgb_in.unsqueeze(2) # [B, C, T, H, W]
        assert rgb_in.ndim == exp_dim, f"Input RGB tensor must have {exp_dim} dimensions."
        assert rgb_in.shape[1] == 3, "Input RGB tensor must have 3 channels (RGB) in dim 1."

        h = self.vae.encoder(rgb_in)
        # Reparametrization trick for differentiable sampling
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * self.latent_scale_factor
        # Squeeze time dimension if exists
        if rgb_latent.ndim == 5:
            rgb_latent = rgb_latent.squeeze(2) # [B, C, H, W]
        assert rgb_latent.ndim == 4, "Encoded RGB latent must have 4 dimensions (B, C, H, W)."

        return rgb_latent
    
    @staticmethod
    def stack_mask_images(mask_in):
        assert mask_in.ndim == 4, "Input depth tensor must have 4 dimensions (B, C, H, W)."
        stacked = mask_in.repeat(1, 3, 1, 1)
        return stacked

    def encode_path_mask(self, mask_in: torch.Tensor) -> torch.Tensor:
        """
        Encodes path masks images to latent representations.

        Args:
            depth_in (torch.Tensor): Input depth image tensor.

        Returns:
            torch.Tensor: Encoded latent representation of the depth image.
        """
        if not isinstance(mask_in, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        assert mask_in.ndim == 4, "Input depth tensor must have 4 dimensions (B, C, H, W)."

        stacked = self.stack_mask_images(mask_in)
        depth_latent = self.encode_rgb(stacked)
        return depth_latent

    def prepare_cfg_latents(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Return the concatenated image & text latents for conditional and
        unconditional branches, respecting each input's `dropout_prob`.
        Keys with `dropout_prob == 0` are *never* dropped.
        
        Returns
        -------
        dict  with  {
            'img_cond'   : (B, C_img, h, w),
            'img_uncond' : (B, C_img, h, w),
            'txt_cond'   : (B, L,   D),
            'txt_uncond' : (B, L,   D),
        }
        """
        img_cond, img_uncond = [], []
        ctrl_cond, ctrl_uncond = [], []
        txt_cond, txt_uncond = None, None

        device = None
        for key, spec in self.backbone_inputs.items():
            x        = inputs[key]
            modality = spec["modality"]
            p_drop   = spec.get("dropout_prob", 0.0)
            if device is None:
                device = x.device

            if self.training:
                # training: use stochastic conditioning dropout
                keep_mask = torch.rand((len(x),), device=device) >= p_drop
            else:
                # (i.e., zeros/empty everywhere)
                keep_mask = torch.zeros((len(x),), device=device, dtype=torch.bool)

            # import pdb; pdb.set_trace()  # Debugging breakpoint
            # rgb = (x[0].permute(1, 2, 0).cpu().numpy() + 1)/2.0
            # rgb = cv2.cvtColor((rgb*255).astype(np.uint8), cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            # ---------- IMAGE inputs ------------------------------------
            if modality == "image":
                if x.ndim == 5:                       # (B,T,C,H,W) → first frame
                    x = x[:, 0]
                
                # Sample probability to drop out this input
                assert x.shape[1] == 3, f"Expected 3 channels for {key}, got {x.shape[1]} channels"
                lat = self.encode_rgb(x)
                img_cond.append(lat)                  # always present
                img_uncond.append(
                    torch.where(keep_mask.view(-1, 1, 1, 1), lat, torch.zeros_like(lat))
                )
            # ---------- TEXT inputs -------------------------------------
            elif modality == "text":
                lat = self.encode_text(x)             # (B,L,D)
                txt_cond = lat                        # (only one text key expected)
                if self.empty_text_embed is None:
                    self.encode_empty_text()
                empty = self.empty_text_embed.repeat(lat.shape[0], 1, 1).to(lat)
                txt_uncond = torch.where(
                    keep_mask.view(-1, 1, 1), lat, empty                     # (B,L,D)
                )
            else:
                raise ValueError(f"Unsupported modality '{modality}' for key '{key}'")

        # ---------- Final concat / defaults -----------------------------
        img_cond_cat   = torch.cat(img_cond,   dim=1)           # (B,C_img,h,w)
        img_uncond_cat = torch.cat(img_uncond, dim=1)

        if txt_cond is None:                                    # no text inputs at all
            if self.empty_text_embed is None:
                self.encode_empty_text()
            txt_cond = txt_uncond = self.empty_text_embed.repeat(
                img_cond_cat.shape[0], 1, 1).to(img_cond_cat)

        out = {
            "img_cond"  : img_cond_cat,
            "img_uncond": img_uncond_cat,
            "txt_cond"  : txt_cond,
            "txt_uncond": txt_uncond,
        }

        if ctrl_cond:
            out["ctrl_cond"] = torch.cat(ctrl_cond,   dim=0)           # (B,C_img,h,w)
            out["ctrl_uncond"] = torch.cat(ctrl_uncond, dim=0)
        return out

    def forward(self, 
        inputs: Dict[str, torch.Tensor],
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 1,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = 'bilinear', 
        cfg_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        generator: Union[torch.Generator, None] = None,
        show_progress_bar: bool = False,
    ):
        """
        Forward pass through model
        Inputs:
            rgb (torch.Tensor): (B, C, H, W), 0-1
            text (str): Text input for conditioning
        Outputs:
            Dict[str, torch.Tensor]: (B, 1, H, W) path mask prediction
        """
        if denoising_steps is None:
            denoising_steps = self.fwd_kwargs['denoising_steps']
        if processing_res is None:
            processing_res = inputs[next(iter(self.backbone_inputs))].shape[-1]

        assert ensemble_size >= 1, "Ensemble size must be >= 1"
        self._check_inference_step(denoising_steps)
        resample_method = get_tv_resample_method(resample_method)

        #1 Preprocess inputs
        for input_key, input_dict in self.backbone_inputs.items():
            assert input_key in inputs, f"Input key '{input_key}' not found in unet inputs."
            modality = input_dict['modality']
            if modality == 'image':
                rgb_in = inputs[input_key]
                if rgb_in.ndim == 5: # Squeeze away time dimension if exists
                    rgb_in = rgb_in.squeeze(1)

                assert rgb_in.ndim == 4, "Input RGB tensor must have 4 dimensions (B, C, H, W)."
                # assert rgb_in.shape[1] == 3, "Input RGB tensor must have 3 channels (RGB)."
                input_size = rgb_in.shape

                if processing_res > 0:
                    rgb_in = resize_max_res(
                        rgb_in,
                        max_edge_resolution=processing_res,
                        resample_method=resample_method
                    )
                assert rgb_in.min() >= -1.0 and rgb_in.max() <= 1.0 
                inputs[input_key] = rgb_in
            elif modality == 'text':
                if isinstance(inputs[input_key], str):
                    inputs[input_key] = [inputs[input_key]]
            else:
                raise ValueError(f"Unsupported modality '{modality}' for input key '{input_key}'.")

        #2 Handle ensemble size
        B = inputs[next(iter(self.backbone_inputs))].shape[0]
        if ensemble_size > 1:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v):
                    # Repeat along batch dim
                    inputs[k] = v.repeat_interleave(ensemble_size, dim=0)
                elif isinstance(v, list):
                    # Repeat each element e times while preserving order
                    inputs[k] = [x for x in v for _ in range(ensemble_size)]
                else:
                    # Leave other types untouched
                    inputs[k] = v

        #3 Inference
        outputs = self.infer(
            inputs=inputs, 
            num_inference_steps=denoising_steps, 
            cfg_scale=cfg_scale,
            guidance_rescale=guidance_rescale,
            generator=generator,
            show_progress_bar=show_progress_bar
        )
        final_pred = outputs['target_pred']  # [B, 3, H, W]

        if match_input_res:
            final_pred = resize(
                final_pred,
                size=input_size[-2:],
                interpolation=resample_method,
                antialias=True
            )

        outputs.update({
            "path_mask_pred": final_pred,  # [B, 3, H, W]
        })

        return outputs

    def infer(
        self,
        inputs: Dict[str, torch.Tensor],
        num_inference_steps: int,
        cfg_scale: float,
        guidance_rescale: float,
        generator: Union[torch.Generator, None],
        show_progress_bar: bool,
        **kwargs
    ):
        """
        Inference method for generating path masks
        """
        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.eta = 0.0 # Make DDIM stochastic
        timesteps = self.scheduler.timesteps
        
        lat_dict = self.prepare_cfg_latents(inputs)
        rgb_cond, rgb_uncond = lat_dict["img_cond"], lat_dict["img_uncond"]
        ctrl_cond, ctrl_uncond = lat_dict.get("ctrl_cond", None), lat_dict.get("ctrl_uncond", None)
        txt_cond, txt_uncond = lat_dict["txt_cond"], lat_dict["txt_uncond"]

        B, _, H, W = rgb_cond.shape

        target_latent = torch.randn(
            (B, self.vae_latent_size, H, W),
            device=rgb_cond.device,
            dtype=rgb_cond.dtype,
            generator=generator
        )

        if show_progress_bar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising"
            )
        else:
            iterable = enumerate(timesteps)  

        for i, t in iterable:
            latent_input = torch.cat(
                [torch.cat([rgb_cond,   target_latent], 1),
                    torch.cat([rgb_uncond, target_latent], 1)],
                dim=0
            ) # [2B, 4, h, w]
            txt_in  = torch.cat([txt_cond, txt_uncond], 0) # [2B, L, D]

            t_device = t.unsqueeze(0).to(latent_input.device)
            eps = self.backbone(
                latent_input, t_device, txt_in
            ).sample
            eps_c, eps_u = eps.chunk(2)

            eps_guided = eps_u + cfg_scale * (eps_c - eps_u) if cfg_scale is not None else eps_c

            if cfg_scale is not None and guidance_rescale > 0.0:
                # Rescale noise according to guidance rescale
                eps_guided = self.rescale_noise_cfg(eps_guided, eps_c, guidance_rescale=guidance_rescale)

            target_latent = self.scheduler.step(
                eps_guided, t, target_latent, generator=generator
            ).prev_sample
        
        rescale_target_latent, target_pred = self.decode_target(target_latent)  # [B, 3, H, W]
        target_pred = torch.clip(target_pred, -1.0, 1.0)
        target_pred = (target_pred + 1.0) / 2.0  # Normalize to [0, 1]
 
        output = {
            "target_latent": rescale_target_latent,  # [B, 4, H, W]
            "target_pred_cont": target_pred,  # [B, 3, H, W]
            "target_pred": (target_pred > 0.5).float(),  # [B, 3, H, W]
            "rgb_cond": rgb_cond,  # [B, 4, H, W
        }
        if ctrl_cond is not None:
            output["ctrl_cond"] = ctrl_cond
        if txt_cond is not None:
            output["txt_cond"] = txt_cond

        return output
    
    def rescale_noise_cfg(self, noise_cfg, noise_pred_cond, guidance_rescale=0.0):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_cond = noise_pred_cond.std(dim=list(range(1, noise_pred_cond.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_cond / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg

    def decode_target(self, target_latent: torch.Tensor, guidance_rescale=0.0) -> torch.Tensor:
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        target_latent = target_latent / self.latent_scale_factor
        # decode
        exp_dim = self.pipeline_cfg['vae']['in_dim']
        if target_latent.ndim == 4 and exp_dim == 5:
            target_latent = target_latent.unsqueeze(2)
        z = self.vae.post_quant_conv(target_latent)
        stacked = self.vae.decoder(z)

        if z.ndim == 5 and z.shape[2]==1:
            z = z[:, :, -1]
            stacked = stacked[:, :, -1]
        # mean of output channels
        target_mean = stacked.mean(dim=1, keepdim=True)
        return z, target_mean