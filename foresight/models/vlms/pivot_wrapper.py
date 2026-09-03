# cotnav/models/vlms/pivot_wrapper.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import copy
from pathlib import Path
import tempfile
import json
from joblib import Parallel, delayed
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .infer_registry import get as get_infer

from omegaconf import OmegaConf
from cotnav.prompts.interface import (parse_and_unify, OutputFormat, ChatQuery, TaskSpec)
from cotnav.geometry.camera import (Calib, project_to_pixel)
from cotnav.geometry.motion import MotionTemplateLibrary
from typing import Iterable, Optional, TypedDict
from cotnav.utils.metric import hausdorff_xyz

ImgLike = Union[str, Path, Image.Image, np.ndarray]
Color = Tuple[int, int, int]

def create_pivot(**kwargs: Any) -> Any:
    return PIVOT(**kwargs)

def build_arcs_bank_and_actions(
    unified_responses: List[Dict[str, Any]],
    motion_arcs: Sequence[Any],
    num_actions: int,
    action_dim: int,
    z_const: float = -0.4,
) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    """
    Build the (K x N x D) arc bank once and map each unified decision response
    to its (N x D) action sequence by indexing into the bank.

    Returns:
        pred_choices: List[int] length B
        action_preds: Tensor [B x N x D]
        arcs_bank:    Tensor [K x N x D]
    """
    # 1) Build arc bank once (K x N x D)
    K = len(motion_arcs)
    arcs_bank = torch.empty((K, num_actions, action_dim), dtype=torch.float32)
    for k, arc in enumerate(motion_arcs):
        arc_xy = torch.from_numpy(arc.sample_along_arc(num_samples=num_actions))  # (N,2)
        if action_dim == 2:
            arcs_bank[k] = arc_xy
        else:
            z = torch.full((arc_xy.shape[0], 1), float(z_const))
            arcs_bank[k] = torch.cat((arc_xy, z), dim=1)

    # 2) Map unified decisions -> actions
    pred_choices: List[int] = []
    actions: List[torch.Tensor] = []
    for u in unified_responses:
        choice = int(u["final_response"]["choice"])
        pred_choices.append(choice)
        actions.append(arcs_bank[choice])  # (N,D)

    action_preds = torch.stack(actions, dim=0)  # [B x N x D]
    return pred_choices, action_preds, arcs_bank

class PIVOT:
    """Minimal wrapper that forwards VQA calls to the configured VLM."""

    def __init__(
        self, *, 
        vlm: Dict[str, Any], 
        motion_parameters: Dict[str, Any], 
        annotation: Dict[str, Any], 
        **kwargs
    ) -> None:
        """
        Args:
            vlm: configuration dict for the VLM instance. See _init_vlm().
            motion_parameters: dict of motion parameter settings for motion templates.
            annotation: (optional) dict of annotation settings (not used yet).
        """
        # Convert to dict if OmegaConf
        if OmegaConf.is_config(vlm):
            vlm = OmegaConf.to_container(vlm, resolve=True, enum_to_str=True)

        self.vlm = self._init_vlm(vlm)
        self._ann_motion_cfg = annotation.get('motion_cfg', {})
        self._ann_goal_cfg = annotation.get('goal_cfg', {})
        generate_defaults = vlm.get("generate_defaults", {})
        self._default_instructions: Optional[str] = generate_defaults.pop("instructions", None)
        self._base_model_args: Dict[str, Any] = vlm.get("model_args", {})
        extra_model_args = generate_defaults.pop("model_args", {})
        if extra_model_args:
            self._base_model_args.update(extra_model_args)
        self._call_defaults: Dict[str, Any] = generate_defaults
        
        print("vlm ", vlm)
        self.vlm_cfg = vlm['provider_kwargs']

        self._default_service_tier: Optional[str] = vlm.get("service_tier")
        self._default_timeout: Optional[float] = vlm.get("timeout")
        self._default_max_retries: int = int(vlm.get("max_retries", 5))

        # Keep a copy of the motion configuration for downstream helpers.
        self.mp_cfg = copy.deepcopy(motion_parameters or {})
        self._initial_motion_cfg = copy.deepcopy(self.mp_cfg)
        self._num_actions = vlm.get("num_actions", 20)
        self._action_dim = vlm.get("action_dim", 3)  # e.g., (x,y,z)    

    def motion_templates(self):
        mp_cfg = copy.deepcopy(self.mp_cfg)
        motion_bank = MotionTemplateLibrary(
            max_curvature=float(mp_cfg['max_curvature']),
            max_path_len=float(mp_cfg['max_free_path_length']),
            num_options=int(mp_cfg['num_options'])
        )
        return motion_bank.arcs()

    def get_gt_choice(self, arcs: np.ndarray, odom: np.ndarray) -> np.ndarray:
        """
        arcs: (B, K, N, 3), odom: (B, M, 3)
        Returns ground truth choice indices (B,)
        """
        if isinstance(arcs, torch.Tensor):
            arcs, odom = arcs.detach().cpu().numpy(), odom.detach().cpu().numpy()
        B, K, N, _ = arcs.shape
        if odom.ndim == 2:
            odom = odom[None, ...]  # (1, M, 3)

        # Compute Hausdorff distances for all combinations (B, K)
        hausdorff_distances = np.zeros((B, K))
        for b in range(B):
            for k in range(K):
                hausdorff_distances[b, k] = hausdorff_xyz(arcs[b, k], odom[b])
        
        # Find ground truth indices (best Hausdorff distance for each batch) (B,)
        ground_truth_indices = np.argmin(hausdorff_distances, axis=1)
        
        return ground_truth_indices

    # -------------------- Public API --------------------

    def preprocess_image(
        self, image: ImgLike, calib: Calib=None, use_arcs=True, goal_hdg=None
    ) -> Image.Image:
        """Convert input image to PIL.Image RGB."""
        img_hw = None
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)

        # Convert to list of lists of PIL images
        if image.ndim == 5: # [B, T, C, H, W]
            image = image.transpose(0, 1, 3, 4, 2) # [B, T, H, W, C]
            images = [ 
                [self._to_pil(image[b, t]) for t in range(image.shape[1])] 
                for b in range(image.shape[0])
            ]
            img_hw = (images[0][0].height, images[0][0].width)
        elif image.ndim == 4: # [T, C, H, W]
            image = image.transpose(0, 2, 3, 1) # [T, H, W, C]
            images = [[self._to_pil(image[t]) for t in range(image.shape[0])]]
            img_hw = (images[0][0].height, images[0][0].width)
        else:
            raise ValueError("Unsupported image shape for preprocess_image.")

        # Annotate last image with motion templates if available
        image_msgs = []
        motion_arcs = self.motion_templates()
        for b in range(len(images)):
            if len(images[b]) == 0:
                continue

            last_img = images[b][-1]
            annotated = last_img.copy()
            if calib is not None and use_arcs:
                annotated, _ = self.annotate_constant_curvature(
                    last_img,
                    arcs=motion_arcs,
                    calib=calib,
                    **self._ann_motion_cfg
                )
            # Annotate goal heading arrow 
            if goal_hdg is not None:
                annotated = self.annotate_goal_heading(
                    annotated, goal_hdg, **self._ann_goal_cfg
                )
                # self.annotate_goal_heading(annotated, goal_hdg, **self._ann_goal_cfg).save("test.jpg")

            images[b][-1] = annotated ### Carl: only the last image is annotated?
            image_msgs.append([ChatQuery("image", "user", img) for img in images[b]])

        # Resize images standard dimensions if needed
        max_dim = self.vlm_cfg.get("max_pixel_dim", img_hw[0])
        # Check if any of thehw are greater than max_dim
        if img_hw is not None and (img_hw[0] > max_dim or img_hw[1] > max_dim):
            for b in range(len(image_msgs)):
                for t in range(len(image_msgs[b])):
                    im = image_msgs[b][t].content
                    if not isinstance(im, Image.Image):
                        im = self._to_pil(im)
                    w, h = im.size
                    m = max(w, h)
                    scale = float(max_dim) / float(m)
                    new_w = max(1, int(round(w * scale)))
                    new_h = max(1, int(round(h * scale)))
                    im = im.resize((new_w, new_h), Image.LANCZOS)
                    image_msgs[b][t] = ChatQuery("image", "user", im) 

        return image_msgs

    def __call__(
        self, 
        rgb_image: ImgLike, 
        task: TaskSpec,
        calib: Calib,
        goal_hdg: torch.Tensor,
        action_label,
        **call_kwargs: Any
    ) -> Dict[str, Any]:
        """Upload the image and run a VLM call with a prompt is given"""
        payload: Dict[str, Any] = {
            "responses": [],   # unified per-batch response
            "costs":    [],    # per-step cost dict
            "image_ctx": [],   # last image per sequence
        }

        image_ctx_b = self.preprocess_image(
            rgb_image, 
            calib,
            goal_hdg
        ) # 1 x T x images

        motion_arcs = self.motion_templates()
        for b in range(len(image_ctx_b)):
            image_ctx = image_ctx_b[b]
            messages = []
            messages.extend(image_ctx)

            intermediate_responses = []
            intermediate_costs = {}
            intermediate_prompts = task.meta['intermediate_prompts']
            for prompt in intermediate_prompts:
                messages.append(ChatQuery("text", "user", prompt))
                response = self.vqa(task.system_prompt, messages)
                raw = getattr(response, "output_parsed", None)
                if hasattr(raw, "model_dump"):
                    raw = raw.model_dump()
                elif raw is None:
                    # try generic fields some providers expose
                    raw = getattr(response, "output_json", None) \
                        or getattr(response, "output_text", None) \
                        or getattr(response, "text", None)
                response_text = json.dumps(raw, ensure_ascii=False) if not isinstance(raw, str) else raw
                messages.append(ChatQuery("text", "assistant", response_text))
                intermediate_responses.append(raw)

                cost, brk = self.vlm.get_cost(
                    self.vlm.get_model_name(),
                    response.usage.input_tokens - response.usage.input_tokens_details.cached_tokens,
                    response.usage.input_tokens_details.cached_tokens,
                    response.usage.output_tokens
                )
                intermediate_costs[f"step_{len(intermediate_responses)}"] = {
                    "cost": cost, "breakdown": brk
                }

            unified = parse_and_unify(
                intermediate_responses[-1],
                task.output_format,
                meta={**(task.meta or {})}
            )

            payload["responses"].append(unified)
            payload["costs"].append(intermediate_costs)
            payload["image_ctx"].append(image_ctx[-1].content)

            # ---- Single DECISIONS block (runs once, after batch loop) ----
            if task.output_format == OutputFormat.DECISIONS_V1:
                B = len(payload["responses"])
                pred_choices, action_preds_tensor, arcs_bank = build_arcs_bank_and_actions(
                    payload["responses"],        # List[unified_response per B]
                    motion_arcs,
                    self._num_actions,
                    self._action_dim,
                )

                # tile bank to [B x K x N x D] for GT comparison if needed
                motion_arcs_xyz_b = arcs_bank.unsqueeze(0).tile(B, 1, 1, 1)

                gt_choices_val = None
                if action_label is not None:
                    gt_choices_val = self.get_gt_choice(
                        motion_arcs_xyz_b, torch.tile(action_label.unsqueeze(0), (B, 1, 1))
                    )

                payload.update({
                    "pred_choices": pred_choices,             # List[int]
                    "action_preds": action_preds_tensor,      # [B x N x D]
                    "motion_arcs": motion_arcs_xyz_b,         # [B x K x N x D]
                })
                if gt_choices_val is not None:
                    payload.update({"gt_choices": gt_choices_val})
            else:
                pass  # other output formats can be added here

        return payload

    def vqa(self, system_prompt: str, prompts: list(ChatQuery), resume: bool = False, **call_kwargs: Any) -> PivotVQAResult:
        """Upload the image and run a single VLM call with the provided question."""  
        inputs = self.vlm.compile_prompt(prompts)
        try:
            response = self.vlm.generate_response(
                system_prompt,
                inputs,
                **call_kwargs
            )
        except Exception as e:
            print(f"VLM call failed with exception: {e}")
            return None
        return response
    
    def batch_vqa(self, system_prompt: str, batch_prompts: list, **call_kwargs: Any) -> list:
        """Upload the image and run a batch of VLM calls with the provided questions."""  
        force_single_response = bool(call_kwargs.pop("force_single_response", False))

        if force_single_response:
            single_n_jobs = int(call_kwargs.pop("single_response_n_jobs", 8))
            output_format = call_kwargs.pop("output_format", None)
            assert output_format is not None, "output_format must be provided"

            def _run_single(prompt_item: list[ChatQuery]):
                # Compile each prompt independently to preserve single-call behavior.
                input_item = self.vlm.compile_prompt(prompt_item)
                try:
                    return self.vlm.generate_response(
                        system_prompt,
                        input_item,
                        output_format,
                        **call_kwargs,
                    )
                except Exception as e:
                    print(f"VLM single-response call failed with exception: {e}")
                    return None

            return Parallel(n_jobs=single_n_jobs, backend="threading")(
                delayed(_run_single)(prompt_item) for prompt_item in batch_prompts
            )
        
        batch_inputs = [self.vlm.compile_prompt(prompts) for prompts in batch_prompts]
        responses = self.vlm.generate_batch_response(
            system_prompt,
            batch_inputs,
            **call_kwargs
        )
        return responses

    def annotate_goal_heading(
        self,
        image: ImgLike,
        heading_angle: float,
        *,
        center: Optional[Tuple[int, int]] = None,
        color: Color = (51, 255, 255),
        thickness: int = 20,
        style: str = "triangle",          # "triangle" | "line" | "chevron"
        length_ratio: float = 0.15,       # fraction of image min(W,H) for shaft length
        head_len_ratio: float = 0.07,     # fraction for arrowhead length
        head_wid_ratio: float = 0.05,     # fraction for arrowhead width
        degrees: bool = True,             # if True, heading_angle is in degrees
        overlay_alpha: float = 0.9        # blend strength over the original image
    ) -> Image.Image:
        """
        Draw a goal-direction arrow near the top of the image.

        Convention (image-centric):
        - 0° (or 0 rad) points UP (north).
        - Positive angles rotate COUNTERCLOCKWISE (so +90° = left/west, +180° = down/south).
        - Set `degrees=False` if `heading_angle` is already in radians.

        Args:
            image: PIL.Image | np.ndarray | str/Path (local file). URLs not supported here.
            heading_angle: heading of goal relative to north-up.
            center: (u, v) pixels of the arrow base. Defaults to (W/2, 10%*H).
            color: (R,G,B)
            thickness: shaft thickness (pixels)
            style: "triangle" (default), "line", or "chevron"
            length_ratio: shaft length relative to min(W,H)
            head_len_ratio: head length relative to min(W,H)
            head_wid_ratio: head width relative to min(W,H)
            degrees: interpret `heading_angle` as degrees if True
            overlay_alpha: 0..1 for blending drawn overlay

        Returns:
            PIL.Image.Image with the arrow rendered.
        """
        base = self._to_pil(image).convert("RGB")
        W, H = base.size
        under = base.copy()

        # Create transparent overlay to draw vector graphics cleanly
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Default center near top
        if center is None:
            c = (int(W * 0.5), int(H * 0.1))
        else:
            c = (int(center[0]), int(center[1]))

        # Scale geometry from image size
        s = float(min(W, H))
        shaft_len = max(8.0, s * float(length_ratio))
        head_len  = max(6.0, s * float(head_len_ratio))
        head_wid  = max(4.0, s * float(head_wid_ratio))

        # Convert heading to radians. Image coords: +x right, +y down.
        # We define 0 rad = up/north, and +angle = counterclockwise -> left/west at +90°.
        import math
        ang = math.radians(heading_angle) if degrees else float(heading_angle)

        ux = -math.sin(ang)   # +90° -> -1 (left)
        uy = -math.cos(ang)   # 0° -> -1 (up)

        # Base and tip of shaft
        x0, y0 = c
        x1 = x0 + ux * shaft_len
        y1 = y0 + uy * shaft_len

        def as_xy(pt):
            return (int(round(pt[0])), int(round(pt[1])))

        # Draw styles
        if style.lower() == "line":
            draw.line([as_xy((x0, y0)), as_xy((x1, y1))], fill=(*color, 255), width=int(thickness))

        elif style.lower() == "chevron":
            # Shaft
            draw.line([as_xy((x0, y0)), as_xy((x1, y1))], fill=(*color, 255), width=int(thickness))
            # Two small fletches at tip
            px, py = -uy, ux
            f = head_len * 0.6
            tip = (x1, y1)
            left  = (x1 - ux * f + px * (head_wid * 0.6), y1 - uy * f + py * (head_wid * 0.6))
            right = (x1 - ux * f - px * (head_wid * 0.6), y1 - uy * f - py * (head_wid * 0.6))
            draw.line([as_xy(tip), as_xy(left)],  fill=(*color, 255), width=int(max(2, thickness - 2)))
            draw.line([as_xy(tip), as_xy(right)], fill=(*color, 255), width=int(max(2, thickness - 2)))

        else:  # "triangle" (default)
            shaft_end = (x1 - ux * (head_len * 0.6), y1 - uy * (head_len * 0.6))
            draw.line([as_xy((x0, y0)), as_xy(shaft_end)], fill=(*color, 255), width=int(thickness))

            # Triangle head at the tip
            px, py = -uy, ux
            tip    = (x1, y1)
            base_c = (x1 - ux * head_len, y1 - uy * head_len)
            left   = (base_c[0] + px * (head_wid * 0.5), base_c[1] + py * (head_wid * 0.5))
            right  = (base_c[0] - px * (head_wid * 0.5), base_c[1] - py * (head_wid * 0.5))
            draw.polygon([as_xy(tip), as_xy(left), as_xy(right)], fill=(*color, 255))

        # Composite overlay
        comp = Image.alpha_composite(under.convert("RGBA"), overlay)
        if 0.0 < overlay_alpha < 1.0:
            comp = Image.blend(under.convert("RGBA"), comp, overlay_alpha)

        return comp.convert("RGB")

    def annotate_constant_curvature(
        self,
        image: ImgLike,
        *,
        points_uv: Optional[Sequence[Tuple[float, float]]] = None, # [u,v] in pixels 
        arcs: Optional[Sequence[Any]] = None, # ConstantCurvatureArc objects
        calib: Optional[Calib] = None, # if arcs given, must provide calib
        # Style / options:
        selected_idx: Optional[int] = None,
        colors: Optional[Sequence[Color]] = None,
        thickness: int = 4,
        endpoint_radius: int = 20,
        overlay_alpha: float = 0.9,
        samples_per_meter: int = 10,
        # Constant height of the template in base frame (meters). Match your Convoi default.
        z_base: float = -0.4,
        label_text: bool = True,
        label_font_size: Optional[int] = 40,
        label_font_color: Optional[Color] = None,
        border_padding: int = 0,
    ) -> Tuple[Image.Image, List[Tuple[int, int]]]:
        """
        Draw numbered endpoint circles (like Convoi's annotate_image) and, if `arcs` are
        provided, draw the projected arc polylines from the robot to each endpoint.

        - If `arcs` is None: only numbered circles from `points_uv` are drawn.
        - If `arcs` is provided: arcs are sampled in base frame, projected using
          `project_xyz_to_uv(...)`, then rasterized as polylines on the image.

        Returns:
            annotated_image (PIL.Image RGB),
            endpoints_px    list of (u, v) ints (the circle centers actually drawn)
        """
        # --- Prepare base image + overlay ---
        base = self._to_pil(image).convert("RGB")
        W, H = base.size
        under = base.copy()
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        endpoints_px: List[Tuple[int, int]] = []
        font = self._get_label_font(label_font_size) if label_text else None
        if label_font_color:
            rgb_vals = tuple(int(c) for c in label_font_color)
            font_rgb = rgb_vals[:3] if len(rgb_vals) >= 3 else (255, 255, 255)
        else:
            font_rgb = (255, 255, 255)

        # --- If arcs are given, draw them (requires projection) ---
        if arcs is not None and len(arcs) > 0:
            if calib is None:
                raise ValueError("If arcs are given, must also provide calib.")

            # Prepare colors
            cols = list(colors) if colors else None

            for i, arc in enumerate(arcs):
                # sample along arc in base frame (meters)
                n = max(2, int(samples_per_meter * float(arc.length)))
                s_vals = np.linspace(0.0, float(arc.length), n)
                xy_local = np.array([arc.xy_at_s(float(s)) for s in s_vals], dtype=np.float32)  # (N,2)
                xyz = np.concatenate([xy_local, np.full((n, 1), z_base, np.float32)], axis=1)   # (N,3)

                # project to image
                uv_all, vis_mask = project_to_pixel(xyz, calib)

                # draw line segments only where both endpoints are visible & inside image
                color_i = cols[i % len(cols)] if cols else (0, 255, 0)
                for j in range(len(uv_all) - 1):
                    if vis_mask[j] and vis_mask[j + 1]:
                        u0, v0 = uv_all[j]
                        u1, v1 = uv_all[j + 1]
                        draw.line([(u0, v0), (u1, v1)], fill=(*color_i, 255), width=thickness)

                # remember the final visible endpoint (fallback to last sample)
                # if caller also supplies points_uv, we'll use those for circle centers instead
                valid_idx = np.flatnonzero(vis_mask)
                if valid_idx.size > 0:
                    u_end, v_end = uv_all[valid_idx[-1]]
                else:
                    u_end, v_end = uv_all[-1]
                endpoints_px.append((int(round(u_end)), int(round(v_end))))

        # --- Determine circle centers: prefer provided points_uv if given ---
        if points_uv is not None and len(points_uv) > 0:
            circle_centers = [(int(round(u)), int(round(v))) for (u, v) in points_uv]
        else:
            circle_centers = endpoints_px

        # --- Draw numbered circles (Convoi style) ---
        circle_color_default = (255, 255, 255)
        circle_color_selected = (0, 255, 0)
        pad = max(0, int(border_padding))
        min_x = pad + endpoint_radius
        min_y = pad + endpoint_radius
        max_x = max(min_x, W - 1 - pad - endpoint_radius)
        max_y = max(min_y, H - 1 - pad - endpoint_radius)

        adjusted_centers: List[Tuple[int, int]] = []

        for i, (cx, cy) in enumerate(circle_centers):
            sel = (selected_idx is not None and i == int(selected_idx))
            color = circle_color_selected if sel else circle_color_default

            cx = max(min_x, min(max_x, cx))
            cy = max(min_y, min(max_y, cy))
            adjusted_centers.append((cx, cy))

            # filled circle
            draw.ellipse(
                [cx - endpoint_radius, cy - endpoint_radius,
                 cx + endpoint_radius, cy + endpoint_radius],
                fill=(*color, 255),
                outline=None
            )

            if label_text:
                label = str(i)
                # center text crudely using textbbox
                try:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except Exception:
                    tw, th = (8, 10)
                tx = cx - int(tw / 2)
                ty = cy - int(th / 2)
                # black outline then white text for legibility
                draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0, 255), font=font)
                draw.text((tx, ty), label, fill=(*font_rgb, 255), font=font)

        # --- Composite like cv2.addWeighted(alpha=0.3) ---
        comp = Image.alpha_composite(under.convert("RGBA"), overlay)
        if 0.0 < overlay_alpha < 1.0:
            comp = Image.blend(under.convert("RGBA"), comp, overlay_alpha)

        # Use whichever centers we actually drew for return value
        ret_centers = adjusted_centers if len(adjusted_centers) > 0 else endpoints_px
        return comp.convert("RGB"), ret_centers

    # -------------------- Context management --------------------

    def cleanup(self) -> None:
        if hasattr(self.vlm, "cleanup"):
            print("Cleaning up VLM temporary files...")
            self.vlm.cleanup()

    def get_context(self) -> Dict[str, Any]:
        """Return the current motion parameter configuration."""
        return {"motion_parameters": copy.deepcopy(self.mp_cfg)}

    def reset_context(self) -> None:
        """Restore the motion parameters to their initial state."""
        self.mp_cfg = copy.deepcopy(self._initial_motion_cfg)

    def restore_context(self, context: Dict[str, Any]) -> None:
        """Restore motion parameters from a previously captured context."""
        if not isinstance(context, dict):
            raise TypeError("context must be a dict produced by get_context().")
        motion_cfg = context.get("motion_parameters")
        if motion_cfg is not None:
            self.mp_cfg = copy.deepcopy(motion_cfg)

    # -------------------- Internals --------------------

    def _init_vlm(self, cfg: Dict[str, Any]) -> Any:
        """
        Accepts:
          - {"instance": <vlm>}
          - {"factory": callable, "kwargs": {...}}
          - {"name": "openai:gpt5", "provider_kwargs": {...}}  # if infer_registry available
        """
        if "instance" in cfg and cfg["instance"] is not None:
            return cfg["instance"]

        if "factory" in cfg and callable(cfg["factory"]):
            return cfg["factory"](**dict(cfg.get("kwargs", {})))

        if "name" in cfg and get_infer is not None:
            return get_infer(cfg["name"], **dict(cfg.get("provider_kwargs", {})))

        raise ValueError(
            "vlm config must provide one of: "
            "'instance', 'factory'+optional 'kwargs', or 'name'+optional 'provider_kwargs'."
        )

    @staticmethod
    def _to_pil(image: ImgLike) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, (str, Path)):
            p = str(image)
            if p.startswith("http://") or p.startswith("https://"):
                raise ValueError("Cannot convert URL images to PIL directly; download first.")
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _get_label_font(self, size: Optional[int]) -> Optional[ImageFont.ImageFont]:
        if size is None:
            try:
                return ImageFont.load_default()
            except Exception:
                return None
        try:
            return ImageFont.truetype("DejaVuSans.ttf", int(size))
        except Exception:
            try:
                return ImageFont.load_default()
            except Exception:
                return None
