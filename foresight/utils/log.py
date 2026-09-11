from __future__ import annotations

import cv2
import copy
import torch
import numpy as np
import wandb
import logging
import joblib
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Callable, Union, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from foresight.utils.draw import draw_polyline, draw_text_band, draw_bev_poses_topdown
from foresight.geometry.camera import Calib, project_to_pixel


@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

def _as_numpy_uint8_hwc(img: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert image to HWC uint8 numpy array."""
    if isinstance(img, torch.Tensor):
        img = img.float().detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] in {1, 3, 4}:  # CHW
        img = img.transpose(1, 2, 0)
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def _as_torch_float_chw(img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert image to CHW float tensor in [0,1] for TensorBoard."""
    if isinstance(img, torch.Tensor):
        t = img.detach().float().cpu()
    else:
        t = torch.from_numpy(np.asarray(img))
    if t.ndim == 3 and t.shape[0] in {1, 3, 4}:
        chw = t
    elif t.ndim == 3:
        chw = t.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected image with 3 dims, got {tuple(t.shape)}")
    if chw.dtype != torch.float32:
        chw = chw.float()
    if chw.max() > 1.0:
        chw = (chw / 255.0).clamp(0.0, 1.0)
    return chw

class LogManager:
    """
    Log non-scalar artifacts (images, videos, text, etc.) from a merged tensor dict.

    Visualization config example:
      {
        "name": "pred_vs_gt_masks",
        "fn": "render_pred_gt_mask",            # resolved from globals() in this module
        "input_keys": ["pred_mask", "gt_mask"], # keys inside merged_dict
        "kwargs": {...},                        # optional
        "every_n_steps": 50,                    # optional gating
        "max_items": 8                          # optional
      }

    A visualization function should return one of:
      - {"wandb": {<tag>: <wandb.Image/Video/...>, ...}}
      - {"images": {<tag>: <HWC uint8 numpy or CHW torch>, ...}}  # LogManager will wrap if possible
      - {"text": {<tag>: "..."}, "tables": {...}, ...}            # up to you

    The `logger` passed to .log(...) should be either:
      - pytorch_lightning.loggers.WandbLogger (preferred)
      - raw wandb module/run with `.log(dict, step=...)`
      - anything with `.log_metrics` for scalars (not used here)
    """

    def __init__(self, visualizations: List[Dict[str, Any]]):
        self._cfgs: Dict[str, Dict[str, Any]] = {}
        self._fns: Dict[str, Callable] = {}

        for cfg in visualizations:
            disp = cfg.get("name") or cfg.get("fn")
            if disp is None:
                raise ValueError(f"Visualization config missing 'name' or 'fn': {cfg}")

            fn_name = cfg.get("fn")
            if fn_name is None:
                raise ValueError(f"Visualization '{disp}' missing 'fn'")

            try:
                fn = globals()[fn_name]
            except KeyError:
                raise ValueError(f"Visualization fn '{fn_name}' not found in globals(). "
                                 f"Define it in log_manager.py or import it into this module.")

            self._cfgs[disp] = cfg
            self._fns[disp] = fn

    def __call__(
        self,
        merged_dict: Dict[str, Any],
        logger: Any,
        step: Optional[int] = None,
        stage: str = "val",
    ) -> None:
        """
        Run all visualizations and write to logger.
        """
        if logger is None:
            return

        for name, fn in self._fns.items():
            cfg = copy.deepcopy(self._cfgs[name])

            # optional gating
            every = int(cfg.get("every_n_steps", 1) or 1)
            if step is not None and every > 1 and (step % every) != 0:
                continue

            # gather inputs from merged_dict
            key_map = cfg['key_map']
            if not isinstance(key_map, list):
                raise ValueError(f"Visualization '{name}' key_map must be a list of keys")
            key_map = { it['from']: it['to'] for it in key_map }

            vis_inputs: Dict[str, Any] = {}
            missing = []
            for k, v in key_map.items():
                if k in merged_dict:
                    vis_inputs[v] = merged_dict[k]
                else:
                    missing.append(k)

            assert len(missing) == 0, f"[LogManager] visualization '{name}' missing keys: {missing}"

            # add extra kwargs
            vis_inputs.update(cfg.get("kwargs", {}))

            # run visualization
            try:
                payload = fn(**vis_inputs)
            except Exception as e:
                print(f"[LogManager] visualization '{name}' failed")
                raise

            if payload is None:
                continue

            # write payload to logger
            self._write_payload(logger, payload, step, stage)


    def _write_payload(self, logger: Any, payload: Dict[str, Any], step: Optional[int], stage: str) -> None:
        exp = getattr(logger, "experiment", logger)  # Lightning loggers expose .experiment

        # 1) explicit wandb objects
        if "wandb" in payload and payload["wandb"] is not None:
            if wandb is None:
                return
            wb_dict = payload["wandb"]
            run = exp if hasattr(exp, "log") else wandb
            if hasattr(run, "log"):
                run.log(wb_dict, step=step)
            return

        # 2) generic images
        if "images" in payload and payload["images"] is not None:
            imgs_dict = payload["images"]

            # W&B path: wrap as wandb.Image
            if wandb is not None and (hasattr(exp, "log") or hasattr(exp, "config")):
                run = exp if hasattr(exp, "log") else wandb
                to_log = {}
                for tag, imgs in imgs_dict.items():
                    if not isinstance(imgs, (list, tuple)):
                        imgs = [imgs]
                    wb_imgs = []
                    for im in imgs:
                        arr = _as_numpy_uint8_hwc(im)
                        wb_imgs.append(wandb.Image(arr))
                    to_log[tag] = wb_imgs
                run.log(to_log, step=step)
                return

            # TensorBoard path: SummaryWriter.add_image/add_images
            if hasattr(exp, "add_image"):
                for tag, imgs in imgs_dict.items():
                    if not isinstance(imgs, (list, tuple)):
                        imgs = [imgs]
                    for i, im in enumerate(imgs):
                        t = _as_torch_float_chw(im)
                        exp.add_image(f"{tag}/{i}", t, global_step=0 if step is None else step)
                return

        # 3) text
        if "text" in payload and payload["text"] is not None:
            txt_dict = payload["text"]

            # W&B
            if wandb is not None and (hasattr(exp, "log") or hasattr(exp, "config")):
                run = exp if hasattr(exp, "log") else wandb
                to_log = {}
                for tag, txt in txt_dict.items():
                    if isinstance(txt, (list, tuple)):
                        txt = "\n".join(map(str, txt))
                    to_log[tag] = str(txt)
                run.log(to_log, step=step)
                return

            # TensorBoard
            if hasattr(exp, "add_text"):
                for tag, txt in txt_dict.items():
                    if isinstance(txt, (list, tuple)):
                        txt = "\n".join(map(str, txt))
                    exp.add_text(tag, str(txt), global_step=0 if step is None else step)
                return

def make_mosaic(imgs: List[np.ndarray], cols: int) -> np.ndarray:
    """imgs: list of HWC uint8 RGB (same H,W,C). returns one HWC uint8 RGB."""
    if not imgs:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    H, W = imgs[0].shape[:2]
    C = imgs[0].shape[2]
    cols = max(1, int(cols))
    rows = int(np.ceil(len(imgs) / cols))

    canvas = np.zeros((rows * H, cols * W, C), dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r * H : (r + 1) * H, c * W : (c + 1) * W] = im
    return canvas    

@torch.no_grad()
def log_trace(
    obs: torch.Tensor, 
    pred: torch.Tensor, 
    gt: torch.Tensor, 
    task: list[str],
    *,
    tag: str = "trace/pred_vs_gt",
    max_items: int = 8,
):
    """
    Log predicted vs ground-truth traces as Images.

    obs: [B, T, 3, H, W] observations in range [0,1]
    pred: { 'trajectory': [B, M, 2] | [B, 0] } predicted pixel coordinates
    gt: [B, N, 2]
    task: list of B strings describing the task
    """
    assert obs.min()>=0 and obs.max()<=1, "obs should be in [0,1] range"

    # Loop over batch and log images in fixed rows
    B, T, _, H, W = obs.shape
    frame = obs[:, -1].clamp(0, 1)  # [B,3,H,W]

    pr_c = pred['trajectory']
    gt_c   = gt.float().detach().cpu().numpy()

    imgs = []
    for b in range(min(B, max_items)):
        img = _as_numpy_uint8_hwc(frame[b])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        gt_b = gt_c[b].tolist()    # [[x,y],...]
        pr_b = pr_c[b]    # [[x,y],...]

        if gt_b is not None and len(gt_b) > 0:
            img = draw_polyline(gt_b, img, color=(255, 255, 51), line_thickness=2, dot_radius=3)

        # draw pred (blue) if present
        if pr_b is not None and len(pr_b) > 0:
            img = draw_polyline(pr_b, img, color=(255, 0, 0), line_thickness=2, dot_radius=3)

        # task text (top-left)
        txt = task[b] if (task is not None and b < len(task)) else ""
        if txt:
            img = draw_text_band(img, task[b], background=False)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img_rgb)
    mosaic = make_mosaic(imgs, cols=4)  # HWC uint8
    return {"images": {tag: mosaic}}

def _overlay_mask(img, mask, colour, alpha=0.4):
    mask = mask.clip(0, 1) * alpha
    return img * (1 - mask) + colour * mask


@torch.no_grad()
def log_waypoint_xyz_to_tb(
    logger: Any,
    obs: torch.Tensor,
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    infos: Optional[List[Dict[str, Any]]],
    extra_info: Optional[List[Dict[str, Any]]],
    epoch: int,
    global_step: int,
    *,
    prefix: str = "val",
    max_items: int = 8,
    images_per_row: int = 4,
    gt_color: Tuple[int, int, int] = (0, 0, 255),
    pred_color: Tuple[int, int, int] = (51, 255, 255),
    bev_xlim: Tuple[float, float] = (-4.8, 4.8),
    bev_ylim: Tuple[float, float] = (-4.8, 4.8),
) -> Optional[np.ndarray]:
    if logger is None:
        return None

    if obs.ndim == 5:
        obs = obs[:, -1]
    if obs.ndim != 4:
        raise ValueError(f"Expected obs shape [B,3,H,W] or [B,T,3,H,W], got {tuple(obs.shape)}")

    bsz = min(int(obs.shape[0]), int(max_items))
    pred_np = pred_xyz.detach().float().cpu().numpy()
    gt_np = gt_xyz.detach().float().cpu().numpy()
    obs_np = obs.detach()

    panels: List[np.ndarray] = []
    for b in range(bsz):
        image_panel = _as_numpy_uint8_hwc(obs_np[b]).copy()

        calib_dict = None
        if isinstance(extra_info, list) and b < len(extra_info) and isinstance(extra_info[b], dict):
            calib_dict = extra_info[b].get("calib")

        if calib_dict is not None:
            try:
                calib = Calib.from_dict(calib_dict)
                gt_pix, gt_valid = project_to_pixel(np.asarray(gt_np[b], dtype=np.float64), calib)
                pred_pix, pred_valid = project_to_pixel(np.asarray(pred_np[b], dtype=np.float64), calib)

                if gt_pix is not None and np.any(gt_valid):
                    image_panel = draw_polyline(
                        gt_pix[gt_valid].tolist(),
                        image_panel,
                        color=gt_color,
                        line_thickness=2,
                        dot_radius=3,
                    )
                if pred_pix is not None and np.any(pred_valid):
                    image_panel = draw_polyline(
                        pred_pix[pred_valid].tolist(),
                        image_panel,
                        color=pred_color,
                        line_thickness=2,
                        dot_radius=3,
                    )
            except Exception:
                pass

        image_hw = (image_panel.shape[0], image_panel.shape[1])
        bev_panel = draw_bev_poses_topdown(
            np.stack(
                [
                    np.asarray(gt_np[b], dtype=np.float64),
                    np.asarray(pred_np[b], dtype=np.float64),
                ],
                axis=0,
            ),
            colors=[gt_color, pred_color],
            image_hw=image_hw,
            title="GT vs Pred XY",
            xlim=bev_xlim,
            ylim=bev_ylim,
        )
        panels.append(np.concatenate([image_panel, bev_panel], axis=1))

    if not panels:
        return None
    mosaic = make_mosaic(panels, cols=max(1, int(images_per_row)))
    tag = f"{prefix}/waypoint_xyz_overlay_bev/epoch_{epoch}"
    wb_img = wandb.Image(mosaic, caption=tag)

    run = getattr(logger, "experiment", logger)
    if hasattr(run, "log"):
        run.log({tag: [wb_img]})
    else:
        wandb.log({tag: [wb_img]})
    return mosaic

@torch.no_grad()
def log_path_mask_to_tb(
    logger,
    tensor_dict: Dict[str, torch.Tensor],
    log_config : List[Dict],
    epoch      : int,
    global_step: int,
    *,
    prefix: str = "val",
    mask_color_rgb: Tuple[float, float, float] = (0.2, 1.0, 1.0),
):
    """
    Log a compact grid where *each row* groups:
        ┌─ RGB⊙GT ──┬─ RGB⊙pred₀ ─┬─ … ┬─ RGB⊙pred_E-1 ─┐
    All tiles keep their native H×W aspect.

    Call signature and cfg unchanged → drop-in replacement.
    """
    if not log_config:
        return

    rgb      = tensor_dict[log_config[0]["name"]].float()        # B×3×H×W
    mask_gt  = tensor_dict[log_config[1]["name"]].float()        # B×1×H×W
    mask_pred = tensor_dict[log_config[2]["name"]]               # B×E×1×H×W or B×1×H×W
    if mask_pred.ndim == 4:
        mask_pred = mask_pred.unsqueeze(1)                       # add ensemble dim

    B, _, H, W = rgb.shape
    E          = mask_pred.shape[1]
    device     = rgb.device
    colour     = torch.tensor(mask_color_rgb, device=device).view(3, 1, 1)

    # map rgb to [0,1]
    rgb = ((rgb + 1) / 2).clamp(0, 1) if rgb.min() < 0 else rgb.clamp(0, 1)

    rows = []
    for b in range(B):
        img_b   = rgb[b]                       # 3×H×W
        gt_b    = mask_gt[b]                   # 1×H×W
        preds_b = mask_pred[b]                 # E×1×H×W

        # ---- ground-truth tile ---------------------------------------
        row_tiles = [_overlay_mask(img_b, gt_b, colour)]     # 3×H×W

        # ---- prediction tiles ---------------------------------------
        # broadcast rgb_b to match E predictions WITHOUT repeat-repeat bug
        img_stack = img_b.unsqueeze(0).expand(E, 3, H, W)    # share storage
        pred_tiles = _overlay_mask(img_stack, preds_b, colour)             # E×3×H×W

        # lay them horizontally (no reshape tricks)
        row_tiles += [t for t in pred_tiles]                 # list of E+1 tensors
        row = torch.cat(row_tiles, dim=2)                    # 3×H×((E+1)·W)
        rows.append(row)
    grid = torch.cat(rows, dim=1) 

    tag = f"{prefix}/path_mask_overlay/epoch_{epoch}"
    
    # CHW [0,1] → HWC uint8
    img = grid.detach().clamp(0, 1).cpu()
    img_hwc = img.permute(1, 2, 0).numpy()  # (H, W, 3)
    wb_img = wandb.Image(img_hwc, caption=tag)

    # `logger` is usually a wandb.Run from Lightning's WandbLogger
    if hasattr(logger, "log"):
        # recommended: value is a list of images
        logger.log({tag: [wb_img]})
    else:
        # fallback: use global wandb.run
        wandb.log({tag: [wb_img]})