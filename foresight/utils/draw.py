
import cv2
import hickle as hkl
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Sequence, Tuple

def abs_to_norm(
    points: List[Sequence[float]] | np.ndarray,
    height: int,
    width: int,
    *,
    clip: bool = True,
    out_dtype = np.float32,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Convert absolute pixel XY coordinates to normalized coordinates in [0,1]
    for an image of (height, width).

    Args:
        points: [[x, y], ...] as list/array. Each (x, y) is in pixel units.
        height, width: image dimensions.
        clip: if True, clip to [0,1].
        out_dtype: dtype for returned array.
        eps: small epsilon to avoid divide-by-zero.

    Returns:
        (N, 2) array of normalized XY coordinates.
    """
    arr = np.asarray(points, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[-1] != 2:
        raise ValueError("points must have shape (N, 2) for XY")

    # x maps from [0, width-1] → [0,1]
    arr[:, 0] /= max(width  - 1, eps)
    arr[:, 1] /= max(height - 1, eps)

    if clip:
        arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)
        arr[:, 1] = np.clip(arr[:, 1], 0.0, 1.0)

    return arr.astype(out_dtype, copy=False)

def norm_to_abs(
    points: List[Sequence[float]] | np.ndarray,
    height: int,
    width: int,
    *,
    clip: bool = True,
    round_pixels: bool = True,
    out_dtype = np.int32,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Convert XY points to absolute pixel coordinates for an image of (height, width).

    Args:
        points: [[x, y], ...] as list/array. If normalized=True, x,y in [0,1].
        height, width: image dimensions.
        normalized: force treat as normalized if True; as absolute if False; auto if None.
        clip: clip to [0,width-1] x [0,height-1] after conversion.
        round_pixels: round to nearest integer.
        out_dtype: dtype for the returned array if round_pixels=True.
        eps: tolerance used in auto-detection.

    Returns:
        (N, 2) array of absolute pixel XY.
    """
    arr = np.asarray(points, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[-1] != 2:
        raise ValueError("points must have shape (N, 2) for XY")

    # x maps to [0, width-1], y maps to [0, height-1]
    arr[:, 0] *= (width  - 1)
    arr[:, 1] *= (height - 1)

    if clip:
        arr[:, 0] = np.clip(arr[:, 0], 0, width  - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, height - 1)

    if round_pixels:
        arr = np.round(arr).astype(out_dtype, copy=False)

    return arr

def draw_polyline(
    points: List[Sequence[float]],
    image: np.ndarray,
    line_thickness: int = 1,
    color: Tuple[int, int, int] = (51, 255, 255),  # BGR aqua/cyan
    dot_radius: int = 2,
) -> np.ndarray:
    """
    Draws a polyline connecting points (p0->p1->...->pN) and dots at each point.

    Args:
        points: list of [x, y]. Either normalized (0..1) or absolute pixels
                (auto-detected). Example: [[0.5,0.5], [1.0,1.0], ...]
        image:  HxWxC (uint8) image (BGR or grayscale).
        line_thickness: thickness of connecting segments.
        color:  BGR color tuple. Default aqua/cyan.
        dot_radius: radius of the dot drawn at each vertex.

    Returns:
        A copy of the image with drawings.
    """
    if image.ndim == 2:  # grayscale -> BGR
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    if len(points) == 0:
        return canvas

    h, w = canvas.shape[:2]

    # Auto-detect normalized vs absolute pixels:
    # If all coordinates are in [0, 1.0 + eps], treat as normalized.
    arr = np.asarray(points, dtype=np.float32)
    eps = 1e-3
    if np.all(arr >= -eps) and np.all(arr <= 1.0 + eps):
        # normalized -> pixel coordinates
        arr = norm_to_abs(arr, h, w, clip=True, round_pixels=True, out_dtype=np.int32)
    else:
        # assume already pixels
        arr[:, 0] = np.clip(arr[:, 0], 0, w - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, h - 1)

    pts = arr.round().astype(int)

    # Draw dots
    for (x, y) in pts:
        cv2.circle(canvas, (int(x), int(y)), dot_radius, color, thickness=-1, lineType=cv2.LINE_AA)

    # Draw connecting segments
    for i in range(len(pts) - 1):
        cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), color, thickness=line_thickness, lineType=cv2.LINE_AA)

    return canvas


def get_distinct_bgr_colors(n: int) -> list[Tuple[int, int, int]]:
    n = max(0, int(n))
    if n == 0:
        return []
    rgb_base = [
        (255, 64, 64),    # red
        (255, 140, 0),    # orange
        (255, 215, 0),    # gold
        (180, 220, 40),   # olive-lime
        (40, 200, 40),    # green
        (255, 0, 200),    # magenta
        (220, 90, 220),   # purple-magenta (no blue tint)
        (255, 105, 180),  # hot pink
        (210, 120, 80),   # brownish orange
        (160, 255, 90),   # light lime
    ]
    bgr_base = [(b, g, r) for (r, g, b) in rgb_base]
    if n <= len(bgr_base):
        return bgr_base[:n]

    colors = list(bgr_base)
    extra = n - len(colors)
    for i in range(extra):
        # Restrict hue to warm/non-blue bands: [0, 30] U [45, 80] U [145, 179]
        seg_sizes = [31, 36, 35]
        total = sum(seg_sizes)
        k = int((i * total) / max(1, extra))
        if k < seg_sizes[0]:
            h = k
        elif k < seg_sizes[0] + seg_sizes[1]:
            h = 45 + (k - seg_sizes[0])
        else:
            h = 145 + (k - seg_sizes[0] - seg_sizes[1])
        hsv = np.uint8([[[int(h), 220, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


def draw_index_badge(
    image: np.ndarray,
    point_xy: Sequence[float],
    index: int,
    color: Tuple[int, int, int],
    *,
    radius: int = 12,
    outline_thickness: int = 1,
    fill_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()
    h, w = canvas.shape[:2]
    pt = np.asarray(point_xy, dtype=np.float32).reshape(-1)
    if pt.size < 2:
        return canvas
    eps = 1e-3
    if -eps <= float(pt[0]) <= 1.0 + eps and -eps <= float(pt[1]) <= 1.0 + eps:
        x = int(np.clip(pt[0], 0.0, 1.0) * max(1, w - 1))
        y = int(np.clip(pt[1], 0.0, 1.0) * max(1, h - 1))
    else:
        x = int(np.clip(pt[0], 0, max(1, w - 1)))
        y = int(np.clip(pt[1], 0, max(1, h - 1)))

    cv2.circle(canvas, (x, y), int(radius), fill_color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(
        canvas,
        (x, y),
        int(radius),
        color,
        thickness=max(1, int(outline_thickness)),
        lineType=cv2.LINE_AA,
    )

    text = str(int(index))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    text_thickness = 2
    # Ensure text stays comfortably inside badge.
    for _ in range(4):
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
        if tw <= int(radius * 1.25) and th <= int(radius * 1.25):
            break
        font_scale *= 0.90
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
    tx = x - tw // 2
    ty = y + th // 2
    cv2.putText(canvas, text, (tx, ty), font, font_scale, color, text_thickness, cv2.LINE_AA)
    return canvas

def draw_text_band(
    img: np.ndarray,
    text: str,
    *,
    band_frac: float = 0.20,
    margin: int = 8,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.3,
    thickness: int = 1,
    line_gap: int = 4,
    fg: Tuple[int, int, int] = (255, 255, 255),
    bg: Tuple[int, int, int] = (0, 0, 0),
    background: bool = True,
) -> np.ndarray:
    """
    Draw wrapped text into the top band of an image without spilling.
    Works for HxW (grayscale) or HxWxC (uint8) images. Returns a copy.

    Args:
      band_frac: fraction of image height reserved for text (e.g., 0.2 = top 20%).
      background: if True, draw a filled background strip behind the text.
    """
    if img.ndim == 2:
        canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        canvas = img.copy()

    if not text:
        return canvas

    H, W = canvas.shape[:2]
    top_h = max(1, min(H, int(round(band_frac * H))))
    max_w = max(1, W - 2 * margin)

    # Wrap to width
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        cand = (cur + " " + w).strip()
        (tw, _), _ = cv2.getTextSize(cand, font, font_scale, thickness)
        if tw <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    # Fit to height
    (_, line_h), baseline = cv2.getTextSize("Ag", font, font_scale, thickness)
    step = line_h + baseline + line_gap
    max_lines = max(1, (top_h - 2 * margin) // max(1, step))
    lines = lines[:max_lines]

    # Optional background strip (only as tall as needed, capped by band)
    used_h = min(top_h, margin + len(lines) * step + margin)
    if background:
        cv2.rectangle(canvas, (0, 0), (W, used_h), bg, thickness=-1)

    # Draw lines
    y = margin + line_h
    for line in lines:
        cv2.putText(canvas, line, (margin, y), font, font_scale, fg, thickness, cv2.LINE_AA)
        y += step
        if y > top_h - margin:
            break

    return canvas

def make_query_panel(payload: dict) -> np.ndarray:
    required = ["sample_id", "goal_cmd", "samples"]
    for k in required:
        if k not in payload:
            raise KeyError(f"make_query_panel payload missing required key '{k}'")
    samples = payload["samples"]
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError("make_query_panel payload['samples'] must be a non-empty list")

    for i, s in enumerate(samples):
        for k in ("image", "reason", "hausdorff", "scores"):
            if k not in s:
                raise KeyError(f"make_query_panel payload['samples'][{i}] missing key '{k}'")
        if not isinstance(s["scores"], dict):
            raise TypeError(f"make_query_panel payload['samples'][{i}]['scores'] must be a dict")

    sample_id = str(payload["sample_id"])
    goal_cmd = str(payload["goal_cmd"])
    distance_key = str(payload.get("distance_key", "HD"))
    tile_height = int(payload.get("tile_height", 220))
    tile_gap = int(payload.get("tile_gap", 8))
    separator_px = int(payload.get("separator_px", 6))
    margin = int(payload.get("margin", 10))
    line_gap = int(payload.get("line_gap", 4))
    block_gap = int(payload.get("block_gap", 10))
    max_text_lines = int(payload.get("max_text_lines", 220))
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = float(payload.get("title_font_scale", 0.42))
    body_scale = float(payload.get("body_font_scale", 0.42))
    header_scale = float(payload.get("header_font_scale", 0.50))
    thickness = int(payload.get("font_thickness", 1))

    def to_rgb_uint8(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise ValueError("make_query_panel expects images with shape HxW or HxWx3(+)")
        arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return arr

    def resize_keep_height(img: np.ndarray, out_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError("make_query_panel received an empty image tile")
        out_w = max(1, int(round(w * float(out_h) / float(h))))
        return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

    def wrap_text_to_width(text: str, max_w_px: int, font_scale: float) -> List[str]:
        if max_w_px <= 4:
            return [text]
        out: List[str] = []
        for raw_line in str(text).splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                out.append("")
                continue
            words = raw_line.split()
            cur = ""
            for word in words:
                cand = (cur + " " + word).strip()
                (tw, _), _ = cv2.getTextSize(cand, font, font_scale, thickness)
                if tw <= max_w_px or not cur:
                    cur = cand
                    continue
                out.append(cur)
                cur = word
            if cur:
                out.append(cur)
        return out if out else [""]

    (title_line_h, title_base) = cv2.getTextSize("Ag", font, title_scale, thickness)[0][1], cv2.getTextSize("Ag", font, title_scale, thickness)[1]
    title_band_h = title_line_h + title_base + 6
    tiles: List[np.ndarray] = []

    for c, sample in enumerate(samples):
        img = resize_keep_height(to_rgb_uint8(sample["image"]), tile_height)
        tile_h, tile_w = img.shape[:2]
        tile = np.zeros((tile_h + title_band_h, tile_w, 3), dtype=np.uint8)
        tile[:title_band_h, :, :] = 24
        tile[title_band_h:, :, :] = img
        hdist = sample["hausdorff"]
        hd_str = "NaN" if hdist is None else f"{hdist:.3f}"
        title = f"q={c} | {distance_key}={hd_str}"
        cv2.putText(tile, title, (6, 4 + title_line_h), font, title_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        tiles.append(tile)

    row_h = tiles[0].shape[0]
    row_w = sum(t.shape[1] for t in tiles) + tile_gap * (len(tiles) - 1)
    image_row = np.zeros((row_h, row_w, 3), dtype=np.uint8)
    x = 0
    for i, tile in enumerate(tiles):
        w = tile.shape[1]
        image_row[:, x:x + w] = tile
        x += w + (tile_gap if i < len(tiles) - 1 else 0)

    text_w = max(40, row_w - 2 * margin)
    lines: List[tuple[str, float]] = []
    lines.extend((line, header_scale) for line in wrap_text_to_width(f"{sample_id} | goal: {goal_cmd}", text_w, header_scale))
    lines.append(("", body_scale))
    for r, sample in enumerate(samples):
        hd_str = "NaN" if sample["hausdorff"] is None else f"{sample['hausdorff']:.3f}"
        lines.extend((line, body_scale) for line in wrap_text_to_width(f"q={r}: {distance_key}={hd_str}", text_w, body_scale))
        for score_name, values in sample["scores"].items():
            if not isinstance(values, list):
                raise TypeError(
                    f"make_query_panel payload['samples'][{r}]['scores']['{score_name}'] must be a list"
                )
            value_str = ", ".join(["NaN" if v is None else f"{float(v):.3f}" for v in values])
            lines.extend((line, body_scale) for line in wrap_text_to_width(f"{score_name}: [{value_str}]", text_w, body_scale))
        lines.extend((line, body_scale) for line in wrap_text_to_width(f"reason: {str(sample['reason'])}", text_w, body_scale))
        if r < len(samples) - 1:
            lines.append(("", body_scale))

    if len(lines) > max_text_lines:
        lines = lines[: max(1, max_text_lines - 1)] + [("... (truncated)", body_scale)]

    total_h = margin
    for text, scale in lines:
        if not text:
            total_h += block_gap
            continue
        (_, h), b = cv2.getTextSize("Ag", font, scale, thickness)
        total_h += h + b + line_gap
    total_h += margin
    text_canvas = np.full((max(1, total_h), row_w, 3), 245, dtype=np.uint8)

    y = margin
    for text, scale in lines:
        if not text:
            y += block_gap
            continue
        (_, h), b = cv2.getTextSize("Ag", font, scale, thickness)
        y_baseline = y + h
        cv2.putText(text_canvas, text, (margin, y_baseline), font, scale, (10, 10, 10), thickness, cv2.LINE_AA)
        y += h + b + line_gap

    sep = np.full((max(0, separator_px), row_w, 3), 220, dtype=np.uint8)
    return np.vstack([image_row, sep, text_canvas])


def draw_bev_poses_topdown(
    poses: np.ndarray | List[Sequence[float]],
    *,
    colors: List[Tuple[int, int, int]] | None = None,
    image_hw: Tuple[int, int] | None = None,
    bev_width: int | None = None,
    title: str = "Top-Down BEV XY",
    xlim: Tuple[float, float] = (-12.8, 12.8),
    ylim: Tuple[float, float] = (-6.4, 6.4),
) -> np.ndarray:
    """
    Render a top-down BEV XY plot from poses and return it as an RGB image.
    Poses are transformed to be with respect to the first pose.

    Args:
        poses: (N,8) [ts, x, y, z, qw, qx, qy, qz], or (N,2+) XY-like array.
        image_hw: Optional (H, W) of the image panel. When provided, output BEV
            is resized to have the same height.
        bev_width: Optional output width for BEV panel. Defaults to image width
            when image_hw is provided, else 512.
        title: Title of the BEV plot.
        xlim: Optional fixed x-axis limits as (xmin, xmax).
        ylim: Optional fixed y-axis limits as (ymin, ymax).

    Returns:
        RGB uint8 image of the BEV plot.
    """
    poses_arr = np.asarray(poses, dtype=np.float64)
    if poses_arr.ndim not in (2, 3):
        raise ValueError(f"poses must be 2D or 3D array, got {poses_arr.shape}")
    if poses_arr.ndim == 2 and poses_arr.shape[0] == 0:
        raise ValueError(f"poses must be non-empty, got {poses_arr.shape}")
    if poses_arr.ndim == 3 and (poses_arr.shape[0] == 0 or poses_arr.shape[1] == 0):
        raise ValueError(f"poses must be non-empty, got {poses_arr.shape}")

    if image_hw is not None:
        target_h = int(image_hw[0])
        default_w = int(image_hw[1])
    else:
        target_h = 512
        default_w = 512
    target_h = max(1, target_h)
    target_w = max(1, int(bev_width) if bev_width is not None else default_w)

    if poses_arr.ndim == 2:
        poses_arr = poses_arr[None, ...]

    # Standard SE3 odometry format: [ts, x, y, z, qw, qx, qy, qz].
    xy_trajs = []
    for traj in poses_arr:
        if traj.shape[1] >= 8:
            # Lazy import prevents circular import with foresight.utils.log -> foresight.utils.draw.
            from foresight.utils.math import odom_to_local_pose
            traj_local = odom_to_local_pose(traj[:, :8])
            xy = traj_local[:, 1:3]
        elif traj.shape[1] >= 2:
            xy = traj[:, :2].copy()
            xy -= xy[:1]
        else:
            raise ValueError(f"poses must have at least 2 columns, got {traj.shape}")
        xy_trajs.append(xy)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    dpi = 140
    fig_w = max(1.0, float(target_w) / float(dpi))
    fig_h = max(1.0, float(target_h) / float(dpi))
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    if len(xy_trajs) == 2:
        labels = ["gt", "pred"]
    else:
        labels = [f"traj_{i}" for i in range(len(xy_trajs))]

    cmap = plt.get_cmap("tab10")

    def _bgr_to_mpl(color_bgr: Tuple[int, int, int]):
        r, g, b = [float(c) for c in color_bgr]
        if max(r, g, b) > 1.0:
            return (r / 255.0, g / 255.0, b / 255.0)
        return (r, g, b)

    for i, xy in enumerate(xy_trajs):
        if colors is not None and i < len(colors):
            color = _bgr_to_mpl(colors[i])
        else:
            color = cmap(i % 10)
        ax.plot(xy[:, 0], xy[:, 1], "-o", color=color, linewidth=1.5, markersize=2.5, label=labels[i])
        ax.scatter(xy[0, 0], xy[0, 1], color=color, s=18, marker="s")
        ax.scatter(xy[-1, 0], xy[-1, 1], color=color, s=18, marker="x")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    if len(xy_trajs) > 1:
        legend = ax.legend(
            loc="lower right",
            bbox_to_anchor=(0.995, 0.01),
            bbox_transform=fig.transFigure,
            borderaxespad=0.0,
            fontsize=8,
        )
        legend.set_in_layout(False)

    fig.tight_layout()
    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
    plt.close(fig)
    if panel.shape[0] != target_h or panel.shape[1] != target_w:
        panel = cv2.resize(panel, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return panel


def plot_density_scatter_from_df(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ax,
    *,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    density_kind: str = "kde",
    density_levels: int = 20,
    density_bins: int = 40,
    scatter_alpha: float = 0.35,
    scatter_size: float = 8.0,
    scatter_color: str = "white",
    scatter_edgecolor: str = "black",
    cmap: str = "viridis",
) -> int:
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(
            f"Missing plot columns. Required ({x_col}, {y_col}), "
            f"available={list(df.columns)}"
        )

    try:
        import seaborn as sns
    except Exception as exc:
        raise RuntimeError(
            "Seaborn is required for density scatter plotting. "
            "Install seaborn in the runtime environment."
        ) from exc

    x_values = pd.to_numeric(df[x_col], errors="coerce")
    y_values = pd.to_numeric(df[y_col], errors="coerce")
    plot_df = pd.DataFrame({x_col: x_values, y_col: y_values}).dropna()
    valid_count = int(len(plot_df))

    plot_title = title or f"{x_col} vs {y_col}"
    x_axis_label = x_label or x_col
    y_axis_label = y_label or y_col
    ax.set_title(plot_title)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)

    if valid_count == 0:
        ax.text(0.5, 0.5, "No valid points", ha="center", va="center", transform=ax.transAxes)
        return 0

    if density_kind == "hist":
        sns.histplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            bins=max(4, int(density_bins)),
            cmap=cmap,
            cbar=True,
            ax=ax,
        )
    else:
        sns.kdeplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            fill=True,
            levels=max(3, int(density_levels)),
            thresh=0.0,
            cmap=cmap,
            ax=ax,
        )

    sns.scatterplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        s=max(1.0, float(scatter_size)),
        alpha=float(np.clip(scatter_alpha, 0.0, 1.0)),
        color=scatter_color,
        edgecolor=scatter_edgecolor,
        linewidth=0.3,
        ax=ax,
    )
    return valid_count


def visualize_unified_sample(
    hkl_path: str | Path,
    *,
    line_thickness: int = 1,
    trace_color: Tuple[int, int, int] = (51, 255, 255),
    dot_radius: int = 2,
    band_frac: float = 0.30,
) -> np.ndarray:
    """
    Load a unified-format .hkl sample, draw its trajectory on the front image,
    and overlay the first available subgoal text at the top.

    Expects the .hkl to contain:
      - 'image': (H, W, 3) uint8 RGB frame
      - 'trace_pts': (N, 2) normalized or pixel XY points
      - 'generations': optional list of dicts with 'subgoal' (language goal)
    """
    sample_path = Path(hkl_path)
    sample = hkl.load(str(sample_path))

    image = sample.get("image", None)
    if image is None:
        raise ValueError(f"'image' key missing in unified sample {sample_path}")
    if image.ndim == 2:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        # Stored as RGB; convert to BGR for OpenCV drawing
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    trace_pts = sample.get("trace_pts", None)
    if trace_pts is not None:
        img_bgr = draw_polyline(
            trace_pts,
            img_bgr,
            line_thickness=line_thickness,
            color=trace_color,
            dot_radius=dot_radius,
        )

    # Extract first non-empty subgoal, if present
    subgoal_text = ""
    generations = sample.get("generations", [])
    if isinstance(generations, list):
        for entry in generations:
            if not isinstance(entry, dict):
                continue
            val = entry.get("subgoal", None)
            if isinstance(val, str) and val.strip():
                subgoal_text = val.strip()
                break

    if subgoal_text:
        img_bgr = draw_text_band(
            img_bgr,
            subgoal_text,
            band_frac=band_frac,
            background=True,
            font_scale=0.4
        )

    return img_bgr