# metric_manager.py
from __future__ import annotations
import copy
import importlib
from typing import Any, Dict, List, Optional, Callable

import numpy as np
from scipy.spatial.distance import cdist
import torch
from omegaconf import OmegaConf
    
class MetricManager:
    """
    Lightweight metric aggregator.

    Each metric config can be one of:
      - {"fn": "package.module:function_name", "name": "optional_alias", "pred_key": "...", "lab_key": "...", "kwargs": {...}}
      - {"name": "compute_accuracy", "pred_key": "...", "lab_key": "...", "kwargs": {...}}  # resolved via `registry`

    registry: optional dict[str, Callable] to resolve short names.
    writer: optional summary writer (must support .add_scalar(tag, scalar_value, global_step=step))
    """

    def __init__(
        self,
        metrics: List[Dict[str, Any]],
        writer: Any = None,
    ):
        self.writer = writer
        self._cfgs: Dict[str, Dict[str, Any]] = {}
        self._metrics: Dict[str, Any] = {}
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

        # Optional streaming store for special metrics (e.g., precision/recall curves)
        self._stream: Dict[str, Dict[str, np.ndarray]] = {}

        for cfg in metrics:
            name = str(cfg.get("name", "")).strip()
            if not name:
                raise ValueError("Each metric config must include a non-empty 'name'.")
            metric_cls = self._resolve_metric_class(cfg)
            metric = metric_cls(
                key_map=cfg.get("key_map", {}),
                kwargs=cfg.get("kwargs", {}),
                name=name,
            )
            self._cfgs[name] = cfg
            self._metrics[name] = metric

    def reset(self):
        for k in self._totals:
            self._totals[k] = 0.0
            self._counts[k] = 0
        self._stream.clear()

    def set_writer(self, writer: Any):
        self.writer = writer

    def update(self,
               tensor_dict: Dict[str, Any],
               step: Optional[int] = None,
               stage: str = "val",
               n: int = 1) -> Dict[str, Any]:
        """
        Compute all metrics for the current sample (or mini-batch),
        update running averages, and return the instantaneous values.

        Each metric config may specify:
          - pred_key: key inside predictions or model_inputs (predictions searched first)
          - lab_key:  key inside predictions or model_inputs (model_inputs searched second)
          - kwargs:   extra args to pass to the metric function

        Special-case: if prediction tensor has ndim==5 (B,E,...) → mean over ensemble (dim=1).
        """
        out: Dict[str, Any] = {}

        # Average over ensemble dimension if given
        for key in tensor_dict.keys():
            if isinstance(tensor_dict[key], torch.Tensor) and tensor_dict[key].ndim == 5:
                tensor_dict[key] = tensor_dict[key].mean(dim=1)

        for metric_name, metric in self._metrics.items():
            try:
                result = metric.forward(tensor_dict)
            except Exception:
                print(f"Warning: metric '{metric_name}' failed.")
                raise
            if not isinstance(result, dict):
                raise TypeError(f"Metric '{metric_name}' must return a dict, got {type(result)}")

            for key, val in result.items():
                key_name = str(key)
                scalar = _to_scalar(val)
                out[key_name] = scalar
                self._totals[key_name] = self._totals.get(key_name, 0.0) + float(scalar) * n
                self._counts[key_name] = self._counts.get(key_name, 0) + n
                if self.writer is not None and hasattr(self.writer, "add_scalar"):
                    self.writer.add_scalar(
                        f"{stage}/{key_name}",
                        float(scalar),
                        global_step=0 if step is None else step,
                    )

        return out

    def averages(self) -> Dict[str, float]:
        """Return running averages for each metric."""
        avg = {}
        for k in self._totals:
            c = max(1, self._counts[k])
            avg[k] = self._totals[k] / c
        return avg

    @staticmethod
    def _resolve_metric_class(cfg: Dict[str, Any]) -> type:
        class_name = cfg.get("class")
        if not class_name:
            raise ValueError(
                f"Metric '{cfg.get('name', '<unnamed>')}' must set 'class' in config."
            )
        if ":" in class_name:
            module_name, symbol_name = class_name.split(":", 1)
            module = importlib.import_module(module_name)
            metric_cls = getattr(module, symbol_name)
        else:
            metric_cls = globals().get(class_name)
        if metric_cls is None:
            raise ValueError(f"Unknown metric class '{class_name}'.")
        return metric_cls


def _resolve_key_path(runtime: Dict[str, Any], key_path: str) -> Any:
    cur: Any = runtime
    for part in key_path.split("/"):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(f"Failed to resolve key path '{key_path}' at '{part}'.")
    return cur


def _to_scalar(val: Any) -> float:
    if isinstance(val, torch.Tensor):
        vv = val.detach().cpu()
        return float(vv.item()) if vv.numel() == 1 else float(vv.mean().item())
    if isinstance(val, np.ndarray):
        return float(val.reshape(())) if val.size == 1 else float(val.mean())
    return float(val)


class HausdorffDistanceMetric:
    def __init__(self, key_map: Dict[str, str], kwargs: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        self.key_map = dict(key_map or {})
        self.kwargs = dict(kwargs or {})
        self.name = name or "hausdorff"

    def forward(self, runtime: Dict[str, Any]) -> Dict[str, float]:
        pred = runtime[self.key_map["pred"]]
        odom = runtime[self.key_map["odom"]]
        val = compute_hausdorff_distance(pred=pred, odom=odom, **self.kwargs)
        return {f"{self.name}/score": float(val)}


class RewardFunctionMetric:
    def __init__(self, key_map: Dict[str, str], kwargs: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        self.key_map = dict(key_map or {})
        self.kwargs = dict(kwargs or {})
        self.name = name or "reward"
        fn_path = self.kwargs.get("fn")
        if not fn_path:
            raise ValueError(f"Metric '{self.name}' requires kwargs.fn for reward function path.")
        module_name, fn_name = str(fn_path).split(":", 1)
        module = importlib.import_module(module_name)
        self.reward_fn = getattr(module, fn_name)

    def forward(self, runtime: Dict[str, Any]) -> Dict[str, float]:
        solution_key = self.key_map["solution_str"]
        ground_truth_key = self.key_map["ground_truth"]

        solution_str = runtime[solution_key]
        ground_truth = runtime[ground_truth_key]
        extra_info_cfg = copy.deepcopy(self.kwargs.get("extra_info", {}))
        if OmegaConf.is_config(extra_info_cfg):
            extra_info_cfg = OmegaConf.to_container(extra_info_cfg)
        if "trace_pts" in self.key_map:
            trace_pts_key = self.key_map["trace_pts"]
            extra_info_cfg["trace_pts"] = runtime[trace_pts_key]
        if "motion_response" in self.key_map:
            motion_response_key = self.key_map["motion_response"]
            extra_info_cfg["motion_response"] = runtime[motion_response_key]
        
        call_kwargs = {
            k: copy.deepcopy(v)
            for k, v in self.kwargs.items()                                                                               
            if k not in {"fn", "extra_info"}
        }

        rewards = self.reward_fn(
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info_cfg,
            **call_kwargs,
        )
        if not isinstance(rewards, dict):
            raise TypeError(
                f"Reward function for metric '{self.name}' must return dict, got {type(rewards)}"
            )
        out: Dict[str, float] = {}
        for k, v in rewards.items():
            out[f"{self.name}/{k}"] = _to_scalar(v)

        return out

def intersection_over_union(pred, target, valid_mask=None):
    """Calculate Intersection over Union (IoU) for segmentation tasks."""
    if valid_mask is not None:
        pred = pred[valid_mask]
        target = target[valid_mask]
    pred = pred > 0.5  
    target = target > 0.5
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection

    if union == 0:
        return torch.tensor(0.0, device=pred.device)

    iou = intersection / union
    return iou

def precision_recall(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask: np.ndarray = None,
    num_bins: int = 100,
    gt_threshold: float = 0.5,
    return_counts: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a precision–recall curve between ground-truth and predicted masks.

    Args:
        gt_arr:     (B,1,H,W) array of ground-truth values in [0,1].
        pred_arr:   (B,1,H,W) array of predicted probabilities in [0,1].
        num_bins:   number of thresholds to sweep between 0 and 1.
        gt_threshold: threshold to binarize ground truth into positives.
        return_counts: if True, return (tp, fp, fn) counts instead of precision/recall.

    Returns:
        If return_counts=False:
            thresholds: (num_bins,) array of thresholds used.
            precision:  (num_bins,) array of precision at each threshold.
            recall:     (num_bins,) array of recall at each threshold.
        If return_counts=True:
            thresholds: (num_bins,) array of thresholds used.
            tp_counts:  (num_bins,) array of true positive counts.
            fp_counts:  (num_bins,) array of false positive counts.
            fn_counts:  (num_bins,) array of false negative counts.
    """
    if isinstance(gt_arr, torch.Tensor):
        gt_arr = gt_arr.detach().cpu().numpy()
    if isinstance(pred_arr, torch.Tensor):
        pred_arr = pred_arr.detach().cpu().numpy()
    if isinstance(valid_mask, torch.Tensor):
        valid_mask = valid_mask.detach().cpu().numpy()

    # flatten to 1D
    gt_flat   = gt_arr.reshape(-1)
    pred_flat = pred_arr.reshape(-1)

    if valid_mask is not None:
        # apply valid mask if provided
        valid_mask = valid_mask.reshape(-1)
        gt_flat   = gt_flat[valid_mask]
        pred_flat = pred_flat[valid_mask]

    # binarize ground truth
    gt_pos = gt_flat >= gt_threshold
    n_pos  = gt_pos.sum()

    thresholds = np.linspace(0.0, 1.0, num_bins)
    
    if return_counts:
        tp_counts = np.empty(num_bins, dtype=int)
        fp_counts = np.empty(num_bins, dtype=int)
        fn_counts = np.empty(num_bins, dtype=int)
        
        for i, thr in enumerate(thresholds):
            pred_pos = pred_flat >= thr
            tp = int(np.logical_and(pred_pos, gt_pos).sum())
            pp = int(pred_pos.sum())
            fp = pp - tp
            fn = int(n_pos - tp)
            
            tp_counts[i] = tp
            fp_counts[i] = fp
            fn_counts[i] = fn
            
        return thresholds, tp_counts, fp_counts, fn_counts
    else:
        precision = np.empty(num_bins, dtype=float)
        recall    = np.empty(num_bins, dtype=float)
        for i, thr in enumerate(thresholds):
            pred_pos = pred_flat >= thr
            tp = int(np.logical_and(pred_pos, gt_pos).sum())
            pp = int(pred_pos.sum())
            fn = int(n_pos - tp)

            precision[i] = tp / pp if pp > 0 else 1.0
            recall[i]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return thresholds, precision, recall

# Loss computations for nontensor training
def compute_accuracy(arcs: np.ndarray, pred: np.ndarray, odom: np.ndarray) -> float:
    """
    arcs: (B, K, N, 3), pred: (B, N, 3), odom: (B, M, 3)
    Returns average accuracy over batch.
    """
    
    if isinstance(arcs, torch.Tensor):
        arcs, pred, odom = arcs.detach().cpu().numpy(), pred.detach().cpu().numpy(), odom.detach().cpu().numpy()
    B, K, N, _ = arcs.shape
    if odom.ndim == 2:
        odom = odom[None, ...]  # (1, M, 3)

    # Get best prediction for each batch element (B,)
    
    # Compute Hausdorff distances for all combinations (B, K)
    hausdorff_distances = np.zeros((B, K))
    hausdorff_distances_pred = np.zeros((B, K))
    for b in range(B):
        for k in range(K):
            hausdorff_distances[b, k] = hausdorff_xyz(arcs[b, k], odom[b])
            hausdorff_distances_pred[b, k] = hausdorff_xyz(arcs[b, k], pred[b])
    
    selected_k_indices = np.argmin(hausdorff_distances_pred, axis=1)
    # Find ground truth indices (best Hausdorff distance for each batch) (B,)
    ground_truth_indices = np.argmin(hausdorff_distances, axis=1)
    
    # Compute accuracy as fraction of correct predictions
    acc = np.mean(selected_k_indices == ground_truth_indices)
    
    return acc

def compute_hausdorff_distance(pred: np.ndarray, odom: np.ndarray, source="pred", oneway=True) -> float:
    """
    arcs: (B, K, N, 3), pred: (B, N, 3), odom: (B, M, 3)
    Returns average hdist(model, odom) over batch.
    """

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(odom, torch.Tensor):
        odom = odom.float().detach().cpu().numpy()
    if isinstance(pred, dict):
        pred = pred['trajectory']
    
    B, N, _ = odom.shape
    if odom.ndim == 2:
        odom = odom[None, ...]  # (1, M, 3)
    total_distance = 0.0
    for b in range(B):
        src, tgt = pred[b], odom[b]
        if source == "odom":
            src, tgt = odom[b], pred[b]
        src = np.array(src, dtype=np.float32)
        tgt = np.array(tgt, dtype=np.float32)
        total_distance += hausdorff_xyz(src, tgt, oneway=oneway)
    return total_distance / B

def compute_relative_hausdorff_distance(arcs: np.ndarray, pred: np.ndarray, odom: np.ndarray, oneway=True) -> float:
    """
    arcs: (B, K, N, 3), pred: (B, N, 3), odom: (B, M, 3)
    Returns average hdist(model arc, odom arc)
    """
    if isinstance(arcs, torch.Tensor):
        arcs, pred, odom = arcs.detach().cpu().numpy(), pred.detach().cpu().numpy(), odom.detach().cpu().numpy()
    B, K, N, _ = arcs.shape
    if odom.ndim == 2:
        odom = odom[None, ...]  # (1, M, 3)
    total_distance = 0.0
    # Compute Hausdorff distances for all combinations (B, K)
    hausdorff_distances = np.zeros((B, K))
    hausdorff_distances_pred = np.zeros((B, K))
    for b in range(B):
        for k in range(K):
            hausdorff_distances[b, k] = hausdorff_xyz(arcs[b, k], odom[b])
            hausdorff_distances_pred[b, k] = hausdorff_xyz(arcs[b, k], pred[b])

    total_distance = np.sum(np.abs(np.min(hausdorff_distances_pred, axis=1) - np.min(hausdorff_distances, axis=1)))
    return total_distance / B

def hausdorff_xyz(A: np.ndarray, B: np.ndarray, oneway=True) -> float:
    """Oneway Hausdorff distance between two polylines A(N,3) and B(M,3)."""
    if A.size == 0 or B.size == 0: 
        return np.inf
    D = cdist(A, B)  # (N,M)
    if oneway:
        return float(D.min(axis=1).max())
    else:
        return float(max(D.min(axis=1).max(), D.min(axis=0).max()))

if __name__ == "__main__":
    print("Testing compute_accuracy with specific probabilities...")