"""
training_utils.py
=================
Shared helpers for fine-tuning scripts:
  - colorize_depth     : float32 depth array → uint8 RGB via matplotlib colormap
  - denormalize_rgb    : undo ImageNet normalization → uint8 HWC numpy array
  - align_depth_pred   : least-squares disparity-space alignment (matches eval_depth_anything_v2.py)
  - log_image_grid     : write [RGB | metric pred | (relative pred) | GT] grid to TensorBoard
  - append_metrics_csv : append one epoch row to a CSV results file
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
import torchvision.utils as vutils

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_METRICS_HEADER = [
    'split', 'variant', 'epoch',
    'd1', 'd2', 'd3',
    'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog',
]


def colorize_depth(depth_np: np.ndarray, vmin: float = None, vmax: float = None) -> np.ndarray:
    """
    Convert a (H, W) float32 depth array to a (H, W, 3) uint8 RGB image
    using the 'plasma' colormap.  Invalid pixels (NaN / <=0) are rendered black.
    """
    valid = np.isfinite(depth_np) & (depth_np > 0)
    lo = float(depth_np[valid].min()) if vmin is None else vmin
    hi = float(depth_np[valid].max()) if vmax is None else vmax
    norm = np.zeros_like(depth_np, dtype=np.float32)
    if hi > lo:
        norm[valid] = (depth_np[valid] - lo) / (hi - lo)
    rgba = (cm.plasma(norm) * 255).astype(np.uint8)   # (H, W, 4)
    rgb  = rgba[..., :3]                               # drop alpha
    rgb[~valid] = 0
    return rgb


def denormalize_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Undo ImageNet normalization on a (3, H, W) float32 CPU tensor.
    Returns a (H, W, 3) uint8 numpy array.
    """
    img = img_tensor.cpu().float() * _IMAGENET_STD + _IMAGENET_MEAN
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def align_depth_pred(
    pred_np: np.ndarray,
    gt_np: np.ndarray,
    min_depth: float = 0.001,
    max_depth: float = 10.0,
) -> np.ndarray:
    """
    Least-squares scale+shift alignment in disparity (1/depth) space.

    Matches the alignment used in eval_depth_anything_v2.py so that the relative
    baseline in the vis grid is directly comparable to the offline eval numbers.

    Parameters
    ----------
    pred_np  : (H, W) float32 — raw relative depth output (arbitrary scale)
    gt_np    : (H, W) float32 — metric GT depth in metres
    min/max_depth : valid GT depth range

    Returns
    -------
    (H, W) float32 aligned metric depth.  Pixels where the aligned disparity is
    non-positive are set to NaN (shown as black in colorize_depth).
    """
    valid = (
        (gt_np > min_depth) & (gt_np <= max_depth)
        & np.isfinite(pred_np) & (pred_np > 0)
    )
    if valid.sum() < 10:
        return np.full_like(pred_np, np.nan)

    dp = pred_np[valid].ravel().astype(np.float64)
    dg = (1.0 / gt_np[valid].ravel()).astype(np.float64)
    A  = np.stack([dp, np.ones_like(dp)], axis=1)
    x, _, _, _ = np.linalg.lstsq(A, dg, rcond=None)
    scale, shift = float(x[0]), float(x[1])

    disp_aligned  = scale * pred_np.astype(np.float64) + shift
    depth_aligned = np.where(disp_aligned > 0, 1.0 / disp_aligned, np.nan)
    return depth_aligned.astype(np.float32)


def log_image_grid(
    writer,
    tag: str,
    rgb_list: list,           # list of (3, H, W) float tensors (ImageNet-normalized)
    pred_list: list,          # list of (H, W) float tensors — finetuned metric pred
    gt_list: list,            # list of (H, W) float tensors — GT metric depth
    step: int,
    n: int = 4,
    rel_list: list = None,    # optional list of (H, W) float tensors — relative baseline pred
                              # (already aligned to GT scale via align_depth_pred)
) -> None:
    """
    Build a side-by-side depth comparison grid and write it to TensorBoard.
    At most `n` samples (rows) are shown.

    Without rel_list  →  3 columns : [RGB input | finetuned metric pred | GT]
    With    rel_list  →  4 columns : [RGB input | finetuned metric pred | relative baseline | GT]

    All depth maps share a [0, GT_max] colour range per row so that pred, relative,
    and GT are directly comparable within each row.
    """
    if writer is None:
        return

    use_rel = rel_list is not None and len(rel_list) > 0
    ncols   = 4 if use_rel else 3

    rows = []
    iters = zip(rgb_list[:n], pred_list[:n], gt_list[:n])
    if use_rel:
        iters = zip(rgb_list[:n], pred_list[:n], gt_list[:n], rel_list[:n])

    for sample in iters:
        if use_rel:
            rgb_t, pred_t, gt_t, rel_t = sample
        else:
            rgb_t, pred_t, gt_t = sample
            rel_t = None

        gt_np   = gt_t.cpu().float().numpy()
        pred_np = pred_t.cpu().float().numpy()

        # Shared depth range anchored to GT so all columns are comparable
        valid = gt_np > 0
        vmax  = float(gt_np[valid].max()) if valid.any() else 1.0
        vmin  = 0.0

        def to_chw(arr):
            return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

        rgb_np   = denormalize_rgb(rgb_t)
        pred_rgb = colorize_depth(pred_np, vmin=vmin, vmax=vmax)
        gt_rgb   = colorize_depth(gt_np,   vmin=vmin, vmax=vmax)

        row = [to_chw(rgb_np), to_chw(pred_rgb)]
        if use_rel:
            rel_np  = rel_t.cpu().float().numpy()
            rel_rgb = colorize_depth(rel_np, vmin=vmin, vmax=vmax)
            row.append(to_chw(rel_rgb))
        row.append(to_chw(gt_rgb))
        rows += row

    if not rows:
        return

    grid = vutils.make_grid(rows, nrow=ncols, padding=4, pad_value=0.5)
    writer.add_image(tag, grid, global_step=step)


def append_metrics_csv(csv_path: str, row: dict) -> None:
    """
    Append one epoch's metrics to a CSV file.  Creates the file with a header
    row on first call; subsequent calls append without writing the header again.
    """
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_METRICS_HEADER, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)