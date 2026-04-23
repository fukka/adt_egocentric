"""
eval_utils.py
=============
Shared utilities for the egocentric depth estimation benchmark.

Provides:
  - Image / depth loading with optional rotation
  - Valid-pixel masking
  - Least-squares affine (scale+shift) alignment for relative-depth models
  - Median-ratio scale-only alignment
  - Standard depth evaluation metrics: AbsRel, SqRel, RMSE, RMSElog, δ1/2/3
  - Side-by-side comparison figure saving
  - CSV result logging
"""

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

ROTATION_MAP = {
    0:   None,
    90:  Image.ROTATE_90,
    180: Image.ROTATE_180,
    270: Image.ROTATE_270,
}

METRICS_HEADER = [
    "model", "variant",
    "AbsRel", "SqRel", "RMSE", "RMSElog",
    "delta1", "delta2", "delta3",
    "alignment",
]


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_rgb(path: str, rotation: int = 0) -> np.ndarray:
    """
    Load an RGB image as a uint8 HxWx3 numpy array.

    Parameters
    ----------
    path     : path to the image file (PNG, JPG, …)
    rotation : clockwise rotation in degrees — one of {0, 90, 180, 270}

    Returns
    -------
    np.ndarray  uint8, shape (H, W, 3), RGB order
    """
    if rotation not in ROTATION_MAP:
        raise ValueError(f"rotation must be one of {list(ROTATION_MAP.keys())}, got {rotation}")
    img = Image.open(path).convert("RGB")
    if rotation != 0:
        img = img.transpose(ROTATION_MAP[rotation])
    return np.array(img)


def load_depth_gt(path: str, depth_scale: float = 1.0,
                  min_depth: float = 0.01, max_depth: float = 100.0) -> np.ndarray:
    """
    Load a ground-truth depth map from a .npy file.

    Parameters
    ----------
    path        : path to the .npy depth map
    depth_scale : multiply all values by this factor (e.g. 0.001 to convert mm→m)
    min_depth   : pixels below this value (after scaling) are treated as invalid
    max_depth   : pixels above this value (after scaling) are treated as invalid

    Returns
    -------
    np.ndarray  float32, shape (H, W); invalid pixels are set to NaN
    """
    depth = np.load(path).astype(np.float32)
    if depth.ndim == 3:
        depth = depth.squeeze(-1)
    depth = depth * depth_scale
    # Mark invalid pixels
    invalid = (depth <= min_depth) | (depth >= max_depth) | ~np.isfinite(depth)
    depth[invalid] = np.nan
    return depth


def get_valid_mask(gt: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels with finite, positive GT depth."""
    return np.isfinite(gt) & (gt > 0)


# ──────────────────────────────────────────────────────────────────────────────
# Depth alignment
# ──────────────────────────────────────────────────────────────────────────────

def align_scale_shift(pred: np.ndarray, gt: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
    """
    Least-squares affine alignment:  pred_aligned = scale * pred + shift
    Solves the 2×2 linear system [scale, shift] to minimise ||scale*p + shift - g||^2
    over valid pixels.

    Suitable for affine-invariant (relative-depth) models such as
    Depth Anything V2 and Marigold.
    """
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    A = np.stack([p, np.ones_like(p)], axis=1)          # (N, 2)
    x, _, _, _ = np.linalg.lstsq(A, g, rcond=None)      # [scale, shift]
    scale, shift = x
    return (scale * pred + shift).astype(np.float32)


def align_scale_only(pred: np.ndarray, gt: np.ndarray,
                     mask: np.ndarray) -> np.ndarray:
    """
    Median-ratio scale alignment:  pred_aligned = median(gt/pred) * pred
    A lighter alternative that preserves the relative depth distribution.
    """
    ratio = np.median(gt[mask] / (pred[mask] + 1e-8))
    return (ratio * pred).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, gt: np.ndarray,
                    mask: np.ndarray) -> dict:
    """
    Compute standard depth evaluation metrics over valid (masked) pixels.

    Parameters
    ----------
    pred  : predicted depth map (float32, H×W), already aligned to GT scale
    gt    : ground-truth depth map (float32, H×W)
    mask  : boolean valid-pixel mask (H×W)

    Returns
    -------
    dict with keys: AbsRel, SqRel, RMSE, RMSElog, delta1, delta2, delta3
    """
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)

    # Clamp predictions to avoid log(0)
    p = np.clip(p, 1e-6, None)
    g = np.clip(g, 1e-6, None)

    diff  = np.abs(p - g)
    diff2 = (p - g) ** 2

    abs_rel = np.mean(diff / g)
    sq_rel  = np.mean(diff2 / g)
    rmse    = np.sqrt(np.mean(diff2))
    rmselog = np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))

    ratio   = np.maximum(p / g, g / p)
    delta1  = np.mean(ratio < 1.25)
    delta2  = np.mean(ratio < 1.25 ** 2)
    delta3  = np.mean(ratio < 1.25 ** 3)

    return {
        "AbsRel":  float(abs_rel),
        "SqRel":   float(sq_rel),
        "RMSE":    float(rmse),
        "RMSElog": float(rmselog),
        "delta1":  float(delta1),
        "delta2":  float(delta2),
        "delta3":  float(delta3),
    }


def print_metrics(metrics: dict, model: str, variant: str = "",
                  alignment: str = "scale+shift") -> None:
    """Pretty-print a metrics dict to stdout."""
    label = f"{model}" + (f" [{variant}]" if variant else "")
    print(f"\n{'='*60}")
    print(f"  {label}  (alignment: {alignment})")
    print(f"{'='*60}")
    print(f"  AbsRel : {metrics['AbsRel']:.4f}")
    print(f"  SqRel  : {metrics['SqRel']:.4f}")
    print(f"  RMSE   : {metrics['RMSE']:.4f}")
    print(f"  RMSElog: {metrics['RMSElog']:.4f}")
    print(f"  δ₁     : {metrics['delta1']*100:.2f}%")
    print(f"  δ₂     : {metrics['delta2']*100:.2f}%")
    print(f"  δ₃     : {metrics['delta3']*100:.2f}%")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

DEPTH_CMAP = "magma_r"   # Warm colours = near; cool = far


def _normalise_depth_for_display(depth: np.ndarray,
                                  vmin: float = None,
                                  vmax: float = None) -> np.ndarray:
    """Normalise depth to [0, 1] for colormap display, ignoring NaN."""
    d = depth.copy().astype(np.float32)
    if vmin is None:
        vmin = float(np.nanpercentile(d, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(d, 98))
    d = np.clip(d, vmin, vmax)
    d = (d - vmin) / (vmax - vmin + 1e-8)
    d[~np.isfinite(depth)] = 0.0
    return d


def save_comparison_figure(rgb: np.ndarray,
                            gt: np.ndarray,
                            pred_aligned: np.ndarray,
                            metrics: dict,
                            model: str,
                            variant: str,
                            alignment: str,
                            output_path: str) -> None:
    """
    Save a 4-panel figure: RGB | GT depth | Predicted depth | Error map.

    Parameters
    ----------
    rgb           : uint8 (H, W, 3) RGB image
    gt            : float32 (H, W) GT depth (NaN = invalid)
    pred_aligned  : float32 (H, W) predicted depth after alignment
    metrics       : dict from compute_metrics()
    model/variant : strings for the title
    alignment     : alignment strategy string for the subtitle
    output_path   : path to save the PNG figure
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Shared depth range based on GT
    vmin = float(np.nanpercentile(gt, 2))
    vmax = float(np.nanpercentile(gt, 98))

    gt_norm   = _normalise_depth_for_display(gt, vmin, vmax)
    pred_norm = _normalise_depth_for_display(pred_aligned, vmin, vmax)

    # Absolute error map (clipped at 95th percentile)
    err = np.abs(pred_aligned - gt)
    err[~np.isfinite(gt)] = np.nan
    err_max = float(np.nanpercentile(err, 95))
    err_norm = np.clip(err, 0, err_max) / (err_max + 1e-8)
    err_norm[~np.isfinite(err)] = 0.0

    cmap_d = plt.get_cmap(DEPTH_CMAP)
    cmap_e = plt.get_cmap("hot")

    fig = plt.figure(figsize=(20, 5))
    fig.patch.set_facecolor("#1a1a2e")
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)

    panels = [
        (rgb,                    "Input RGB",        None),
        (cmap_d(gt_norm)[..., :3],  "GT Depth",      f"[{vmin:.2f} – {vmax:.2f} m]"),
        (cmap_d(pred_norm)[..., :3], f"Predicted ({model})", f"alignment: {alignment}"),
        (cmap_e(err_norm)[..., :3], "Absolute Error", f"(0 – {err_max:.2f} m)"),
    ]

    for i, (img_data, title, subtitle) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        if i == 0:
            ax.imshow(img_data)
        else:
            ax.imshow(img_data, vmin=0, vmax=1)
        ax.set_title(title, color="white", fontsize=11, pad=4)
        if subtitle:
            ax.set_xlabel(subtitle, color="#aaaaaa", fontsize=8)
        ax.axis("off")

    # Metrics text box
    label = f"{model}" + (f"  [{variant}]" if variant else "")
    metrics_str = (
        f"{label}\n"
        f"AbsRel={metrics['AbsRel']:.4f}  SqRel={metrics['SqRel']:.4f}\n"
        f"RMSE={metrics['RMSE']:.4f}  RMSElog={metrics['RMSElog']:.4f}\n"
        f"δ₁={metrics['delta1']*100:.1f}%  δ₂={metrics['delta2']*100:.1f}%  δ₃={metrics['delta3']*100:.1f}%"
    )
    fig.text(0.5, 0.01, metrics_str, ha="center", va="bottom",
             color="white", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#2d2d44", alpha=0.85))

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [viz] Saved → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CSV result logging
# ──────────────────────────────────────────────────────────────────────────────

def append_to_csv(csv_path: str, model: str, variant: str,
                  metrics: dict, alignment: str) -> None:
    """Append one result row to a CSV file (creates header if file is new)."""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "model":     model,
            "variant":   variant,
            "AbsRel":    f"{metrics['AbsRel']:.6f}",
            "SqRel":     f"{metrics['SqRel']:.6f}",
            "RMSE":      f"{metrics['RMSE']:.6f}",
            "RMSElog":   f"{metrics['RMSElog']:.6f}",
            "delta1":    f"{metrics['delta1']:.6f}",
            "delta2":    f"{metrics['delta2']:.6f}",
            "delta3":    f"{metrics['delta3']:.6f}",
            "alignment": alignment,
        })
    print(f"  [csv] Results appended → {csv_path}")
