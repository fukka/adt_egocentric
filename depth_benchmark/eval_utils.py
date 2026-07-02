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
  - 3D Geometric Metric (3DGM) from GeoNet / GeoNet++:
      depth → back-projection → surface normals → angular error vs GT normals
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

NORMAL_METRICS_HEADER = [
    "model", "variant",
    "mean_deg", "median_deg", "rmse_deg",
    "pct_11.25", "pct_22.5", "pct_30", "pct_45",
    "normal_source",
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
    rotation : counter-clockwise rotation in degrees — one of {0, 90, 180, 270}

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


def align_scale_shift_disparity(pred: np.ndarray, gt: np.ndarray,
                                mask: np.ndarray) -> np.ndarray:
    """
    Affine alignment in inverse-depth (disparity) space for relative-depth models.

    Solves:  scale * pred + shift ≈ 1/gt  over GT-valid pixels,
    then returns:  1 / (scale * pred + shift)  as metric depth.

    For models like Depth Anything V2 that output disparity-like values
    (larger = closer), aligning in disparity space guarantees a positive
    scale and avoids the sign-flip that depth-space alignment produces when
    pred and GT have inverted polarity.  Pixels where the aligned disparity
    is ≤ 0 are set to NaN and excluded from metrics and normal computation.

    Reference: MiDaS / DPT evaluation protocol (Ranftl et al., TPAMI 2022).
    """
    p   = pred[mask].astype(np.float64)
    g_d = 1.0 / gt[mask].astype(np.float64)           # GT disparity, always > 0
    A   = np.stack([p, np.ones_like(p)], axis=1)       # (N, 2)
    x, _, _, _ = np.linalg.lstsq(A, g_d, rcond=None)  # [scale, shift]
    scale, shift = x
    pred_disp  = scale * pred.astype(np.float64) + shift
    pred_depth = np.where(pred_disp > 0, 1.0 / pred_disp, np.nan)
    return pred_depth.astype(np.float32)


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

    # Shared depth range anchored to GT so both panels use the same colour scale.
    # This is critical for honest visual comparison: if pred_aligned diverges
    # from GT (e.g. RMSE 2+ m), the error will be visible as a hue difference
    # rather than hidden by independent per-map normalisation.
    vmin = float(np.nanpercentile(gt, 2))
    vmax = float(np.nanpercentile(gt, 98))

    gt_norm   = _normalise_depth_for_display(gt,           vmin, vmax)
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


def save_normal_comparison_figure(rgb: np.ndarray,
                                   gt_normals: np.ndarray,
                                   pred_normals: np.ndarray,
                                   metrics_3dgm: dict,
                                   model: str,
                                   variant: str,
                                   normal_source: str,
                                   output_path: str) -> None:
    """
    Save a 4-panel figure: RGB | GT normals | Predicted normals | Angular error.

    Parameters
    ----------
    rgb           : uint8 (H, W, 3) RGB image
    gt_normals    : float32 (H, W, 3) GT surface normals (NaN = invalid)
    pred_normals  : float32 (H, W, 3) predicted normals derived from depth
    metrics_3dgm  : dict from compute_3dgm_metrics()
    model/variant : strings for the title
    normal_source : description of how pred_normals were obtained
    output_path   : path to save the PNG figure
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    def normals_to_rgb(n: np.ndarray) -> np.ndarray:
        """Map unit normals [-1,1]³ → RGB [0,1]³; invalid pixels are black."""
        out = np.zeros((*n.shape[:2], 3), dtype=np.float32)
        valid = np.all(np.isfinite(n), axis=-1)
        out[valid] = np.clip((n[valid] + 1.0) / 2.0, 0.0, 1.0)
        return out

    gt_rgb   = normals_to_rgb(gt_normals)
    pred_rgb = normals_to_rgb(pred_normals)

    # Per-pixel angular error map
    both_valid = (np.all(np.isfinite(pred_normals), axis=-1)
                  & np.all(np.isfinite(gt_normals), axis=-1))
    dot = np.where(both_valid,
                   np.clip(np.sum(pred_normals * gt_normals, axis=-1), -1.0, 1.0),
                   np.nan)
    angle_deg = np.where(both_valid, np.degrees(np.arccos(dot)), np.nan)
    err_max = float(np.nanpercentile(angle_deg, 95))
    err_norm = np.nan_to_num(np.clip(angle_deg / (err_max + 1e-8), 0.0, 1.0), nan=0.0)

    cmap_e = plt.get_cmap("hot")

    fig = plt.figure(figsize=(20, 5))
    fig.patch.set_facecolor("#1a1a2e")
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)

    panels = [
        (rgb,                           "Input RGB",                    None),
        (gt_rgb,                        "GT Normals",                   "(XYZ → RGB)"),
        (pred_rgb,                      f"Predicted Normals ({model})", normal_source),
        (cmap_e(err_norm)[..., :3],     "Angular Error",                f"(0 – {err_max:.1f}°)"),
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

    label = f"{model}" + (f"  [{variant}]" if variant else "")
    m = metrics_3dgm
    metrics_str = (
        f"{label}\n"
        f"Mean={m['mean_deg']:.2f}°  Median={m['median_deg']:.2f}°  RMSE={m['rmse_deg']:.2f}°\n"
        f"≤11.25°={m['pct_11.25']*100:.1f}%  ≤22.5°={m['pct_22.5']*100:.1f}%"
        f"  ≤30°={m['pct_30']*100:.1f}%  ≤45°={m['pct_45']*100:.1f}%"
    )
    fig.text(0.5, 0.01, metrics_str, ha="center", va="bottom",
             color="white", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#2d2d44", alpha=0.85))

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [viz] Saved → {output_path}")


def save_normal_comparison_figure_2(pred_depth: np.ndarray,
                                   gt_normals: np.ndarray,
                                   pred_normals: np.ndarray,
                                   metrics_3dgm: dict,
                                   model: str,
                                   variant: str,
                                   normal_source: str,
                                   output_path: str,
                                   alignment: str = "scale+shift (least-squares)") -> None:
    """
    Save a 4-panel figure: pred depth | GT normals | Predicted normals | Angular error.

    Parameters
    ----------
    pred_depth    : float32 (H, W) aligned predicted depth map
    gt_normals    : float32 (H, W, 3) GT surface normals (NaN = invalid)
    pred_normals  : float32 (H, W, 3) predicted normals derived from depth
    metrics_3dgm  : dict from compute_3dgm_metrics()
    model/variant : strings for the title
    normal_source : description of how pred_normals were obtained
    output_path   : path to save the PNG figure
    alignment     : alignment strategy label shown on the depth panel subtitle
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    def normals_to_rgb(n: np.ndarray) -> np.ndarray:
        """Map unit normals [-1,1]³ → RGB [0,1]³; invalid pixels are black."""
        out = np.zeros((*n.shape[:2], 3), dtype=np.float32)
        valid = np.all(np.isfinite(n), axis=-1)
        out[valid] = np.clip((n[valid] + 1.0) / 2.0, 0.0, 1.0)
        return out

    gt_rgb   = normals_to_rgb(gt_normals)
    pred_rgb = normals_to_rgb(pred_normals)

    cmap_d = plt.get_cmap(DEPTH_CMAP)
    vmin = float(np.nanpercentile(pred_depth, 2))
    vmax = float(np.nanpercentile(pred_depth, 98))
    pred_norm = _normalise_depth_for_display(pred_depth, vmin, vmax)

    # Per-pixel angular error map
    both_valid = (np.all(np.isfinite(pred_normals), axis=-1)
                  & np.all(np.isfinite(gt_normals), axis=-1))
    dot = np.where(both_valid,
                   np.clip(np.sum(pred_normals * gt_normals, axis=-1), -1.0, 1.0),
                   np.nan)
    angle_deg = np.where(both_valid, np.degrees(np.arccos(dot)), np.nan)
    err_max = float(np.nanpercentile(angle_deg, 95))
    err_norm = np.nan_to_num(np.clip(angle_deg / (err_max + 1e-8), 0.0, 1.0), nan=0.0)

    cmap_e = plt.get_cmap("hot")

    fig = plt.figure(figsize=(20, 5))
    fig.patch.set_facecolor("#1a1a2e")
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)

    panels = [
        (cmap_d(pred_norm)[..., :3],    "Predicted Depth",              f"alignment: {alignment}"),
        (gt_rgb,                        "GT Normals",                   "(XYZ → RGB)"),
        (pred_rgb,                      f"Predicted Normals ({model})", normal_source),
        (cmap_e(err_norm)[..., :3],     "Angular Error",                f"(0 – {err_max:.1f}°)"),
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

    label = f"{model}" + (f"  [{variant}]" if variant else "")
    m = metrics_3dgm
    metrics_str = (
        f"{label}\n"
        f"Mean={m['mean_deg']:.2f}°  Median={m['median_deg']:.2f}°  RMSE={m['rmse_deg']:.2f}°\n"
        f"≤11.25°={m['pct_11.25']*100:.1f}%  ≤22.5°={m['pct_22.5']*100:.1f}%"
        f"  ≤30°={m['pct_30']*100:.1f}%  ≤45°={m['pct_45']*100:.1f}%"
    )
    fig.text(0.5, 0.01, metrics_str, ha="center", va="bottom",
             color="white", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#2d2d44", alpha=0.85))

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [viz] Saved → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3D Geometric Metric (3DGM) — GeoNet / GeoNet++ protocol
# ──────────────────────────────────────────────────────────────────────────────
# Reference: Yin & Shi, "GeoNet: Unsupervised Learning of Dense Depth,
#   Optical Flow and Camera Pose", CVPR 2018.
# Metric definition: back-project predicted depth to 3D using camera intrinsics,
#   compute surface normals via central-difference cross-products, then evaluate
#   angular error against ground-truth normals.
# Angular error metrics match the standard NYU-Depth v2 normal evaluation:
#   mean/median/RMSE of error in degrees + fraction below 11.25°/22.5°/30°.


def load_normal_gt(path: str) -> np.ndarray:
    """
    Load a ground-truth surface normal map and return it as-is (no coordinate
    transform applied).

    For the ADT dataset the .npy files contain Blender world-space normals
    (ADT Y-up frame, from the Blender Cycles Normal pass).  To compare against
    camera-space normals produced by depth_to_normals() the caller must rotate
    these into camera space:
        n_cam = R_world_to_cam @ n_world   (per pixel, ignoring Blender↔OpenCV axis flip)
    where R_world_to_cam is the 3×3 rotation part of the camera extrinsic matrix.

    Supported formats
    -----------------
    .npy : (H, W, 3) float32 — unit normals (values in [-1, 1])
    .png : (H, W, 3) uint8   — normals encoded as (n + 1) / 2 × 255

    Returns
    -------
    np.ndarray float32 (H, W, 3), unit normals in the file's native frame;
    invalid (near-zero) pixels are NaN.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        normals = np.load(path).astype(np.float32)
    elif ext in (".png", ".jpg", ".jpeg"):
        img = np.array(Image.open(path).convert("RGB")).astype(np.float32)
        normals = img / 127.5 - 1.0   # [0,255] → [-1,1]
    else:
        raise ValueError(f"Unsupported normal map format: {ext}  (use .npy or .png)")

    if normals.ndim != 3 or normals.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) normal map, got {normals.shape}")

    # Re-normalise to unit length; mark near-zero vectors as invalid.
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    valid = norms[..., 0] > 1e-4
    normals = np.where(valid[..., None], normals / (norms + 1e-8), np.nan)
    return normals


def depth_to_normals(depth: np.ndarray,
                     fx: float, fy: float,
                     cx: float, cy: float) -> np.ndarray:
    """
    Back-project a depth map to camera-space 3D points, then estimate surface
    normals via central-difference cross-products (GeoNet protocol).

    Convention
    ----------
    Camera frame: +X right, +Y down, +Z into scene.
    cross(dPu, dPv) yields Z > 0 for camera-facing surfaces (nz ≈ +1 for a
    flat wall facing the camera), matching the D2NT / GeoNet++ evaluation
    convention.  No sign flip is applied; (n+1)/2 RGB encoding gives blue
    for camera-facing surfaces.

    Parameters
    ----------
    depth : float32 (H, W) — depth in metres; invalid pixels are NaN / ≤ 0
    fx, fy, cx, cy : camera intrinsics (pixels)

    Returns
    -------
    np.ndarray float32 (H, W, 3) — unit normals (Z < 0 for camera-facing
    surfaces); boundary and invalid pixels are NaN.
    """
    H, W = depth.shape
    d = depth.astype(np.float64)

    # Back-project to 3D
    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    X = (uu - cx) * d / fx
    Y = (vv - cy) * d / fy
    Z = d.copy()

    # Mark invalid depth as NaN in all three channels
    bad = ~np.isfinite(d) | (d <= 0)
    X[bad] = np.nan
    Y[bad] = np.nan
    Z[bad] = np.nan

    P = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)

    # Central differences — span 2 pixels, so boundaries are excluded
    dPu = np.full_like(P, np.nan)
    dPv = np.full_like(P, np.nan)
    dPu[:, 1:-1] = P[:, 2:] - P[:, :-2]   # right − left
    dPv[1:-1, :] = P[2:, :] - P[:-2, :]   # below − above

    # Cross product: n = dPu × dPv
    # (+X right) × (+Y down) → nz > 0 (pointing forward / into the scene).
    # This is the standard convention used by D2NT and matches (n+1)/2 RGB encoding
    # where a flat wall facing the camera appears blue (nx≈0, ny≈0, nz≈+1 → B≈1).
    normals = np.cross(dPu, dPv)   # (H, W, 3), nz > 0 for camera-facing surfaces

    # Normalise
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    valid = (norms[..., 0] > 1e-8) & np.isfinite(norms[..., 0])
    normals = np.where(valid[..., None],
                       normals / (norms + 1e-8),
                       np.nan)

    return normals.astype(np.float32)


def estimate_intrinsics(h: int, w: int,
                        fov_diag_deg: float = 55.0) -> tuple:
    """
    Heuristic pinhole intrinsics when none are provided.
    Assumes a given diagonal FoV (default 55° — reasonable for wearable cameras).
    Returns (fx, fy, cx, cy).
    """
    diag = np.sqrt(h ** 2 + w ** 2)
    f = (diag / 2.0) / np.tan(np.radians(fov_diag_deg) / 2.0)
    return f, f, w / 2.0, h / 2.0


def compute_3dgm_metrics(pred_normals: np.ndarray,
                          gt_normals: np.ndarray,
                          depth_mask: np.ndarray) -> dict:
    """
    Compute the 3D Geometric Metric (3DGM) angular-error statistics.

    Parameters
    ----------
    pred_normals : float32 (H, W, 3) — predicted normals (NaN = invalid)
    gt_normals   : float32 (H, W, 3) — GT normals      (NaN = invalid)
    depth_mask   : bool   (H, W)     — True where GT depth is valid

    Returns
    -------
    dict with keys: mean_deg, median_deg, rmse_deg, pct_11.25, pct_22.5, pct_30,
                    pct_45, n_valid (number of valid pixels used)
    """
    # Combined validity: depth valid + both normal vectors finite
    valid = (depth_mask
             & np.all(np.isfinite(pred_normals), axis=-1)
             & np.all(np.isfinite(gt_normals),   axis=-1))

    if valid.sum() == 0:
        raise RuntimeError(
            "No valid pixels for 3DGM. Check that depth, GT normals, and "
            "intrinsics are consistent."
        )

    pn = pred_normals[valid].astype(np.float64)
    gn = gt_normals[valid].astype(np.float64)

    # Clamp dot product to [-1, 1] to guard against floating-point drift
    dot = np.clip(np.sum(pn * gn, axis=-1), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot))   # (N,)

    return {
        "mean_deg":   float(np.mean(angle_deg)),
        "median_deg": float(np.median(angle_deg)),
        "rmse_deg":   float(np.sqrt(np.mean(angle_deg ** 2))),
        "pct_11.25":  float(np.mean(angle_deg <= 11.25)),
        "pct_22.5":   float(np.mean(angle_deg <= 22.5)),
        "pct_30":     float(np.mean(angle_deg <= 30.0)),
        "pct_45":     float(np.mean(angle_deg <= 45.0)),
        "n_valid":    int(valid.sum()),
    }


def print_3dgm_metrics(metrics: dict, model: str,
                        variant: str = "", normal_source: str = "") -> None:
    """Pretty-print 3DGM metrics to stdout."""
    label = f"{model}" + (f" [{variant}]" if variant else "")
    src   = f"  (normals from: {normal_source})" if normal_source else ""
    print(f"\n{'='*60}")
    print(f"  {label}  — 3D Geometric Metric (GeoNet){src}")
    print(f"{'='*60}")
    print(f"  Valid pixels : {metrics['n_valid']}")
    print(f"  Mean  error  : {metrics['mean_deg']:.3f}°")
    print(f"  Median error : {metrics['median_deg']:.3f}°")
    print(f"  RMSE  error  : {metrics['rmse_deg']:.3f}°")
    print(f"  δ ≤ 11.25°   : {metrics['pct_11.25']*100:.2f}%")
    print(f"  δ ≤ 22.5°    : {metrics['pct_22.5']*100:.2f}%")
    print(f"  δ ≤ 30°      : {metrics['pct_30']*100:.2f}%")
    print(f"  δ ≤ 45°      : {metrics['pct_45']*100:.2f}%")
    print(f"{'='*60}\n")


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


def append_3dgm_to_csv(csv_path: str, model: str, variant: str,
                        metrics: dict, normal_source: str) -> None:
    """Append one 3DGM result row to a CSV file (separate from depth metrics)."""
    stem, ext = os.path.splitext(csv_path)
    path_3dgm = f"{stem}_3dgm{ext}"
    write_header = not os.path.exists(path_3dgm)
    with open(path_3dgm, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NORMAL_METRICS_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "model":         model,
            "variant":       variant,
            "mean_deg":      f"{metrics['mean_deg']:.4f}",
            "median_deg":    f"{metrics['median_deg']:.4f}",
            "rmse_deg":      f"{metrics['rmse_deg']:.4f}",
            "pct_11.25":     f"{metrics['pct_11.25']:.6f}",
            "pct_22.5":      f"{metrics['pct_22.5']:.6f}",
            "pct_30":        f"{metrics['pct_30']:.6f}",
            "pct_45":        f"{metrics['pct_45']:.6f}",
            "normal_source": normal_source,
        })
    print(f"  [csv] 3DGM results appended → {path_3dgm}")
