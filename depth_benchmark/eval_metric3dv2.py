"""
eval_metric3dv2.py
==================
Benchmark baseline: Metric3D v2  (IEEE TPAMI 2024)
  Paper : https://arxiv.org/abs/2404.15506
  Repo  : https://github.com/YvanYin/Metric3D
  Hub   : torch.hub  →  'YvanYin/Metric3D'

Metric3D v2 produces METRIC depth and surface normals by normalising all
images into a canonical camera space, making it truly camera-agnostic.
Camera intrinsics are required (or approximated from image dimensions).

Evaluation strategy
-------------------
  1. Direct metric comparison — primary.
  2. Scale+shift alignment — secondary (for like-for-like comparison).

Usage
-----
  python eval_metric3dv2.py \\
      --rgb       /path/to/frame_0000.png \\
      --depth_gt  /path/to/frame_0000.npy \\
      --output_dir /path/to/output \\
      [--variant  vit_small|vit_large|vit_giant2]  (default: vit_large) \\
      [--rotation 0|90|180|270]                    (default: 0) \\
      [--depth_scale 1.0]                          (default: 1.0) \\
      [--max_depth 10.0]                           (default: 10.0 m) \\
      [--intrinsics fx fy cx cy]                   (optional; estimated if omitted) \\
      [--device   cuda|cpu]                        (default: auto)

Dependencies
------------
  pip install torch torchvision pillow numpy matplotlib
  # Metric3D is loaded via torch.hub (downloads automatically on first run)
  # For GPU: CUDA-enabled PyTorch is strongly recommended
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from eval_utils import (
    load_rgb, load_depth_gt, get_valid_mask,
    align_scale_shift, align_scale_only,
    compute_metrics, print_metrics,
    save_comparison_figure, append_to_csv,
)

# Canonical focal length used internally by Metric3D
METRIC3D_CANONICAL_FOCAL = 1000.0

VARIANT_HUB_NAMES = {
    "vit_small":  "metric3d_vit_small",
    "vit_large":  "metric3d_vit_large",
    "vit_giant2": "metric3d_vit_giant2",
}


# ──────────────────────────────────────────────────────────────────────────────

def estimate_intrinsics_from_image(h: int, w: int) -> tuple:
    """
    Heuristic intrinsics when none are provided:
    assume 55° diagonal FoV (reasonable for wearable cameras).
    """
    diag = np.sqrt(h**2 + w**2)
    fov_diag_rad = np.radians(55.0)
    f = (diag / 2.0) / np.tan(fov_diag_rad / 2.0)
    cx, cy = w / 2.0, h / 2.0
    return f, f, cx, cy


def preprocess_for_metric3d(rgb_np: np.ndarray,
                             fx: float, fy: float,
                             device: str):
    """
    Rescale image to Metric3D canonical space and return:
      - input tensor (1, 3, H_new, W_new)
      - rescaled intrinsics (fx_new, fy_new, cx_new, cy_new)
      - pad info for cropping back
    """
    h, w = rgb_np.shape[:2]

    # Target scale such that the shorter side ≈ 616 (Metric3D standard)
    short = min(h, w)
    scale = 616.0 / short
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    # Pad to multiple of 14 (ViT patch size)
    pad_h = (14 - new_h % 14) % 14
    pad_w = (14 - new_w % 14) % 14

    img_resized = np.array(Image.fromarray(rgb_np).resize((new_w, new_h), Image.BILINEAR))
    img_padded  = np.pad(img_resized, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    # Normalise with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_padded.astype(np.float32) / 255.0 - mean) / std

    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)

    # Rescale intrinsics
    fx_new = fx * scale
    fy_new = fy * scale
    cx_new = (w / 2.0) * scale
    cy_new = (h / 2.0) * scale

    return tensor, (fx_new, fy_new, cx_new, cy_new), (new_h, new_w), (h, w)


def run_metric3dv2(rgb_np: np.ndarray,
                   variant: str,
                   device: str,
                   intrinsics: list = None) -> np.ndarray:
    """
    Run Metric3D v2 inference.

    Returns
    -------
    np.ndarray float32 (H, W) — metric depth in metres
    """
    hub_name = VARIANT_HUB_NAMES[variant]
    print(f"  [Metric3Dv2] Loading: {hub_name} via torch.hub …")

    model = torch.hub.load("YvanYin/Metric3D", hub_name, pretrain=True,
                           trust_repo=True)
    model.to(device).eval()

    h, w = rgb_np.shape[:2]

    # Intrinsics
    if intrinsics is not None:
        fx, fy, cx, cy = intrinsics
    else:
        fx, fy, cx, cy = estimate_intrinsics_from_image(h, w)
        print(f"  [Metric3Dv2] Estimated intrinsics: fx={fx:.1f}, fy={fy:.1f}, "
              f"cx={cx:.1f}, cy={cy:.1f}")

    # Preprocess
    tensor, (fx_s, fy_s, cx_s, cy_s), (new_h, new_w), (orig_h, orig_w) = \
        preprocess_for_metric3d(rgb_np, fx, fy, device)

    # Build intrinsics in canonical space for the model
    intrinsic_tensor = torch.tensor(
        [[fx_s, 0.0, cx_s], [0.0, fy_s, cy_s], [0.0, 0.0, 1.0]],
        dtype=torch.float32, device=device
    ).unsqueeze(0)  # (1, 3, 3)

    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference(
            {"input": tensor, "intrinsic": intrinsic_tensor}
        )

    # pred_depth: (1, 1, H_pad, W_pad) in metres
    pred = pred_depth.squeeze().cpu().float().numpy()

    # Crop to resized (unpadded) dimensions
    pred = pred[:new_h, :new_w]

    # Resize back to original resolution
    pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
    pred_orig = F.interpolate(pred_t, size=(orig_h, orig_w),
                              mode="bilinear", align_corners=False)
    pred_orig = pred_orig.squeeze().numpy()

    return pred_orig.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Metric3D v2 on a single egocentric RGB+depth pair."
    )
    parser.add_argument("--rgb",         required=True)
    parser.add_argument("--depth_gt",    required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--variant",     default="vit_large",
                        choices=["vit_small", "vit_large", "vit_giant2"])
    parser.add_argument("--rotation",    type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--max_depth",   type=float, default=10.0)
    parser.add_argument("--intrinsics",  type=float, nargs=4,
                        metavar=("fx", "fy", "cx", "cy"), default=None,
                        help="Camera intrinsics. Estimated from image size if omitted.")
    parser.add_argument("--device",      default=None)
    parser.add_argument("--csv",         default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [Metric3Dv2] Device: {args.device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load inputs ─────────────────────────────────────────────────────────
    rgb = load_rgb(args.rgb, rotation=args.rotation)
    gt  = load_depth_gt(args.depth_gt, depth_scale=args.depth_scale,
                        max_depth=args.max_depth)
    if args.rotation != 0:
        k = {90: 1, 180: 2, 270: 3}[args.rotation]
        gt = np.rot90(gt, k=k).copy()
    mask = get_valid_mask(gt)
    print(f"  [Metric3Dv2] RGB: {rgb.shape}  GT: {gt.shape}  valid: {mask.sum()}")

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"  [Metric3Dv2] Running inference (variant={args.variant}) …")
    pred_metric = run_metric3dv2(rgb, variant=args.variant,
                                  device=args.device,
                                  intrinsics=args.intrinsics)
    print(f"  [Metric3Dv2] Pred range: [{pred_metric[mask].min():.3f}, {pred_metric[mask].max():.3f}] m")

    # ── Evaluate: no alignment ───────────────────────────────────────────────
    alignment_direct = "none (metric)"
    metrics_direct = compute_metrics(pred_metric, gt, mask)
    print_metrics(metrics_direct, "Metric3D v2", variant=args.variant,
                  alignment=alignment_direct)

    # ── Evaluate: scale+shift alignment (secondary) ──────────────────────────
    pred_aligned = align_scale_shift(pred_metric, gt, mask)
    alignment_aff = "scale+shift (least-squares)"
    metrics_aff = compute_metrics(pred_aligned, gt, mask)
    print_metrics(metrics_aff, "Metric3D v2", variant=f"{args.variant} [aff-aligned]",
                  alignment=alignment_aff)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_stem = f"metric3dv2_{args.variant}"
    save_comparison_figure(
        rgb=rgb, gt=gt, pred_aligned=pred_aligned,
        metrics=metrics_aff,
        model="Metric3D v2", variant=args.variant,
        alignment=alignment_aff,
        output_path=os.path.join(args.output_dir, f"{out_stem}_comparison.png"),
    )
    np.save(os.path.join(args.output_dir, f"{out_stem}_pred_metric.npy"), pred_metric)

    csv_path = args.csv or os.path.join(args.output_dir, "results.csv")
    append_to_csv(csv_path, "Metric3D_v2", args.variant,             metrics_direct, alignment_direct)
    append_to_csv(csv_path, "Metric3D_v2", f"{args.variant}_aligned", metrics_aff,    alignment_aff)

    return metrics_direct


if __name__ == "__main__":
    main()
