"""
eval_depth_anything_v2.py
=========================
Benchmark baseline: Depth Anything V2  (NeurIPS 2024)
  Paper : https://arxiv.org/abs/2406.09414
  Models: depth-anything/Depth-Anything-V2-{Small,Base,Large}-hf  (HuggingFace)

Depth Anything V2 is an affine-invariant (relative) depth estimator.
Predicted depth is aligned to GT via least-squares scale+shift before evaluation.

Usage
-----
  python eval_depth_anything_v2.py \\
      --rgb       /path/to/frame_0000.png \\
      --depth_gt  /path/to/frame_0000.npy \\
      --output_dir /path/to/output \\
      [--variant  small|base|large]         (default: large) \\
      [--rotation 0|90|180|270]             (default: 0) \\
      [--depth_scale 1.0]                   (default: 1.0; use 0.001 for mm→m) \\
      [--max_depth 10.0]                    (default: 10.0 m) \\
      [--device    cuda|cpu]                (default: auto-detect)

Dependencies
------------
  pip install torch torchvision transformers pillow numpy matplotlib
"""

import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image

# Shared utilities (must be in the same directory or on PYTHONPATH)
sys.path.insert(0, os.path.dirname(__file__))
from eval_utils import (
    load_rgb, load_depth_gt, get_valid_mask,
    align_scale_shift,
    compute_metrics, print_metrics,
    save_comparison_figure, append_to_csv,
)

MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


# ──────────────────────────────────────────────────────────────────────────────

def run_depth_anything_v2(rgb_np: np.ndarray,
                           variant: str,
                           device: str) -> np.ndarray:
    """
    Run Depth Anything V2 inference.

    Parameters
    ----------
    rgb_np  : uint8 (H, W, 3) RGB numpy array
    variant : 'small' | 'base' | 'large'
    device  : 'cuda' | 'cpu'

    Returns
    -------
    np.ndarray float32 (H, W) — relative depth (arbitrary scale, NOT metric)
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_id = MODEL_IDS[variant]
    print(f"  [DAv2] Loading model: {model_id}")

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id,
                                                        torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device).eval()

    image = Image.fromarray(rgb_np)
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # interpolate to original resolution
        pred = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=rgb_np.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().float().numpy()

    return pred


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Depth Anything V2 on a single egocentric RGB+depth pair."
    )
    parser.add_argument("--rgb",        required=True, help="Path to input RGB image")
    parser.add_argument("--depth_gt",   required=True, help="Path to GT depth .npy")
    parser.add_argument("--output_dir", required=True, help="Directory for outputs")
    parser.add_argument("--variant",    default="large", choices=["small", "base", "large"],
                        help="Model variant (default: large)")
    parser.add_argument("--rotation",   type=int, default=0, choices=[0, 90, 180, 270],
                        help="Clockwise rotation applied to the RGB image before inference (default: 0)")
    parser.add_argument("--depth_scale", type=float, default=1.0,
                        help="Multiply GT depth by this factor (e.g. 0.001 for mm→m, default: 1.0)")
    parser.add_argument("--max_depth",   type=float, default=10.0,
                        help="GT depth values above this (metres) are treated as invalid (default: 10.0)")
    parser.add_argument("--device",     default=None,
                        help="'cuda' or 'cpu' (default: auto-detect)")
    parser.add_argument("--csv",        default=None,
                        help="Path to append CSV result row (optional)")
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [DAv2] Device: {args.device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load inputs ─────────────────────────────────────────────────────────
    print(f"  [DAv2] Loading RGB  : {args.rgb}  (rotation={args.rotation}°)")
    rgb = load_rgb(args.rgb, rotation=args.rotation)
    print(f"         RGB shape   : {rgb.shape}")

    print(f"  [DAv2] Loading GT   : {args.depth_gt}")
    gt = load_depth_gt(args.depth_gt, depth_scale=args.depth_scale,
                       max_depth=args.max_depth)
    mask = get_valid_mask(gt)
    print(f"         GT shape    : {gt.shape}  valid px: {mask.sum()} / {mask.size}")

    # If rotation applied to RGB, also rotate the GT
    if args.rotation != 0:
        from eval_utils import ROTATION_MAP
        gt_pil = Image.fromarray(np.where(np.isfinite(gt), gt, 0).astype(np.float32))
        # PIL doesn't support float arrays directly — rotate via numpy
        k = {90: 1, 180: 2, 270: 3}[args.rotation]
        gt = np.rot90(gt, k=k).copy()
        mask = get_valid_mask(gt)
        print(f"         GT rotated  : {gt.shape}  valid px: {mask.sum()}")

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"  [DAv2] Running inference (variant={args.variant}) …")
    pred_raw = run_depth_anything_v2(rgb, variant=args.variant, device=args.device)
    print(f"         Pred shape  : {pred_raw.shape}")

    # ── Alignment (affine: scale + shift) ───────────────────────────────────
    alignment = "scale+shift (least-squares)"
    pred_aligned = align_scale_shift(pred_raw, gt, mask)
    print(f"  [DAv2] Aligned depth range: [{pred_aligned[mask].min():.3f}, {pred_aligned[mask].max():.3f}] m")

    # ── Metrics ──────────────────────────────────────────────────────────────
    metrics = compute_metrics(pred_aligned, gt, mask)
    print_metrics(metrics, model="Depth Anything V2", variant=args.variant,
                  alignment=alignment)

    # ── Save visualisation ───────────────────────────────────────────────────
    out_stem = f"dav2_{args.variant}"
    fig_path = os.path.join(args.output_dir, f"{out_stem}_comparison.png")
    save_comparison_figure(
        rgb=rgb, gt=gt, pred_aligned=pred_aligned,
        metrics=metrics,
        model="Depth Anything V2", variant=args.variant,
        alignment=alignment,
        output_path=fig_path,
    )

    # Optionally save raw + aligned depth as npy
    np.save(os.path.join(args.output_dir, f"{out_stem}_pred_raw.npy"), pred_raw)
    np.save(os.path.join(args.output_dir, f"{out_stem}_pred_aligned.npy"), pred_aligned)

    # ── CSV logging ──────────────────────────────────────────────────────────
    csv_path = args.csv or os.path.join(args.output_dir, "results.csv")
    append_to_csv(csv_path, "Depth_Anything_V2", args.variant, metrics, alignment)

    return metrics


if __name__ == "__main__":
    main()
