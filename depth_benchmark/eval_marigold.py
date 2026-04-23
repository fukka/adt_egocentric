"""
eval_marigold.py
================
Benchmark baseline: Marigold  (CVPR 2024 Oral)
  Paper : https://arxiv.org/abs/2312.02145
  Model : prs-eth/marigold-depth-v1-0        (standard, ~20 denoising steps)
          prs-eth/marigold-depth-lcm-v1-0     (LCM fast variant, ~4 steps)

Marigold is an affine-invariant (relative) depth estimator built on
Stable Diffusion. Predicted depth is aligned to GT via least-squares
scale+shift before evaluation.

Usage
-----
  python eval_marigold.py \\
      --rgb       /path/to/frame_0000.png \\
      --depth_gt  /path/to/frame_0000.npy \\
      --output_dir /path/to/output \\
      [--variant  standard|lcm]              (default: standard) \\
      [--steps    20]                         (default: 20 for standard, 4 for lcm) \\
      [--ensemble 10]                         (default: 10 for standard, 1 for lcm) \\
      [--rotation 0|90|180|270]              (default: 0) \\
      [--depth_scale 1.0]                    (default: 1.0) \\
      [--max_depth 10.0]                     (default: 10.0 m) \\
      [--device   cuda|cpu]                  (default: auto-detect)

Dependencies
------------
  pip install torch diffusers transformers accelerate pillow numpy matplotlib
"""

import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from eval_utils import (
    load_rgb, load_depth_gt, get_valid_mask,
    align_scale_shift,
    compute_metrics, print_metrics,
    save_comparison_figure, append_to_csv,
)

MODEL_IDS = {
    "standard": "prs-eth/marigold-depth-v1-0",
    "lcm":      "prs-eth/marigold-depth-lcm-v1-0",
}
DEFAULT_STEPS    = {"standard": 20, "lcm": 4}
DEFAULT_ENSEMBLE = {"standard": 10, "lcm": 1}


# ──────────────────────────────────────────────────────────────────────────────

def run_marigold(rgb_np: np.ndarray,
                 variant: str,
                 steps: int,
                 ensemble_size: int,
                 device: str) -> np.ndarray:
    """
    Run Marigold depth inference.

    Parameters
    ----------
    rgb_np        : uint8 (H, W, 3) RGB
    variant       : 'standard' | 'lcm'
    steps         : number of denoising steps
    ensemble_size : number of predictions to ensemble
    device        : 'cuda' | 'cpu'

    Returns
    -------
    np.ndarray float32 (H, W) — disparity in [0, 1] (higher = closer)
                                 NOTE: disparity, not depth; alignment handles inversion
    """
    from diffusers import MarigoldDepthPipeline

    model_id = MODEL_IDS[variant]
    print(f"  [Marigold] Loading: {model_id}")

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = MarigoldDepthPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if (device == "cuda") else None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    image = Image.fromarray(rgb_np)

    with torch.no_grad():
        output = pipe(
            image,
            denoising_steps=steps,
            ensemble_size=ensemble_size,
            processing_res=768,          # internal processing resolution
            match_input_res=True,        # upsample back to original resolution
            color_map=None,
            show_progress_bar=False,
        )

    # depth_np: float32 (H, W), values in [0,1] — 0=far, 1=near (disparity)
    depth_np = np.array(output.depth_np, dtype=np.float32)
    return depth_np


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Marigold on a single egocentric RGB+depth pair."
    )
    parser.add_argument("--rgb",         required=True, help="Path to input RGB image")
    parser.add_argument("--depth_gt",    required=True, help="Path to GT depth .npy")
    parser.add_argument("--output_dir",  required=True, help="Directory for outputs")
    parser.add_argument("--variant",     default="standard", choices=["standard", "lcm"],
                        help="Model variant (default: standard)")
    parser.add_argument("--steps",       type=int, default=None,
                        help="Denoising steps (default: 20 for standard, 4 for lcm)")
    parser.add_argument("--ensemble",    type=int, default=None,
                        help="Ensemble size (default: 10 for standard, 1 for lcm)")
    parser.add_argument("--rotation",    type=int, default=0, choices=[0, 90, 180, 270],
                        help="Clockwise rotation of the RGB before inference (default: 0)")
    parser.add_argument("--depth_scale", type=float, default=1.0,
                        help="Multiply GT depth by this factor (default: 1.0)")
    parser.add_argument("--max_depth",   type=float, default=10.0,
                        help="GT depth cap in metres (default: 10.0)")
    parser.add_argument("--device",      default=None,
                        help="'cuda' or 'cpu' (default: auto)")
    parser.add_argument("--csv",         default=None,
                        help="Path to append CSV result row (optional)")
    args = parser.parse_args()

    # Defaults
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.steps    is None: args.steps    = DEFAULT_STEPS[args.variant]
    if args.ensemble is None: args.ensemble = DEFAULT_ENSEMBLE[args.variant]
    print(f"  [Marigold] Device: {args.device}  steps={args.steps}  ensemble={args.ensemble}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load inputs ─────────────────────────────────────────────────────────
    print(f"  [Marigold] Loading RGB : {args.rgb}  (rotation={args.rotation}°)")
    rgb = load_rgb(args.rgb, rotation=args.rotation)

    print(f"  [Marigold] Loading GT  : {args.depth_gt}")
    gt = load_depth_gt(args.depth_gt, depth_scale=args.depth_scale,
                       max_depth=args.max_depth)

    if args.rotation != 0:
        k = {90: 1, 180: 2, 270: 3}[args.rotation]
        gt = np.rot90(gt, k=k).copy()

    mask = get_valid_mask(gt)
    print(f"         GT shape    : {gt.shape}  valid px: {mask.sum()}")

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"  [Marigold] Running inference …")
    pred_disp = run_marigold(rgb, variant=args.variant,
                              steps=args.steps,
                              ensemble_size=args.ensemble,
                              device=args.device)

    # Marigold outputs disparity ([0,1], near=1). Convert to pseudo-depth by
    # flipping, then align affinely — the alignment handles scale & shift.
    # Some pipelines already output depth; if so, remove the flip.
    pred_pseudo_depth = 1.0 - pred_disp   # higher value = farther away

    # ── Alignment ────────────────────────────────────────────────────────────
    alignment = "scale+shift (least-squares)"
    pred_aligned = align_scale_shift(pred_pseudo_depth, gt, mask)
    print(f"  [Marigold] Aligned range: [{pred_aligned[mask].min():.3f}, {pred_aligned[mask].max():.3f}] m")

    # ── Metrics ──────────────────────────────────────────────────────────────
    metrics = compute_metrics(pred_aligned, gt, mask)
    print_metrics(metrics, model="Marigold", variant=args.variant, alignment=alignment)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_stem = f"marigold_{args.variant}"
    fig_path = os.path.join(args.output_dir, f"{out_stem}_comparison.png")
    save_comparison_figure(
        rgb=rgb, gt=gt, pred_aligned=pred_aligned,
        metrics=metrics,
        model="Marigold", variant=args.variant,
        alignment=alignment,
        output_path=fig_path,
    )
    np.save(os.path.join(args.output_dir, f"{out_stem}_pred_raw.npy"), pred_disp)
    np.save(os.path.join(args.output_dir, f"{out_stem}_pred_aligned.npy"), pred_aligned)

    csv_path = args.csv or os.path.join(args.output_dir, "results.csv")
    append_to_csv(csv_path, "Marigold", args.variant, metrics, alignment)

    return metrics


if __name__ == "__main__":
    main()
