"""
eval_depth_anything_v2.py
=========================
Benchmark baseline: Depth Anything V2  (NeurIPS 2024)
  Paper : https://arxiv.org/abs/2406.09414
  Repo  : https://github.com/DepthAnything/Depth-Anything-V2

Depth Anything V2 is an affine-invariant (relative) depth estimator.
Predicted depth is aligned to GT via least-squares scale+shift before evaluation.

Model loading
-------------
Uses the original PyTorch repo directly (no HuggingFace Hub download).
Before running:
  1. Clone the repo:
       git clone https://github.com/DepthAnything/Depth-Anything-V2.git
  2. Download the checkpoint(s) you need and place them in one directory:
       depth_anything_v2_vits.pth   (Small)
       depth_anything_v2_vitb.pth   (Base)
       depth_anything_v2_vitl.pth   (Large)
     Checkpoints are linked from the GitHub README (GitHub Releases page).
  3. Pass --repo_dir and --ckpt_dir (or set the constants at the top of this
     file) so the script can find the code and weights.

Usage
-----
  python eval_depth_anything_v2.py \\
      --rgb       /path/to/frame_0000.png \\
      --depth_gt  /path/to/frame_0000.npy \\
      --output_dir /path/to/output \\
      --repo_dir  /path/to/Depth-Anything-V2 \\
      --ckpt_dir  /path/to/checkpoints \\
      [--variant  small|base|large]         (default: large) \\
      [--rotation 0|90|180|270]             (default: 0) \\
      [--depth_scale 1.0]                   (default: 1.0; use 0.001 for mm→m) \\
      [--max_depth 10.0]                    (default: 10.0 m) \\
      [--device    cuda|cpu]                (default: auto-detect)

Dependencies
------------
  pip install torch torchvision pillow numpy matplotlib
  (transformers is no longer required)
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

# ── Model configuration ────────────────────────────────────────────────────
# encoder architecture and DPT head dimensions for each variant
MODEL_CONFIGS = {
    "small": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192,  384]},
    "base":  {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384,  768]},
    "large": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

# Checkpoint filename for each variant (placed in --ckpt_dir)
CKPT_NAMES = {
    "small": "depth_anything_v2_vits.pth",
    "base":  "depth_anything_v2_vitb.pth",
    "large": "depth_anything_v2_vitl.pth",
}


# ──────────────────────────────────────────────────────────────────────────────

def run_depth_anything_v2(rgb_np: np.ndarray,
                           variant: str,
                           device: str,
                           repo_dir: str,
                           ckpt_dir: str) -> np.ndarray:
    """
    Run Depth Anything V2 inference using the original PyTorch repo.

    Parameters
    ----------
    rgb_np   : uint8 (H, W, 3) RGB numpy array
    variant  : 'small' | 'base' | 'large'
    device   : 'cuda' | 'cpu'
    repo_dir : path to the cloned Depth-Anything-V2 repo root
    ckpt_dir : directory containing the .pth checkpoint files

    Returns
    -------
    np.ndarray float32 (H, W) — relative depth (arbitrary scale, NOT metric)
    """
    # Make the repo's own modules importable
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError as e:
        raise ImportError(
            f"Could not import DepthAnythingV2 from '{repo_dir}'. "
            "Make sure --repo_dir points to the cloned "
            "https://github.com/DepthAnything/Depth-Anything-V2 root."
        ) from e

    ckpt_path = os.path.join(ckpt_dir, CKPT_NAMES[variant])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Download '{CKPT_NAMES[variant]}' and place it in --ckpt_dir."
        )

    print(f"  [DAv2] Loading checkpoint: {ckpt_path}")
    model = DepthAnythingV2(**MODEL_CONFIGS[variant])
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device).eval()

    # infer_image handles internal preprocessing and returns an HxW float32 array
    with torch.no_grad():
        pred = model.infer_image(rgb_np)

    return pred.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Depth Anything V2 on a single egocentric RGB+depth pair."
    )
    parser.add_argument("--rgb",        required=True, help="Path to input RGB image")
    parser.add_argument("--depth_gt",   required=True, help="Path to GT depth .npy")
    parser.add_argument("--output_dir", required=True, help="Directory for outputs")
    parser.add_argument("--repo_dir",   required=True,
                        help="Path to the cloned Depth-Anything-V2 repo root "
                             "(https://github.com/DepthAnything/Depth-Anything-V2)")
    parser.add_argument("--ckpt_dir",   required=True,
                        help="Directory containing depth_anything_v2_vit{s,b,l}.pth checkpoints")
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
    pred_raw = run_depth_anything_v2(rgb, variant=args.variant, device=args.device,
                                     repo_dir=args.repo_dir, ckpt_dir=args.ckpt_dir)
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
