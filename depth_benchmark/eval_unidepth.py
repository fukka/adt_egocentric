"""
eval_unidepth.py
================
Benchmark baseline: UniDepth v2  (CVPR 2024)
  Paper : https://arxiv.org/abs/2403.18913
  Repo  : https://github.com/lpiccinelli-eth/UniDepth
  Models: lpiccinelli/unidepth-v2-vitl14   (ViT-L/14 — recommended)
          lpiccinelli/unidepth-v2-vits14   (ViT-S/14 — lightweight)

UniDepth jointly estimates METRIC depth and camera intrinsics from a single
image — no focal length required at inference. Output is in metres.

Evaluation strategy
-------------------
  1. Direct metric comparison (no alignment) — primary result.
  2. Scale-only alignment (median ratio) — shown as secondary for fair
     comparison against relative-depth baselines.

Usage
-----
  python eval_unidepth.py \\
      --rgb       /path/to/frame_0000.png \\
      --depth_gt  /path/to/frame_0000.npy \\
      --output_dir /path/to/output \\
      [--variant  vitl14|vits14]             (default: vitl14) \\
      [--rotation 0|90|180|270]             (default: 0) \\
      [--depth_scale 1.0]                   (default: 1.0) \\
      [--max_depth 10.0]                    (default: 10.0 m) \\
      [--intrinsics fx fy cx cy]            (optional, if known) \\
      [--device   cuda|cpu]                 (default: auto)

Dependencies
------------
  pip install torch torchvision pillow numpy matplotlib
  pip install git+https://github.com/lpiccinelli-eth/UniDepth.git
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
    align_scale_only, align_scale_shift,
    compute_metrics, print_metrics,
    save_comparison_figure, append_to_csv,
)

MODEL_IDS = {
    "vitl14": "lpiccinelli/unidepth-v2-vitl14",
    "vits14": "lpiccinelli/unidepth-v2-vits14",
}


# ──────────────────────────────────────────────────────────────────────────────

def build_intrinsics_tensor(fx: float, fy: float,
                             cx: float, cy: float,
                             device: str) -> torch.Tensor:
    """Pack camera intrinsics into a (3,3) tensor."""
    K = torch.tensor([[fx, 0.0, cx],
                       [0.0, fy, cy],
                       [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    return K


def run_unidepth(rgb_np: np.ndarray,
                 variant: str,
                 device: str,
                 intrinsics: list = None) -> np.ndarray:
    """
    Run UniDepth v2 inference.

    Parameters
    ----------
    rgb_np     : uint8 (H, W, 3) RGB
    variant    : 'vitl14' | 'vits14'
    device     : 'cuda' | 'cpu'
    intrinsics : [fx, fy, cx, cy] or None (model estimates them)

    Returns
    -------
    np.ndarray float32 (H, W) — metric depth in metres
    """
    from unidepth.models import UniDepthV2

    model_id = MODEL_IDS[variant]
    print(f"  [UniDepth] Loading: {model_id}")

    model = UniDepthV2.from_pretrained(model_id)
    model.to(device).eval()

    # UniDepth expects a (1, 3, H, W) float32 tensor in [0, 1]
    rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0
    rgb_tensor = rgb_tensor.unsqueeze(0).to(device)   # (1, 3, H, W)

    camera_tensor = None
    if intrinsics is not None:
        fx, fy, cx, cy = intrinsics
        camera_tensor = build_intrinsics_tensor(fx, fy, cx, cy, device)
        camera_tensor = camera_tensor.unsqueeze(0)    # (1, 3, 3)
        print(f"  [UniDepth] Using provided intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    else:
        print("  [UniDepth] No intrinsics provided — model will estimate them.")

    with torch.no_grad():
        predictions = model.infer(rgb_tensor, camera_tensor)

    # predictions["depth"]: (1, 1, H, W) in metres
    depth_pred = predictions["depth"].squeeze().cpu().float().numpy()

    # Log estimated intrinsics if no GT provided
    if intrinsics is None and "intrinsics" in predictions:
        K_est = predictions["intrinsics"].squeeze().cpu().numpy()
        print(f"  [UniDepth] Estimated intrinsics (fx,fy,cx,cy): "
              f"{K_est[0,0]:.1f}, {K_est[1,1]:.1f}, {K_est[0,2]:.1f}, {K_est[1,2]:.1f}")

    return depth_pred


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate UniDepth v2 on a single egocentric RGB+depth pair."
    )
    parser.add_argument("--rgb",         required=True)
    parser.add_argument("--depth_gt",    required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--variant",     default="vitl14", choices=["vitl14", "vits14"])
    parser.add_argument("--rotation",    type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--max_depth",   type=float, default=10.0)
    parser.add_argument("--intrinsics",  type=float, nargs=4,
                        metavar=("fx", "fy", "cx", "cy"),
                        default=None,
                        help="Camera intrinsics (optional). If omitted, model estimates them.")
    parser.add_argument("--device",      default=None)
    parser.add_argument("--csv",         default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [UniDepth] Device: {args.device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load inputs ─────────────────────────────────────────────────────────
    rgb = load_rgb(args.rgb, rotation=args.rotation)
    gt  = load_depth_gt(args.depth_gt, depth_scale=args.depth_scale,
                        max_depth=args.max_depth)
    if args.rotation != 0:
        k = {90: 1, 180: 2, 270: 3}[args.rotation]
        gt = np.rot90(gt, k=k).copy()
    mask = get_valid_mask(gt)
    print(f"  [UniDepth] RGB: {rgb.shape}  GT: {gt.shape}  valid px: {mask.sum()}")

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"  [UniDepth] Running inference (variant={args.variant}) …")
    pred_metric = run_unidepth(rgb, variant=args.variant,
                                device=args.device,
                                intrinsics=args.intrinsics)
    print(f"  [UniDepth] Predicted range: [{pred_metric[mask].min():.3f}, {pred_metric[mask].max():.3f}] m")

    # ── Evaluate: no alignment (metric) ─────────────────────────────────────
    alignment_direct = "none (metric)"
    metrics_direct = compute_metrics(pred_metric, gt, mask)
    print_metrics(metrics_direct, "UniDepth", variant=args.variant,
                  alignment=alignment_direct)

    # ── Evaluate: scale-only alignment (secondary) ──────────────────────────
    pred_scale_aligned = align_scale_only(pred_metric, gt, mask)
    alignment_scale = "scale-only (median ratio)"
    metrics_scale = compute_metrics(pred_scale_aligned, gt, mask)
    print_metrics(metrics_scale, "UniDepth", variant=f"{args.variant} [scale-aligned]",
                  alignment=alignment_scale)

    # ── Save visualisation (use scale-aligned pred for best visual) ──────────
    out_stem = f"unidepth_{args.variant}"
    save_comparison_figure(
        rgb=rgb, gt=gt, pred_aligned=pred_scale_aligned,
        metrics=metrics_scale,
        model="UniDepth", variant=args.variant,
        alignment=alignment_scale,
        output_path=os.path.join(args.output_dir, f"{out_stem}_comparison.png"),
    )
    np.save(os.path.join(args.output_dir, f"{out_stem}_pred_metric.npy"), pred_metric)

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = args.csv or os.path.join(args.output_dir, "results.csv")
    append_to_csv(csv_path, "UniDepth", args.variant,           metrics_direct, alignment_direct)
    append_to_csv(csv_path, "UniDepth", f"{args.variant}_scale_aligned", metrics_scale,  alignment_scale)

    return metrics_direct


if __name__ == "__main__":
    main()
