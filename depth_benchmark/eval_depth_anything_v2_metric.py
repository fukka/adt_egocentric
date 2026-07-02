"""
eval_depth_anything_v2_metric.py
=================================
Benchmark baseline: Depth Anything V2 — Metric Depth variant
  Paper : https://arxiv.org/abs/2406.09414
  Repo  : https://github.com/DepthAnything/Depth-Anything-V2

Is it reasonable to use DAv2 Metric for egocentric indoor evaluation?
----------------------------------------------------------------------
YES, with important caveats:

  1. Domain gap — Indoor models are trained on Hypersim (synthetic) with
     pseudo-labelled real images from DUSt3R. Egocentric/wearable camera
     footage (wide FOV, hand proximity, unconventional viewpoints) is out-
     of-distribution. Expect a systematic scale bias of ~5–15%.

  2. No camera intrinsics used — Unlike Metric3D v2, DAv2 Metric does not
     condition on focal length. Scale accuracy therefore degrades for cameras
     whose FOV deviates from the training distribution (mainly ~60° HFOV
     perspective cameras). Wide-angle wearable cameras (≥90° HFOV) will see
     larger scale errors than Metric3D v2.

  3. Close-range objects — Training data rarely includes objects closer than
     ~30 cm. Expect lower accuracy for wrist/hand regions in ego datasets.

  4. Depth cap — Indoor models cap at 20 m; outdoor models at 80 m. Use the
     indoor variant for most egocentric scenes.

  5. Alignment comparison — This script evaluates BOTH:
       (a) Direct metric output — no alignment, tests absolute scale accuracy.
       (b) Scale+shift aligned  — removes global scale/offset bias to isolate
           structural/relative accuracy.
     The gap between (a) and (b) quantifies the scale bias on your data.

Model loading
-------------
Uses the original PyTorch repo (same as eval_depth_anything_v2.py).
Metric-depth checkpoints are separate files from the relative-depth ones.

Before running:
  1. Clone: https://github.com/DepthAnything/Depth-Anything-V2
  2. Download metric checkpoints from the GitHub README and place in --ckpt_dir:
       depth_anything_v2_metric_hypersim_vits.pth   (Small, indoor ≤20 m)
       depth_anything_v2_metric_hypersim_vitb.pth   (Base,  indoor ≤20 m)
       depth_anything_v2_metric_hypersim_vitl.pth   (Large, indoor ≤20 m)
       depth_anything_v2_metric_vkitti_vits.pth     (Small, outdoor ≤80 m)
       depth_anything_v2_metric_vkitti_vitb.pth     (Base,  outdoor ≤80 m)
       depth_anything_v2_metric_vkitti_vitl.pth     (Large, outdoor ≤80 m)

Usage
-----
  python eval_depth_anything_v2_metric.py \\
      --rgb        /path/to/frame_0000.png \\
      --depth_gt   /path/to/frame_0000.npy \\
      --output_dir /path/to/output \\
      --repo_dir   /path/to/Depth-Anything-V2 \\
      --ckpt_dir   /path/to/checkpoints \\
      [--variant   small|base|large]    (default: large) \\
      [--domain    indoor|outdoor]      (default: indoor) \\
      [--rotation  0|90|180|270]        (default: 0; counter-clockwise) \\
      [--depth_scale 1.0]               (default: 1.0; use 0.001 for mm→m) \\
      [--max_depth 10.0]                (default: 10.0 m) \\
      [--device    cuda|cpu]            (default: auto-detect)

Dependencies
------------
  pip install torch torchvision pillow numpy matplotlib
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

# ── Model configuration ────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "small": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192,  384]},
    "base":  {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384,  768]},
    "large": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

# Checkpoint filename pattern: metric_hypersim (indoor) or metric_vkitti (outdoor)
CKPT_NAMES = {
    ("small", "indoor"):  "depth_anything_v2_metric_hypersim_vits.pth",
    ("base",  "indoor"):  "depth_anything_v2_metric_hypersim_vitb.pth",
    ("large", "indoor"):  "depth_anything_v2_metric_hypersim_vitl.pth",
    ("small", "outdoor"): "depth_anything_v2_metric_vkitti_vits.pth",
    ("base",  "outdoor"): "depth_anything_v2_metric_vkitti_vitb.pth",
    ("large", "outdoor"): "depth_anything_v2_metric_vkitti_vitl.pth",
}

# Maximum depth (metres) reported by each domain's training set
DOMAIN_MAX_DEPTH = {"indoor": 20.0, "outdoor": 80.0}


# ──────────────────────────────────────────────────────────────────────────────

def _patch_for_metric_depth(model, max_depth: float) -> None:
    """
    Patch a DepthAnythingV2 built without max_depth support to produce metric output.

    Why this is needed
    ------------------
    Older repo versions build the DPT head with a final ReLU and run
        depth = F.relu(depth)
    in forward(). The metric checkpoint weights were trained expecting
        depth = sigmoid(logits) * max_depth
    as the final step. Loading metric weights into the ReLU head means logits
    designed for sigmoid (often negative for shallow pixels) get clipped to zero,
    producing an all-zero depth map.

    What this patch does
    --------------------
    1. Finds and removes the final ReLU in output_conv2 so the raw logits from
       the metric checkpoint reach forward() unclamped.
    2. Replaces forward() with a version that applies sigmoid(x) * max_depth
       instead of relu(x).
    """
    import types
    import torch.nn as nn
    import torch.nn.functional as F

    # Step 1: Remove the final ReLU from the DPT output head.
    # The metric checkpoint's last conv was trained without an output clamp;
    # the sigmoid in forward() serves that role. Leaving ReLU here would clip
    # negative logits before they ever reach sigmoid.
    oconv = model.depth_head.scratch.output_conv2
    for i in range(len(oconv) - 1, -1, -1):
        if isinstance(oconv[i], nn.ReLU):
            oconv[i] = nn.Identity()
            break

    # Step 2: Replace forward() so it applies sigmoid scaling instead of relu.
    # We store max_depth on the model to keep infer_image() working unchanged
    # (it calls self.forward() internally).
    model._metric_max_depth = max_depth

    def _metric_forward(self, x):
        h, w = x.shape[-2:]
        patch_size = getattr(self.pretrained, "patch_size", 14)
        patch_h, patch_w = h // patch_size, w // patch_size
        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = torch.sigmoid(depth) * self._metric_max_depth
        return depth.squeeze(1)

    model.forward = types.MethodType(_metric_forward, model)


def run_depth_anything_v2_metric(rgb_np: np.ndarray,
                                  variant: str,
                                  domain: str,
                                  device: str,
                                  repo_dir: str,
                                  ckpt_dir: str) -> np.ndarray:
    """
    Run Depth Anything V2 Metric inference using the original PyTorch repo.

    Parameters
    ----------
    rgb_np   : uint8 (H, W, 3) RGB numpy array
    variant  : 'small' | 'base' | 'large'
    domain   : 'indoor' | 'outdoor'  — selects the fine-tuned checkpoint
    device   : 'cuda' | 'cpu'
    repo_dir : path to the cloned Depth-Anything-V2 repo root
    ckpt_dir : directory containing the metric .pth checkpoint files

    Returns
    -------
    np.ndarray float32 (H, W) — metric depth in metres (no alignment needed)
    """
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

    ckpt_name = CKPT_NAMES[(variant, domain)]
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Download '{ckpt_name}' and place it in --ckpt_dir.\n"
            f"Metric checkpoints are listed under 'Metric Depth Estimation' "
            f"in the Depth-Anything-V2 GitHub README."
        )

    max_depth = DOMAIN_MAX_DEPTH[domain]
    print(f"  [DAv2-M] Loading checkpoint: {ckpt_path}")

    # Build the model, then apply the sigmoid-scaling patch for metric output.
    # The patch is always applied: repos that natively accept max_depth in the
    # constructor already wire up sigmoid correctly, but the patch is idempotent
    # (it just replaces the last ReLU with Identity and overrides forward()).
    # Repos that don't accept max_depth need the patch to avoid all-zero output.
    model = DepthAnythingV2(**MODEL_CONFIGS[variant])
    _patch_for_metric_depth(model, max_depth)
    print(f"  [DAv2-M] Metric patch applied: sigmoid(x) * {max_depth} m")

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    with torch.no_grad():
        pred = model.infer_image(rgb_np)   # HxW float32, metres

    return pred.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Depth Anything V2 Metric on a single egocentric RGB+depth pair. "
            "Outputs both direct-metric and scale+shift-aligned results."
        )
    )
    parser.add_argument("--rgb",        required=False, help="Path to input RGB image",
                        default='/user/f.zhang2/Documents/projectaria_tools_adt_data/Apartment_release_golden_skeleton_seq100_10s_sample_M1292/exocentric_rendered/overhead/frame_0000.png'
                        # default='/user/f.zhang2/Documents/projectaria_tools_adt_data/Apartment_release_golden_skeleton_seq100_10s_sample_M1292/blender_rendered_maps/pinhole/frame_0000.png'
                        )
    parser.add_argument("--depth_gt",   required=False, help="Path to GT depth .npy",
                        default='/user/f.zhang2/Documents/projectaria_tools_adt_data/Apartment_release_golden_skeleton_seq100_10s_sample_M1292/exocentric_rendered/overhead/depth_maps/frame_0000.npy',
                        # default='/user/f.zhang2/Documents/projectaria_tools_adt_data/Apartment_release_golden_skeleton_seq100_10s_sample_M1292/blender_rendered_maps/pinhole/depth_maps/frame_0000.npy',
                        )
    parser.add_argument("--output_dir", required=False, help="Directory for outputs",
                        default='/user/f.zhang2/Documents/projectaria_tools_adt_data/Apartment_release_golden_skeleton_seq100_10s_sample_M1292/exocentric_rendered/overhead/depth_anything_v2_results',
                        # default='/user/f.zhang2/Documents/projectaria_tools_adt_data/Apartment_release_golden_skeleton_seq100_10s_sample_M1292/blender_rendered_maps/pinhole/depth_anything_v2_results',
                        )
    parser.add_argument("--repo_dir",   required=False,
                        help="Path to the cloned Depth-Anything-V2 repo root "
                             "(https://github.com/DepthAnything/Depth-Anything-V2)",
                        default='/user/f.zhang2/projects/adt_egocentric/depth_benchmark/Depth-Anything-V2',
                        )
    parser.add_argument("--ckpt_dir",   required=False,
                        help="Directory containing depth_anything_v2_vit{s,b,l}.pth checkpoints",
                        default='/user/f.zhang2/projects/adt_egocentric/depth_benchmark/Depth-Anything-V2/checkpoints',
                        )
    parser.add_argument("--variant",    default="large", choices=["small", "base", "large"],
                        help="Model variant (default: large)")
    parser.add_argument("--domain",     default="indoor", choices=["indoor", "outdoor"],
                        help="Checkpoint domain: 'indoor' (Hypersim, ≤20 m) or "
                             "'outdoor' (VKITTI, ≤80 m). Use 'indoor' for egocentric "
                             "scenes (default: indoor)")
    parser.add_argument("--rotation",   type=int, default=0, choices=[0, 90, 180, 270],
                        help="Counter-clockwise rotation applied to RGB before inference (default: 0)")
    parser.add_argument("--depth_scale", type=float, default=1.0,
                        help="Multiply GT depth by this factor (e.g. 0.001 for mm→m, default: 1.0)")
    parser.add_argument("--max_depth",   type=float, default=10.0,
                        help="GT depth values above this (metres) are treated as invalid (default: 10.0)")
    parser.add_argument("--device",     default=None,
                        help="'cuda' or 'cpu' (default: auto-detect)")
    parser.add_argument("--csv",        default=None,
                        help="Path to append CSV result rows (optional)")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [DAv2-M] Device : {args.device}")
    print(f"  [DAv2-M] Domain : {args.domain}  (max training depth: "
          f"{DOMAIN_MAX_DEPTH[args.domain]} m)")
    if args.domain == "outdoor":
        print("  [DAv2-M] WARNING: 'outdoor' checkpoint is trained on Virtual KITTI "
              "(driving scenes). Accuracy on indoor egocentric data will be poor.")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load inputs ───────────────────────────────────────────────────────────
    print(f"  [DAv2-M] Loading RGB : {args.rgb}  (rotation={args.rotation}°)")
    rgb = load_rgb(args.rgb, rotation=args.rotation)
    print(f"           RGB shape  : {rgb.shape}")

    print(f"  [DAv2-M] Loading GT  : {args.depth_gt}")
    gt   = load_depth_gt(args.depth_gt, depth_scale=args.depth_scale,
                         max_depth=args.max_depth)
    if args.rotation != 0:
        k  = {90: 1, 180: 2, 270: 3}[args.rotation]
        gt = np.rot90(gt, k=k).copy()
    mask = get_valid_mask(gt)
    print(f"           GT shape   : {gt.shape}  valid px: {mask.sum()} / {mask.size}")

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"  [DAv2-M] Running inference (variant={args.variant}, domain={args.domain}) …")
    pred_metric = run_depth_anything_v2_metric(
        rgb, variant=args.variant, domain=args.domain,
        device=args.device, repo_dir=args.repo_dir, ckpt_dir=args.ckpt_dir,
    )
    print(f"           Pred shape : {pred_metric.shape}")
    if pred_metric.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_metric.shape} vs gt {gt.shape}"
        )
    print(f"  [DAv2-M] Pred range  : "
          f"[{pred_metric[mask].min():.3f}, {pred_metric[mask].max():.3f}] m")

    # ── Evaluate: direct metric (primary) ────────────────────────────────────
    # No alignment — tests whether the model predicts correct absolute depth.
    alignment_direct = "none (metric)"
    metrics_direct   = compute_metrics(pred_metric, gt, mask, max_depth=args.max_depth)
    print_metrics(metrics_direct, "Depth Anything V2 Metric",
                  variant=f"{args.variant}/{args.domain}", alignment=alignment_direct)

    # ── Evaluate: scale+shift aligned (secondary) ────────────────────────────
    # Removes global affine bias to isolate structural / relative accuracy.
    # The gap between this and the direct result quantifies scale error.
    alignment_aff  = "scale+shift (least-squares)"
    pred_aligned   = align_scale_shift(pred_metric, gt, mask)
    metrics_aff    = compute_metrics(pred_aligned, gt, mask, max_depth=args.max_depth)
    print_metrics(metrics_aff, "Depth Anything V2 Metric",
                  variant=f"{args.variant}/{args.domain} [aff-aligned]",
                  alignment=alignment_aff)

    scale_gap = metrics_direct["AbsRel"] - metrics_aff["AbsRel"]
    print(f"  [DAv2-M] Scale bias  : ΔAbsRel = {scale_gap:+.4f}  "
          f"({'large — notable domain gap' if scale_gap > 0.05 else 'small — good metric scale'})")

    # ── Save visualisation ────────────────────────────────────────────────────
    out_stem  = f"dav2_metric_{args.variant}_{args.domain}"
    fig_path  = os.path.join(args.output_dir, f"{out_stem}_comparison.png")
    save_comparison_figure(
        rgb=rgb, gt=gt, pred_aligned=pred_aligned,
        metrics=metrics_aff,
        model="Depth Anything V2 Metric",
        variant=f"{args.variant}/{args.domain}",
        alignment=alignment_aff,
        output_path=fig_path,
    )
    np.save(os.path.join(args.output_dir, f"{out_stem}_pred_metric.npy"),  pred_metric)
    np.save(os.path.join(args.output_dir, f"{out_stem}_pred_aligned.npy"), pred_aligned)

    # ── CSV logging ───────────────────────────────────────────────────────────
    csv_path = args.csv or os.path.join(args.output_dir, "results.csv")
    variant_tag = f"{args.variant}_{args.domain}"
    append_to_csv(csv_path, "DAv2_Metric", variant_tag,               metrics_direct, alignment_direct)
    append_to_csv(csv_path, "DAv2_Metric", f"{variant_tag}_aligned",  metrics_aff,    alignment_aff)

    return metrics_direct


if __name__ == "__main__":
    main()
