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

──────────────────────────────────────────────────────────────────────────────
Single-frame mode (original behaviour — backward compatible)
──────────────────────────────────────────────────────────────────────────────
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

──────────────────────────────────────────────────────────────────────────────
Batch mode  --mode ego-fisheye | ego-pinhole | exo-pinhole
                 | exo-pinhole-newscene | real | real-newscene
──────────────────────────────────────────────────────────────────────────────
Runs over all saved frames for a hard-coded list of sequences.

Two sequence lists are configured at the top of this file:
  BATCH_SEQ_DIRS    — used by ego-fisheye, ego-pinhole, exo-pinhole, real
  NEWSCENE_SEQ_DIRS — used by exo-pinhole-newscene, real-newscene

Directory conventions expected on disk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ego-fisheye / ego-pinhole  (render_from_poses_blender.py output):
  {seq_dir}/blender_rendered_maps/{mode_subdir}/videos_rgb/frame_XXXXXX_TTTTT.png
  {seq_dir}/blender_rendered_maps/{mode_subdir}/depth_maps/frame_XXXXXX_TTTTT.npy
  {seq_dir}/blender_rendered_maps/{mode_subdir}/normal_maps/frame_XXXXXX_TTTTT.npy   (optional)
  where mode_subdir = 'fisheye' (ego-fisheye) or 'pinhole' (ego-pinhole)

exo-pinhole  (render_exocentric_blender.py output, uses BATCH_SEQ_DIRS):
  {seq_dir}/exocentric_rendered/{cam_name}/videos_rgb/frame_XXXXXX_TTTTT.png
  {seq_dir}/exocentric_rendered/{cam_name}/depth_maps/frame_XXXXXX_TTTTT.npy
  {seq_dir}/exocentric_rendered/{cam_name}/normal_maps/frame_XXXXXX_TTTTT.npy  (optional)
  All cameras in {seq_dir}/exocentric_rendered/ are included unless
  --exo_camera is specified.

exo-pinhole-newscene  (render_exocentric_blender.py output, uses NEWSCENE_SEQ_DIRS):
  Same directory layout as exo-pinhole but loads from NEWSCENE_SEQ_DIRS and
  only includes camera subdirs whose name starts with 'bed_' (bedroom cameras).
  --exo_camera further restricts within that filtered set.

Intrinsics (used for 3DGM / normal estimation from depth)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ego-fisheye : real Aria RGB focal, treated as a central-pixel pinhole. Fisheye
                back-projection is non-linear, so this is only a central-region
                approximation — ego-fisheye 3DGM is NOT comparable to pinhole-mode
                3DGM. Depth metrics are intrinsics-free and unaffected. For exact
                wide-angle normals, rectify fisheye→pinhole or use fisheye rays.
  ego-pinhole : ADT Aria focal ≈ 611 px @ 1408 px → scaled to actual render size
  exo-pinhole : focal = (H/2) / tan(VFOV/2),  default VFOV = 70°

Example usage:
  python eval_depth_anything_v2.py \\
      --mode ego-pinhole \\
      --repo_dir /path/to/Depth-Anything-V2 \\
      --ckpt_dir /path/to/checkpoints \\
      [--variant large] \\
      [--max_depth 10.0] \\
      [--exo_camera overhead]   # exo-pinhole only: restrict to one camera \\
      [--output_dir /override/output/root]   # default: inside each seq dir

Metrics
-------
Let T be the set of valid pixels (GT depth finite, 0 < GT ≤ max_depth,
and aligned prediction finite, 0 < pred ≤ max_depth).
With --mask_invalid_input, pixels whose input RGB is a black/blank border
(e.g. outside the Aria fisheye circle) are also removed from T *before*
alignment, so they cannot skew the scale+shift fit or the metrics.
Let d̂ᵢ = predicted depth (metres) and dᵢ = GT depth (metres) for pixel i ∈ T.

── Depth metrics (Eigen et al., NIPS 2014) ──────────────────────────────────

  AbsRel  =  (1/|T|) · Σᵢ |d̂ᵢ − dᵢ| / dᵢ

    Mean absolute error relative to GT depth.  Scale-sensitive: a 0.5 m error
    at 1 m (AbsRel = 0.5) is penalised the same as a 5 m error at 10 m.
    Lower is better.  Typical good results: < 0.10.

  SqRel   =  (1/|T|) · Σᵢ (d̂ᵢ − dᵢ)² / dᵢ

    Squared error normalised by GT depth (not GT² — see Eigen et al.).
    Has units of metres.  Amplifies large absolute errors and amplifies errors
    at near pixels (small dᵢ denominator).  A single very wrong near pixel can
    dominate; watch for scenes with objects < 0.5 m from the camera.
    Lower is better.

  RMSE    =  sqrt( (1/|T|) · Σᵢ (d̂ᵢ − dᵢ)² )

    Root-mean-squared error in metres.  Sensitive to large absolute errors
    but, unlike SqRel, not amplified by near pixels.
    Lower is better.

  RMSElog =  sqrt( (1/|T|) · Σᵢ (log d̂ᵢ − log dᵢ)² )

    RMSE computed in log-depth space.  Scale-invariant: a 2× error at 1 m
    is penalised identically to a 2× error at 10 m.  Robust to large
    absolute depth values.  Lower is better.  Typical good results: < 0.15.

  δ₁      =  (1/|T|) · |{ i : max(d̂ᵢ/dᵢ, dᵢ/d̂ᵢ) < 1.25   }|
  δ₂      =  (1/|T|) · |{ i : max(d̂ᵢ/dᵢ, dᵢ/d̂ᵢ) < 1.25²  }|
  δ₃      =  (1/|T|) · |{ i : max(d̂ᵢ/dᵢ, dᵢ/d̂ᵢ) < 1.25³  }|

    Fraction of pixels whose depth ratio is within a threshold factor of 1.
    δ₁ (threshold 1.25) is the standard quality gate; δ₂/δ₃ are looser.
    Higher is better.  Typical good results: δ₁ > 0.90.

── Alignment (applied before depth metrics) ─────────────────────────────────

  The raw model output r̂ᵢ is affine-invariant (arbitrary scale and shift).
  Alignment is performed in inverse-depth (disparity) space:

    Solve:  scale · r̂ᵢ + shift  ≈  1/dᵢ    (least-squares over T)
    Then:   d̂ᵢ  =  1 / (scale · r̂ᵢ + shift)

  Aligning in disparity space guarantees a positive scale when the model
  outputs disparity-like values (larger = closer), avoiding the sign-flip
  that depth-space alignment produces.  Pixels where (scale·r̂+shift) ≤ 0
  or the resulting depth > max_depth are excluded from T.

── 3D Geometric Metric — 3DGM (GeoNet / GeoNet++ protocol) ─────────────────

  Both GT and predicted depth maps are back-projected to camera-space 3D
  point clouds and surface normals nᵢ, n̂ᵢ ∈ ℝ³ (unit vectors) are estimated
  via central-difference cross-products.

  Angular error:  θᵢ  =  arccos( clamp(n̂ᵢ · nᵢ, −1, 1) )   [degrees]

  Reported statistics over all pixels where both normals are finite:

    Mean    =  (1/|V|) · Σᵢ θᵢ
    Median  =  median of { θᵢ }
    RMSE    =  sqrt( (1/|V|) · Σᵢ θᵢ² )
    ≤11.25° =  fraction of pixels with θᵢ ≤ 11.25°
    ≤22.5°  =  fraction of pixels with θᵢ ≤ 22.5°
    ≤30°    =  fraction of pixels with θᵢ ≤ 30°
    ≤45°    =  fraction of pixels with θᵢ ≤ 45°

  Lower mean / RMSE and higher percentages are better.
  Normal convention: camera frame (+X right, +Y down, +Z into scene);
  nz > 0 for surfaces facing the camera; encoded as RGB = (n+1)/2.

Dependencies
------------
  pip install torch torchvision pillow numpy matplotlib
"""

import argparse
import os
import sys
import glob
import math
import numpy as np
import torch
from PIL import Image

# Shared utilities (must be in the same directory or on PYTHONPATH)
sys.path.insert(0, os.path.dirname(__file__))
from eval_utils import (
    load_rgb, load_depth_gt, get_valid_mask, get_input_valid_mask,
    valid_eval_mask,
    align_scale_shift_disparity,
    compute_metrics, print_metrics,
    save_comparison_figure, append_to_csv,
    depth_to_normals, estimate_intrinsics,
    compute_3dgm_metrics, print_3dgm_metrics, append_3dgm_to_csv,
    # save_normal_comparison_figure,
    save_normal_comparison_figure_2,
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

# ── Hard-coded sequence lists ───────────────────────────────────────────────
# BATCH_SEQ_DIRS  — used by: ego-fisheye, ego-pinhole, exo-pinhole, real
# NEWSCENE_SEQ_DIRS — used by: exo-pinhole-newscene, real-newscene
#
# Add or remove sequence directories in each list as needed.
# Each entry is the root of one ADT sequence (the folder that contains
# blender_rendered_maps/, exocentric_rendered/, videos_rgb/, depth_npy/, …).
_ADT_ROOT = '/user/f.zhang2/Documents/projectaria_tools_adt_data_clean'
BATCH_SEQ_DIRS = [
    f'{_ADT_ROOT}/Apartment_release_clean_seq131_M1292',
    # f'{_ADT_ROOT}/Apartment_release_decoration_seq132_M1292',
]

# Sequences that contain the "new scene" data (bedroom exocentric renders,
# real Aria frames annotated as bedroom, etc.).
NEWSCENE_SEQ_DIRS: list[str] = [
    f'{_ADT_ROOT}/Apartment_release_decoration_seq132_M1292',
]

# Camera-name prefix used by exo-pinhole-newscene to select bedroom cameras.
_NEWSCENE_CAM_PREFIX = 'bed_'

# Rooms to include when running in real-newscene mode.
# Edit this list to select which room labels (as annotated with the room
# annotator tool) should be kept for evaluation.
NEWSCENE_ROOMS: list[str] = ['bedroom']

# ADT Aria RGB intrinsics (focal length at native 1408 px square crop)
_ARIA_FOCAL_PX_AT_1408 = 611.0
_ARIA_CALIB_SIZE_PX    = 1408.0

# Default exo-pinhole vertical FoV (matches render_exocentric_blender.py default)
_EXO_DEFAULT_VFOV_DEG = 70.0


# ──────────────────────────────────────────────────────────────────────────────

def _load_model(args):
    """
    Load and return a DAv2 model in eval mode.

    Handles both the pretrained relative-depth checkpoint (args.ckpt_dir) and
    a finetuned metric checkpoint (args.finetuned_from).  The returned model
    is on args.device and in .eval() mode.
    """
    if args.repo_dir not in sys.path:
        sys.path.insert(0, args.repo_dir)
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError as e:
        raise ImportError(
            f"Could not import DepthAnythingV2 from '{args.repo_dir}'. "
            "Make sure --repo_dir points to the cloned "
            "https://github.com/DepthAnything/Depth-Anything-V2 root."
        ) from e

    if args.finetuned_from:
        # ── Finetuned (metric) checkpoint ────────────────────────────────────
        # Prefer metric_depth/depth_anything_v2/dpt.py which adds max_depth to
        # __init__.  Fall back to the root class with a max_depth patch.
        _metric_dir = os.path.join(args.repo_dir, 'metric_depth')
        if os.path.isdir(_metric_dir):
            _evicted = {k: sys.modules.pop(k)
                        for k in list(sys.modules) if 'depth_anything_v2' in k}
            sys.path.insert(0, _metric_dir)
            try:
                from depth_anything_v2.dpt import (  # noqa: PLC0415
                    DepthAnythingV2 as _MetricDAv2)
            except Exception:
                sys.modules.update(_evicted)
                raise
            finally:
                sys.path.remove(_metric_dir)
            _ModelCls    = _MetricDAv2
            _model_kwargs = {**MODEL_CONFIGS[args.variant], 'max_depth': args.max_depth}
        else:
            print("  [DAv2] metric_depth/ not found in repo — "
                  "using root class with max_depth patch")
            _ModelCls    = DepthAnythingV2
            _model_kwargs = MODEL_CONFIGS[args.variant]

        ckpt_path = args.finetuned_from
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Finetuned checkpoint not found: {ckpt_path}")
        print(f"  [DAv2] Loading finetuned checkpoint: {ckpt_path}")
        raw   = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = raw['model'] if (isinstance(raw, dict) and 'model' in raw) else raw
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
        m = _ModelCls(**_model_kwargs)
        missing, unexpected = m.load_state_dict(state, strict=False)
        if missing:
            print(f"    [WARN] Missing keys ({len(missing)}): {missing[:5]} …")
        if unexpected:
            print(f"    [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")
        if not hasattr(m, 'max_depth'):
            m.max_depth = args.max_depth
    else:
        # ── Pretrained (relative-depth) checkpoint ────────────────────────────
        ckpt_path = os.path.join(args.ckpt_dir, CKPT_NAMES[args.variant])
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Download '{CKPT_NAMES[args.variant]}' and place it in --ckpt_dir."
            )
        print(f"  [DAv2] Loading pretrained checkpoint: {ckpt_path}")
        m = DepthAnythingV2(**MODEL_CONFIGS[args.variant])
        m.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))

    return m.to(args.device).eval()


# ── Batch-mode helpers ────────────────────────────────────────────────────────

def _ego_intrinsics(h: int, w: int) -> tuple:
    """Pinhole intrinsics for ADT Aria RGB camera scaled to render size."""
    scale = w / _ARIA_CALIB_SIZE_PX
    fx = fy = _ARIA_FOCAL_PX_AT_1408 * scale
    cx, cy = w / 2.0, h / 2.0
    return fx, fy, cx, cy


def _exo_intrinsics(h: int, w: int, vfov_deg: float = _EXO_DEFAULT_VFOV_DEG) -> tuple:
    """Pinhole intrinsics for a render with a given vertical FoV."""
    fy = (h / 2.0) / math.tan(math.radians(vfov_deg / 2.0))
    fx = fy   # square pixels, symmetric lens
    cx, cy = w / 2.0, h / 2.0
    return fx, fy, cx, cy


def _restrict_to_valid_input(mask: np.ndarray, rgb: np.ndarray,
                             gt_shape: tuple, args) -> np.ndarray:
    """
    Optionally AND the GT-valid *mask* with the valid-input-region mask.

    When --mask_invalid_input is set, pixels whose input RGB is a black/blank
    border (e.g. outside the Aria fisheye circle) are removed so they cannot
    corrupt the least-squares alignment or the reported metrics.  The input
    mask is derived from *rgb* (RGB/prediction resolution) and resized with
    nearest-neighbour to the GT resolution when the two differ.
    """
    if not getattr(args, 'mask_invalid_input', False):
        return mask
    input_valid = get_input_valid_mask(rgb, args.input_black_thresh,
                                       args.input_valid_erode)
    if input_valid.shape != tuple(gt_shape):
        import cv2  # available via eval_utils' dependency set
        input_valid = cv2.resize(
            input_valid.astype(np.uint8), (gt_shape[1], gt_shape[0]),
            interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask & input_valid



def _load_room_filter(rgb_dir: str, allowed_rooms: list[str]) -> set[str] | None:
    """
    Read ``room_annotations.csv`` from *rgb_dir* (written by the room annotator
    tool) and return the set of frame **stems** (filename without extension)
    whose ``room`` column is in *allowed_rooms*.

    Returns ``None`` if the CSV does not exist so callers can distinguish
    "no annotation file" from "annotation file with zero matching frames".

    CSV format expected (one row per frame)::

        frame_number,timestamp,filepath,room
        400,13863283646787,frame_000400_13863283646787.jpg,bedroom
        ...
    """
    import csv as _csv
    csv_path = os.path.join(rgb_dir, 'room_annotations.csv')
    if not os.path.exists(csv_path):
        return None

    allowed = {r.strip().lower() for r in allowed_rooms}
    stems: set[str] = set()
    with open(csv_path, newline='') as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if row['room'].strip().lower() in allowed:
                stems.add(os.path.splitext(row['filepath'].strip())[0])
    return stems


def _collect_frames(rgb_dir: str, depth_dir: str, normal_dir):
    """
    Scan rgb_dir for image files (.png, .jpg, .jpeg); match each with a depth
    .npy of the same stem.  Returns a list of dicts with keys: rgb, depth_gt,
    normal_gt (may be None), stem.  Only frames with a matching depth file are
    returned.  Stems are deduplicated so a frame present as both .png and .jpg
    is included only once (first alphabetically).
    """
    seen_stems = set()
    frames = []
    all_rgb = sorted(
        glob.glob(os.path.join(rgb_dir, '*.png'))
        + glob.glob(os.path.join(rgb_dir, '*.jpg'))
        + glob.glob(os.path.join(rgb_dir, '*.jpeg'))
    )
    for rgb_path in all_rgb:
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        if stem in seen_stems:
            continue
        depth_path = os.path.join(depth_dir, f'{stem}.npy')
        if not os.path.exists(depth_path):
            continue  # no GT depth → skip
        normal_path = None
        if normal_dir is not None:
            candidate = os.path.join(normal_dir, f'{stem}.npy')
            if os.path.exists(candidate):
                normal_path = candidate
        seen_stems.add(stem)
        frames.append({'rgb': rgb_path, 'depth_gt': depth_path,
                        'normal_gt': normal_path, 'stem': stem})
    return frames



def _print_aggregate(metrics_mean: dict, metrics_3dgm_mean,
                     n_frames: int, model: str, variant: str,
                     alignment: str) -> None:
    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS — {model} ({variant})")
    print(f"  Frames evaluated: {n_frames}")
    print(f"{'='*70}")
    print_metrics(metrics_mean, model=model, variant=variant,
                  alignment=f"{alignment} [pixel-weighted over {n_frames} frames]")
    if metrics_3dgm_mean is not None:
        print_3dgm_metrics(metrics_3dgm_mean, model, variant=variant,
                           normal_source="depth_to_normals(aligned) [pixel-weighted]")
    print(f"{'='*70}\n")


def _save_aggregate_csv(csv_path: str, model: str, variant: str,
                        metrics_mean: dict, metrics_3dgm_mean,
                        n_frames: int, alignment: str) -> None:
    append_to_csv(csv_path, f"{model}_AGGREGATE_N{n_frames}", variant,
                  metrics_mean,
                  f"{alignment} [pixel-weighted over {n_frames} frames]")
    if metrics_3dgm_mean is not None:
        append_3dgm_to_csv(csv_path, f"{model}_AGGREGATE_N{n_frames}", variant,
                           metrics_3dgm_mean,
                           "depth_to_normals(aligned) pixel-weighted")


# ── Batch evaluation entry point ──────────────────────────────────────────────

def run_batch(args) -> None:
    """Batch evaluation over all frames of all hard-coded sequences."""
    mode = args.mode
    # 'ego-fisheye' | 'ego-pinhole' | 'exo-pinhole'
    # | 'exo-pinhole-newscene' | 'real' | 'real-newscene'

    # Choose which sequence list to iterate over.
    # exo-pinhole-newscene and real-newscene target the "new scene" sequences;
    # all other modes use the standard benchmark list.
    if mode in ('exo-pinhole-newscene', 'real-newscene'):
        seq_dirs = NEWSCENE_SEQ_DIRS
    else:
        seq_dirs = BATCH_SEQ_DIRS

    print(f"\n[DAv2 Batch]  mode={mode}  variant={args.variant}")
    print(f"  Sequences  : {len(seq_dirs)}")
    for d in seq_dirs:
        print(f"    {d}")

    # ── Build the list of (rgb_dir, depth_dir, normal_dir, cam_label, room_filter) tuples ──
    # room_filter is a set of frame stems to keep, or None to keep all frames.
    job_roots = []

    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir)
        if mode in ('ego-fisheye', 'ego-pinhole'):
            render_root = args.render_dir or os.path.join(seq_dir,
                                                          'blender_rendered_maps')
            sub = 'fisheye' if mode == 'ego-fisheye' else 'pinhole'
            rgb_dir    = os.path.join(render_root, sub, 'videos_rgb')
            depth_dir  = os.path.join(render_root, sub, 'depth_maps')
            normal_dir = os.path.join(render_root, sub, 'normal_maps')
            if not os.path.isdir(rgb_dir):
                print(f"  [WARN] Not found: {rgb_dir} — skipping {seq_name}")
                continue
            normal_dir = normal_dir if os.path.isdir(normal_dir) else None
            cam_label  = f'{seq_name}/{sub}'
            job_roots.append((rgb_dir, depth_dir, normal_dir, cam_label, None))

        elif mode == 'exo-pinhole':
            render_root = args.render_dir or os.path.join(seq_dir,
                                                          'exocentric_rendered')
            if not os.path.isdir(render_root):
                print(f"  [WARN] Not found: {render_root} — skipping {seq_name}")
                continue
            # Enumerate camera subdirs (each contains videos_rgb/)
            cam_dirs = sorted([
                d for d in os.listdir(render_root)
                if os.path.isdir(os.path.join(render_root, d, 'videos_rgb'))
            ])
            if args.exo_camera:
                wanted = set(args.exo_camera)
                cam_dirs = [c for c in cam_dirs if c in wanted]
            if not cam_dirs:
                print(f"  [WARN] No camera dirs found in {render_root} "
                      f"(exo_camera filter={args.exo_camera}) — skipping {seq_name}")
                continue
            for cam in cam_dirs:
                cam_root   = os.path.join(render_root, cam)
                rgb_dir    = os.path.join(cam_root, 'videos_rgb')
                depth_dir  = os.path.join(cam_root, 'depth_maps')
                normal_dir = os.path.join(cam_root, 'normal_maps')
                normal_dir = normal_dir if os.path.isdir(normal_dir) else None
                cam_label  = f'{seq_name}/{cam}'
                job_roots.append((rgb_dir, depth_dir, normal_dir, cam_label, None))

        elif mode == 'exo-pinhole-newscene':
            # Bedroom exocentric renders from render_exocentric_blender.py.
            # Layout identical to exo-pinhole:
            #   {seq_dir}/exocentric_rendered/{cam_name}/videos_rgb/
            #   {seq_dir}/exocentric_rendered/{cam_name}/depth_maps/
            #   {seq_dir}/exocentric_rendered/{cam_name}/normal_maps/  (optional)
            # Camera filter: only subdirs whose name starts with _NEWSCENE_CAM_PREFIX
            # ('bed_').  --exo_camera further restricts within that filtered set.
            render_root = args.render_dir or os.path.join(seq_dir,
                                                          'exocentric_rendered')
            if not os.path.isdir(render_root):
                print(f"  [WARN] Not found: {render_root} — skipping {seq_name}")
                continue
            # Enumerate all camera subdirs that have a videos_rgb/ folder.
            all_cam_dirs = sorted([
                d for d in os.listdir(render_root)
                if os.path.isdir(os.path.join(render_root, d, 'videos_rgb'))
            ])
            # Step 1: keep only bedroom cameras (bed_* prefix).
            cam_dirs = [c for c in all_cam_dirs
                        if c.startswith(_NEWSCENE_CAM_PREFIX)]
            if not cam_dirs:
                print(f"  [WARN] No '{_NEWSCENE_CAM_PREFIX}*' camera dirs found in "
                      f"{render_root} — skipping {seq_name}")
                continue
            # Step 2: honour --exo_camera on top of the prefix filter.
            if args.exo_camera:
                wanted = set(args.exo_camera)
                cam_dirs = [c for c in cam_dirs if c in wanted]
            if not cam_dirs:
                print(f"  [WARN] No cameras match prefix '{_NEWSCENE_CAM_PREFIX}' "
                      f"AND --exo_camera={args.exo_camera} in {seq_name}")
                continue
            print(f"  [exo-pinhole-newscene] {seq_name}: "
                  f"{len(cam_dirs)} bedroom camera(s): {cam_dirs}")
            for cam in cam_dirs:
                cam_root   = os.path.join(render_root, cam)
                rgb_dir    = os.path.join(cam_root, 'videos_rgb')
                depth_dir  = os.path.join(cam_root, 'depth_maps')
                normal_dir = os.path.join(cam_root, 'normal_maps')
                normal_dir = normal_dir if os.path.isdir(normal_dir) else None
                cam_label  = f'{seq_name}/{cam}'
                job_roots.append((rgb_dir, depth_dir, normal_dir, cam_label, None))

        elif mode == 'real':
            # Real Aria sensor frames: {seq_dir}/videos_rgb/  +  {seq_dir}/depth_npy/
            # Matches the layout expected by _gather_real() in dataset/adt.py.
            rgb_dir   = args.render_dir or os.path.join(seq_dir, 'videos_rgb')
            depth_dir = os.path.join(seq_dir, 'depth_npy')
            if not os.path.isdir(rgb_dir):
                print(f"  [WARN] Not found: {rgb_dir} — skipping {seq_name}")
                continue
            if not os.path.isdir(depth_dir):
                print(f"  [WARN] Not found: {depth_dir} — skipping {seq_name}")
                continue
            cam_label = f'{seq_name}/real'
            job_roots.append((rgb_dir, depth_dir, None, cam_label, None))

        elif mode == 'real-newscene':
            # Same layout as 'real', but restricted to frames annotated as one
            # of the rooms in NEWSCENE_ROOMS via room_annotations.csv produced
            # by the room annotator tool (saved inside the videos_rgb folder).
            rgb_dir   = args.render_dir or os.path.join(seq_dir, 'videos_rgb')
            depth_dir = os.path.join(seq_dir, 'depth_npy')
            if not os.path.isdir(rgb_dir):
                print(f"  [WARN] Not found: {rgb_dir} — skipping {seq_name}")
                continue
            if not os.path.isdir(depth_dir):
                print(f"  [WARN] Not found: {depth_dir} — skipping {seq_name}")
                continue
            room_filter = _load_room_filter(rgb_dir, NEWSCENE_ROOMS)
            if room_filter is None:
                print(f"  [WARN] room_annotations.csv not found in {rgb_dir} "
                      f"— run the room annotator tool first, skipping {seq_name}")
                continue
            print(f"  [real-newscene] rooms={NEWSCENE_ROOMS}  "
                  f"matching stems: {len(room_filter)}")
            if not room_filter:
                print(f"  [WARN] No frames match rooms {NEWSCENE_ROOMS} "
                      f"in {seq_name} — skipping")
                continue
            cam_label = f'{seq_name}/real-newscene'
            job_roots.append((rgb_dir, depth_dir, None, cam_label, room_filter))

    if not job_roots:
        print("[ERROR] No valid render directories found. Check sequence paths "
              "and ensure renders have been generated first.")
        sys.exit(1)

    # ── Collect all frames across all jobs ────────────────────────────────────
    all_frames = []
    for rgb_dir, depth_dir, normal_dir, cam_label, room_filter in job_roots:
        frames = _collect_frames(rgb_dir, depth_dir, normal_dir)
        if room_filter is not None:
            before = len(frames)
            frames = [f for f in frames if f['stem'] in room_filter]
            print(f"  {cam_label}: {len(frames)}/{before} frames "
                  f"kept after room filter {NEWSCENE_ROOMS}")
        if args.num_frames is not None and len(frames) > args.num_frames:
            indices = np.linspace(0, len(frames) - 1, args.num_frames, dtype=int)
            frames  = [frames[i] for i in indices]
            print(f"  {cam_label}: {len(frames)} frames sampled (--num_frames)")
        else:
            print(f"  {cam_label}: {len(frames)} frames with GT depth")
        for f in frames:
            f['cam_label'] = cam_label
            # Stash depth/normal dir paths for intrinsics
            f['depth_dir']  = depth_dir
            f['normal_dir'] = normal_dir
        all_frames.extend(frames)

    if not all_frames:
        print("[ERROR] No frames with GT depth found. "
              "Did you render with --no_depth disabled?")
        sys.exit(1)

    print(f"\n  Total frames to evaluate: {len(all_frames)}")

    # ── Set up output directory ────────────────────────────────────────────────
    _ckpt_tag = (os.path.splitext(os.path.basename(args.finetuned_from))[0]
                 if args.finetuned_from else args.variant)
    out_root = args.output_dir or os.path.join(
        seq_dirs[0], 'depth_benchmark_results',
        f'dav2_{_ckpt_tag}_{mode}'
    )
    os.makedirs(out_root, exist_ok=True)
    csv_path = os.path.join(out_root, 'results_per_frame.csv')

    # ── Resolve alignment strategy ────────────────────────────────────────────
    # Pretrained (relative-depth) model → align in disparity space.
    # Finetuned (metric) model → output is already in metres; skip alignment.
    # --no_align always wins; absence of --no_align with a finetuned ckpt
    # still skips alignment automatically.
    _no_align = args.no_align or (args.finetuned_from is not None)
    _alignment_label = "none (metric)" if _no_align else "scale+shift-disparity"
    print(f"  [DAv2] alignment={_alignment_label}")

    # ── Load model once ───────────────────────────────────────────────────────
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [DAv2] Device: {args.device}")

    _model_holder = [None]

    def _infer(rgb_np: np.ndarray) -> np.ndarray:
        if _model_holder[0] is None:
            _model_holder[0] = _load_model(args)
        with torch.no_grad():
            return _model_holder[0].infer_image(rgb_np).astype(np.float32)

    # ── Resolve per-mode rotation and depth_scale ─────────────────────────────
    # --rotation / --depth_scale override everything when explicitly set.
    # Otherwise fall back to sensible per-mode defaults:
    #   ego/exo : rotation=0,   depth_scale=1.0   (Blender renders, already metres)
    #   real    : rotation=270, depth_scale=0.001 (Aria sensor orientation, depth in mm)
    if mode in ('real', 'real-newscene'):
        _rotation    = args.rotation    if args.rotation    is not None else args.real_rotation
        _depth_scale = args.depth_scale if args.depth_scale is not None else args.real_depth_scale
        print(f"  [{mode}] rotation={_rotation}°  depth_scale={_depth_scale}"
              f"  (override with --rotation / --depth_scale if needed)")
    else:
        _rotation    = args.rotation    if args.rotation    is not None else 0
        _depth_scale = args.depth_scale if args.depth_scale is not None else 1.0

    # ── Pixel-level accumulators for aggregate metrics ────────────────────────
    # Depth: exact pixel-weighted sums (no approximation)
    px_depth = {'n': 0, 'abs_rel': 0.0, 'sq_rel': 0.0,
                'sq_diff': 0.0, 'sq_log': 0.0,
                'd1': 0, 'd2': 0, 'd3': 0}
    # 3DGM: exact pixel-weighted sums; median approximated as pixel-count-weighted
    #       mean of per-frame medians (exact median requires storing all values)
    px_3dgm = {'n': 0, 'deg': 0.0, 'sq_deg': 0.0,
               'n11': 0, 'n22': 0, 'n30': 0, 'n45': 0,
               'med_frames': []}   # list of (median_deg, n_pixels) per frame

    # ── Evaluate frame by frame ───────────────────────────────────────────────
    n_ok = 0

    for i, frame in enumerate(all_frames):
        stem      = frame['stem']
        cam_label = frame['cam_label']
        print(f"\n  [{i+1}/{len(all_frames)}] {cam_label} / {stem}")

        # Load RGB
        rgb = load_rgb(frame['rgb'], rotation=_rotation)

        # Load GT depth
        gt   = load_depth_gt(frame['depth_gt'],
                              depth_scale=_depth_scale,
                              max_depth=args.max_depth)
        if _rotation != 0:
            k  = {90: 1, 180: 2, 270: 3}[_rotation]
            gt = np.rot90(gt, k=k).copy()
        mask = get_valid_mask(gt)
        # Optionally drop black/blank input-border pixels (e.g. outside the Aria
        # fisheye circle) so they cannot corrupt alignment or metrics.
        mask = _restrict_to_valid_input(mask, rgb, gt.shape, args)
        if mask.sum() < 100:
            print(f"    [WARN] Only {mask.sum()} valid GT pixels — skipping")
            continue

        # Infer
        pred_raw = _infer(rgb)
        if pred_raw.shape != gt.shape:
            # Resize pred to GT resolution with bilinear + align_corners=True,
            # matching the official DAv2 validation interpolation convention.
            import torch.nn.functional as _F
            pred_raw = _F.interpolate(
                torch.from_numpy(pred_raw).unsqueeze(0).unsqueeze(0),
                size=(gt.shape[0], gt.shape[1]),
                mode='bilinear', align_corners=True,
            ).squeeze().numpy()
            print(f"    [WARN] pred resized from model output to GT shape {gt.shape}")

        # Alignment
        if _no_align:
            pred_aligned = pred_raw.copy()
        else:
            pred_aligned = align_scale_shift_disparity(pred_raw, gt, mask)

        # Common valid-pixel set (GT-in-range ∩ pred-in-range) — the SAME rule
        # every baseline uses, so no model is scored on a different pixel set.
        depth_mask = valid_eval_mask(pred_aligned, gt, mask, max_depth=args.max_depth)
        metrics    = compute_metrics(pred_aligned, gt, depth_mask)
        n_ok += 1

        # Pixel-weighted accumulation for depth aggregate
        _p = np.clip(pred_aligned[depth_mask].astype(np.float64), 1e-6, None)
        _g = np.clip(gt[depth_mask].astype(np.float64), 1e-6, None)
        _n = len(_p)
        px_depth['n']       += _n
        px_depth['abs_rel'] += float(np.sum(np.abs(_p - _g) / _g))
        px_depth['sq_rel']  += float(np.sum((_p - _g)**2 / _g))
        px_depth['sq_diff'] += float(np.sum((_p - _g)**2))
        px_depth['sq_log']  += float(np.sum((np.log(_p) - np.log(_g))**2))
        _ratio = np.maximum(_p / _g, _g / _p)
        px_depth['d1'] += int(np.sum(_ratio < 1.25))
        px_depth['d2'] += int(np.sum(_ratio < 1.25**2))
        px_depth['d3'] += int(np.sum(_ratio < 1.25**3))

        # Per-frame CSV
        append_to_csv(csv_path, f"DAv2_{cam_label}/{stem}", args.variant,
                      metrics, _alignment_label)

        # Per-frame comparison figure (saved to per-cam subdir)
        cam_out = os.path.join(out_root, cam_label.replace('/', '__'))
        os.makedirs(cam_out, exist_ok=True)
        fig_path = os.path.join(cam_out, f'dav2_{_ckpt_tag}_{stem}_cmp.png')
        save_comparison_figure(
            rgb=rgb, gt=gt, pred_aligned=pred_aligned,
            metrics=metrics,
            model="Depth Anything V2", variant=args.variant,
            alignment=_alignment_label,
            output_path=fig_path,
        )
        np.save(os.path.join(cam_out, f'dav2_{_ckpt_tag}_{stem}_pred_raw.npy'),
                pred_raw)
        np.save(os.path.join(cam_out, f'dav2_{_ckpt_tag}_{stem}_pred_aligned.npy'),
                pred_aligned)
        Image.fromarray(rgb).save(
            os.path.join(cam_out, f'dav2_{_ckpt_tag}_{stem}_input.png'))

        # 3DGM
        h, w = pred_aligned.shape
        if args.intrinsics is not None:
            fx, fy, cx, cy = args.intrinsics
        elif mode in ('exo-pinhole', 'exo-pinhole-newscene'):
            fx, fy, cx, cy = _exo_intrinsics(h, w,
                                              args.exo_vfov or _EXO_DEFAULT_VFOV_DEG)
        elif mode in ('ego-pinhole', 'real', 'real-newscene'):
            # real / real-newscene use the Aria RGB camera — same focal-length
            # model as ego-pinhole
            fx, fy, cx, cy = _ego_intrinsics(h, w)
        else:  # ego-fisheye
            # Use the REAL Aria RGB focal (central-pixel pinhole model) rather
            # than an arbitrary 55° FoV. This is still only a central-pixel
            # approximation of a fisheye lens, so ego-fisheye 3DGM is NOT
            # comparable to pinhole-mode 3DGM (documented in the header). For
            # geometrically exact wide-angle normals, rectify fisheye→pinhole
            # (RGB+GT) or back-project with per-pixel fisheye ray directions.
            fx, fy, cx, cy = _ego_intrinsics(h, w)

        # Derive normals from depth; pred_aligned already has NaN where invalid
        # (disparity ≤ 0), so depth_to_normals propagates those as NaN normals
        # and they are excluded from compute_3dgm_metrics via depth_mask.
        gt_normals   = depth_to_normals(gt, fx, fy, cx, cy)
        pred_normals = depth_to_normals(pred_aligned, fx, fy, cx, cy)
        m3 = compute_3dgm_metrics(pred_normals, gt_normals, depth_mask)
        append_3dgm_to_csv(csv_path, f"DAv2_{cam_label}/{stem}", args.variant,
                           m3, "depth_to_normals(aligned)")

        # Pixel-weighted accumulation for 3DGM aggregate
        _valid_3dgm = (depth_mask
                       & np.all(np.isfinite(pred_normals), axis=-1)
                       & np.all(np.isfinite(gt_normals),   axis=-1))
        if _valid_3dgm.sum() > 0:
            _pn  = pred_normals[_valid_3dgm].astype(np.float64)
            _gn  = gt_normals[_valid_3dgm].astype(np.float64)
            _ang = np.degrees(
                np.arccos(np.clip(np.sum(_pn * _gn, axis=-1), -1.0, 1.0)))
            _n3  = len(_ang)
            px_3dgm['n']    += _n3
            px_3dgm['deg']  += float(np.sum(_ang))
            px_3dgm['sq_deg'] += float(np.sum(_ang**2))
            px_3dgm['n11']  += int(np.sum(_ang <= 11.25))
            px_3dgm['n22']  += int(np.sum(_ang <= 22.5))
            px_3dgm['n30']  += int(np.sum(_ang <= 30.0))
            px_3dgm['n45']  += int(np.sum(_ang <= 45.0))
            px_3dgm['med_frames'].append((float(np.median(_ang)), _n3))

        normal_fig = os.path.join(cam_out,
                                  f'dav2_{_ckpt_tag}_{stem}_normals.png')
        save_normal_comparison_figure_2(
            pred_depth=pred_aligned, gt_normals=gt_normals,
            pred_normals=pred_normals,
            metrics_3dgm=m3,
            model="Depth Anything V2", variant=args.variant,
            normal_source="depth_to_normals(aligned)",
            output_path=normal_fig,
            alignment=_alignment_label,
        )

    if n_ok == 0:
        print("[ERROR] No frames were successfully evaluated.")
        sys.exit(1)

    # ── Aggregate report (pixel-weighted) ────────────────────────────────────
    _n = max(px_depth['n'], 1)
    depth_mean = {
        'AbsRel':  px_depth['abs_rel'] / _n,
        'SqRel':   px_depth['sq_rel']  / _n,
        'RMSE':    float(np.sqrt(px_depth['sq_diff'] / _n)),
        'RMSElog': float(np.sqrt(px_depth['sq_log']  / _n)),
        'delta1':  px_depth['d1'] / _n,
        'delta2':  px_depth['d2'] / _n,
        'delta3':  px_depth['d3'] / _n,
        'n_valid': px_depth['n'],
    }

    dgm_mean = None
    if px_3dgm['n'] > 0:
        _n3 = px_3dgm['n']
        # Median: pixel-count-weighted mean of per-frame medians
        _total_w = sum(w for _, w in px_3dgm['med_frames'])
        _med_agg = (sum(m * w for m, w in px_3dgm['med_frames'])
                    / max(_total_w, 1))
        dgm_mean = {
            'mean_deg':   px_3dgm['deg']    / _n3,
            'median_deg': _med_agg,
            'rmse_deg':   float(np.sqrt(px_3dgm['sq_deg'] / _n3)),
            'pct_11.25':  px_3dgm['n11']   / _n3,
            'pct_22.5':   px_3dgm['n22']   / _n3,
            'pct_30':     px_3dgm['n30']   / _n3,
            'pct_45':     px_3dgm['n45']   / _n3,
            'n_valid':    _n3,
        }

    _print_aggregate(depth_mean, dgm_mean, n_ok,
                     "Depth Anything V2", args.variant, _alignment_label)
    _save_aggregate_csv(
        os.path.join(out_root, 'results_aggregate.csv'),
        "Depth_Anything_V2", args.variant,
        depth_mean, dgm_mean, n_ok, _alignment_label,
    )
    print(f"  Per-frame CSV  : {csv_path}")
    print(f"  Aggregate CSV  : {os.path.join(out_root, 'results_aggregate.csv')}")
    print(f"  Visualisations : {out_root}/")


# ── Single-frame mode (original behaviour) ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Depth Anything V2.\n\n"
            "Without --mode: single-frame evaluation (original behaviour).\n"
            "With    --mode: batch evaluation over all sequences in BATCH_SEQ_DIRS."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Batch mode switch ──────────────────────────────────────────────────────
    parser.add_argument(
        "--mode", default=None,
        choices=["ego-fisheye", "ego-pinhole", "exo-pinhole",
                 "exo-pinhole-newscene", "real", "real-newscene"],
        help=(
            "Batch evaluation mode. "
            "ego-fisheye          : frames from render_from_poses_blender.py --fisheye; "
            "ego-pinhole          : frames from render_from_poses_blender.py (no --fisheye); "
            "exo-pinhole          : all exocentric cameras from render_exocentric_blender.py "
            "                       (uses BATCH_SEQ_DIRS); "
            "exo-pinhole-newscene : bedroom exocentric cameras only (name prefix 'bed_') "
            "                       from render_exocentric_blender.py "
            "                       (uses NEWSCENE_SEQ_DIRS; --exo_camera further restricts); "
            "real                 : real Aria sensor frames "
            "                       ({seq_dir}/videos_rgb/ + {seq_dir}/depth_npy/, "
            "                       uses BATCH_SEQ_DIRS); "
            "real-newscene        : same layout as real but loads from NEWSCENE_SEQ_DIRS "
            "                       and restricts to frames annotated as one of the rooms "
            "                       in NEWSCENE_ROOMS (reads room_annotations.csv from "
            "                       {seq_dir}/videos_rgb/). "
            "When omitted the script runs in single-frame mode "
            "(--rgb / --depth_gt required)."
        ),
    )
    parser.add_argument(
        "--render_dir", default=None,
        help=(
            "Override the render output directory for batch mode. "
            "Default: {seq_dir}/blender_rendered_maps  (ego) "
            "or  {seq_dir}/exocentric_rendered  (exo)."
        ),
    )
    parser.add_argument(
        "--exo_camera", nargs="+", default=None,
        metavar="CAM",
        help="For exo-pinhole: restrict to these camera names (default: all cameras).",
    )
    parser.add_argument(
        "--exo_vfov", type=float, default=None,
        metavar="DEGREES",
        help=f"For exo-pinhole: vertical FoV in degrees used when rendering "
             f"(default: {_EXO_DEFAULT_VFOV_DEG}°).  Only needed for 3DGM.",
    )
    parser.add_argument(
        "--num_frames", type=int, default=None,
        metavar="N",
        help="Maximum number of frames to evaluate per sequence/camera. "
             "Frames are sampled uniformly across the full temporal range. "
             "Default: use all frames.",
    )
    parser.add_argument(
        "--real_rotation", type=int, default=270,
        choices=[0, 90, 180, 270],
        help="For real mode: CCW rotation applied to RGB and depth to correct "
             "Aria sensor orientation (default: 270).  "
             "Override with --rotation to apply the same value to all modes.",
    )
    parser.add_argument(
        "--real_depth_scale", type=float, default=0.001,
        help="For real mode: multiply raw depth_npy values by this factor. "
             "Default 0.001 converts uint16 millimetres → metres.  "
             "Override with --depth_scale to apply the same value to all modes.",
    )

    # ── Single-frame inputs (ignored in batch mode) ────────────────────────────
    parser.add_argument("--rgb", required=False, help="Path to input RGB image",
                        default='/user/f.zhang2/Documents/projectaria_tools_adt_data/'
                                'Apartment_release_golden_skeleton_seq100_10s_sample_M1292/'
                                'exocentric_rendered/overhead/frame_0000.png')
    parser.add_argument("--depth_gt", required=False, help="Path to GT depth .npy",
                        default='/user/f.zhang2/Documents/projectaria_tools_adt_data/'
                                'Apartment_release_golden_skeleton_seq100_10s_sample_M1292/'
                                'exocentric_rendered/overhead/depth_maps/frame_0000.npy')
    parser.add_argument("--output_dir", required=False,
                        help="Directory for outputs (single-frame) or root output dir (batch)",
                        default=None)

    # ── Model / inference ──────────────────────────────────────────────────────
    parser.add_argument("--repo_dir", required=False,
                        help="Path to the cloned Depth-Anything-V2 repo root",
                        default='/user/f.zhang2/projects/adt_egocentric/'
                                'depth_benchmark/Depth-Anything-V2')
    parser.add_argument("--ckpt_dir", required=False,
                        help="Directory containing depth_anything_v2_vit{s,b,l}.pth",
                        default='/user/f.zhang2/projects/adt_egocentric/'
                                'depth_benchmark/Depth-Anything-V2/checkpoints')
    parser.add_argument("--finetuned_from", default=None,
                        help="Path to a finetuned checkpoint produced by finetune_adt.py "
                             "(best_<split>_<variant>.pth or latest.pth). "
                             "When set, the model is loaded as a metric model using "
                             "--max_depth and alignment is skipped by default.")
    parser.add_argument("--no_align", action="store_true", default=False,
                        help="Skip scale+shift alignment and evaluate raw model output "
                             "directly. Enabled automatically when --finetuned_from is set.")
    parser.add_argument("--variant", default="large",
                        choices=["small", "base", "large"])
    parser.add_argument("--rotation", type=int, default=None,
                        choices=[0, 90, 180, 270],
                        help="Counter-clockwise rotation applied to RGB before inference. "
                             "Overrides per-mode defaults (0 for ego/exo, 270 for real).")
    parser.add_argument("--depth_scale", type=float, default=None,
                        help="Multiply GT depth by this factor. "
                             "Overrides per-mode defaults (1.0 for ego/exo, 0.001 for real).")
    parser.add_argument("--max_depth", type=float, default=10.0,
                        help="GT depth values above this (metres) are invalid")
    parser.add_argument("--mask_invalid_input", action="store_true", default=False,
                        help="Exclude pixels whose input RGB is a black/blank border "
                             "(e.g. outside the Aria fisheye circle) from alignment, "
                             "depth metrics and 3DGM. Pixels where GT depth is invalid "
                             "are always excluded regardless of this flag.")
    parser.add_argument("--input_black_thresh", type=int, default=8,
                        help="With --mask_invalid_input: an input pixel is treated as "
                             "invalid when max(R,G,B) <= this value (default: 8).")
    parser.add_argument("--input_valid_erode", type=int, default=0,
                        help="With --mask_invalid_input: erode the valid-input region by "
                             "this many pixels to also drop the noisy ring at the fisheye "
                             "circle boundary (default: 0 = disabled).")
    parser.add_argument("--device", default=None,
                        help="'cuda' or 'cpu' (default: auto-detect)")
    parser.add_argument("--csv", default=None,
                        help="Path to append CSV result row (single-frame only)")

    # ── Normals / 3DGM ────────────────────────────────────────────────────────
    parser.add_argument("--normal_gt", default=None,
                        help="GT surface normal map (.npy or .png) for 3DGM (single-frame only)")
    parser.add_argument("--intrinsics", type=float, nargs=4,
                        metavar=("fx", "fy", "cx", "cy"), default=None,
                        help="Camera intrinsics for 3DGM back-projection. "
                             "Estimated automatically in batch mode if omitted.")

    args = parser.parse_args()

    # ── Route to batch or single-frame mode ───────────────────────────────────
    if args.mode is not None:
        run_batch(args)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # Single-frame mode
    # ══════════════════════════════════════════════════════════════════════════

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [DAv2] Device: {args.device}")

    # Resolve alignment strategy (mirrors batch-mode logic)
    _no_align = args.no_align or (args.finetuned_from is not None)
    alignment  = "none (metric)" if _no_align else "scale+shift-disparity (least-squares)"
    print(f"  [DAv2] alignment={alignment}")

    out_dir = args.output_dir or '/user/f.zhang2/Documents/projectaria_tools_adt_data/' \
              'Apartment_release_golden_skeleton_seq100_10s_sample_M1292/' \
              'exocentric_rendered/overhead/depth_anything_v2_results'
    os.makedirs(out_dir, exist_ok=True)

    # Single-frame mode: rotation/depth_scale default to 0/1.0 (rendered data).
    _sf_rotation    = args.rotation    if args.rotation    is not None else 0
    _sf_depth_scale = args.depth_scale if args.depth_scale is not None else 1.0

    # ── Load inputs ─────────────────────────────────────────────────────────
    print(f"  [DAv2] Loading RGB  : {args.rgb}  (rotation={_sf_rotation}°)")
    rgb = load_rgb(args.rgb, rotation=_sf_rotation)
    print(f"         RGB shape   : {rgb.shape}")

    print(f"  [DAv2] Loading GT   : {args.depth_gt}")
    gt = load_depth_gt(args.depth_gt, depth_scale=_sf_depth_scale,
                       max_depth=args.max_depth)
    mask = get_valid_mask(gt)
    print(f"         GT shape    : {gt.shape}  valid px: {mask.sum()} / {mask.size}")

    # If rotation applied to RGB, also rotate the GT
    if _sf_rotation != 0:
        k = {90: 1, 180: 2, 270: 3}[_sf_rotation]
        gt = np.rot90(gt, k=k).copy()
        mask = get_valid_mask(gt)
        print(f"         GT rotated  : {gt.shape}  valid px: {mask.sum()}")

    # Optionally drop black/blank input-border pixels (e.g. outside the Aria
    # fisheye circle) so they cannot corrupt alignment or metrics.
    if args.mask_invalid_input:
        mask = _restrict_to_valid_input(mask, rgb, gt.shape, args)
        print(f"         Input-masked: valid px: {mask.sum()} "
              f"(black_thresh={args.input_black_thresh}, "
              f"erode={args.input_valid_erode})")

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"  [DAv2] Running inference (variant={args.variant}) …")
    model = _load_model(args)
    with torch.no_grad():
        pred_raw = model.infer_image(rgb).astype(np.float32)
    print(f"         Pred shape  : {pred_raw.shape}")
    if pred_raw.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred_raw.shape} vs gt {gt.shape}")

    # ── Alignment ────────────────────────────────────────────────────────────
    if _no_align:
        pred_aligned = pred_raw.copy()
    else:
        pred_aligned = align_scale_shift_disparity(pred_raw, gt, mask)

    depth_mask = valid_eval_mask(pred_aligned, gt, mask, max_depth=args.max_depth)
    if not _no_align:
        valid_depths = pred_aligned[depth_mask]
        print(f"  [DAv2] Aligned depth range: [{valid_depths.min():.3f}, "
              f"{valid_depths.max():.3f}] m  "
              f"({depth_mask.sum()} valid px, "
              f"{mask.sum()-depth_mask.sum()} masked as invalid)")

    # ── Metrics ──────────────────────────────────────────────────────────────
    metrics = compute_metrics(pred_aligned, gt, depth_mask)
    print_metrics(metrics, model="Depth Anything V2", variant=args.variant,
                  alignment=alignment)

    # ── Save visualisation ───────────────────────────────────────────────────
    out_stem = f"dav2_{args.variant}"
    fig_path = os.path.join(out_dir, f"{out_stem}_comparison.png")
    save_comparison_figure(
        rgb=rgb, gt=gt, pred_aligned=pred_aligned,
        metrics=metrics,
        model="Depth Anything V2", variant=args.variant,
        alignment=alignment,
        output_path=fig_path,
    )

    # Optionally save raw + aligned depth as npy
    np.save(os.path.join(out_dir, f"{out_stem}_pred_raw.npy"), pred_raw)
    np.save(os.path.join(out_dir, f"{out_stem}_pred_aligned.npy"), pred_aligned)
    Image.fromarray(rgb).save(os.path.join(out_dir, f"{out_stem}_input.png"))

    # ── CSV logging ──────────────────────────────────────────────────────────
    csv_path = args.csv or os.path.join(out_dir, "results.csv")
    append_to_csv(csv_path, "Depth_Anything_V2", args.variant, metrics, alignment)

    # ── 3D Geometric Metric (GeoNet protocol) ─────────────────────────────────
    print(f"\n  [DAv2] Computing 3DGM …")

    h, w = pred_aligned.shape
    if args.intrinsics is not None:
        fx, fy, cx, cy = args.intrinsics
        print(f"  [DAv2] Intrinsics: fx={fx:.1f} fy={fy:.1f} "
              f"cx={cx:.1f} cy={cy:.1f}")
    else:
        fx, fy, cx, cy = estimate_intrinsics(h, w)
        print(f"  [DAv2] Intrinsics estimated (55° diag FoV): "
              f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    # Both normals derived from depth (GT already rotated above).
    # pred_aligned has NaN where disparity alignment failed (pred_disp ≤ 0);
    # depth_to_normals propagates those as NaN normals, excluded via depth_mask.
    gt_normals   = depth_to_normals(gt, fx, fy, cx, cy)
    pred_normals = depth_to_normals(pred_aligned, fx, fy, cx, cy)
    normal_source = (f"depth_to_normals({'metric' if _no_align else 'disp-aligned'}, "
                     f"variant={args.variant})")

    metrics_3dgm = compute_3dgm_metrics(pred_normals, gt_normals, depth_mask)
    print_3dgm_metrics(metrics_3dgm, "Depth Anything V2",
                       variant=args.variant, normal_source=normal_source)

    np.save(os.path.join(out_dir,
                         f"dav2_{args.variant}_pred_normals.npy"), pred_normals)
    append_3dgm_to_csv(csv_path, "Depth_Anything_V2", args.variant,
                       metrics_3dgm, normal_source)

    normal_fig_path = os.path.join(out_dir,
                                   f"{out_stem}_normal_comparison.png")
    save_normal_comparison_figure_2(
        pred_depth=pred_aligned, gt_normals=gt_normals, pred_normals=pred_normals,
        metrics_3dgm=metrics_3dgm,
        model="Depth Anything V2", variant=args.variant,
        normal_source=normal_source,
        output_path=normal_fig_path,
        alignment=alignment,
    )

    return metrics


if __name__ == "__main__":
    main()