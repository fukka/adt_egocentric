"""
run_sam2.py — SAM 2.1 Large Segmentation Evaluation on ADT Egocentric Frames
=============================================================================

Runs SAM 2.1 Large (automatic mask generation) on a real Aria egocentric RGB
frame from the Aria Digital Twin (ADT) dataset and evaluates performance against
the ADT ground-truth instance segmentation.

Pipeline
--------
1. Load the real RGB frame from ADT (stream 214-1, 1408×1408) and apply a 90°
   clockwise rotation so objects appear upright (the Aria camera is mounted
   sideways — raw frames are rotated 90° CCW from natural orientation).
2. Load the GT instance segmentation from ADT (stream 400-1, 1408×1408 uint64,
   same 90° CW rotation applied).
3. Resize the image to 1024×1024 (SAM 2's native resolution) to avoid OOM on
   machines with limited RAM, then run SAM2AutomaticMaskGenerator.
4. Rescale predicted masks back to 1408×1408 using nearest-neighbour for fair
   pixel-level IoU comparison against GT.
5. For each GT instance with ≥ MIN_PX pixels, find the best-matching SAM mask
   by intersection-over-union (IoU).
6. Compute summary metrics:
     - Mean / median best-match IoU across all evaluated GT instances
     - Recall at fixed IoU thresholds: @0.25, @0.50, @0.75
     - **Final score: mAP@[0.5:0.95]** — COCO-style mean recall over 10 IoU
       thresholds {0.50, 0.55, …, 0.95}. Measures overall instance-level
       segmentation quality across strict to lenient matching criteria.
7. Save all outputs (masks, JSON results, overlay PNG, report PNG) to ADT_DIR.

Outputs
-------
  real_rot90_f0.png       Real RGB frame, 90° CW rotated
  gt_seg_rot_f0.npy       GT segmentation array (1408×1408 uint64), same rotation
  sam2_masks_f0.npy       SAM 2 masks rescaled to 1408×1408 — shape (N, H, W) bool
  sam2_masks_f0.json      Per-mask metadata (area, predicted IoU, stability score)
  sam2_overlay_f0.png     SAM 2 masks overlaid on the real frame (1024×1024)
  sam2_iou_results.json   Per-instance IoU + summary + mAP@[0.5:0.95] final score
  sam2_report_f0.png      Visual report: real frame / GT / SAM 2 overlay + IoU table

Usage
-----
  # Run from the SAM 2 repo root (so Hydra can resolve config paths)
  cd /path/to/sam2
  python /path/to/repo/run_sam2.py

Configuration
-------------
  Edit the three path constants in the CONFIG section below:
    ADT_DIR  — path to the ADT sequence directory
    SAM2_DIR — path to the sam2 repo root (for sys.path / Hydra config resolution)
    CKPT     — path to the sam2.1_hiera_large.pt checkpoint file

Requirements
------------
  projectaria-tools==2.1.1, numpy, Pillow, torch, sam2 (SAM 2.1)
  See README.md for install instructions.

Notes on memory
---------------
  SAM 2.1 Large uses ~3–4 GB RAM on CPU. To fit within tight memory budgets:
    - Input is resized from 1408 to 1024 (SAM's native size) before inference.
    - crop_n_layers=0 disables multi-scale cropping (biggest RAM saving).
    - points_per_batch=32 reduces peak activation memory during grid sampling.
  Masks are upsampled back to 1408×1408 after inference for accurate GT comparison.
"""

# ── CONFIG — update these three paths ──────────────────────────────────────
ADT_DIR  = '/path/to/ADT/Apartment_release_golden_skeleton_seq100_10s_sample_M1292'
SAM2_DIR = '/path/to/sam2'           # repo root; must contain configs/sam2.1/
CKPT     = '/path/to/sam2_weights/sam2.1_hiera_large.pt'
# ───────────────────────────────────────────────────────────────────────────

import sys, os, json
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import random

# Make sam2 importable; configs are resolved relative to the repo root
sys.path.insert(0, SAM2_DIR)
os.chdir(SAM2_DIR)   # Hydra requires cwd == sam2 repo root for config lookup

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId

# ── Constants ───────────────────────────────────────────────────────────────
FRAME_IDX     = 0      # which RGB frame to evaluate (0 = first frame)
SAM_NATIVE    = 1024   # SAM 2's native input resolution
MIN_PX        = 500    # ignore GT instances smaller than this (in pixels)
MAP_THRESHOLDS = np.arange(0.50, 1.00, 0.05)  # 0.50, 0.55, …, 0.95


# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD REAL RGB FRAME
# ════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Step 1: Loading real RGB frame from ADT")

# Stream 214-1 = Aria left-RGB camera (1408×1408, fisheye)
p_ego    = data_provider.create_vrs_data_provider(f'{ADT_DIR}/main_recording.vrs')
img_data = p_ego.get_image_data_by_index(StreamId('214-1'), FRAME_IDX)
ts_ns    = img_data[1].capture_timestamp_ns
raw      = img_data[0].to_numpy_array()   # (1408, 1408, 3) uint8, sideways

# The Aria RGB camera is mounted 90° CCW from natural upright.
# Rotate 90° CW (k=-1) to restore the upright orientation used throughout.
real_rot = np.rot90(raw, k=-1).copy()
Image.fromarray(real_rot).save(f'{ADT_DIR}/real_rot90_f0.png')
print(f"  Saved real_rot90_f0.png  shape={real_rot.shape}")


# ════════════════════════════════════════════════════════════════════════════
# 2. LOAD GT INSTANCE SEGMENTATION
# ════════════════════════════════════════════════════════════════════════════

print("\nStep 2: Loading GT instance segmentation")

# Stream 400-1 = per-pixel uint64 instance ID for the RGB camera
p_seg    = data_provider.create_vrs_data_provider(f'{ADT_DIR}/segmentations.vrs')
seg_data = p_seg.get_image_data_by_index(StreamId('400-1'), FRAME_IDX)
seg_raw  = seg_data[0].to_numpy_array()   # (1408, 1408) uint64

# Same 90° CW rotation — numpy is used here because PIL cannot handle uint64
gt_seg = np.rot90(seg_raw, k=-1).copy()
np.save(f'{ADT_DIR}/gt_seg_rot_f0.npy', gt_seg)

unique_ids = np.unique(gt_seg)
n_instances = len(unique_ids[unique_ids != 0])
print(f"  Saved gt_seg_rot_f0.npy  instances={n_instances}  "
      f"shape={gt_seg.shape}  dtype={gt_seg.dtype}")

# Build a name lookup: instance UID → object name from instances.json
uid_to_name = {}
try:
    with open(f'{ADT_DIR}/groundtruth/instances.json') as _f:
        _instances = json.load(_f)
    uid_to_name = {
        info['instance_id']: info.get('instance_name', str(info['instance_id']))
        for info in _instances.values()
        if 'instance_id' in info
    }
    print(f"  Loaded {len(uid_to_name)} instance names from instances.json")
except Exception as e:
    print(f"  Warning: could not load instance names ({e}); using raw UIDs")


# ════════════════════════════════════════════════════════════════════════════
# 3. RUN SAM 2.1 LARGE AUTOMATIC MASK GENERATION
# ════════════════════════════════════════════════════════════════════════════

print("\nStep 3: Running SAM 2.1 Large")

# Resize to SAM's native 1024×1024 to avoid OOM on low-RAM machines.
# SAM internally works at this resolution, so no quality is lost.
real_1024 = np.array(
    Image.fromarray(real_rot).resize((SAM_NATIVE, SAM_NATIVE), Image.LANCZOS)
)
print(f"  Input shape: {real_1024.shape}")

# Config path is resolved relative to the sam2 repo root (Hydra convention).
cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'

print("  Loading checkpoint...")
with torch.inference_mode():
    model = build_sam2(cfg, CKPT, device='cpu')
    generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=32,          # grid density for prompt points
        points_per_batch=32,         # smaller batches → lower peak RAM
        pred_iou_thresh=0.70,        # discard masks SAM rates < 70% IoU
        stability_score_thresh=0.92, # discard unstable masks
        stability_score_offset=0.7,
        crop_n_layers=0,             # no multi-scale crops — saves ~2 GB RAM
        min_mask_region_area=100,    # drop tiny noise masks
    )
    print("  Generating masks (this may take several minutes on CPU)...")
    masks = generator.generate(real_1024)

print(f"  -> {len(masks)} masks generated")

# Rescale each predicted mask back to the original 1408×1408 resolution
# using nearest-neighbour to preserve crisp binary boundaries.
H, W = real_rot.shape[:2]
masks_full = []
for m in masks:
    small = m['segmentation'].astype(np.uint8) * 255
    full  = np.array(Image.fromarray(small).resize((W, H), Image.NEAREST)) > 127
    masks_full.append(full)

masks_arr = np.stack(masks_full, axis=0)   # (N, 1408, 1408) bool
np.save(f'{ADT_DIR}/sam2_masks_f0.npy', masks_arr)

# Save per-mask metadata
mask_meta = [{
    'id': i,
    'area_1024': int(m['area']),
    'area_1408': int(masks_full[i].sum()),
    'predicted_iou': float(m['predicted_iou']),
    'stability_score': float(m['stability_score']),
    'bbox_xywh': [int(x) for x in m['bbox']],
} for i, m in enumerate(masks)]
with open(f'{ADT_DIR}/sam2_masks_f0.json', 'w') as f:
    json.dump(mask_meta, f, indent=2)

# Quick overlay visualisation (saved at 1024 resolution for compact file size)
np.random.seed(0)
overlay = np.zeros_like(real_1024, dtype=float)
for m in masks:
    col = np.random.randint(60, 240, 3).astype(float)
    overlay[m['segmentation']] = col
blended = (real_1024.astype(float) * 0.45 + overlay * 0.55).clip(0, 255).astype(np.uint8)
Image.fromarray(blended).save(f'{ADT_DIR}/sam2_overlay_f0.png')

print(f"  Saved sam2_masks_f0.npy {masks_arr.shape}, sam2_masks_f0.json, sam2_overlay_f0.png")


# ════════════════════════════════════════════════════════════════════════════
# 4. IoU EVALUATION — GT vs SAM 2 MASKS
# ════════════════════════════════════════════════════════════════════════════

print("\nStep 4: Computing per-instance IoU")

per_instance = []
for uid in np.unique(gt_seg):
    if uid == 0:
        continue   # background

    gt_mask  = (gt_seg == uid)
    gt_px    = int(gt_mask.sum())
    if gt_px < MIN_PX:
        continue   # skip tiny / heavily occluded instances

    name = uid_to_name.get(uid, str(uid))

    # For each SAM mask, compute IoU = |intersection| / |union|
    best_iou = 0.0
    best_idx = -1
    best_meta = {}
    for i, sm in enumerate(masks_full):
        inter = int((gt_mask & sm).sum())
        if inter == 0:
            continue
        union = int((gt_mask | sm).sum())
        iou   = inter / union
        if iou > best_iou:
            best_iou  = iou
            best_idx  = i
            best_meta = mask_meta[i]

    per_instance.append({
        'uid':            str(uid),
        'name':           name,
        'gt_px':          gt_px,
        'gt_pct':         round(100.0 * gt_px / (H * W), 6),
        'best_iou':       round(best_iou, 6),
        'sam_pred_iou':   round(best_meta.get('predicted_iou', 0.0), 6),
        'sam_stability':  round(best_meta.get('stability_score', 0.0), 6),
        'sam_area':       best_meta.get('area_1408', 0),
        'best_sam_idx':   best_idx,
    })

# Sort by GT pixel count (largest first) for easy reading
per_instance.sort(key=lambda x: x['gt_px'], reverse=True)
print(f"  Evaluated {len(per_instance)} GT instances (≥{MIN_PX} px)")


# ════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY METRICS + FINAL SCORE
# ════════════════════════════════════════════════════════════════════════════

ious = [inst['best_iou'] for inst in per_instance]
n    = len(ious)

mean_iou   = float(np.mean(ious))
median_iou = float(np.median(ious))
n_25  = sum(1 for v in ious if v >= 0.25)
n_50  = sum(1 for v in ious if v >= 0.50)
n_75  = sum(1 for v in ious if v >= 0.75)

# ── Build sam_to_gt_map for AP and precision metrics ───────────────────────
# Maps each SAM mask index to the (uid, best_iou) pairs of GT instances for
# which it is the best-matching mask.  Consistent with per-instance matching
# above: each GT independently chose its best SAM mask; duplicates are kept.
sam_to_gt_map = {}
for inst in per_instance:
    idx = inst['best_sam_idx']
    if idx < 0:
        continue
    sam_to_gt_map.setdefault(idx, []).append((inst['uid'], inst['best_iou']))

n_sam = len(masks)

# SAM masks sorted by predicted_iou descending — confidence proxy for AP curve
sam_sorted = sorted(range(n_sam),
                    key=lambda i: masks[i]['predicted_iou'], reverse=True)


def _compute_ap(sam_sorted, sam_to_gt_map, n_gt, iou_thr):
    """
    AP at a single IoU threshold, COCO 101-point interpolation.

    Predictions are processed highest-confidence first.  A mask is TP if it
    is the best match for at least one GT with best_iou ≥ iou_thr (duplicate
    matching preserved: same mask may cover multiple GTs simultaneously).
    Recall = fraction of distinct GT instances covered so far.
    """
    if n_gt == 0:
        return 0.0

    gts_covered   = set()
    cum_tp = cum_fp = 0
    prec_pts = []
    rec_pts  = []

    for sam_idx in sam_sorted:
        matched_uids = [uid for uid, iou in sam_to_gt_map.get(sam_idx, [])
                        if iou >= iou_thr]
        if matched_uids:
            cum_tp += 1
            gts_covered.update(matched_uids)
        else:
            cum_fp += 1
        prec_pts.append(cum_tp / (cum_tp + cum_fp))
        rec_pts.append(len(gts_covered) / n_gt)

    # 101-point interpolation: for each recall level r, take max precision ≥ r
    ap = 0.0
    for r_thr in np.linspace(0, 1, 101):
        precs = [p for p, r in zip(prec_pts, rec_pts) if r >= r_thr]
        ap   += max(precs) if precs else 0.0
    return ap / 101


# mAP@[0.5:0.95] — true COCO-style mean AP across 10 IoU thresholds
ap_per_thr = [_compute_ap(sam_sorted, sam_to_gt_map, n, t) for t in MAP_THRESHOLDS]
map_score  = float(np.mean(ap_per_thr))

# mAR@[0.5:0.95] — mean recall over the same thresholds (kept from before)
mar_recalls = [sum(1 for v in ious if v >= t) / n for t in MAP_THRESHOLDS]
mar_score   = float(np.mean(mar_recalls))

# ── Precision@0.50 / 0.75 ─────────────────────────────────────────────────
# Fraction of SAM masks that are the best match for at least one GT at IoU ≥ t.
# Masks with no GT match (or only low-IoU matches) count as false positives.
def _precision_at(thr):
    tp = sum(1 for matches in sam_to_gt_map.values()
             if any(iou >= thr for _, iou in matches))
    return tp / n_sam if n_sam > 0 else 0.0

prec_50 = _precision_at(0.50)
prec_75 = _precision_at(0.75)

# ── Recall@0.50 / 0.75 (fraction form, for F1) ────────────────────────────
rec_50 = n_50 / n if n > 0 else 0.0
rec_75 = n_75 / n if n > 0 else 0.0

# ── F1@0.50 / 0.75 ────────────────────────────────────────────────────────
def _f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

f1_50 = _f1(prec_50, rec_50)
f1_75 = _f1(prec_75, rec_75)

summary = {
    'n_gt_instances':      n,
    'min_px_threshold':    MIN_PX,
    'sam_masks_generated': n_sam,
    'mean_iou':            round(mean_iou, 6),
    'median_iou':          round(median_iou, 6),
    'n_iou_25':            n_25,
    'n_iou_50':            n_50,
    'n_iou_75':            n_75,
    # Recall / Precision / F1 at fixed thresholds
    'recall_50':           round(rec_50, 6),
    'recall_75':           round(rec_75, 6),
    'precision_50':        round(prec_50, 6),
    'precision_75':        round(prec_75, 6),
    'f1_50':               round(f1_50, 6),
    'f1_75':               round(f1_75, 6),
    # mAP@[0.5:0.95] — true COCO AP (precision-recall AUC per threshold, averaged)
    'mAP_50_95':           round(map_score, 6),
    'mAP_per_threshold':   {f'{t:.2f}': round(ap, 4)
                            for t, ap in zip(MAP_THRESHOLDS, ap_per_thr)},
    # mAR@[0.5:0.95] — mean recall over thresholds (was previously labelled mAP)
    'mAR_50_95':           round(mar_score, 6),
    'mAR_per_threshold':   {f'{t:.2f}': round(r, 4)
                            for t, r in zip(MAP_THRESHOLDS, mar_recalls)},
}

results = {'summary': summary, 'per_instance': per_instance}
with open(f'{ADT_DIR}/sam2_iou_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print(f"SAM 2.1 Large  |  Frame {FRAME_IDX}  |  GT instances (≥{MIN_PX} px): {n}")
print(f"  Mean IoU:    {mean_iou:.3f}  |  Median IoU:  {median_iou:.3f}")
print(f"  Recall@0.50: {rec_50:.3f}  |  Precision@0.50: {prec_50:.3f}  |  F1@0.50: {f1_50:.3f}")
print(f"  Recall@0.75: {rec_75:.3f}  |  Precision@0.75: {prec_75:.3f}  |  F1@0.75: {f1_75:.3f}")
print(f"  ── mAP@[0.5:0.95] = {map_score:.4f}  |  mAR@[0.5:0.95] = {mar_score:.4f} ──")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════════════
# 6. VISUAL REPORT
# ════════════════════════════════════════════════════════════════════════════

print("\nStep 6: Generating visual report...")

rng1 = random.Random(42)
gt_colored = np.zeros((*gt_seg.shape, 3), dtype=np.uint8)
for uid in np.unique(gt_seg):
    if uid == 0:
        continue
    col = tuple(rng1.randint(60, 240) for _ in range(3))
    gt_colored[gt_seg == uid] = col

sam_overlay = real_rot.copy()
rng2 = random.Random(123)
for sm in masks_full:
    col = np.array([rng2.randint(60, 240) for _ in range(3)])
    sam_overlay[sm] = (sam_overlay[sm].astype(float) * 0.45 + col * 0.55).astype(np.uint8)

fig = plt.figure(figsize=(22, 14), facecolor='#111111')
gs  = gridspec.GridSpec(2, 3, figure=fig,
                        top=0.90, bottom=0.02, left=0.02, right=0.98,
                        hspace=0.08, wspace=0.04)
ax_real  = fig.add_subplot(gs[0, 0])
ax_gt    = fig.add_subplot(gs[0, 1])
ax_sam   = fig.add_subplot(gs[0, 2])
ax_table = fig.add_subplot(gs[1, :])

for ax, img, title in [
    (ax_real,  real_rot,    'Real Frame (90° CW rotated)'),
    (ax_gt,    gt_colored,  'GT Instance Segmentation'),
    (ax_sam,   sam_overlay, 'SAM 2.1 Large – Predicted Masks'),
]:
    ax.imshow(img); ax.set_title(title, color='white', fontsize=13, pad=6,
                                  fontweight='bold'); ax.axis('off')

fig.text(0.5, 0.955,
         'SAM 2.1 Large — Egocentric Frame 0 Segmentation Evaluation',
         ha='center', fontsize=17, color='white', fontweight='bold')
fig.text(0.5, 0.924,
         f"GT: {n} inst ≥{MIN_PX}px  |  SAM masks: {n_sam}  |  "
         f"Mean IoU: {mean_iou:.3f}  |  "
         f"P@0.50: {prec_50:.3f}  R@0.50: {rec_50:.3f}  F1@0.50: {f1_50:.3f}  |  "
         f"P@0.75: {prec_75:.3f}  R@0.75: {rec_75:.3f}  F1@0.75: {f1_75:.3f}  |  "
         f"mAP@[0.5:0.95]: {map_score:.4f}  mAR@[0.5:0.95]: {mar_score:.4f}",
         ha='center', fontsize=10.5, color='#dddddd')

# Per-instance table
ax_table.set_facecolor('#1a1a1a')
ax_table.axis('off')
ax_table.set_xlim(0, 1); ax_table.set_ylim(0, 1)
col_hdrs = ['Object Name', 'GT Area (px)', 'GT %', 'Best IoU', 'Result']
col_x    = [0.01, 0.29, 0.44, 0.54, 0.69]
for hdr, x in zip(col_hdrs, col_x):
    ax_table.text(x, 0.97, hdr, fontsize=9.5, color='#aaaaaa',
                  fontweight='bold', va='top', transform=ax_table.transAxes)

MAX_ROWS = 30
row_h = 0.88 / MAX_ROWS
for i, inst in enumerate(per_instance[:MAX_ROWS]):
    y   = 0.93 - (i + 1) * row_h
    iou = inst['best_iou']
    col = ('#6ee86e' if iou >= 0.75 else '#f5c842' if iou >= 0.50 else
           '#f07070' if iou >= 0.25 else '#cc4444')
    tag = ('✓ Good' if iou >= 0.75 else '~ Partial' if iou >= 0.50 else
           '✗ Weak' if iou >= 0.25 else '✗ Fail')
    bg  = '#222222' if i % 2 == 0 else '#1a1a1a'
    ax_table.add_patch(mpatches.FancyBboxPatch(
        (0, y), 1, row_h, boxstyle='square,pad=0',
        facecolor=bg, edgecolor='none',
        transform=ax_table.transAxes, clip_on=True))
    for v, x, tc in zip(
        [inst['name'], f"{inst['gt_px']:,}", f"{inst['gt_pct']:.2f}%",
         f"{iou:.3f}", tag],
        col_x, ['white', '#aaaaaa', '#aaaaaa', col, col]
    ):
        ax_table.text(x, y + row_h * 0.45, v, fontsize=7.8, color=tc,
                      va='center', transform=ax_table.transAxes)

ax_table.set_title(
    f'Per-Instance IoU — Top {MAX_ROWS} by GT Area  '
    f'[mAP@[0.5:0.95]={map_score:.4f}  mAR@[0.5:0.95]={mar_score:.4f}  '
    f'F1@0.50={f1_50:.3f}  F1@0.75={f1_75:.3f}]',
    color='white', fontsize=11, pad=6, fontweight='bold')
ax_table.legend(handles=[
    mpatches.Patch(color='#6ee86e', label='IoU ≥ 0.75 (Good)'),
    mpatches.Patch(color='#f5c842', label='0.50 ≤ IoU < 0.75 (Partial)'),
    mpatches.Patch(color='#f07070', label='0.25 ≤ IoU < 0.50 (Weak)'),
    mpatches.Patch(color='#cc4444', label='IoU < 0.25 (Fail)'),
], loc='lower right', fontsize=8, framealpha=0.3, labelcolor='white',
   facecolor='#111111', edgecolor='#444444')

fig.savefig(f'{ADT_DIR}/sam2_report_f0.png', dpi=130,
            bbox_inches='tight', facecolor='#111111')
plt.close()
print(f"  Saved sam2_report_f0.png")

print("\nDone. All outputs written to:", ADT_DIR)
