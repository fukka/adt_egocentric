"""
eval_by_size.py
===============
Break down SAM 2 evaluation metrics by GT instance pixel-count bracket.

Loads the JSON outputs produced by run_sam2.py and re-computes every metric
independently for each size bracket, revealing how detection/segmentation
quality varies with object size.

Metrics reported per bracket
----------------------------
  mAP@[0.5:0.95]  — true COCO-style AP (precision-recall AUC per threshold)
  mAR@[0.5:0.95]  — mean recall over the same 10 thresholds
  Precision@0.50 / 0.75
  Recall@0.50    / 0.75
  F1@0.50        / 0.75
  Mean best-match IoU
  N               — number of GT instances in the bracket

Key design choice — per-bracket AP
-----------------------------------
For bracket B, ALL SAM masks are treated as predictions (same pool as the
global evaluation).  A mask is TP at threshold t if it is the best match for
≥1 GT in B with best_iou ≥ t; otherwise it is FP.  Masks that match GTs
OUTSIDE B do not count as TP for B's evaluation (analogous to COCO
per-category AP).  Duplicate matching is preserved: one mask can
simultaneously satisfy multiple GTs in B.

Usage
-----
  # Metrics only
  python eval_by_size.py \\
      --results  /path/to/sam2_iou_results.json \\
      --masks    /path/to/sam2_masks_f0.json \\
      [--brackets 0 100 500 2000 10000] \\
      [--output_dir /path/to/output]

  # Metrics + bracket-coloured mask overlays
  python eval_by_size.py \\
      --results    /path/to/sam2_iou_results.json \\
      --masks      /path/to/sam2_masks_f0.json \\
      --rgb        /path/to/real_rot90_f0.png \\
      [--overlay_alpha 0.45]

  The two companion .npy files are resolved from --masks automatically,
  preserving the shared frame suffix (f0, f150, …):
    sam2_masks_f0.json  →  sam2_masks_f0.npy   (SAM predicted masks)
                        →  gt_seg_rot_f0.npy    (GT segmentation, same frame)

Outputs
-------
  <output_dir>/by_size_metrics.json      — full per-bracket metrics + per-instance listing
  <output_dir>/by_size_metrics.png       — grouped bar chart across brackets
  <output_dir>/by_size_overlay.png       — GT + SAM prediction masks coloured by bracket
                                           (only when --rgb is given)
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from PIL import Image

MAP_THRESHOLDS = np.arange(0.50, 1.00, 0.05)   # 0.50, 0.55, …, 0.95

# ── Bracket palette (R, G, B) in 0-255, up to 8 brackets ─────────────────────
BRACKET_PALETTE = [
    ( 64, 224, 208),   # turquoise
    (144, 238,  80),   # lime green
    (255, 165,   0),   # orange
    (255,  80, 180),   # hot pink
    (255,  69,   0),   # red-orange
    (147, 112, 219),   # medium purple
    ( 30, 144, 255),   # dodger blue
    (255, 215,   0),   # gold
]


# ══════════════════════════════════════════════════════════════════════════════
# Bracket helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_brackets(edges: list):
    """
    Convert a sorted list of left edges into (lo, hi, label) triples.
    The last bracket is open-ended (hi = +inf).

    Example: edges=[0,100,500,2000,10000]
      → [(0,100,'0–99 px'), (100,500,'100–499 px'), …, (10000,inf,'>10,000 px')]
    """
    brackets = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else float("inf")
        if hi == float("inf"):
            label = f">{lo:,} px"
        else:
            label = f"{lo:,}–{hi - 1:,} px"
        brackets.append((lo, hi, label))
    return brackets


def assign_to_bracket(gt_px: int, brackets: list) -> int:
    """Return the index of the bracket that contains gt_px."""
    for i, (lo, hi, _) in enumerate(brackets):
        if lo <= gt_px < hi:
            return i
    return len(brackets) - 1   # overflow → last bracket


# ══════════════════════════════════════════════════════════════════════════════
# Metric computation  (mirrors run_sam2.py logic, scoped to a bracket)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_ap(sam_sorted, sam_to_gt_map, n_gt, iou_thr):
    """
    AP at one IoU threshold via 101-point COCO interpolation.

    sam_sorted     : SAM mask indices sorted by predicted_iou descending
    sam_to_gt_map  : {sam_idx: [(uid, best_iou), …]}  — restricted to bracket GTs
    n_gt           : number of GT instances in this bracket
    """
    if n_gt == 0:
        return 0.0

    gts_covered = set()
    cum_tp = cum_fp = 0
    prec_pts = []
    rec_pts = []

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

    ap = 0.0
    for r_thr in np.linspace(0, 1, 101):
        precs = [p for p, r in zip(prec_pts, rec_pts) if r >= r_thr]
        ap += max(precs) if precs else 0.0
    return ap / 101


def compute_bracket_metrics(instances, sam_to_gt_map_full, sam_sorted, n_sam):
    """
    Compute all metrics for the GT instances that fall in one size bracket.

    Parameters
    ----------
    instances          : list of per-instance dicts (from sam2_iou_results.json)
    sam_to_gt_map_full : global {sam_idx: [(uid, best_iou)]} built from ALL GTs
    sam_sorted         : all SAM mask indices sorted by confidence descending
    n_sam              : total number of SAM masks (denominator for precision)

    Returns
    -------
    dict of metrics, or None if the bracket is empty.
    """
    n_gt = len(instances)
    if n_gt == 0:
        return None

    uids_in_bracket = {inst["uid"] for inst in instances}
    ious = [inst["best_iou"] for inst in instances]

    # Restrict the SAM→GT map to only GTs in this bracket.
    # Masks that happen to be best-match for GTs outside the bracket
    # count as FP within this bracket's AP computation.
    sam_to_gt_map_b = {}
    for sam_idx, matches in sam_to_gt_map_full.items():
        filtered = [(uid, iou) for uid, iou in matches if uid in uids_in_bracket]
        if filtered:
            sam_to_gt_map_b[sam_idx] = filtered

    # ── mAP@[0.5:0.95] ───────────────────────────────────────────────────────
    ap_per_thr = [_compute_ap(sam_sorted, sam_to_gt_map_b, n_gt, t)
                  for t in MAP_THRESHOLDS]
    map_score = float(np.mean(ap_per_thr))

    # ── mAR@[0.5:0.95] ───────────────────────────────────────────────────────
    mar_recalls = [sum(1 for v in ious if v >= t) / n_gt for t in MAP_THRESHOLDS]
    mar_score = float(np.mean(mar_recalls))

    # ── Precision@t  (TP SAM masks / all SAM masks) ───────────────────────────
    def _prec(thr):
        tp = sum(1 for matches in sam_to_gt_map_b.values()
                 if any(iou >= thr for _, iou in matches))
        return tp / n_sam if n_sam > 0 else 0.0

    # ── Recall@t  (fraction of bracket GTs matched) ───────────────────────────
    def _rec(thr):
        return sum(1 for v in ious if v >= thr) / n_gt

    # ── F1 ────────────────────────────────────────────────────────────────────
    def _f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    p50, p75 = _prec(0.50), _prec(0.75)
    r50, r75 = _rec(0.50),  _rec(0.75)

    return {
        "n_gt":         n_gt,
        "mean_best_iou": round(float(np.mean(ious)), 4),
        "median_best_iou": round(float(np.median(ious)), 4),
        "mAP_50_95":    round(map_score, 4),
        "mAR_50_95":    round(mar_score, 4),
        "precision_50": round(p50, 4),
        "precision_75": round(p75, 4),
        "recall_50":    round(r50, 4),
        "recall_75":    round(r75, 4),
        "f1_50":        round(_f1(p50, r50), 4),
        "f1_75":        round(_f1(p75, r75), 4),
        "mAP_per_threshold": {
            f"{t:.2f}": round(ap, 4)
            for t, ap in zip(MAP_THRESHOLDS, ap_per_thr)
        },
        "mAR_per_threshold": {
            f"{t:.2f}": round(r, 4)
            for t, r in zip(MAP_THRESHOLDS, mar_recalls)
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Console table
# ══════════════════════════════════════════════════════════════════════════════

def print_table(bracket_results: list):
    """Print a compact aligned table of per-bracket metrics."""
    col_w = 18
    metrics = [
        ("N",          "n_gt"),
        ("MeanIoU",    "mean_best_iou"),
        ("mAP[.5:.95]","mAP_50_95"),
        ("mAR[.5:.95]","mAR_50_95"),
        ("P@0.50",     "precision_50"),
        ("R@0.50",     "recall_50"),
        ("F1@0.50",    "f1_50"),
        ("P@0.75",     "precision_75"),
        ("R@0.75",     "recall_75"),
        ("F1@0.75",    "f1_75"),
    ]

    label_w = max(len(r["label"]) for r in bracket_results) + 2
    header = f"{'Bracket':<{label_w}}" + "".join(f"{h:>{col_w}}" for h, _ in metrics)
    sep    = "─" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)
    for row in bracket_results:
        m = row["metrics"]
        if m is None:
            line = f"{row['label']:<{label_w}}" + f"{'(no instances)':>{col_w}}"
        else:
            vals = []
            for _, key in metrics:
                v = m[key]
                vals.append(f"{v:>{col_w}}" if isinstance(v, int)
                            else f"{v:>{col_w}.4f}")
            line = f"{row['label']:<{label_w}}" + "".join(vals)
        print(line)
    print(sep + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def save_figure(bracket_results: list, output_path: str):
    """
    Save a grouped bar chart showing key metrics across size brackets.

    Top row: mAP@[0.5:0.95], mAR@[0.5:0.95], Mean IoU
    Bottom row: P/R/F1 at 0.50, P/R/F1 at 0.75
    """
    valid = [r for r in bracket_results if r["metrics"] is not None]
    if not valid:
        print("  [fig] No brackets with data — skipping figure.")
        return

    labels = [r["label"] for r in valid]
    n      = len(labels)
    x      = np.arange(n)

    def _get(key):
        return np.array([r["metrics"][key] for r in valid], dtype=float)

    fig = plt.figure(figsize=(max(14, n * 2.2), 10), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                             top=0.88, bottom=0.12, left=0.07, right=0.97)

    bar_kw = dict(width=0.25, alpha=0.88)
    COLORS = {
        "mAP":  "#4fc3f7",
        "mAR":  "#81c784",
        "IoU":  "#ffb74d",
        "P":    "#f48fb1",
        "R":    "#80deea",
        "F1":   "#ce93d8",
    }

    def _bar_ax(ax, data_dict, title, ylim=(0, 1)):
        """Draw grouped bars; data_dict = {legend_label: (color, array)}."""
        n_bars = len(data_dict)
        offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_kw["width"]
        for offset, (lbl, (col, arr)) in zip(offsets, data_dict.items()):
            ax.bar(x + offset, arr, color=col, label=lbl, **bar_kw)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right",
                            fontsize=7.5, color="white")
        ax.set_ylim(*ylim)
        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(colors="white", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color("#555577")
        ax.set_facecolor("#12122a")
        ax.yaxis.label.set_color("white")
        ax.legend(fontsize=8, framealpha=0.25, labelcolor="white",
                  facecolor="#1a1a2e", edgecolor="#555577")
        # Annotate each bar with its value
        for bar_group in ax.containers:
            for bar in bar_group:
                h = bar.get_height()
                if h > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                            f"{h:.2f}", ha="center", va="bottom",
                            fontsize=6, color="white", alpha=0.8)

    # ── Row 0, col 0: mAP vs mAR ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    _bar_ax(ax, {
        "mAP@[.5:.95]": (COLORS["mAP"], _get("mAP_50_95")),
        "mAR@[.5:.95]": (COLORS["mAR"], _get("mAR_50_95")),
    }, "mAP & mAR  @[0.5:0.95]")

    # ── Row 0, col 1: Mean IoU ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    _bar_ax(ax, {
        "Mean IoU":   (COLORS["IoU"], _get("mean_best_iou")),
        "Median IoU": (COLORS["F1"],  _get("median_best_iou")),
    }, "Best-match IoU")

    # ── Row 0, col 2: Instance counts (bar chart, raw count) ─────────────────
    ax = fig.add_subplot(gs[0, 2])
    counts = _get("n_gt")
    ax.bar(x, counts, color=COLORS["P"], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5, color="white")
    ax.set_title("GT Instance Count per Bracket", color="white",
                 fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#555577")
    ax.set_facecolor("#12122a")
    ax.set_ylim(0, max(counts) * 1.2)
    for i, c in enumerate(counts):
        ax.text(i, c + 0.2, str(int(c)), ha="center", va="bottom",
                fontsize=9, color="white")

    # ── Row 1, col 0: P / R / F1  @0.50 ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    _bar_ax(ax, {
        "Precision": (COLORS["P"], _get("precision_50")),
        "Recall":    (COLORS["R"], _get("recall_50")),
        "F1":        (COLORS["F1"], _get("f1_50")),
    }, "Precision / Recall / F1  @IoU=0.50")

    # ── Row 1, col 1: P / R / F1  @0.75 ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    _bar_ax(ax, {
        "Precision": (COLORS["P"], _get("precision_75")),
        "Recall":    (COLORS["R"], _get("recall_75")),
        "F1":        (COLORS["F1"], _get("f1_75")),
    }, "Precision / Recall / F1  @IoU=0.75")

    # ── Row 1, col 2: mAP per IoU threshold (line per bracket) ───────────────
    ax = fig.add_subplot(gs[1, 2])
    cmap_lines = plt.get_cmap("tab10")
    thr_labels = [f"{t:.2f}" for t in MAP_THRESHOLDS]
    thr_x = np.arange(len(thr_labels))
    for i, row in enumerate(valid):
        ap_vals = [row["metrics"]["mAP_per_threshold"][k] for k in thr_labels]
        ax.plot(thr_x, ap_vals, marker="o", markersize=3,
                color=cmap_lines(i / max(len(valid) - 1, 1)),
                label=row["label"], linewidth=1.5)
    ax.set_xticks(thr_x)
    ax.set_xticklabels(thr_labels, rotation=45, fontsize=7, color="white")
    ax.set_ylim(0, 1)
    ax.set_title("AP per IoU Threshold by Bracket", color="white",
                 fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#555577")
    ax.set_facecolor("#12122a")
    ax.legend(fontsize=7, framealpha=0.25, labelcolor="white",
              facecolor="#1a1a2e", edgecolor="#555577")

    fig.suptitle("SAM 2  —  Metrics by GT Instance Size Bracket",
                 color="white", fontsize=14, fontweight="bold", y=0.95)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [fig] Saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Bracket-coloured mask overlay
# ══════════════════════════════════════════════════════════════════════════════

def _blend(canvas: np.ndarray, mask: np.ndarray,
           color_rgb: tuple, alpha: float) -> None:
    """In-place alpha-blend color onto canvas pixels where mask is True."""
    r, g, b = color_rgb
    for c, val in enumerate((r, g, b)):
        canvas[:, :, c] = np.where(
            mask,
            np.clip((1 - alpha) * canvas[:, :, c] + alpha * val, 0, 255).astype(np.uint8),
            canvas[:, :, c],
        )


def save_overlay_figures(bracket_results: list,
                         rgb_np: np.ndarray,
                         gt_seg: np.ndarray,
                         sam_masks: np.ndarray,
                         output_path: str,
                         alpha: float = 0.45) -> None:
    """
    Save a side-by-side figure with two panels:
      Left  — GT instance masks coloured by size bracket
      Right — Best-matching SAM prediction masks, same bracket colours

    Parameters
    ----------
    bracket_results : list produced by main() (instances include best_sam_idx)
    rgb_np          : (H, W, 3) uint8 RGB image
    gt_seg          : (H, W) uint64 GT segmentation UID map
    sam_masks       : (N, H, W) bool predicted masks from SAM
    output_path     : destination PNG path
    alpha           : blending transparency for mask fill  (default 0.45)
    """
    H, W = rgb_np.shape[:2]
    gt_canvas   = rgb_np.copy()
    pred_canvas = rgb_np.copy()

    # ── Collect bracket info and draw masks ───────────────────────────────────
    legend_patches = []
    seen_brackets  = set()

    for b_idx, row in enumerate(bracket_results):
        if row["metrics"] is None:
            continue   # empty bracket

        color = BRACKET_PALETTE[b_idx % len(BRACKET_PALETTE)]
        label = row["label"]

        for inst in row["instances"]:
            # ── GT mask ───────────────────────────────────────────────────────
            try:
                uid_int  = int(inst["uid"])
                gt_mask  = (gt_seg == uid_int)
                if gt_mask.any():
                    _blend(gt_canvas, gt_mask, color, alpha)
            except (ValueError, OverflowError):
                pass

            # ── Best-matching SAM prediction mask ─────────────────────────────
            sam_idx = inst.get("best_sam_idx", -1)
            if sam_idx >= 0 and sam_idx < len(sam_masks):
                pred_mask = sam_masks[sam_idx]   # (H, W) bool
                if pred_mask.shape != (H, W):
                    # Nearest-neighbour resize to match RGB
                    pred_mask = np.array(
                        Image.fromarray(pred_mask.astype(np.uint8) * 255)
                        .resize((W, H), Image.NEAREST)
                    ).astype(bool)
                if pred_mask.any():
                    _blend(pred_canvas, pred_mask, color, alpha)

        # Legend entry (one per non-empty bracket)
        if label not in seen_brackets:
            seen_brackets.add(label)
            r, g, b = color
            legend_patches.append(
                mpatches.Patch(
                    facecolor=(r / 255, g / 255, b / 255),
                    label=label,
                    alpha=0.85,
                )
            )

    # ── Draw contours on both panels (thin white outline per instance) ─────────
    try:
        import cv2
        for b_idx, row in enumerate(bracket_results):
            if row["metrics"] is None:
                continue
            color = BRACKET_PALETTE[b_idx % len(BRACKET_PALETTE)]
            bgr   = (color[2], color[1], color[0])

            for inst in row["instances"]:
                # GT contour
                try:
                    uid_int = int(inst["uid"])
                    gt_mask = (gt_seg == uid_int).astype(np.uint8)
                    cnts, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(gt_canvas, cnts, -1, bgr, 1)
                except (ValueError, OverflowError):
                    pass

                # Pred contour
                sam_idx = inst.get("best_sam_idx", -1)
                if sam_idx >= 0 and sam_idx < len(sam_masks):
                    pred_mask = sam_masks[sam_idx]
                    if pred_mask.shape != (H, W):
                        pred_mask = np.array(
                            Image.fromarray(pred_mask.astype(np.uint8) * 255)
                            .resize((W, H), Image.NEAREST)
                        ).astype(bool)
                    pmask_u8 = pred_mask.astype(np.uint8)
                    cnts, _ = cv2.findContours(pmask_u8, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(pred_canvas, cnts, -1, bgr, 1)
    except ImportError:
        pass   # cv2 optional — contours skipped

    # ── Compose figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(22, 11), facecolor="#1a1a2e")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08,
                        wspace=0.04)

    for ax, canvas, title in [
        (axes[0], gt_canvas,   "GT Instances  (coloured by size bracket)"),
        (axes[1], pred_canvas, "SAM Best-match Predictions  (same bracket colours)"),
    ]:
        ax.imshow(canvas)
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")
        ax.set_facecolor("#1a1a2e")

    # Shared legend below both panels
    if legend_patches:
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=min(len(legend_patches), 6),
            fontsize=11,
            framealpha=0.3,
            labelcolor="white",
            facecolor="#1a1a2e",
            edgecolor="#555577",
            bbox_to_anchor=(0.5, 0.01),
        )

    fig.suptitle("SAM 2  —  Mask Overlay by GT Instance Size Bracket",
                 color="white", fontsize=15, fontweight="bold", y=0.97)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [overlay] Saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Report SAM 2 metrics broken down by GT instance pixel size.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results",    required=True,
                        help="Path to sam2_iou_results.json (output of run_sam2.py)")
    parser.add_argument("--masks",      required=True,
                        help="Path to sam2_masks_f0.json (output of run_sam2.py)")
    parser.add_argument("--brackets",   type=int, nargs="+",
                        default=[0, 100, 500, 2000, 10000],
                        help="Left edges of size brackets in pixels. The rightmost "
                             "bracket is open-ended. (default: 0 100 500 2000 10000)")
    parser.add_argument("--output_dir", default=None,
                        help="Directory for output files. Defaults to the directory "
                             "containing --results.")
    # ── Optional overlay visualisation ────────────────────────────────────────
    parser.add_argument("--rgb",        default=None,
                        help="Path to the input RGB image (e.g. real_rot90_f0.png).  "
                             "When given, saves by_size_overlay.png.  "
                             "gt_seg_rot_f0.npy and the SAM masks .npy are "
                             "resolved automatically from --results / --masks paths.")
    parser.add_argument("--overlay_alpha", type=float, default=0.45,
                        help="Transparency for mask fill in overlay (default: 0.45).")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.results))
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load inputs ───────────────────────────────────────────────────────────
    print(f"Loading {args.results} …")
    with open(args.results) as f:
        results_json = json.load(f)

    print(f"Loading {args.masks} …")
    with open(args.masks) as f:
        masks_meta = json.load(f)   # list of {id, predicted_iou, …}

    per_instance = results_json["per_instance"]
    n_sam        = len(masks_meta)

    print(f"  GT instances : {len(per_instance)}")
    print(f"  SAM masks    : {n_sam}")

    # ── Build sam_to_gt_map  (mirrors run_sam2.py) ────────────────────────────
    sam_to_gt_map = {}
    for inst in per_instance:
        idx = inst["best_sam_idx"]
        if idx < 0:
            continue
        sam_to_gt_map.setdefault(idx, []).append((inst["uid"], inst["best_iou"]))

    # Sort SAM masks by predicted_iou descending (confidence proxy)
    sam_sorted = sorted(range(n_sam),
                        key=lambda i: masks_meta[i]["predicted_iou"], reverse=True)

    # ── Build brackets and assign instances ───────────────────────────────────
    edges    = sorted(set(args.brackets))
    brackets = make_brackets(edges)
    print(f"\nBrackets ({len(brackets)}):")
    for lo, hi, label in brackets:
        print(f"  {label}")

    # Assign each instance to a bracket
    grouped = {label: [] for _, _, label in brackets}
    unassigned = []
    for inst in per_instance:
        idx = assign_to_bracket(inst["gt_px"], brackets)
        if idx < len(brackets):
            grouped[brackets[idx][2]].append(inst)
        else:
            unassigned.append(inst)

    if unassigned:
        print(f"  Warning: {len(unassigned)} instances not assigned to any bracket")

    # ── Compute metrics per bracket ───────────────────────────────────────────
    print("\nComputing per-bracket metrics …")
    bracket_results = []
    for lo, hi, label in brackets:
        instances = grouped[label]
        metrics   = compute_bracket_metrics(instances, sam_to_gt_map,
                                            sam_sorted, n_sam)
        bracket_results.append({
            "label":    label,
            "lo_px":    lo,
            "hi_px":    hi if hi != float("inf") else None,
            "metrics":  metrics,
            "instances": [
                {
                    "uid":          inst["uid"],
                    "name":         inst["name"],
                    "gt_px":        inst["gt_px"],
                    "best_iou":     inst["best_iou"],
                    "best_sam_idx": inst.get("best_sam_idx", -1),
                }
                for inst in instances
            ],
        })

    # ── Print table ───────────────────────────────────────────────────────────
    print_table(bracket_results)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = os.path.join(args.output_dir, "by_size_metrics.json")
    with open(out_json, "w") as f:
        json.dump(bracket_results, f, indent=2)
    print(f"  [json] Saved → {out_json}")

    # ── Save metrics figure ───────────────────────────────────────────────────
    out_fig = os.path.join(args.output_dir, "by_size_metrics.png")
    save_figure(bracket_results, out_fig)

    # ── Save overlay figure (optional — only needs --rgb) ────────────────────
    if args.rgb:
        # Both .npy files sit alongside sam2_masks_f<N>.json and share the
        # same frame suffix — sam2_masks_f0 ↔ gt_seg_rot_f0, etc.
        masks_abs      = os.path.abspath(args.masks)
        masks_dir      = os.path.dirname(masks_abs)
        masks_stem     = os.path.splitext(os.path.basename(masks_abs))[0]   # sam2_masks_f0
        gt_seg_stem    = masks_stem.replace("sam2_masks", "gt_seg_rot")     # gt_seg_rot_f0
        gt_seg_path    = os.path.join(masks_dir, gt_seg_stem + ".npy")
        sam_masks_path = os.path.join(masks_dir, masks_stem + ".npy")

        missing = [
            (name, path) for name, path in [
                ("gt_seg_rot_f0.npy",  gt_seg_path),
                (os.path.basename(sam_masks_path), sam_masks_path),
            ] if not os.path.isfile(path)
        ]
        if missing:
            print(f"\n  [overlay] Skipping — companion .npy file(s) not found:")
            for name, path in missing:
                print(f"    {name}  →  {path}")
        else:
            print(f"\nLoading overlay inputs …")
            print(f"  GT seg   : {gt_seg_path}")
            print(f"  SAM masks: {sam_masks_path}")
            rgb_np    = np.array(Image.open(args.rgb).convert("RGB"))
            gt_seg    = np.load(gt_seg_path)
            sam_masks = np.load(sam_masks_path)
            print(f"  RGB      : {rgb_np.shape}")
            print(f"  GT seg   : {gt_seg.shape}  dtype={gt_seg.dtype}")
            print(f"  SAM masks: {sam_masks.shape}  dtype={sam_masks.dtype}")
            out_overlay = os.path.join(args.output_dir, "by_size_overlay.png")
            save_overlay_figures(bracket_results, rgb_np, gt_seg, sam_masks,
                                 out_overlay, alpha=args.overlay_alpha)

    # ── Quick summary ─────────────────────────────────────────────────────────
    print("\nSummary  (brackets with ≥1 instance):")
    for row in bracket_results:
        m = row["metrics"]
        if m is None:
            continue
        print(f"  {row['label']:<22s}  n={m['n_gt']:3d}  "
              f"mAP={m['mAP_50_95']:.3f}  mAR={m['mAR_50_95']:.3f}  "
              f"F1@50={m['f1_50']:.3f}  F1@75={m['f1_75']:.3f}  "
              f"MeanIoU={m['mean_best_iou']:.3f}")


if __name__ == "__main__":
    main()
