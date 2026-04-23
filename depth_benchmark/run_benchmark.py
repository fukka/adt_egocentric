"""
run_benchmark.py
================
Aggregate benchmark runner for egocentric depth estimation.

Runs all baselines in sequence, collects their metrics, and produces:
  - results.csv        — one row per model/variant/alignment
  - summary_table.png  — visual comparison table
  - summary_bar.png    — bar chart comparing AbsRel and δ₁ across models

Baselines included
------------------
  1. Depth Anything V2  (small / base / large)
  2. Marigold           (standard / lcm)
  3. UniDepth v2        (vitl14)
  4. Metric3D v2        (vit_large)

Each baseline can be toggled with --skip-* flags.

Usage
-----
  python run_benchmark.py \\
      --rgb        /Users/fengjiazhang/Desktop/ADT/maps_opaque_test/rgb/frame_0000.png \\
      --depth_gt   /Users/fengjiazhang/Desktop/ADT/maps_opaque_test/depth_maps/frame_0000.npy \\
      --output_dir /Users/fengjiazhang/Desktop/ADT/benchmark_results \\
      [--rotation  0|90|180|270]       (default: 0) \\
      [--depth_scale 1.0]              (default: 1.0; use 0.001 for mm→m) \\
      [--max_depth   10.0]             (default: 10.0 m) \\
      [--intrinsics fx fy cx cy]       (optional; used by metric models) \\
      [--device     cuda|cpu]          (default: auto) \\
      [--dav2_variant  small|base|large]  (default: large) \\
      [--skip_dav2]                    \\
      [--skip_marigold]                \\
      [--skip_unidepth]                \\
      [--skip_metric3d]

Dependencies
------------
  See requirements.txt
"""

import argparse
import os
import sys
import csv
import subprocess
import importlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add benchmark dir to path
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH_DIR)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def run_baseline(module_name: str, argv: list) -> None:
    """Import and run a baseline module with a patched sys.argv."""
    original_argv = sys.argv
    try:
        sys.argv = [module_name + ".py"] + argv
        mod = importlib.import_module(module_name)
        mod.main()
    except SystemExit:
        pass
    except Exception as e:
        print(f"\n  [WARNING] {module_name} failed: {e}\n")
    finally:
        sys.argv = original_argv


def read_results_csv(csv_path: str) -> list:
    """Read results.csv and return a list of row dicts."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_summary_table_figure(rows: list, output_path: str) -> None:
    """
    Render a clean comparison table image.
    Rows with alignment == 'none (metric)' or 'scale+shift (least-squares)'
    are shown; scale-only and secondary rows are greyed.
    """
    if not rows:
        print("  [summary] No results to plot.")
        return

    # Filter to primary rows
    primary_alignments = {"none (metric)", "scale+shift (least-squares)"}
    primary = [r for r in rows if r.get("alignment", "") in primary_alignments]
    if not primary:
        primary = rows

    headers = ["Model", "Variant", "Alignment",
               "AbsRel↓", "SqRel↓", "RMSE↓", "RMSElog↓",
               "δ₁↑(%)", "δ₂↑(%)", "δ₃↑(%)"]

    table_data = []
    for r in primary:
        table_data.append([
            r.get("model", ""),
            r.get("variant", ""),
            r.get("alignment", ""),
            f"{float(r['AbsRel']):.4f}",
            f"{float(r['SqRel']):.4f}",
            f"{float(r['RMSE']):.4f}",
            f"{float(r['RMSElog']):.4f}",
            f"{float(r['delta1'])*100:.1f}",
            f"{float(r['delta2'])*100:.1f}",
            f"{float(r['delta3'])*100:.1f}",
        ])

    fig, ax = plt.subplots(figsize=(18, max(3, 0.5 * len(table_data) + 1.5)))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")

    tbl = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Style header
    for j in range(len(headers)):
        cell = tbl[0, j]
        cell.set_facecolor("#2E75B6")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colours; highlight best metric per column
    metric_cols = [3, 4, 5, 6, 7, 8, 9]  # 0-indexed in headers
    # Extract numeric values per column
    lower_better = {3, 4, 5, 6}   # AbsRel, SqRel, RMSE, RMSElog
    upper_better = {7, 8, 9}       # δ1, δ2, δ3

    col_vals = {}
    for ci in metric_cols:
        try:
            vals = [float(table_data[ri][ci]) for ri in range(len(table_data))]
            col_vals[ci] = vals
        except Exception:
            col_vals[ci] = []

    for i in range(len(table_data)):
        bg = "#2d2d44" if i % 2 == 0 else "#1a1a2e"
        for j in range(len(headers)):
            cell = tbl[i + 1, j]
            cell.set_facecolor(bg)
            cell.set_text_props(color="white")

    # Highlight best in each metric column
    for ci in metric_cols:
        vals = col_vals.get(ci, [])
        if not vals:
            continue
        if ci in lower_better:
            best_i = int(np.argmin(vals))
        else:
            best_i = int(np.argmax(vals))
        cell = tbl[best_i + 1, ci]
        cell.set_facecolor("#1a6b3c")
        cell.set_text_props(color="#90ee90", fontweight="bold")

    ax.set_title("Depth Estimation Benchmark — Summary", color="white",
                 fontsize=13, pad=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [summary] Table saved → {output_path}")


def build_bar_chart(rows: list, output_path: str) -> None:
    """Bar chart comparing AbsRel and δ₁ across primary model configurations."""
    if not rows:
        return

    primary_alignments = {"none (metric)", "scale+shift (least-squares)"}
    primary = [r for r in rows if r.get("alignment", "") in primary_alignments]
    if not primary:
        primary = rows

    labels  = [f"{r['model']}\n[{r['variant']}]" for r in primary]
    absrel  = [float(r["AbsRel"]) for r in primary]
    delta1  = [float(r["delta1"]) * 100 for r in primary]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, len(labels) * 2.5), 5))
    fig.patch.set_facecolor("#1a1a2e")

    colors = ["#2E75B6", "#E8763A", "#2ECC71", "#9B59B6",
              "#F39C12", "#1ABC9C", "#E74C3C"]

    for ax in (ax1, ax2):
        ax.set_facecolor("#2d2d44")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#555")
        ax.spines["left"].set_color("#555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # AbsRel (lower is better)
    bars1 = ax1.bar(x, absrel, color=colors[:len(labels)], edgecolor="#1a1a2e", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, color="white", fontsize=7)
    ax1.set_ylabel("AbsRel  (↓ lower is better)", color="white", fontsize=9)
    ax1.set_title("Absolute Relative Error", color="white", fontsize=10)
    ax1.yaxis.label.set_color("white")
    for bar, val in zip(bars1, absrel):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", color="white", fontsize=7)

    # δ₁ (higher is better)
    bars2 = ax2.bar(x, delta1, color=colors[:len(labels)], edgecolor="#1a1a2e", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color="white", fontsize=7)
    ax2.set_ylabel("δ₁ accuracy  (% pixels, ↑ higher is better)", color="white", fontsize=9)
    ax2.set_title("δ₁ = % pixels with max(p/g, g/p) < 1.25", color="white", fontsize=10)
    ax2.set_ylim(0, 105)
    for bar, val in zip(bars2, delta1):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", color="white", fontsize=7)

    fig.suptitle("Egocentric Depth Estimation Benchmark", color="white", fontsize=13, y=1.02)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [summary] Bar chart saved → {output_path}")


def print_summary_table(rows: list) -> None:
    """Print a formatted comparison table to stdout."""
    if not rows:
        return
    print("\n" + "=" * 100)
    print(f"  {'MODEL':<25} {'VARIANT':<25} {'ALIGN':<22} "
          f"{'AbsRel':>8} {'RMSE':>8} {'δ₁%':>8} {'δ₂%':>8}")
    print("=" * 100)
    primary_alignments = {"none (metric)", "scale+shift (least-squares)"}
    for r in rows:
        if r.get("alignment", "") not in primary_alignments:
            continue
        print(f"  {r['model']:<25} {r['variant']:<25} {r['alignment']:<22} "
              f"{float(r['AbsRel']):>8.4f} {float(r['RMSE']):>8.4f} "
              f"{float(r['delta1'])*100:>7.1f}% {float(r['delta2'])*100:>7.1f}%")
    print("=" * 100 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all depth estimation baselines and aggregate results."
    )
    # Required
    parser.add_argument("--rgb",        required=True, help="Path to input RGB image")
    parser.add_argument("--depth_gt",   required=True, help="Path to GT depth .npy")
    parser.add_argument("--output_dir", required=True, help="Directory for all outputs")

    # Shared options
    parser.add_argument("--rotation",    type=int,   default=0, choices=[0, 90, 180, 270],
                        help="Clockwise rotation applied to RGB (and GT) before inference")
    parser.add_argument("--depth_scale", type=float, default=1.0,
                        help="GT depth multiplier (e.g. 0.001 for mm→m, default: 1.0)")
    parser.add_argument("--max_depth",   type=float, default=10.0,
                        help="GT depth cap in metres (default: 10.0)")
    parser.add_argument("--intrinsics",  type=float, nargs=4,
                        metavar=("fx", "fy", "cx", "cy"), default=None,
                        help="Camera intrinsics for metric models (optional)")
    parser.add_argument("--device",      default=None, help="'cuda' or 'cpu' (default: auto)")

    # Per-model options
    parser.add_argument("--dav2_variant", default="large",
                        choices=["small", "base", "large"],
                        help="Depth Anything V2 variant (default: large)")

    # Skip flags
    parser.add_argument("--skip_dav2",     action="store_true", help="Skip Depth Anything V2")
    parser.add_argument("--skip_marigold", action="store_true", help="Skip Marigold")
    parser.add_argument("--skip_unidepth", action="store_true", help="Skip UniDepth")
    parser.add_argument("--skip_metric3d", action="store_true", help="Skip Metric3D v2")

    args = parser.parse_args()

    import torch
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results.csv")

    # Remove old CSV so we start fresh
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # ── Shared args builder ──────────────────────────────────────────────────
    def shared_args():
        a = [
            "--rgb",        args.rgb,
            "--depth_gt",   args.depth_gt,
            "--output_dir", args.output_dir,
            "--rotation",   str(args.rotation),
            "--depth_scale",str(args.depth_scale),
            "--max_depth",  str(args.max_depth),
            "--device",     args.device,
            "--csv",        csv_path,
        ]
        if args.intrinsics:
            a += ["--intrinsics"] + [str(v) for v in args.intrinsics]
        return a

    # ── Depth Anything V2 ────────────────────────────────────────────────────
    if not args.skip_dav2:
        print("\n" + "▶" * 60)
        print(f"  Running: Depth Anything V2 [{args.dav2_variant}]")
        print("▶" * 60)
        run_baseline("eval_depth_anything_v2",
                     shared_args() + ["--variant", args.dav2_variant])

    # ── Marigold ─────────────────────────────────────────────────────────────
    if not args.skip_marigold:
        for mv in ["standard", "lcm"]:
            print("\n" + "▶" * 60)
            print(f"  Running: Marigold [{mv}]")
            print("▶" * 60)
            run_baseline("eval_marigold",
                         shared_args() + ["--variant", mv])

    # ── UniDepth ─────────────────────────────────────────────────────────────
    if not args.skip_unidepth:
        print("\n" + "▶" * 60)
        print(f"  Running: UniDepth v2 [vitl14]")
        print("▶" * 60)
        run_baseline("eval_unidepth",
                     shared_args() + ["--variant", "vitl14"])

    # ── Metric3D v2 ──────────────────────────────────────────────────────────
    if not args.skip_metric3d:
        print("\n" + "▶" * 60)
        print(f"  Running: Metric3D v2 [vit_large]")
        print("▶" * 60)
        run_baseline("eval_metric3dv2",
                     shared_args() + ["--variant", "vit_large"])

    # ── Aggregate results ─────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  BENCHMARK COMPLETE — Aggregating results …")
    print("═" * 60)

    rows = read_results_csv(csv_path)
    if rows:
        print_summary_table(rows)
        build_summary_table_figure(rows,
            output_path=os.path.join(args.output_dir, "summary_table.png"))
        build_bar_chart(rows,
            output_path=os.path.join(args.output_dir, "summary_bar.png"))
    else:
        print("  No results found in CSV. Check for errors above.")

    print(f"\n  All outputs saved to: {args.output_dir}")
    print(f"  CSV results       : {csv_path}")


if __name__ == "__main__":
    main()
