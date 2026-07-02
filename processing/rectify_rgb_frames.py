"""
rectify_rgb_frames.py
=====================
Read already-extracted RGB fisheye images (from extract_rgb_frames.py) for a
single ADT sequence and undistort them to a pinhole (LINEAR) model using the
calibration embedded in the sequence's VRS file.

Input frames:   <seq_dir>/<in_dir>/frame_<XXXXXX>_<timestamp_ns>.<ext>
Rectified output: <seq_dir>/<out_dir>/frame_<XXXXXX>_<timestamp_ns>.png

Usage
-----
  # Rectify with defaults
  python rectify_rgb_frames.py

  # Point at a specific sequence folder
  python rectify_rgb_frames.py --seq_dir ~/Documents/projectaria_tools_adt_data/MySequence

  # Override resolution and focal length
  python rectify_rgb_frames.py --output_size 512 --focal 300

  # Dry-run — show what would be done
  python rectify_rgb_frames.py --dry_run

Defaults
--------
  --seq_dir      ~/Documents/projectaria_tools_adt_data/Apartment_release_golden_skeleton_seq100_10s_sample_M1292
  --vrs_name     main_recording.vrs   (used only for calibration)
  --in_dir       videos_rgb
  --out_dir      videos_rgb_rectified
  --output_size  512
  --focal        300.0
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image


DEFAULT_SEQ_DIR = os.path.expanduser(
    "~/Documents/projectaria_tools_adt_data/"
    "Apartment_release_clean_seq131_M1292"
)
DEFAULT_VRS_NAME = "video.vrs"
DEFAULT_IN_DIR   = "videos_rgb_selected"
DEFAULT_OUT_DIR  = "videos_rgb_selected_rectified"


# ──────────────────────────────────────────────────────────────────────────────
# Calibration helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_ego_calibration(vrs_path: str):
    try:
        from projectaria_tools.core import data_provider
    except ImportError:
        sys.exit("ERROR: projectaria_tools is not installed.\n  pip install projectaria-tools")
    dp = data_provider.create_vrs_data_provider(vrs_path)
    return dp.get_device_calibration().get_camera_calib('camera-rgb')


def rescale_calibration(cam_calib, actual_w: int, actual_h: int):
    calib_w, calib_h = cam_calib.get_image_size()
    if calib_w == actual_w and calib_h == actual_h:
        return cam_calib
    scale = actual_w / calib_w
    print(f"  Rescaling calibration {calib_w}x{calib_h} -> {actual_w}x{actual_h} (factor={scale:.4f})")
    return cam_calib.rescale(np.array([actual_w, actual_h], dtype=np.int32), scale)


def build_linear_calibration(src_calib, out_size: int, focal_px: float):
    try:
        from projectaria_tools.core import calibration
    except ImportError:
        sys.exit("ERROR: projectaria_tools is not installed.")
    return calibration.get_linear_camera_calibration(
        out_size, out_size, focal_px,
        'camera-rgb-linear',
        src_calib.get_transform_device_camera(),
    )


def rectify_frame(frame_np: np.ndarray, src_calib, dst_calib) -> np.ndarray:
    from projectaria_tools.core import calibration
    return calibration.distort_by_calibration(frame_np, dst_calib, src_calib).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rectify extracted RGB fisheye images for one ADT sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--seq_dir", default=DEFAULT_SEQ_DIR,
                        help="Path to the sequence folder  (default: DEFAULT_SEQ_DIR)")
    parser.add_argument("--vrs_name", default=DEFAULT_VRS_NAME,
                        help=f"VRS filename for calibration  (default: {DEFAULT_VRS_NAME})")
    parser.add_argument("--in_dir", default=DEFAULT_IN_DIR,
                        help=f"Sub-folder of extracted fisheye images  (default: {DEFAULT_IN_DIR})")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                        help=f"Sub-folder for rectified output  (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--output_size", type=int, default=1408,
                        help="Square output resolution in pixels  (default: 512)")
    parser.add_argument("--focal", type=float, default=300.0,
                        help="Target pinhole focal length in pixels  (default: 300.0)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be done without writing any files.")
    args = parser.parse_args()

    seq_dir  = os.path.expanduser(args.seq_dir)
    in_dir   = os.path.join(seq_dir, args.in_dir)
    out_dir  = os.path.join(seq_dir, args.out_dir)
    vrs_path = os.path.join(seq_dir, args.vrs_name)

    # ── Validate inputs ────────────────────────────────────────────────────
    if not os.path.isdir(seq_dir):
        sys.exit(f"ERROR: seq_dir not found: {seq_dir}")
    if not os.path.isdir(in_dir):
        sys.exit(f"ERROR: in_dir not found: {in_dir}")
    if not os.path.isfile(vrs_path):
        sys.exit(f"ERROR: VRS file not found: {vrs_path}")

    frames = sorted(
        f for f in os.listdir(in_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    n_total = len(frames)
    if n_total == 0:
        sys.exit(f"ERROR: No images found in {in_dir}")

    print(f"Sequence : {seq_dir}")
    print(f"Input    : {in_dir}  ({n_total} frames)")
    print(f"Output   : {out_dir}")
    print(f"Settings : output_size={args.output_size}, focal={args.focal}")

    if dry_run := args.dry_run:
        print("\n(DRY RUN — no files will be written)")
        return

    # ── Check if already done ──────────────────────────────────────────────
    if os.path.isdir(out_dir):
        existing = [f for f in os.listdir(out_dir) if f.lower().endswith(".png")]
        if len(existing) == n_total:
            print(f"\nAlready rectified ({n_total} files) — skipping.")
            return

    # ── Load calibration ───────────────────────────────────────────────────
    print(f"\nLoading calibration from {vrs_path} ...")
    src_calib = load_ego_calibration(vrs_path)
    print(f"  model     : {src_calib.get_model_name()}")
    print(f"  image_size: {src_calib.get_image_size()}")

    first_img = np.array(Image.open(os.path.join(in_dir, frames[0])).convert("RGB"))
    actual_h, actual_w = first_img.shape[:2]
    src_calib = rescale_calibration(src_calib, actual_w, actual_h)
    dst_calib = build_linear_calibration(src_calib, args.output_size, args.focal)

    os.makedirs(out_dir, exist_ok=True)

    # ── Rectify frames ─────────────────────────────────────────────────────
    print(f"\nRectifying {n_total} frames ...")
    t0 = time.time()
    for done_idx, fname in enumerate(frames):
        img  = np.array(Image.open(os.path.join(in_dir, fname)).convert("RGB"))
        rect = rectify_frame(img, src_calib, dst_calib)

        stem = os.path.splitext(fname)[0]
        Image.fromarray(rect).save(os.path.join(out_dir, f"{stem}.png"))

        if (done_idx + 1) % 50 == 0 or (done_idx + 1) == n_total:
            elapsed = time.time() - t0
            avg     = elapsed / (done_idx + 1)
            remain  = avg * (n_total - done_idx - 1)
            print(f"  {done_idx+1}/{n_total}  ({elapsed:.1f}s elapsed, ~{remain:.1f}s remaining)")

    print(f"\nDone. {n_total} rectified frames saved to: {out_dir}")


if __name__ == "__main__":
    main()