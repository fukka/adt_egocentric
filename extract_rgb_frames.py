"""
extract_rgb_frames.py
=====================
Extract all RGB frames from the 214-1 (camera-rgb) stream of main_recording.vrs
for every ADT sequence found under --data_root.

Frames are saved as:
    <data_root>/<sequence_name>/videos_rgb/frame_<XXXXXX>_<timestamp_ns>.jpg

Usage
-----
  # Extract all sequences (JPEG, every frame)
  python extract_rgb_frames.py

  # Dry-run — show what would be done, touch nothing
  python extract_rgb_frames.py --dry_run

  # Only sequences whose name contains 'skeleton'
  python extract_rgb_frames.py --filter skeleton

  # Save as lossless PNG instead of JPEG
  python extract_rgb_frames.py --format png

  # Keep only every 5th frame (stride)
  python extract_rgb_frames.py --stride 5

  # Override default data root
  python extract_rgb_frames.py --data_root ~/Documents/projectaria_tools_adt_data

  # Override VRS filename (default: main_recording.vrs)
  python extract_rgb_frames.py --vrs_name video.vrs

Defaults
--------
  --data_root  ~/Documents/projectaria_tools_adt_data
  --vrs_name   main_recording.vrs
  --stream     214-1
  --format     jpg
  --stride     1  (every frame)
  --out_dir    videos_rgb  (subfolder inside each sequence directory)
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DATA_ROOT = os.path.expanduser("~/Documents/projectaria_tools_adt_data")
DEFAULT_VRS_NAME  = "main_recording.vrs"
DEFAULT_STREAM    = "214-1"
DEFAULT_OUT_DIR   = "videos_rgb"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def find_sequences(data_root: str, vrs_name: str) -> list[str]:
    """
    Return sorted list of sequence names (direct subdirectories of data_root)
    that contain the target VRS file.
    """
    sequences = []
    try:
        entries = sorted(os.listdir(data_root))
    except FileNotFoundError:
        sys.exit(f"ERROR: data_root not found: {data_root}")

    for entry in entries:
        seq_dir = os.path.join(data_root, entry)
        if not os.path.isdir(seq_dir):
            continue
        vrs_path = os.path.join(seq_dir, vrs_name)
        if os.path.isfile(vrs_path):
            sequences.append(entry)
    return sequences


def is_already_extracted(out_dir: str, expected_count: int) -> bool:
    """
    Return True if out_dir already contains exactly expected_count image files.
    (Loose check — a different count means extraction was interrupted or uses a
    different stride; we re-extract in that case.)
    """
    if not os.path.isdir(out_dir):
        return False
    existing = [f for f in os.listdir(out_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return len(existing) == expected_count


def extract_sequence(seq_dir: str, vrs_name: str, stream_id_str: str,
                     out_subdir: str, stride: int, fmt: str,
                     dry_run: bool) -> tuple[int, int]:
    """
    Extract frames from one sequence.

    Returns (n_saved, n_total) tuple.
    """
    # Lazy import so the script can be imported without projectaria installed
    try:
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.stream_id import StreamId
    except ImportError:
        sys.exit(
            "ERROR: projectaria_tools is not installed.\n"
            "  pip install projectaria-tools"
        )

    vrs_path  = os.path.join(seq_dir, vrs_name)
    out_dir   = os.path.join(seq_dir, out_subdir)
    ext       = "jpg" if fmt == "jpg" else "png"
    jpeg_qual = 95   # only used for JPEG

    # Open VRS
    dp   = data_provider.create_vrs_data_provider(vrs_path)
    sid  = StreamId(stream_id_str)
    n_total = dp.get_num_data(sid)

    # Determine which frame indices we'll save
    indices = list(range(0, n_total, stride))
    n_save  = len(indices)

    if dry_run:
        print(f"    VRS frames : {n_total}  →  would save {n_save} "
              f"(stride={stride}) to {out_dir}")
        return n_save, n_total

    # Skip if already complete
    if is_already_extracted(out_dir, n_save):
        print(f"    Already extracted ({n_save} files) — skipping.")
        return 0, n_total

    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    for done_idx, frame_idx in enumerate(indices):
        img_data, meta = dp.get_image_data_by_index(sid, frame_idx)
        img_np = img_data.to_numpy_array()          # (H, W, 3) uint8, RGB

        ts_ns  = meta.capture_timestamp_ns
        fname  = f"frame_{frame_idx:06d}_{ts_ns}.{ext}"
        fpath  = os.path.join(out_dir, fname)

        if fmt == "jpg":
            # cv2 expects BGR
            cv2.imwrite(fpath, img_np[..., ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_qual])
        else:
            # PNG: cv2 expects BGR
            cv2.imwrite(fpath, img_np[..., ::-1])

        # Progress every 50 saves
        if (done_idx + 1) % 50 == 0 or (done_idx + 1) == n_save:
            elapsed = time.time() - t0
            avg     = elapsed / (done_idx + 1)
            remain  = avg * (n_save - done_idx - 1)
            print(f"    {done_idx+1}/{n_save} frames saved  "
                  f"({elapsed:.1f}s elapsed, ~{remain:.1f}s remaining)")

    return n_save, n_total


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract 214-1 RGB frames from main_recording.vrs for all ADT sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_root", default=DEFAULT_DATA_ROOT,
        help=f"Root folder containing one sub-directory per sequence  "
             f"(default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--vrs_name", default=DEFAULT_VRS_NAME,
        help=f"VRS filename within each sequence directory  "
             f"(default: {DEFAULT_VRS_NAME})",
    )
    parser.add_argument(
        "--stream", default=DEFAULT_STREAM,
        help=f"VRS stream ID to extract  (default: {DEFAULT_STREAM})",
    )
    parser.add_argument(
        "--out_dir", default=DEFAULT_OUT_DIR,
        help=f"Output sub-folder name inside each sequence directory  "
             f"(default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--format", default="jpg", choices=["jpg", "png"],
        help="Output image format  (default: jpg)",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Save every Nth frame  (default: 1 = every frame)",
    )
    parser.add_argument(
        "--filter", default=None, metavar="SUBSTRING",
        help="Only process sequences whose name contains this substring "
             "(case-sensitive).  E.g. --filter skeleton",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be done without writing any files.",
    )
    args = parser.parse_args()

    data_root = os.path.expanduser(args.data_root)

    # ── Discover sequences ─────────────────────────────────────────────────
    all_sequences = find_sequences(data_root, args.vrs_name)
    print(f"Sequences with {args.vrs_name} : {len(all_sequences)}")

    if args.filter:
        sequences = [s for s in all_sequences if args.filter in s]
        print(f'After --filter "{args.filter}"      : {len(sequences)} sequences')
    else:
        sequences = all_sequences

    if not sequences:
        sys.exit("No sequences matched. Exiting.")

    if args.dry_run:
        print("(DRY RUN — no files will be written)\n")

    # ── Process each sequence ──────────────────────────────────────────────
    n_seq      = len(sequences)
    total_saved = 0
    failed      = []
    t_global    = time.time()

    for i, seq_name in enumerate(sequences, 1):
        seq_dir = os.path.join(data_root, seq_name)
        print(f"\n[{i}/{n_seq}]  {seq_name}")

        try:
            n_saved, n_total = extract_sequence(
                seq_dir     = seq_dir,
                vrs_name    = args.vrs_name,
                stream_id_str = args.stream,
                out_subdir  = args.out_dir,
                stride      = args.stride,
                fmt         = args.format,
                dry_run     = args.dry_run,
            )
            total_saved += n_saved
            if not args.dry_run and n_saved > 0:
                print(f"  ✓ Saved {n_saved} / {n_total} frames")
        except Exception as exc:
            print(f"  !! FAILED: {exc}")
            failed.append(seq_name)

        # Overall progress estimate
        elapsed_g = time.time() - t_global
        avg_g     = elapsed_g / i
        remain_g  = avg_g * (n_seq - i)
        print(f"  Progress {i}/{n_seq} — "
              f"elapsed {elapsed_g/60:.1f} min  "
              f"~{remain_g/60:.1f} min remaining")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"DRY RUN complete.  Would process {n_seq} sequence(s).")
    else:
        ok = n_seq - len(failed)
        print(f"Done.  {ok}/{n_seq} sequences processed  |  "
              f"{total_saved} frames saved total.")
        if failed:
            print(f"\nFailed sequences ({len(failed)}):")
            for s in failed:
                print(f"  {s}")
    print("=" * 60)


if __name__ == "__main__":
    main()
