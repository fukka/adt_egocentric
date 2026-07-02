"""
extract_rgb_frames.py
=====================
Extract RGB frames from the 214-1 (camera-rgb) stream of main_recording.vrs
for every ADT sequence found under --data_root.

Only frames within the trajectory-valid window are extracted (same range used
by render_from_poses_blender.py).  The RGB VRS starts several seconds before
tracking begins; those warmup frames are skipped so that videos_rgb/ aligns
frame-for-frame with seg_npy/ and depth_npy/ produced by
extract_seg_depth_frames.py.

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
  --data_root  ~/Documents/projectaria_tools_adt_data_clean
  --vrs_name   main_recording.vrs
  --stream     214-1
  --format     png
  --stride     1  (every frame)
  --out_dir    videos_rgb  (subfolder inside each sequence directory)
"""

import argparse
import bisect
import csv
import os
import sys
import time

import cv2
import numpy as np


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DATA_ROOT  = os.path.expanduser("~/Documents/projectaria_tools_adt_data_clean")
DEFAULT_VRS_NAME   = "main_recording.vrs"
DEFAULT_ALIGN_VRS  = "main_recording.vrs"
DEFAULT_STREAM     = "214-1"
DEFAULT_OUT_DIR    = "videos_rgb"

_MAX_ALIGN_NS = 50_000_000  # 50 ms — accepts ~97 µs valid pairs, rejects warmup frames


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_trajectory_range_us(seq_dir: str):
    """
    Return (traj_t0_us, traj_t1_us) from groundtruth/aria_trajectory.csv,
    or None if the file is absent.
    """
    csv_path = os.path.join(seq_dir, "groundtruth", "aria_trajectory.csv")
    if not os.path.isfile(csv_path):
        return None
    t0 = t1 = None
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            ts = int(row['tracking_timestamp_us'])
            if t0 is None:
                t0 = ts
            t1 = ts
    return (t0, t1) if t0 is not None else None


def find_valid_frame_range(dp, sid, n_total: int,
                           traj_t0_us: int, traj_t1_us: int) -> tuple:
    """
    Binary-search the VRS stream for the first and last frame whose timestamp
    falls within [traj_t0_us, traj_t1_us] (microseconds).

    Mirrors render_from_poses_blender.py exactly.
    Returns (first_valid_frame, last_valid_frame).
    """
    def ts_us(idx):
        return dp.get_image_data_by_index(sid, idx)[1].capture_timestamp_ns // 1000

    lo, hi = 0, n_total - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if ts_us(mid) < traj_t0_us:
            lo = mid + 1
        else:
            hi = mid
    first = lo

    lo, hi = 0, n_total - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ts_us(mid) > traj_t1_us:
            hi = mid - 1
        else:
            lo = mid
    last = lo

    return first, last


def load_rgb_timestamps(seq_dir: str, align_vrs: str, rgb_dir: str = "videos_rgb"):
    """
    Return RGB reference timestamps for alignment.

    Tries align_vrs first (dense list indexed by frame number).
    Falls back to parsing filenames in rgb_dir (dict {rgb_idx: ts_ns}).
    Returns None if neither source is available.
    """
    vrs_path = os.path.join(seq_dir, align_vrs)
    if os.path.isfile(vrs_path):
        try:
            from projectaria_tools.core import data_provider
            from projectaria_tools.core.stream_id import StreamId
        except ImportError:
            return None
        dp  = data_provider.create_vrs_data_provider(vrs_path)
        sid = StreamId("214-1")
        n   = dp.get_num_data(sid)
        return [dp.get_image_data_by_index(sid, i)[1].capture_timestamp_ns
                for i in range(n)]

    # Fallback: parse frame_XXXXXX_TTTTTTTTTTTTTTTTTT.{ext} filenames
    dir_path = os.path.join(seq_dir, rgb_dir)
    if not os.path.isdir(dir_path):
        return None
    result = {}
    for fname in os.listdir(dir_path):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = fname.rsplit(".", 1)[0].split("_")
        if len(stem) >= 3 and stem[0] == "frame":
            try:
                result[int(stem[1])] = int(stem[2])
            except ValueError:
                continue
    return result if result else None


def _nearest_idx(rgb_ts_ns: int, timestamps: list) -> tuple:
    """Return (index, abs_delta_ns) of the nearest frame in timestamps."""
    pos = bisect.bisect_left(timestamps, rgb_ts_ns)
    if pos == 0:
        idx = 0
    elif pos >= len(timestamps):
        idx = len(timestamps) - 1
    else:
        before = rgb_ts_ns - timestamps[pos - 1]
        after  = timestamps[pos] - rgb_ts_ns
        idx = pos - 1 if before <= after else pos
    return idx, abs(rgb_ts_ns - timestamps[idx])


def find_sequences(data_root: str, vrs_name: str) -> list:
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
                     rgb_timestamps, dry_run: bool) -> tuple:
    """
    Extract trajectory-valid frames from one sequence.

    When rgb_timestamps is provided (a list of capture_timestamp_ns from
    main_recording.vrs 214-1), the extractor iterates RGB frame indices with
    the given stride and looks up the nearest frame in the target VRS by
    timestamp.  Output filenames use the RGB frame index so they match
    videos_rgb/ exactly.  Pairs whose delta exceeds _MAX_ALIGN_NS are skipped.

    When rgb_timestamps is None (extracting main_recording.vrs itself), the
    original index-based path is used.

    Returns (n_saved, n_total) tuple.
    """
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
    jpeg_qual = 95

    dp      = data_provider.create_vrs_data_provider(vrs_path)
    sid     = StreamId(stream_id_str)
    n_total = dp.get_num_data(sid)

    # ── RGB-driven alignment path ──────────────────────────────────────────
    if rgb_timestamps is not None:
        tgt_timestamps = [
            dp.get_image_data_by_index(sid, i)[1].capture_timestamp_ns
            for i in range(n_total)
        ]

        if isinstance(rgb_timestamps, dict):
            # Loaded from videos_rgb/ filenames — already filtered and strided.
            # Iterate the existing frame indices directly; no stride re-applied.
            print(f"    RGB ref source    : videos_rgb/ ({len(rgb_timestamps)} frames)")
            rgb_items = sorted(rgb_timestamps.items())  # [(rgb_idx, ts_ns), ...]
            valid_pairs = []
            for rgb_i, rgb_ts_ns in rgb_items:
                tgt_i, dt = _nearest_idx(rgb_ts_ns, tgt_timestamps)
                if dt <= _MAX_ALIGN_NS:
                    valid_pairs.append((rgb_i, tgt_i, rgb_ts_ns))
        else:
            # Loaded from VRS — dense list; apply stride and trajectory filter.
            print(f"    RGB ref source    : VRS ({len(rgb_timestamps)} frames)")
            traj_range = load_trajectory_range_us(seq_dir)
            if traj_range:
                traj_t0_ns = traj_range[0] * 1000
                traj_t1_ns = traj_range[1] * 1000
                rgb_first = bisect.bisect_left(rgb_timestamps, traj_t0_ns)
                rgb_last  = bisect.bisect_right(rgb_timestamps, traj_t1_ns) - 1
                rgb_first = max(0, min(rgb_first, len(rgb_timestamps) - 1))
                rgb_last  = max(0, min(rgb_last,  len(rgb_timestamps) - 1))
            else:
                rgb_first, rgb_last = 0, len(rgb_timestamps) - 1
            valid_pairs = []
            for rgb_i in range(0, len(rgb_timestamps), stride):
                if not (rgb_first <= rgb_i <= rgb_last):
                    continue
                tgt_i, dt = _nearest_idx(rgb_timestamps[rgb_i], tgt_timestamps)
                if dt <= _MAX_ALIGN_NS:
                    valid_pairs.append((rgb_i, tgt_i, rgb_timestamps[rgb_i]))

        n_save = len(valid_pairs)
        print(f"    Target VRS frames : {n_total}")
        print(f"    RGB-aligned pairs : {n_save}  (max delta {_MAX_ALIGN_NS//1_000_000} ms)")

        if dry_run:
            print(f"    Would save {n_save} frames (stride={stride}) to {out_dir}")
            return n_save, n_total

        if is_already_extracted(out_dir, n_save):
            print(f"    Already extracted ({n_save} files) — skipping.")
            return 0, n_total

        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        for done_idx, (rgb_i, tgt_i, ts_ns) in enumerate(valid_pairs):
            img_data, _ = dp.get_image_data_by_index(sid, tgt_i)
            img_np = img_data.to_numpy_array()
            fname  = f"frame_{rgb_i:06d}_{ts_ns}.{ext}"
            fpath  = os.path.join(out_dir, fname)
            if fmt == "jpg":
                cv2.imwrite(fpath, img_np[..., ::-1],
                            [cv2.IMWRITE_JPEG_QUALITY, jpeg_qual])
            else:
                cv2.imwrite(fpath, img_np[..., ::-1])
            if (done_idx + 1) % 50 == 0 or (done_idx + 1) == n_save:
                elapsed = time.time() - t0
                remain  = elapsed / (done_idx + 1) * (n_save - done_idx - 1)
                print(f"    {done_idx+1}/{n_save} frames saved  "
                      f"({elapsed:.1f}s elapsed, ~{remain:.1f}s remaining)")
        return n_save, n_total

    # ── Native index path (main_recording.vrs itself) ──────────────────────
    traj_range = load_trajectory_range_us(seq_dir)
    if traj_range:
        traj_t0_us, traj_t1_us = traj_range
        first_valid, last_valid = find_valid_frame_range(
            dp, sid, n_total, traj_t0_us, traj_t1_us)
        skipped_start = first_valid
        skipped_end   = n_total - 1 - last_valid
        print(f"    Trajectory range  : {traj_t0_us} … {traj_t1_us} us")
        print(f"    Valid frame range : [{first_valid}, {last_valid}]  "
              f"({last_valid - first_valid + 1} frames"
              + (f", skipping {skipped_start} early" if skipped_start else "")
              + (f", {skipped_end} late"             if skipped_end   else "")
              + ")")
    else:
        first_valid, last_valid = 0, n_total - 1
        print(f"    aria_trajectory.csv not found — extracting all {n_total} frames")

    indices = [i for i in range(0, n_total, stride)
               if first_valid <= i <= last_valid]
    n_save  = len(indices)

    if dry_run:
        print(f"    Would save {n_save} frames (stride={stride}) to {out_dir}")
        return n_save, n_total

    if is_already_extracted(out_dir, n_save):
        print(f"    Already extracted ({n_save} files) — skipping.")
        return 0, n_total

    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    for done_idx, frame_idx in enumerate(indices):
        img_data, meta = dp.get_image_data_by_index(sid, frame_idx)
        img_np = img_data.to_numpy_array()

        ts_ns = meta.capture_timestamp_ns
        fname = f"frame_{frame_idx:06d}_{ts_ns}.{ext}"
        fpath = os.path.join(out_dir, fname)

        if fmt == "jpg":
            cv2.imwrite(fpath, img_np[..., ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_qual])
        else:
            cv2.imwrite(fpath, img_np[..., ::-1])

        if (done_idx + 1) % 50 == 0 or (done_idx + 1) == n_save:
            elapsed = time.time() - t0
            remain  = elapsed / (done_idx + 1) * (n_save - done_idx - 1)
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
        "--format", default="png", choices=["jpg", "png"],
        help="Output image format  (default: png)",
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
        "--align_vrs", default=DEFAULT_ALIGN_VRS, metavar="VRS_FILENAME",
        help="VRS file whose 214-1 RGB timestamps are used as the reference "
             "clock; nearest frame in --vrs_name is found by timestamp and "
             "output filenames use the reference RGB frame index so all "
             "outputs align with videos_rgb/.  Skipped automatically when "
             "--vrs_name equals --align_vrs (i.e. extracting the reference "
             "itself).  "
             f"(default: {DEFAULT_ALIGN_VRS})",
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
    n_seq       = len(sequences)
    total_saved = 0
    failed      = []
    t_global    = time.time()

    for i, seq_name in enumerate(sequences, 1):
        seq_dir = os.path.join(data_root, seq_name)
        print(f"\n[{i}/{n_seq}]  {seq_name}")

        # Load RGB reference timestamps unless extracting the reference itself
        rgb_ts = None
        use_align = args.align_vrs and (args.align_vrs != args.vrs_name)
        if use_align:
            vrs_present = os.path.isfile(os.path.join(seq_dir, args.align_vrs))
            src = args.align_vrs if vrs_present else f"{DEFAULT_OUT_DIR}/ (VRS absent)"
            print(f"    Loading RGB timestamps from {src} …")
            rgb_ts = load_rgb_timestamps(seq_dir, args.align_vrs)
            if rgb_ts is None:
                print(f"    !! Neither {args.align_vrs} nor {DEFAULT_OUT_DIR}/ found — skipping")
                failed.append(seq_name)
                continue
            print(f"    RGB timestamps loaded: {len(rgb_ts)} frames")

        try:
            n_saved, n_total = extract_sequence(
                seq_dir       = seq_dir,
                vrs_name      = args.vrs_name,
                stream_id_str = args.stream,
                out_subdir    = args.out_dir,
                stride        = args.stride,
                fmt           = args.format,
                rgb_timestamps = rgb_ts,
                dry_run       = args.dry_run,
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