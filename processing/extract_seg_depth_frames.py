"""
extract_seg_depth_frames.py
===========================
Extract ground-truth segmentation and depth frames from ADT VRS files for
every sequence found under --data_root.

For each of the three camera streams (400-1 RGB, 400-2 SLAM-left, 400-3 SLAM-right
for segmentation; 345-1/2/3 for depth) the script writes:

  Segmentation
  ------------
  <seq>/seg_npy/<stream>/frame_<XXXXXX>_<ts_ns>.npy   uint64  (H×W)
      Each pixel = (object_id << 32) | instance_id.
      0 = background / no object.
  <seq>/seg_jpg/<stream>/frame_<XXXXXX>_<ts_ns>.jpg
      Pseudo-colour: each instance_id gets a stable MD5-derived RGB colour.

  Depth
  -----
  <seq>/depth_npy/<stream>/frame_<XXXXXX>_<ts_ns>.npy   uint16  (H×W)
      Depth in millimetres.  0 = invalid / no measurement.
  <seq>/depth_jpg/<stream>/frame_<XXXXXX>_<ts_ns>.jpg
      False-colour depth map (TURBO colormap); invalid pixels shown in black.

Camera-stream mapping
---------------------
  *-1   RGB camera      1408 × 1408
  *-2   SLAM left        480 ×  640
  *-3   SLAM right       480 ×  640

Usage
-----
  # Dry-run
  python extract_seg_depth_frames.py --dry_run

  # Extract everything (seg + depth, all cameras)
  python extract_seg_depth_frames.py

  # Segmentation only
  python extract_seg_depth_frames.py --mode seg

  # Depth only, every 5th frame
  python extract_seg_depth_frames.py --mode depth --stride 5

  # Only RGB-camera stream (400-1 / 345-1)
  python extract_seg_depth_frames.py --streams 1

  # Filter to specific sequences
  python extract_seg_depth_frames.py --filter skeleton

  # Custom data root
  python extract_seg_depth_frames.py --data_root ~/my_data/adt

Defaults
--------
  --data_root   ~/Documents/projectaria_tools_adt_data
  --mode        both   (seg and depth)
  --streams     1      (RGB camera only)
  --stride      1      (every frame)
  --save_jpg    off    (visualisation JPEGs not written by default)
"""

import argparse
import bisect
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


# SPACE
# python3 processing/extract_seg_depth_frames.py  --data_root /group-volume/Fengjia/data/projectaria_tools_adt_data_clean
# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DATA_ROOT = os.path.expanduser("~/Documents/projectaria_tools_adt_data_clean")
SEG_VRS_NAME      = "segmentation/segmentations.vrs"
DEPTH_VRS_NAME    = "depth/depth_images.vrs"
SEG_STREAM_PREFIX   = "400"
DEPTH_STREAM_PREFIX = "345"


# ── Colour helpers ─────────────────────────────────────────────────────────────

def colorize_segmentation(arr: np.ndarray) -> np.ndarray:
    """
    Convert a uint64 segmentation map to a uint8 RGB visualisation.

    Each unique instance_id (lower 32 bits) gets a stable MD5-derived colour.
    Background (value 0) is black.

    Vectorised implementation: uses np.searchsorted for a single-pass lookup
    instead of a per-instance boolean scan.  ~12× faster than the naive loop
    on 1408×1408 frames with ~100 instances.
    """
    instance_ids = (arr & 0xFFFFFFFF).astype(np.int64)   # H×W int64
    unique_ids   = np.unique(instance_ids)                 # sorted ascending

    # Build a compact colour table (one RGB row per unique id)
    color_table = np.zeros((len(unique_ids), 3), dtype=np.uint8)
    for i, uid in enumerate(unique_ids):
        if uid != 0:
            d = hashlib.md5(str(uid).encode()).digest()
            color_table[i] = (d[0], d[1], d[2])

    # Vectorised lookup: searchsorted maps each pixel to its position in
    # unique_ids → index into color_table  (O(n_pixels × log n_unique))
    idx = np.searchsorted(unique_ids, instance_ids.ravel()).reshape(instance_ids.shape)
    return color_table[idx]   # H×W×3 uint8


def colorize_depth(arr: np.ndarray) -> np.ndarray:
    """
    Convert a uint16 depth map (mm) to a uint8 RGB visualisation.

    Uses the TURBO colormap.  Invalid pixels (value 0) are rendered black.
    Normalises to the [1st, 99th] percentile of valid pixels for contrast.
    """
    valid_mask = arr > 0
    if valid_mask.any():
        lo = np.percentile(arr[valid_mask], 1)
        hi = np.percentile(arr[valid_mask], 99)
        hi = max(hi, lo + 1)
        norm = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    else:
        norm = np.zeros(arr.shape, dtype=np.float32)

    norm_u8 = (norm * 255).astype(np.uint8)
    rgb = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)   # BGR
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb[~valid_mask] = 0    # black out invalid pixels
    return rgb


# ── Sequence discovery ────────────────────────────────────────────────────────

def find_sequences(data_root: str, mode: str, stream_indices: list) -> list:
    """
    Return sorted sequence names whose directory contains the required VRS file(s).
    """
    try:
        entries = sorted(os.listdir(data_root))
    except FileNotFoundError:
        sys.exit(f"ERROR: data_root not found: {data_root}")

    need_seg   = mode in ("seg",   "both")
    need_depth = mode in ("depth", "both")

    sequences = []
    for entry in entries:
        seq_dir = os.path.join(data_root, entry)
        if not os.path.isdir(seq_dir):
            continue
        has_seg   = os.path.isfile(os.path.join(seq_dir, SEG_VRS_NAME))
        has_depth = os.path.isfile(os.path.join(seq_dir, DEPTH_VRS_NAME))
        if need_seg and not has_seg:
            continue
        if need_depth and not has_depth:
            continue
        sequences.append(entry)
    return sequences


def is_already_extracted(out_dir: str, n_expected: int) -> bool:
    if not os.path.isdir(out_dir):
        return False
    existing = [f for f in os.listdir(out_dir)
                if f.endswith(".npy") or f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return len(existing) == n_expected


# ── RGB timestamp loader ──────────────────────────────────────────────────────

def load_rgb_timestamps(seq_dir: str) -> list:
    """
    Return the list of capture timestamps from the RGB stream (214-1) of
    main_recording.vrs.  Returns None if the VRS is absent.
    """
    try:
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.stream_id import StreamId
    except ImportError:
        return None

    vrs_path = os.path.join(seq_dir, "video.vrs")
    if not os.path.isfile(vrs_path):
        return None

    dp  = data_provider.create_vrs_data_provider(vrs_path)
    sid = StreamId("214-1")
    n   = dp.get_num_data(sid)
    return [dp.get_image_data_by_index(sid, i)[1].capture_timestamp_ns
            for i in range(n)]


# ── Per-VRS extraction ────────────────────────────────────────────────────────

# Max allowed time gap between an RGB frame and its nearest seg/depth frame.
# Warmup RGB frames (before trajectory) are 10+ s away → safely rejected.
# Valid paired frames are ~97 µs apart → safely accepted.
_MAX_ALIGN_NS = 1_000_000_000   # 1 second


def _nearest_seg_idx(rgb_ts_ns: int, seg_timestamps: list) -> tuple:
    """Return (seg_frame_idx, delta_ns) for the seg frame closest to rgb_ts_ns."""
    pos = bisect.bisect_left(seg_timestamps, rgb_ts_ns)
    if pos == 0:
        idx = 0
    elif pos >= len(seg_timestamps):
        idx = len(seg_timestamps) - 1
    else:
        before = rgb_ts_ns - seg_timestamps[pos - 1]
        after  = seg_timestamps[pos] - rgb_ts_ns
        idx = pos - 1 if before <= after else pos
    return idx, abs(rgb_ts_ns - seg_timestamps[idx])


def extract_vrs(seq_dir: str, vrs_name: str, stream_prefix: str,
                stream_indices: list, npy_subdir: str, jpg_subdir: str,
                colorize_fn, stride: int, save_jpg: bool,
                rgb_timestamps: list, dry_run: bool) -> tuple:
    """
    Extract frames from one VRS file into npy and jpg sub-directories.

    Filename alignment with videos_rgb/
    ------------------------------------
    The seg/depth VRS starts at the trajectory boundary; main_recording.vrs
    starts ~10 s earlier with warmup frames, so seg frame index N ≠ RGB N.

    When rgb_timestamps is provided we ITERATE BY RGB FRAME INDEX (applying
    the same stride from 0 as extract_rgb_frames.py does), then look up the
    nearest seg/depth frame by timestamp.  This guarantees that for any
    stride, every file produced here has an exact filename match in
    videos_rgb/:

        seg_npy/frame_000400_T.npy  ↔  videos_rgb/frame_000400_T.jpg  ✓

    RGB frames that predate or postdate the trajectory (no close seg/depth
    frame within 1 s) are silently skipped.

    Returns (total_npy_saved, total_jpg_saved).
    """
    try:
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.stream_id import StreamId
    except ImportError:
        sys.exit("ERROR: projectaria_tools not installed.  pip install projectaria-tools")

    vrs_path = os.path.join(seq_dir, vrs_name)
    dp = data_provider.create_vrs_data_provider(vrs_path)
    if dp is None:
        raise RuntimeError(f"Cannot open {vrs_path}")

    total_npy = 0
    total_jpg = 0

    for stream_idx in stream_indices:
        sid_str = f"{stream_prefix}-{stream_idx}"
        sid     = StreamId(sid_str)
        n_total = dp.get_num_data(sid)
        if n_total == 0:
            print(f"    [{sid_str}] no frames — skipping")
            continue

        npy_dir = os.path.join(seq_dir, npy_subdir)
        jpg_dir = os.path.join(seq_dir, jpg_subdir)

        if rgb_timestamps:
            # ── RGB-driven iteration (correct for any stride) ─────────────────
            # Load seg timestamps once for nearest-neighbour lookup
            seg_timestamps = [
                dp.get_image_data_by_index(sid, i)[1].capture_timestamp_ns
                for i in range(n_total)
            ]
            # Iterate over the same RGB frame indices that extract_rgb_frames.py
            # would produce, skipping those with no close seg frame.
            rgb_indices = range(0, len(rgb_timestamps), stride)
            valid_pairs = []   # (rgb_frame_i, seg_frame_i)
            for rgb_i in rgb_indices:
                seg_i, dt = _nearest_seg_idx(rgb_timestamps[rgb_i], seg_timestamps)
                if dt <= _MAX_ALIGN_NS:
                    valid_pairs.append((rgb_i, seg_i))
            n_save = len(valid_pairs)

            if dry_run:
                jpg_note = f"  jpg→{jpg_dir}" if save_jpg else "  (jpg disabled)"
                print(f"    [{sid_str}] {n_total} seg frames  "
                      f"RGB stride={stride} → {n_save} aligned pairs  "
                      f"npy→{npy_dir}{jpg_note}")
                total_npy += n_save
                if save_jpg:
                    total_jpg += n_save
                continue

            # Skip if already complete
            npy_done = is_already_extracted(npy_dir, n_save)
            jpg_done = is_already_extracted(jpg_dir, n_save) if save_jpg else True
            if npy_done and jpg_done:
                print(f"    [{sid_str}] already extracted ({n_save} frames) — skipping")
                continue

            os.makedirs(npy_dir, exist_ok=True)
            if save_jpg:
                os.makedirs(jpg_dir, exist_ok=True)

            t0 = time.time()
            with ThreadPoolExecutor(max_workers=2) as io_pool:
                for done_i, (rgb_i, seg_i) in enumerate(valid_pairs):
                    img_data = dp.get_image_data_by_index(sid, seg_i)
                    arr = img_data[0].to_numpy_array()
                    ts  = rgb_timestamps[rgb_i]        # use RGB timestamp
                    stem = f"frame_{rgb_i:06d}_{ts}"

                    if not npy_done:
                        io_pool.submit(np.save,
                                       os.path.join(npy_dir, f"{stem}.npy"), arr)
                        total_npy += 1
                    if save_jpg and not jpg_done:
                        rgb_img = colorize_fn(arr)
                        bgr     = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                        io_pool.submit(cv2.imwrite,
                                       os.path.join(jpg_dir, f"{stem}.jpg"),
                                       bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        total_jpg += 1

                    if (done_i + 1) % 100 == 0 or (done_i + 1) == n_save:
                        elapsed = time.time() - t0
                        remain  = elapsed / (done_i + 1) * (n_save - done_i - 1)
                        print(f"    [{sid_str}] {done_i+1}/{n_save} frames  "
                              f"({elapsed:.1f}s elapsed, ~{remain:.1f}s remaining)")

        else:
            # ── Fallback: no RGB timestamps → iterate seg frames directly ─────
            indices = list(range(0, n_total, stride))
            n_save  = len(indices)

            if dry_run:
                jpg_note = f"  jpg→{jpg_dir}" if save_jpg else "  (jpg disabled)"
                print(f"    [{sid_str}] {n_total} frames → save {n_save}  "
                      f"npy→{npy_dir}{jpg_note}  (native timestamps)")
                total_npy += n_save
                if save_jpg:
                    total_jpg += n_save
                continue

            npy_done = is_already_extracted(npy_dir, n_save)
            jpg_done = is_already_extracted(jpg_dir, n_save) if save_jpg else True
            if npy_done and jpg_done:
                print(f"    [{sid_str}] already extracted ({n_save} frames) — skipping")
                continue

            os.makedirs(npy_dir, exist_ok=True)
            if save_jpg:
                os.makedirs(jpg_dir, exist_ok=True)

            t0 = time.time()
            with ThreadPoolExecutor(max_workers=2) as io_pool:
                for done_i, frame_i in enumerate(indices):
                    img_data, meta = dp.get_image_data_by_index(sid, frame_i)
                    arr  = img_data.to_numpy_array()
                    stem = f"frame_{frame_i:06d}_{meta.capture_timestamp_ns}"

                    if not npy_done:
                        io_pool.submit(np.save,
                                       os.path.join(npy_dir, f"{stem}.npy"), arr)
                        total_npy += 1
                    if save_jpg and not jpg_done:
                        rgb_img = colorize_fn(arr)
                        bgr     = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                        io_pool.submit(cv2.imwrite,
                                       os.path.join(jpg_dir, f"{stem}.jpg"),
                                       bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        total_jpg += 1

                    if (done_i + 1) % 100 == 0 or (done_i + 1) == n_save:
                        elapsed = time.time() - t0
                        remain  = elapsed / (done_i + 1) * (n_save - done_i - 1)
                        print(f"    [{sid_str}] {done_i+1}/{n_save} frames  "
                              f"({elapsed:.1f}s elapsed, ~{remain:.1f}s remaining)")

    return total_npy, total_jpg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract ADT ground-truth segmentation and depth frames to npy + jpg.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_root", default=DEFAULT_DATA_ROOT,
        help=f"Root folder with one sub-directory per sequence  (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--mode", default="both", choices=["seg", "depth", "both"],
        help="Which modality to extract  (default: both)",
    )
    parser.add_argument(
        "--streams", type=int, nargs="+", default=[1],
        metavar="N",
        help="Camera stream indices to extract: 1=RGB, 2=SLAM-left, 3=SLAM-right  "
             "(default: 1)",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Save every Nth frame  (default: 1 = every frame)",
    )
    parser.add_argument(
        "--filter", default=None, metavar="SUBSTRING",
        help="Only process sequences whose name contains this substring  "
             "(case-sensitive).  E.g. --filter skeleton",
    )
    parser.add_argument(
        "--save_jpg", action="store_true", default=False,
        help="Also write colour-visualisation JPEGs alongside the npy files  "
             "(default: off — npy only).",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be done without writing any files.",
    )
    args = parser.parse_args()

    data_root = os.path.expanduser(args.data_root)
    do_seg    = args.mode in ("seg",   "both")
    do_depth  = args.mode in ("depth", "both")

    # ── Discover sequences ─────────────────────────────────────────────────────
    all_sequences = find_sequences(data_root, args.mode, args.streams)
    print(f"Sequences found : {len(all_sequences)}")

    if args.filter:
        sequences = [s for s in all_sequences if args.filter in s]
        print(f'After --filter "{args.filter}": {len(sequences)} sequences')
    else:
        sequences = all_sequences

    if not sequences:
        sys.exit("No sequences matched. Exiting.")

    if args.dry_run:
        print("(DRY RUN — no files will be written)\n")

    # ── Process ────────────────────────────────────────────────────────────────
    n_seq       = len(sequences)
    total_npy   = 0
    total_jpg   = 0
    failed      = []
    t_global    = time.time()

    for i, seq_name in enumerate(sequences, 1):
        seq_dir = os.path.join(data_root, seq_name)
        print(f"\n[{i}/{n_seq}]  {seq_name}")

        try:
            # ── Load RGB timestamps for nearest-neighbour alignment ────────────
            # The seg/depth VRS starts at the trajectory boundary; main_recording.vrs
            # starts ~10 s earlier with warmup frames.  We use nearest-neighbour
            # timestamp matching to map each seg/depth frame to the correct RGB
            # frame index (and its timestamp) so filenames align with videos_rgb/.
            rgb_ts = load_rgb_timestamps(seq_dir)
            if rgb_ts:
                print(f"  RGB timestamps loaded: {len(rgb_ts)} frames  "
                      f"(nearest-neighbour naming enabled)")
            else:
                print(f"  main_recording.vrs not found — using native frame indices")

            if do_seg:
                print(f"  [segmentation]")
                n, j = extract_vrs(
                    seq_dir        = seq_dir,
                    vrs_name       = SEG_VRS_NAME,
                    stream_prefix  = SEG_STREAM_PREFIX,
                    stream_indices = args.streams,
                    npy_subdir     = "seg_npy",
                    jpg_subdir     = "seg_jpg",
                    colorize_fn    = colorize_segmentation,
                    stride         = args.stride,
                    save_jpg       = args.save_jpg,
                    rgb_timestamps = rgb_ts,
                    dry_run        = args.dry_run,
                )
                total_npy += n
                total_jpg += j
                if not args.dry_run:
                    jpg_msg = f"  {j} jpg" if args.save_jpg else ""
                    print(f"  ✓ seg  {n} npy{jpg_msg}")

            if do_depth:
                print(f"  [depth]")
                n, j = extract_vrs(
                    seq_dir        = seq_dir,
                    vrs_name       = DEPTH_VRS_NAME,
                    stream_prefix  = DEPTH_STREAM_PREFIX,
                    stream_indices = args.streams,
                    npy_subdir     = "depth_npy",
                    jpg_subdir     = "depth_jpg",
                    colorize_fn    = colorize_depth,
                    stride         = args.stride,
                    save_jpg       = args.save_jpg,
                    rgb_timestamps = rgb_ts,
                    dry_run        = args.dry_run,
                )
                total_npy += n
                total_jpg += j
                if not args.dry_run:
                    jpg_msg = f"  {j} jpg" if args.save_jpg else ""
                    print(f"  ✓ depth  {n} npy{jpg_msg}")

        except Exception as exc:
            import traceback
            print(f"  !! FAILED: {exc}")
            traceback.print_exc()
            failed.append(seq_name)

        elapsed_g = time.time() - t_global
        remain_g  = elapsed_g / i * (n_seq - i)
        print(f"  Progress {i}/{n_seq} — "
              f"elapsed {elapsed_g/60:.1f} min  ~{remain_g/60:.1f} min remaining")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"DRY RUN complete.  Would process {n_seq} sequence(s).")
        print(f"  ~{total_npy} npy files  |  ~{total_jpg} jpg files")
    else:
        ok = n_seq - len(failed)
        jpg_summary = f"  |  {total_jpg} jpg saved" if args.save_jpg else ""
        print(f"Done.  {ok}/{n_seq} sequences  |  {total_npy} npy saved{jpg_summary}")
        if failed:
            print(f"\nFailed sequences ({len(failed)}):")
            for s in failed:
                print(f"  {s}")
    print("=" * 60)


if __name__ == "__main__":
    main()