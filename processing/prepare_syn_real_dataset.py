"""
prepare_syn_real_dataset.py
===========================
Extract ALL paired synthetic + real frames for one or more ADT sequences.

For each sequence:
  1. Downloads synthetic_video.vrs if absent (reads URL from ADT_download_urls.json).
  2. Extracts ALL trajectory-valid real frames from main_recording.vrs.
  3. For each real frame, finds the timestamp-nearest frame in synthetic_video.vrs
     and saves it with the same filename stem.
  4. Output layout:
       {seq_dir}/syn_real_pairs/real/       frame_{XXXXXX}_{ts_ns}.png
       {seq_dir}/syn_real_pairs/synthetic/  frame_{XXXXXX}_{ts_ns}.png
  5. Idempotent: skips sequences whose pair directories already contain the expected count.

Usage
-----
  # Extract all frames for seq131 (synthetic_video.vrs already present)
  python prepare_syn_real_dataset.py \\
      --seq_dirs ~/Documents/projectaria_tools_adt_data/Apartment_release_clean_seq131_M1292 \\
      --skip_download

  # Download synthetic + extract for multiple sequences
  python prepare_syn_real_dataset.py \\
      --seq_dirs /path/to/seq131 /path/to/seq148

  # Every-5th-frame (stride=5)
  python prepare_syn_real_dataset.py --stride 5

  # Dry-run
  python prepare_syn_real_dataset.py --dry_run
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
import zipfile

import numpy as np
import requests
from PIL import Image

# ── Defaults ──────────────────────────────────────────────────────────────────
_ADT_ROOT = os.path.expanduser('~/Documents/projectaria_tools_adt_data_clean')
if not os.path.exists(_ADT_ROOT):
    _ADT_ROOT = '/user/f.zhang2/Documents/projectaria_tools_adt_data_clean'
    if not os.path.exists(_ADT_ROOT):
        _ADT_ROOT = '/group-volume/Fengjia/data/projectaria_tools_adt_data_clean'

DEFAULT_URLS_JSON = os.path.expanduser('~/Documents/projectaria_sandbox/projectaria_tools/ADT_download_urls.json')
DEFAULT_SEQ_DIRS  = [
    os.path.join(_ADT_ROOT, 'Apartment_release_clean_seq131_M1292'),
    os.path.join(_ADT_ROOT, 'Apartment_release_clean_seq133_M1292'),
    os.path.join(_ADT_ROOT, 'Apartment_release_clean_seq134_M1292'),
    os.path.join(_ADT_ROOT, 'Apartment_release_clean_seq135_M1292'),
    os.path.join(_ADT_ROOT, 'Apartment_release_clean_seq136_M1292'),
    os.path.join(_ADT_ROOT, 'Apartment_release_clean_seq137_M1292'),
    os.path.join(_ADT_ROOT, 'Apartment_release_clean_seq138_M1292'),
    os.path.join(_ADT_ROOT, 'Apartment_release_clean_seq140_M1292'),
    os.path.join(_ADT_ROOT, 'Apartment_release_clean_seq141_M1292'),
    os.path.join(_ADT_ROOT, 'Apartment_release_clean_seq142_M1292'),
    os.path.join(_ADT_ROOT, 'Apartment_release_clean_seq143_M1292'),
    os.path.join(_ADT_ROOT, 'Apartment_release_clean_seq144_M1292'),
    os.path.join(_ADT_ROOT, 'Apartment_release_clean_seq145_M1292'),
    os.path.join(_ADT_ROOT, 'Apartment_release_clean_seq146_M1292'),
    os.path.join(_ADT_ROOT, 'Apartment_release_clean_seq147_M1292'),
    os.path.join(_ADT_ROOT, 'Apartment_release_clean_seq148_M1292'),
    os.path.join(_ADT_ROOT, 'Apartment_release_clean_seq149_M1292'),
    os.path.join(_ADT_ROOT, 'Apartment_release_clean_seq150_M1292'),
    os.path.join(_ADT_ROOT, 'Apartment_release_decoration_seq132_M1292'),
]

REAL_VRS_NAME = 'main_recording.vrs'
SYN_VRS_NAME  = 'synthetic_video.vrs'
RGB_STREAM    = '214-1'


# ── Download / extract helpers (from prepare_syn_real_pairs.py) ───────────────

def _download_file(url: str, dest_path: str):
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    warnings.filterwarnings('ignore', category=InsecureRequestWarning)

    print(f'  Downloading → {dest_path}')
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with requests.get(url, stream=True, verify=False, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        done  = 0
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f'    {done/1e6:.0f} / {total/1e6:.0f} MB  '
                          f'({done/total*100:.1f}%)\r', end='', flush=True)
    print()


def _extract_zip(zip_path: str, dest_dir: str):
    print(f'  Extracting {os.path.basename(zip_path)} …')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = [m for m in zf.namelist() if m.endswith('.vrs')]
        for m in members:
            out = os.path.join(dest_dir, os.path.basename(m))
            with zf.open(m) as src, open(out, 'wb') as dst:
                dst.write(src.read())
            print(f'    Extracted → {out}')


def ensure_synthetic_vrs(seq_dir: str, urls_json: str, skip_download: bool) -> str:
    """Return path to synthetic_video.vrs, downloading if necessary."""
    syn_vrs = os.path.join(seq_dir, SYN_VRS_NAME)
    if os.path.isfile(syn_vrs):
        return syn_vrs

    if skip_download:
        print(f'  WARN: {syn_vrs} not found and --skip_download set — skipping sequence.')
        return None

    seq_name = os.path.basename(seq_dir)
    with open(urls_json) as f:
        urls = json.load(f)
    if seq_name not in urls['sequences']:
        print(f'  WARN: {seq_name} not found in {urls_json} — skipping.')
        return None
    info = urls['sequences'][seq_name].get('synthetic')
    if not info:
        print(f'  WARN: no "synthetic" entry for {seq_name} in URLs — skipping.')
        return None

    zip_path = os.path.join(seq_dir, info['filename'])
    if not os.path.isfile(zip_path):
        print(f'  Downloading synthetic zip ({info["file_size_bytes"]/1e6:.0f} MB) …')
        _download_file(info['download_url'], zip_path)

    _extract_zip(zip_path, seq_dir)

    if not os.path.isfile(syn_vrs):
        # Try fuzzy match
        candidates = [f for f in os.listdir(seq_dir)
                      if 'synthetic' in f and f.endswith('.vrs')]
        if candidates:
            syn_vrs = os.path.join(seq_dir, candidates[0])
        else:
            print(f'  ERROR: could not find synthetic VRS after extraction.')
            return None
    return syn_vrs


# ── Trajectory helpers (from extract_rgb_frames.py) ───────────────────────────

def _load_trajectory_range_us(seq_dir: str):
    csv_path = os.path.join(seq_dir, 'groundtruth', 'aria_trajectory.csv')
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


def _find_valid_range(dp, sid, n_total: int, traj_t0_us: int, traj_t1_us: int):
    def ts_us(i):
        return dp.get_image_data_by_index(sid, i)[1].capture_timestamp_ns // 1000

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
    return first, lo


# ── Per-sequence extraction ───────────────────────────────────────────────────

def _is_complete(out_dir: str, expected: int) -> bool:
    if not os.path.isdir(out_dir):
        return False
    n = sum(1 for f in os.listdir(out_dir) if f.endswith('.png'))
    return n == expected


def extract_sequence_pairs(seq_dir: str, stride: int, dry_run: bool) -> int:
    """
    Extract real + synthetic pairs for one sequence.
    Returns number of pairs saved (0 if skipped or dry-run).
    """
    try:
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.stream_id import StreamId
    except ImportError:
        sys.exit('ERROR: projectaria_tools not installed — pip install projectaria-tools')

    real_vrs = os.path.join(seq_dir, REAL_VRS_NAME)
    syn_vrs  = os.path.join(seq_dir, SYN_VRS_NAME)
    pair_dir = os.path.join(seq_dir, 'syn_real_pairs')
    real_out = os.path.join(pair_dir, 'real')
    syn_out  = os.path.join(pair_dir, 'synthetic')

    if not os.path.isfile(real_vrs):
        print(f'  WARN: {real_vrs} not found — skipping.')
        return 0
    if not os.path.isfile(syn_vrs):
        print(f'  WARN: {syn_vrs} not found — skipping.')
        return 0

    # ── Open real VRS ──────────────────────────────────────────────────────
    dp_real = data_provider.create_vrs_data_provider(real_vrs)
    sid     = StreamId(RGB_STREAM)
    n_real  = dp_real.get_num_data(sid)

    traj = _load_trajectory_range_us(seq_dir)
    if traj:
        first, last = _find_valid_range(dp_real, sid, n_real, traj[0], traj[1])
        print(f'  Real VRS: {n_real} frames total, valid [{first}, {last}]')
    else:
        first, last = 0, n_real - 1
        print(f'  Real VRS: {n_real} frames (no trajectory — using all)')

    indices = [i for i in range(0, n_real, stride) if first <= i <= last]
    n_pairs = len(indices)

    if _is_complete(real_out, n_pairs) and _is_complete(syn_out, n_pairs):
        print(f'  Already complete ({n_pairs} pairs) — skipping.')
        return 0

    if dry_run:
        print(f'  Would extract {n_pairs} pairs (stride={stride}).')
        return n_pairs

    os.makedirs(real_out, exist_ok=True)
    os.makedirs(syn_out,  exist_ok=True)

    # ── Open synthetic VRS + compute timestamp mapping ─────────────────────
    dp_syn = data_provider.create_vrs_data_provider(syn_vrs)

    # Find the RGB stream in synthetic VRS
    syn_sid = None
    for sid_str in ['214-1', '1201-1', '211-1', '247-1']:
        try:
            s = StreamId(sid_str)
            if dp_syn.get_num_data(s) > 0:
                syn_sid = s
                break
        except Exception:
            pass
    if syn_sid is None:
        print(f'  ERROR: no RGB stream found in {syn_vrs}')
        return 0

    n_syn = dp_syn.get_num_data(syn_sid)
    _, m0 = dp_syn.get_image_data_by_index(syn_sid, 0)
    _, m1 = dp_syn.get_image_data_by_index(syn_sid, 1)
    syn_t0 = m0.capture_timestamp_ns
    syn_dt = m1.capture_timestamp_ns - syn_t0

    def real_ts_to_syn_idx(real_ts_ns: int) -> int:
        idx = round((real_ts_ns - syn_t0) / syn_dt)
        return max(0, min(n_syn - 1, idx))

    # ── Extract pairs ──────────────────────────────────────────────────────
    t0 = time.time()
    for done, frame_idx in enumerate(indices):
        # Real frame
        real_data, real_meta = dp_real.get_image_data_by_index(sid, frame_idx)
        real_ts  = real_meta.capture_timestamp_ns
        real_arr = real_data.to_numpy_array()
        fname    = f'frame_{frame_idx:06d}_{real_ts}.png'

        Image.fromarray(real_arr).save(os.path.join(real_out, fname))

        # Synthetic frame (timestamp-matched)
        syn_idx  = real_ts_to_syn_idx(real_ts)
        syn_data, _ = dp_syn.get_image_data_by_index(syn_sid, syn_idx)
        syn_arr  = syn_data.to_numpy_array()

        Image.fromarray(syn_arr).save(os.path.join(syn_out, fname))

        if (done + 1) % 100 == 0 or (done + 1) == n_pairs:
            elapsed = time.time() - t0
            remain  = elapsed / (done + 1) * (n_pairs - done - 1)
            print(f'  {done+1}/{n_pairs}  ({elapsed:.0f}s elapsed, ~{remain:.0f}s remaining)')

    return n_pairs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--seq_dirs', nargs='+', default=DEFAULT_SEQ_DIRS,
                        help='Sequence root directories to process.')
    parser.add_argument('--urls_json', default=DEFAULT_URLS_JSON,
                        help='Path to ADT_download_urls.json for downloads.')
    parser.add_argument('--stride', type=int, default=1,
                        help='Save every Nth real frame (default 1 = every frame).')
    parser.add_argument('--skip_download', action='store_true',
                        help='Do not download missing synthetic_video.vrs; skip instead.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be done without writing files.')
    args = parser.parse_args()

    urls_json = os.path.expanduser(args.urls_json)
    if args.dry_run:
        print('(DRY RUN)\n')

    total_saved = 0
    for seq_dir in args.seq_dirs:
        seq_dir = os.path.expanduser(seq_dir)
        seq_name = os.path.basename(seq_dir)
        print(f'\n── {seq_name} ──────────────────────')

        if not os.path.isdir(seq_dir):
            print(f'  WARN: directory not found: {seq_dir}')
            continue

        # Ensure synthetic VRS is present
        if not args.dry_run:
            syn_vrs = ensure_synthetic_vrs(seq_dir, urls_json, args.skip_download)
            if syn_vrs is None:
                continue

        n = extract_sequence_pairs(seq_dir, args.stride, args.dry_run)
        if not args.dry_run and n > 0:
            print(f'  ✓ {n} pairs saved to {os.path.join(seq_dir, "syn_real_pairs")}')
        total_saved += n

    print(f'\n{"(dry run) " if args.dry_run else ""}Total pairs: {total_saved}')


if __name__ == '__main__':
    main()
