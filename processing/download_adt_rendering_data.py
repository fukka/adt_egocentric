"""
download_adt_rendering_data.py
==============================
Download and extract the ground-truth and MPS data required for Blender
rendering for every sequence listed in the ADT download-URLs JSON file.

This script reads download URLs directly from ADT_download_urls.json and
handles extraction into the subdirectory layout expected by the render scripts
(render_from_poses_blender_maps.py, render_from_poses.py):

  <seq>/groundtruth/       ← main_groundtruth  (instances.json, scene_objects.csv …)
  <seq>/mps/slam/          ← mps_slam_trajectories, mps_slam_calibration, …
  <seq>/mps/eye_gaze/      ← mps_eye_gaze
  <seq>/                   ← segmentation, depth, synthetic VRS files

Supported data-type keys (passed to --data_types)
--------------------------------------------------
  main_groundtruth      instances.json, aria_trajectory.csv, scene_objects.csv,
                        3d_bounding_box.csv, metadata.json, skeleton files …
  mps_slam_trajectories closed_loop_trajectory.csv, open_loop_trajectory.csv
  mps_slam_calibration  online_calibration.jsonl
  mps_slam_points       semidense_points.csv.gz
  mps_slam_summary      summary.json
  mps_eye_gaze          general_eye_gaze.csv
  mps_artifacts         full MPS artifacts bundle
  segmentation          segmentations.vrs, segmentations_with_skeleton.vrs
  depth                 depth_images.vrs, depth_images_with_skeleton.vrs
  synthetic             synthetic_video.vrs
  main_vrs              main Aria VRS recording
  video_main_rgb        RGB preview MP4

Usage
-----
  # Dry-run
  python download_adt_rendering_data.py --dry_run

  # Download default rendering data (groundtruth + SLAM trajectories)
  python download_adt_rendering_data.py

  # Download only groundtruth (instances.json, scene_objects.csv …)
  python download_adt_rendering_data.py --data_types main_groundtruth

  # Download groundtruth + SLAM trajectories + calibration
  python download_adt_rendering_data.py \\
      --data_types main_groundtruth mps_slam_trajectories mps_slam_calibration

  # Filter to skeleton sequences only
  python download_adt_rendering_data.py --filter skeleton

  # Override default paths
  python download_adt_rendering_data.py \\
      --urls_json  ~/my_path/ADT_download_urls.json \\
      --output_dir ~/my_data/adt

Defaults
--------
  --urls_json   ~/Documents/projectaria_sandbox/projectaria_tools/ADT_download_urls.json
  --output_dir  ~/Documents/projectaria_tools_adt_data
  --data_types  main_groundtruth mps_slam_trajectories
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.request
from zipfile import is_zipfile, ZipFile

# SPACE
# python3 processing/download_adt_rendering_data.py --output_dir /group-volume/Fengjia/data/projectaria_tools_adt_data_clean/ --filter Apartment_release_clean_ --urls_json /group-volume/Fengjia/data/projectaria_sandbox/projectaria_tools/ADT_download_urls.json
# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_URLS_JSON  = os.path.expanduser(
    '~/Documents/projectaria_sandbox/projectaria_tools/ADT_download_urls.json'
)
DEFAULT_OUTPUT_DIR = os.path.expanduser(
    '~/Documents/projectaria_tools_adt_data_clean'
)
# DEFAULT_DATA_TYPES = ['main_groundtruth', 'mps_slam_trajectories', 'depth', 'segmentation']
DEFAULT_DATA_TYPES = ['main_groundtruth']

ALL_DATA_TYPES = [
    'main_groundtruth',
    'mps_slam_trajectories',
    'mps_slam_calibration',
    'mps_slam_points',
    'mps_slam_summary',
    'mps_eye_gaze',
    'mps_artifacts',
    'segmentation',
    'depth',
    'synthetic',
    'main_vrs',
    'video_main_rgb',
]

# Subdirectory inside the sequence folder where each data type is extracted.
# Files from the zip are placed inside this subdirectory.
EXTRACT_SUBDIR = {
    'main_groundtruth':      'groundtruth',
    'mps_slam_trajectories': 'mps/slam',
    'mps_slam_calibration':  'mps/slam',
    'mps_slam_points':       'mps/slam',
    'mps_slam_summary':      'mps/slam',
    'mps_eye_gaze':          'mps/eye_gaze',
    'mps_artifacts':         'mps',
    'segmentation':          'segmentation',   # extract directly into seq root
    'depth':                 'depth',
    'synthetic':             '',
    'main_vrs':              '',   # single file, no extraction
    'video_main_rgb':        '',
}

# Sentinel file (relative to sequence dir) that indicates a completed download.
SENTINEL = {
    'main_groundtruth':      'groundtruth/instances.json',
    'mps_slam_trajectories': 'mps/slam/closed_loop_trajectory.csv',
    'mps_slam_calibration':  'mps/slam/online_calibration.jsonl',
    'mps_slam_points':       'mps/slam/semidense_points.csv.gz',
    'mps_slam_summary':      'mps/slam/summary.json',
    'mps_eye_gaze':          'mps/eye_gaze/general_eye_gaze.csv',
    'mps_artifacts':         'mps/hand_tracking/wrist_and_palm_poses.csv',
    'segmentation':          'segmentations.vrs',
    'depth':                 'depth_images.vrs',
    'synthetic':             'synthetic_video.vrs',
    'main_vrs':              'main_recording.vrs',
    'video_main_rgb':        'preview_rgb.mp4',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_sequences(urls_json: str) -> dict:
    with open(urls_json) as f:
        data = json.load(f)
    return data.get('sequences', data)


def is_downloaded(seq_dir: str, data_type: str) -> bool:
    sentinel = SENTINEL.get(data_type)
    if not sentinel:
        return False
    return os.path.isfile(os.path.join(seq_dir, sentinel))


def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest_path: str, expected_bytes: int = 0) -> None:
    """Download url to dest_path with a simple progress indicator."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(dest_path, 'wb') as out:
        total    = int(resp.headers.get('Content-Length', expected_bytes) or 0)
        received = 0
        chunk_sz = 131072  # 128 KB
        t0       = time.time()
        while True:
            chunk = resp.read(chunk_sz)
            if not chunk:
                break
            out.write(chunk)
            received += len(chunk)
            if total:
                pct     = received / total * 100
                elapsed = time.time() - t0
                speed   = received / elapsed / 1e6 if elapsed > 0 else 0
                print(f'\r  {pct:5.1f}%  {received/1e6:.0f}/{total/1e6:.0f} MB'
                      f'  {speed:.1f} MB/s', end='', flush=True)
    print()


def download_and_extract(url: str, filename: str, sha1sum: str,
                         seq_dir: str, data_type: str,
                         expected_bytes: int = 0,
                         dry_run: bool = False) -> bool:
    """Download a file and extract it (if zip) to the correct subdirectory."""
    subdir = EXTRACT_SUBDIR.get(data_type, '')
    extract_dir = os.path.join(seq_dir, subdir) if subdir else seq_dir
    os.makedirs(extract_dir, exist_ok=True)

    # Download to a temp file in the sequence dir so we can inspect it
    tmp_path = os.path.join(seq_dir, filename)

    if dry_run:
        print(f'  [DRY RUN] would download {filename} → {extract_dir}/')
        return True

    print(f'  Downloading {filename} ({expected_bytes/1e9:.2f} GB) …')
    try:
        download_file(url, tmp_path, expected_bytes)
    except Exception as e:
        print(f'  !! Download failed: {e}')
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False

    # SHA-1 check
    if sha1sum:
        actual = sha1_of_file(tmp_path)
        if actual != sha1sum:
            print(f'  !! SHA-1 mismatch: expected {sha1sum}, got {actual}')
            os.remove(tmp_path)
            return False

    # Extract zip or move single file
    if is_zipfile(tmp_path):
        print(f'  Extracting to {extract_dir}/ …')
        with ZipFile(tmp_path, 'r') as zf:
            zf.extractall(extract_dir)
        os.remove(tmp_path)
    else:
        # Single file (e.g. .vrs, .mp4): rename to canonical name
        dest = os.path.join(extract_dir, SENTINEL.get(data_type, filename).split('/')[-1])
        os.rename(tmp_path, dest)

    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Download and extract ADT ground-truth and MPS data for rendering.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--urls_json', default=DEFAULT_URLS_JSON,
        help=f'Path to ADT_download_urls.json  (default: {DEFAULT_URLS_JSON})',
    )
    parser.add_argument(
        '--output_dir', default=DEFAULT_OUTPUT_DIR,
        help=f'Root output directory  (default: {DEFAULT_OUTPUT_DIR})',
    )
    parser.add_argument(
        '--data_types', nargs='+', default=DEFAULT_DATA_TYPES,
        metavar='KEY',
        choices=ALL_DATA_TYPES,
        help=(
            'One or more data-type keys to download (space-separated). '
            f'Default: {" ".join(DEFAULT_DATA_TYPES)}. '
            f'Available: {", ".join(ALL_DATA_TYPES)}'
        ),
    )
    parser.add_argument(
        '--filter', default=None, metavar='SUBSTRING',
        help='Only process sequences whose name contains this substring '
             '(case-sensitive).  E.g. --filter skeleton',
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Print what would be downloaded without executing anything.',
    )
    parser.add_argument(
        '--skip_existing', action='store_true', default=True,
        help='Skip sequences where the sentinel file already exists  (default: on)',
    )
    parser.add_argument(
        '--no_skip_existing', dest='skip_existing', action='store_false',
        help='Re-download even if sentinel files already exist.',
    )
    args = parser.parse_args()

    urls_json  = os.path.expanduser(args.urls_json)
    output_dir = os.path.expanduser(args.output_dir)

    if not os.path.isfile(urls_json):
        sys.exit(f'ERROR: URLs JSON not found: {urls_json}')
    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)

    print(f'Data types : {", ".join(args.data_types)}')

    sequences = load_sequences(urls_json)
    all_names = sorted(sequences.keys())
    print(f'Total sequences in JSON: {len(all_names)}')

    if args.filter:
        names = [s for s in all_names if args.filter in s]
        print(f'After --filter "{args.filter}": {len(names)} sequences')
    else:
        names = all_names

    if not names:
        sys.exit('No sequences matched. Exiting.')

    # ── Build job list ─────────────────────────────────────────────────────────
    jobs_todo    = []
    jobs_skipped = []

    for seq in names:
        seq_dir = os.path.join(output_dir, seq)
        seq_data = sequences[seq]
        for dt in args.data_types:
            if dt not in seq_data:
                print(f'  [skip] {seq}: {dt} not available in JSON')
                continue
            if args.skip_existing and is_downloaded(seq_dir, dt):
                jobs_skipped.append((seq, dt))
            else:
                jobs_todo.append((seq, dt))

    print(f'Already downloaded (skip): {len(jobs_skipped)}')
    print(f'To download              : {len(jobs_todo)}')
    if args.dry_run:
        print('(DRY RUN — nothing will be downloaded)\n')

    if not jobs_todo:
        print('Nothing to do.')
        return

    # ── Download ───────────────────────────────────────────────────────────────
    n       = len(jobs_todo)
    failed  = []
    t_start = time.time()

    for i, (seq, dt) in enumerate(jobs_todo, 1):
        seq_dir  = os.path.join(output_dir, seq)
        meta     = sequences[seq][dt]
        url      = meta['download_url']
        filename = meta['filename']
        sha1sum  = meta.get('sha1sum', '')
        nbytes   = meta.get('file_size_bytes', 0)

        print(f'\n[{i}/{n}]  {seq}  [{dt}]')

        ok = download_and_extract(
            url=url,
            filename=filename,
            sha1sum=sha1sum,
            seq_dir=seq_dir,
            data_type=dt,
            expected_bytes=nbytes,
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            if ok:
                print(f'  ✓ done')
            else:
                failed.append((seq, dt))

            elapsed   = time.time() - t_start
            avg       = elapsed / i
            remaining = avg * (n - i)
            print(f'  Progress {i}/{n} — elapsed {elapsed/60:.1f} min  '
                  f'~{remaining/60:.1f} min remaining')

    # ── Summary ────────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    if args.dry_run:
        print(f'DRY RUN complete.  Would download {n} file(s).')
    else:
        print(f'Done.  {n - len(failed)}/{n} jobs completed successfully.')
        if failed:
            print(f'\nFailed jobs ({len(failed)}):')
            for seq, dt in failed:
                print(f'  {seq}  [{dt}]')
            print('\nRe-run to retry (failed items have no sentinel file).')
    print('=' * 60)


if __name__ == '__main__':
    main()
