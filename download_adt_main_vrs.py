"""
download_adt_main_vrs.py
========================
Download the main VRS recording for every sequence listed in the ADT
download-URLs JSON file.

Reads all sequence names from the JSON, then calls aria_dataset_downloader
once per sequence with -d 0 (main_vrs).  Already-downloaded sequences are
skipped automatically.

Usage
-----
  # Dry-run — print every command without executing
  python download_adt_main_vrs.py --dry_run

  # Download everything
  python download_adt_main_vrs.py

  # Download only Apartment sequences that contain 'skeleton'
  python download_adt_main_vrs.py --filter skeleton

  # Download only Lite sequences
  python download_adt_main_vrs.py --filter Lite_release

  # Override default paths
  python download_adt_main_vrs.py \\
      --urls_json  ~/my_path/ADT_download_urls.json \\
      --output_dir ~/my_data/adt

Defaults
--------
  --urls_json  ~/Documents/projectaria_sandbox/projectaria_tools/ADT_download_urls.json
  --output_dir ~/Documents/projectaria_tools_adt_data
  --data_type  0   (main_vrs)
"""

import argparse
import json
import os
import subprocess
import sys
import time


# ── Defaults (mirror the sample command) ──────────────────────────────────────
DEFAULT_URLS_JSON  = os.path.expanduser(
    '~/Documents/projectaria_sandbox/projectaria_tools/ADT_download_urls.json'
)
DEFAULT_OUTPUT_DIR = os.path.expanduser(
    '~/Documents/projectaria_tools_adt_data'
)
DEFAULT_DATA_TYPE  = 0   # 0 = main_vrs


def load_sequence_names(urls_json: str) -> list[str]:
    """Return a sorted list of all sequence names in the download-URLs JSON."""
    with open(urls_json) as f:
        data = json.load(f)
    sequences = data.get('sequences', data)   # handle both wrapped and flat formats
    return sorted(sequences.keys())


def is_already_downloaded(output_dir: str, sequence_name: str) -> bool:
    """
    Return True if the sequence directory already exists and contains a
    main_recording.vrs file — the canonical output of -d 0.
    """
    seq_dir = os.path.join(output_dir, sequence_name)
    vrs_path = os.path.join(seq_dir, 'main_recording.vrs')
    return os.path.isfile(vrs_path)


def build_command(urls_json: str, output_dir: str,
                  data_type: int, sequence_name: str) -> list[str]:
    return [
        'aria_dataset_downloader',
        '-c', urls_json,
        '-o', output_dir,
        '-d', str(data_type),
        '-l', sequence_name,
    ]


def main():
    parser = argparse.ArgumentParser(
        description='Download main VRS for all ADT sequences.',
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
        '--data_type', type=int, default=DEFAULT_DATA_TYPE,
        help='Data-type index passed to -d  (default: 0 = main_vrs)',
    )
    parser.add_argument(
        '--filter', default=None, metavar='SUBSTRING',
        help='Only download sequences whose name contains this substring '
             '(case-sensitive).  E.g. --filter skeleton',
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Print every command that would run without executing anything.',
    )
    parser.add_argument(
        '--skip_existing', action='store_true', default=True,
        help='Skip sequences that already have main_recording.vrs  (default: on)',
    )
    parser.add_argument(
        '--no_skip_existing', dest='skip_existing', action='store_false',
        help='Re-download even if main_recording.vrs already exists.',
    )
    args = parser.parse_args()

    urls_json  = os.path.expanduser(args.urls_json)
    output_dir = os.path.expanduser(args.output_dir)

    # ── Validate paths ─────────────────────────────────────────────────────────
    if not os.path.isfile(urls_json):
        sys.exit(f'ERROR: URLs JSON not found: {urls_json}')

    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)

    # ── Load sequence list ─────────────────────────────────────────────────────
    all_sequences = load_sequence_names(urls_json)
    print(f'Total sequences in JSON : {len(all_sequences)}')

    if args.filter:
        sequences = [s for s in all_sequences if args.filter in s]
        print(f'After --filter "{args.filter}" : {len(sequences)} sequences')
    else:
        sequences = all_sequences

    if not sequences:
        sys.exit('No sequences matched. Exiting.')

    # ── Classify: skip / download ──────────────────────────────────────────────
    to_download = []
    skipped     = []
    for seq in sequences:
        if args.skip_existing and is_already_downloaded(output_dir, seq):
            skipped.append(seq)
        else:
            to_download.append(seq)

    print(f'Already downloaded (skip): {len(skipped)}')
    print(f'To download               : {len(to_download)}')
    if args.dry_run:
        print('(DRY RUN — no commands will be executed)\n')

    if not to_download:
        print('Nothing to do.')
        return

    # ── Run downloads ──────────────────────────────────────────────────────────
    n        = len(to_download)
    failed   = []
    t_start  = time.time()

    for i, seq in enumerate(to_download, 1):
        cmd = build_command(urls_json, output_dir, args.data_type, seq)
        cmd_str = ' '.join(cmd)

        print(f'\n[{i}/{n}]  {seq}')
        print(f'  $ {cmd_str}')

        if args.dry_run:
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f'  !! FAILED  (returncode={result.returncode})')
            failed.append(seq)
        else:
            print(f'  ✓ done')

        # Brief progress estimate
        elapsed = time.time() - t_start
        avg     = elapsed / i
        remaining = avg * (n - i)
        print(f'  Progress {i}/{n} — elapsed {elapsed/60:.1f} min  '
              f'~{remaining/60:.1f} min remaining')

    # ── Summary ────────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    if args.dry_run:
        print(f'DRY RUN complete.  Would have run {n} download command(s).')
    else:
        print(f'Done.  {n - len(failed)}/{n} sequences downloaded successfully.')
        if failed:
            print(f'\nFailed sequences ({len(failed)}):')
            for s in failed:
                print(f'  {s}')
            print('\nRe-run the script to retry failed sequences '
                  '(they will not be skipped since their VRS is absent).')
    print('=' * 60)


if __name__ == '__main__':
    main()
