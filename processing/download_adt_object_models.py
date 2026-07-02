"""
download_adt_object_models.py
=============================
Download the shared ADT 3D object models (.glb files) from the
DTC_objects_ADT_download_urls.json file.

All 400 objects are shared across every ADT sequence, so they are
downloaded once to a single central directory rather than per-sequence.
Pass --models_dir to the render scripts instead of each sequence's
object_models/ folder.

Total download: ~5.4 GB (400 .glb files)

Usage
-----
  # Dry-run — print what would be downloaded
  python download_adt_object_models.py --dry_run

  # Download all objects
  python download_adt_object_models.py

  # Download to a custom directory
  python download_adt_object_models.py --models_dir ~/my_data/adt_object_models

  # Download only objects whose name contains a substring
  python download_adt_object_models.py --filter Bowl

Defaults
--------
  --objects_json  ~/Documents/projectaria_sandbox/projectaria_tools/DTC_objects_ADT_download_urls.json
  --models_dir    ~/Documents/projectaria_tools_adt_data/object_models
"""

import argparse
import hashlib
import json
import os
import sys
import time
import warnings

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings('ignore', category=InsecureRequestWarning)


DEFAULT_OBJECTS_JSON = os.path.expanduser(
    '~/Documents/projectaria_sandbox/projectaria_tools/DTC_objects_ADT_download_urls.json'
)
DEFAULT_MODELS_DIR = os.path.expanduser(
    '~/Documents/projectaria_tools_adt_data/object_models'
)


def load_objects(objects_json: str) -> dict:
    with open(objects_json) as f:
        data = json.load(f)
    return data['releases']['ADT']['objects']


def check_url_expiry(objects: dict) -> None:
    import datetime
    from urllib.parse import urlparse, parse_qs
    for meta in objects.values():
        url = meta.get('3d-asset_glb', {}).get('download_url', '')
        if not url:
            continue
        oe_vals = parse_qs(urlparse(url).query).get('oe', [])
        if not oe_vals:
            return
        expiry = int(oe_vals[0], 16)
        if expiry < int(time.time()):
            exp_str = datetime.datetime.utcfromtimestamp(expiry).strftime('%Y-%m-%d %H:%M UTC')
            sys.exit(
                f'\nERROR: The signed download URLs expired on {exp_str}.\n'
                f'Please download a fresh copy of DTC_objects_ADT_download_urls.json\n'
                f'from the ADT website and re-run with --objects_json pointing to it.\n'
            )
        return


def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest_path: str, expected_bytes: int = 0) -> None:
    chunk_sz = 131072
    t0 = time.time()
    with requests.get(url, stream=True, verify=False, timeout=60) as resp:
        resp.raise_for_status()
        total    = int(resp.headers.get('Content-Length', expected_bytes) or 0)
        received = 0
        with open(dest_path, 'wb') as out:
            for chunk in resp.iter_content(chunk_sz):
                if not chunk:
                    continue
                out.write(chunk)
                received += len(chunk)
                if total:
                    pct   = received / total * 100
                    speed = received / (time.time() - t0) / 1e6
                    print(f'\r  {pct:5.1f}%  {received/1e6:.1f}/{total/1e6:.1f} MB'
                          f'  {speed:.1f} MB/s', end='', flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Download shared ADT 3D object models (.glb) to a central directory.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--objects_json', default=DEFAULT_OBJECTS_JSON,
        help=f'Path to DTC_objects_ADT_download_urls.json  (default: {DEFAULT_OBJECTS_JSON})',
    )
    parser.add_argument(
        '--models_dir', default=DEFAULT_MODELS_DIR,
        help=f'Directory to save .glb files  (default: {DEFAULT_MODELS_DIR})',
    )
    parser.add_argument(
        '--filter', default=None, metavar='SUBSTRING',
        help='Only download objects whose name contains this substring (case-sensitive).',
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Print what would be downloaded without executing anything.',
    )
    parser.add_argument(
        '--skip_existing', action='store_true', default=True,
        help='Skip objects whose .glb already exists  (default: on)',
    )
    parser.add_argument(
        '--no_skip_existing', dest='skip_existing', action='store_false',
        help='Re-download even if .glb already exists.',
    )
    args = parser.parse_args()

    objects_json = os.path.expanduser(args.objects_json)
    models_dir   = os.path.expanduser(args.models_dir)

    if not os.path.isfile(objects_json):
        sys.exit(f'ERROR: objects JSON not found: {objects_json}')
    if not args.dry_run:
        os.makedirs(models_dir, exist_ok=True)

    objects = load_objects(objects_json)
    if not args.dry_run:
        check_url_expiry(objects)

    all_names = sorted(objects.keys())
    print(f'Total objects in JSON: {len(all_names)}')

    if args.filter:
        names = [n for n in all_names if args.filter in n]
        print(f'After --filter "{args.filter}": {len(names)} objects')
    else:
        names = all_names

    if not names:
        sys.exit('No objects matched. Exiting.')

    # ── Build job list ─────────────────────────────────────────────────────────
    to_download = []
    skipped     = []
    for name in names:
        glb_path = os.path.join(models_dir, f'{name}.glb')
        if args.skip_existing and os.path.isfile(glb_path):
            skipped.append(name)
        else:
            to_download.append(name)

    total_bytes = sum(
        objects[n]['3d-asset_glb'].get('file_size_bytes', 0) for n in to_download
    )
    print(f'Already downloaded (skip): {len(skipped)}')
    print(f'To download              : {len(to_download)}  ({total_bytes/1e9:.2f} GB)')
    if args.dry_run:
        print('(DRY RUN — nothing will be downloaded)\n')

    if not to_download:
        print('Nothing to do.')
        return

    # ── Download ───────────────────────────────────────────────────────────────
    n       = len(to_download)
    failed  = []
    t_start = time.time()

    for i, name in enumerate(to_download, 1):
        meta     = objects[name]['3d-asset_glb']
        url      = meta['download_url']
        sha1sum  = meta.get('sha1sum', '')
        nbytes   = meta.get('file_size_bytes', 0)
        glb_path = os.path.join(models_dir, f'{name}.glb')

        print(f'\n[{i}/{n}]  {name}  ({nbytes/1e6:.1f} MB)')

        if args.dry_run:
            print(f'  [DRY RUN] would download → {glb_path}')
            continue

        try:
            download_file(url, glb_path, nbytes)
        except Exception as e:
            print(f'  !! Download failed: {e}')
            if os.path.exists(glb_path):
                os.remove(glb_path)
            failed.append(name)
            continue

        if sha1sum:
            actual = sha1_of_file(glb_path)
            if actual != sha1sum:
                print(f'  !! SHA-1 mismatch — removing file')
                os.remove(glb_path)
                failed.append(name)
                continue

        print(f'  ✓ done')
        elapsed   = time.time() - t_start
        remaining = (elapsed / i) * (n - i)
        print(f'  Progress {i}/{n} — elapsed {elapsed/60:.1f} min  '
              f'~{remaining/60:.1f} min remaining')

    # ── Summary ────────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    if args.dry_run:
        print(f'DRY RUN complete.  Would download {n} .glb file(s).')
    else:
        print(f'Done.  {n - len(failed)}/{n} objects downloaded successfully.')
        print(f'Models directory: {models_dir}')
        if failed:
            print(f'\nFailed ({len(failed)}):')
            for name in failed:
                print(f'  {name}')
            print('\nRe-run to retry failed objects.')
    print('=' * 60)


if __name__ == '__main__':
    main()