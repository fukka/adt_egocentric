"""
Download ApartmentEnv.glb for testing.

Reads the URL from DTC_objects_ADT_download_urls.json and downloads to
the seq131 object_models directory (creates it if needed).

Usage:
    python download_apartment_env.py

    # Download to a specific directory
    python download_apartment_env.py --out_dir /path/to/object_models
"""

import argparse
import hashlib
import json
import os
import sys
import time
import warnings

try:
    import requests
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    warnings.filterwarnings('ignore', category=InsecureRequestWarning)
except ImportError:
    sys.exit('ERROR: requests library not found.  Install with:  pip install requests')

URLS_JSON = os.path.join(os.path.dirname(__file__), 'DTC_objects_ADT_download_urls.json')
BASE      = os.path.expanduser(
    '~/Documents/projectaria_tools_adt_data/'
    'Apartment_release_clean_seq131_M1292'
)
DEFAULT_OUT = os.path.join(BASE, 'object_models')


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, expected_bytes=0):
    t0 = time.time()
    with requests.get(url, stream=True, verify=False, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get('Content-Length', expected_bytes) or 0)
        received = 0
        with open(dest, 'wb') as out:
            for chunk in resp.iter_content(131072):
                if not chunk:
                    continue
                out.write(chunk)
                received += len(chunk)
                if total:
                    pct   = received / total * 100
                    speed = received / (time.time() - t0 + 1e-6) / 1e6
                    print(f'\r  {pct:5.1f}%  {received/1e6:.1f}/{total/1e6:.1f} MB'
                          f'  {speed:.1f} MB/s', end='', flush=True)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default=DEFAULT_OUT,
                        help=f'Directory to save ApartmentEnv.glb  (default: {DEFAULT_OUT})')
    parser.add_argument('--urls_json', default=URLS_JSON,
                        help=f'Path to DTC_objects_ADT_download_urls.json')
    args = parser.parse_args()

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    dest = os.path.join(out_dir, 'ApartmentEnv.glb')
    if os.path.exists(dest):
        print(f'Already exists: {dest}')
        return

    urls_json = os.path.expanduser(args.urls_json)
    if not os.path.isfile(urls_json):
        sys.exit(f'ERROR: URLs JSON not found: {urls_json}')

    with open(urls_json) as f:
        data = json.load(f)
    meta = data['releases']['ADT']['objects']['ApartmentEnv']['3d-asset_glb']
    url     = meta['download_url']
    sha1sum = meta.get('sha1sum', '')
    nbytes  = meta.get('file_size_bytes', 0)

    # Check URL expiry
    from urllib.parse import urlparse, parse_qs
    oe_vals = parse_qs(urlparse(url).query).get('oe', [])
    if oe_vals and int(oe_vals[0], 16) < int(time.time()):
        import datetime
        exp_str = datetime.datetime.utcfromtimestamp(int(oe_vals[0], 16)).strftime('%Y-%m-%d %H:%M UTC')
        print(f'WARNING: URL expired on {exp_str}.')
        print('Please download a fresh DTC_objects_ADT_download_urls.json from the ADT website.')
        sys.exit(1)

    print(f'Downloading ApartmentEnv.glb ({nbytes/1e6:.1f} MB) → {dest}')
    try:
        download(url, dest, nbytes)
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        sys.exit(f'Download failed: {e}')

    if sha1sum:
        print('Verifying SHA-1...', end=' ', flush=True)
        actual = sha1_file(dest)
        if actual != sha1sum:
            os.remove(dest)
            sys.exit(f'SHA-1 mismatch (expected {sha1sum}, got {actual}). File removed.')
        print('OK')

    print(f'Done: {dest}')


if __name__ == '__main__':
    main()
