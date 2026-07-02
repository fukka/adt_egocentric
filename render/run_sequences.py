"""
run_sequences.py
================
Batch runner: render every ADT sequence listed in SEQUENCES by calling
render_from_poses_blender.py (or render_exocentric_blender.py) as a subprocess
with the sequence directory passed via --base_dir.

Edit SEQUENCES to add / remove sequences.  Set the DEFAULT_*_FLAGS lists for
flags that apply to every sequence.  Per-sequence overrides go in 'extra_args'.

Modes
-----
    ego-fisheye   render_from_poses_blender.py  --fisheye          (default)
    ego-pinhole   render_from_poses_blender.py  (no --fisheye)
    exo           render_exocentric_blender.py  --camera all
    all           all three modes in sequence order

Usage
-----
    # Ego-fisheye for all sequences (default):
    python run_sequences.py

    # Ego-pinhole only:
    python run_sequences.py --mode ego-pinhole

    # Exocentric (all 8 cameras) for every sequence:
    python run_sequences.py --mode exo

    # All three modes:
    python run_sequences.py --mode all

    # Dry-run — print commands without executing:
    python run_sequences.py --dry_run

    # Single named sequence only:
    python run_sequences.py --seq seq131

    # Override a render flag for all sequences (append after --):
    python run_sequences.py -- --cycles_device OPTIX --cycles_samples 16
"""

import argparse
import subprocess
import sys
import os

# ── Sequence registry ─────────────────────────────────────────────────────────
# Central ADT data root — adjust if your data lives elsewhere.
ADT_DATA_ROOT = '/user/f.zhang2/Documents/projectaria_tools_adt_data'
SAVE_ROOT = '/user/f.zhang2/Documents/projectaria_tools_adt_data_rendered'

# Each entry: required keys = name, base.
# Optional keys:
#   extra_args : list[str]  — extra CLI flags forwarded only to this sequence
#   skip       : bool       — True to skip without removing from the list
SEQUENCES = [
    # {
    #     'name': 'seq131',
    #     'base': f'{ADT_DATA_ROOT}/Apartment_release_clean_seq131_M1292',
    # },
    {
        'name': 'seq134',
        'base': f'{ADT_DATA_ROOT}/Apartment_release_clean_seq134_M1292',
        'extra_args': ['--output_dir', f'{SAVE_ROOT}/Apartment_release_clean_seq134_M1292', '--no_normals'],
    },
    # {
    # Apartment_release_clean_seq134_M1292
    #     'name': 'golden_skeleton_seq100',
    #     'base': f'{ADT_DATA_ROOT}/Apartment_release_golden_skeleton_seq100_10s_sample_M1292',
    # },
    # ── Add more sequences below ──────────────────────────────────────────
    # {
    #     'name': 'seq_xxx',
    #     'base': f'{ADT_DATA_ROOT}/Apartment_release_xxx',
    #     'extra_args': ['--frame_step', '15'],
    # },
    # {
    #     'name': 'some_other_seq',
    #     'base': f'{ADT_DATA_ROOT}/...',
    #     'skip': True,   # temporarily disable without deleting
    # },
]

# ── Default flags (applied to every sequence per mode) ────────────────────────
DEFAULT_EGO_FISHEYE_FLAGS: list[str] = [
    '--fisheye',
    '--frame_step', '100',
    '--cycles_samples', '32',
    '--cycles_device', 'OPTIX',          # uncomment for GPU rendering
    # '--cycles_denoiser', 'OPENIMAGEDENOISE',
]

DEFAULT_EGO_PINHOLE_FLAGS: list[str] = [
    # no --fisheye
    '--frame_step', '100',
    '--cycles_samples', '32',
    '--cycles_device', 'OPTIX',
]

DEFAULT_EXO_FLAGS: list[str] = [
    '--camera', 'all',
    '--cycles_samples', '32',
    '--cycles_device', 'OPTIX',
]

# ── Script paths (resolved relative to this file) ─────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
EGO_SCRIPT  = os.path.join(_HERE, 'render_from_poses_blender.py')
EXO_SCRIPT  = os.path.join(_HERE, 'render_exocentric_blender.py')


def build_cmd(script: str, seq: dict, default_flags: list[str],
              extra_global: list[str]) -> list[str]:
    return ([sys.executable, script, '--base_dir', seq['base']]
            + default_flags
            + seq.get('extra_args', [])
            + extra_global)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode',
                        choices=['ego-fisheye', 'ego-pinhole', 'exo', 'all'],
                        default='ego-fisheye',
                        help='Renderer to run (default: ego-fisheye)')
    parser.add_argument('--seq', default=None,
                        help='Render only this named sequence (default: all)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('extra_global', nargs=argparse.REMAINDER,
                        help='Extra flags forwarded to every render call '
                             '(place after -- to avoid argparse conflicts)')
    args = parser.parse_args()

    # Strip leading '--' separator if present
    extra = [a for a in args.extra_global if a != '--']

    seqs = [s for s in SEQUENCES if not s.get('skip', False)]
    if args.seq:
        seqs = [s for s in seqs if s['name'] == args.seq]
        if not seqs:
            sys.exit(f'ERROR: sequence {args.seq!r} not found. '
                     f'Available: {[s["name"] for s in SEQUENCES]}')

    mode_scripts = {
        'ego-fisheye': [(EGO_SCRIPT, DEFAULT_EGO_FISHEYE_FLAGS)],
        'ego-pinhole': [(EGO_SCRIPT, DEFAULT_EGO_PINHOLE_FLAGS)],
        'exo':         [(EXO_SCRIPT, DEFAULT_EXO_FLAGS)],
        'all':         [(EGO_SCRIPT, DEFAULT_EGO_FISHEYE_FLAGS),
                        (EGO_SCRIPT, DEFAULT_EGO_PINHOLE_FLAGS),
                        (EXO_SCRIPT, DEFAULT_EXO_FLAGS)],
    }

    failed = []
    for seq in seqs:
        for script, flags in mode_scripts[args.mode]:
            cmd = build_cmd(script, seq, flags, extra)
            label = f'[{seq["name"]}  {os.path.basename(script)}]'
            print(f'\n{label}  {" ".join(cmd)}')
            if args.dry_run:
                continue
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f'  *** {label} FAILED (returncode={result.returncode})')
                failed.append((seq['name'], os.path.basename(script)))

    if args.dry_run:
        print('\nDry run complete — no renders executed.')
    elif failed:
        print(f'\nDone with {len(failed)} failure(s):')
        for name, script in failed:
            print(f'  {name}  {script}')
        sys.exit(1)
    else:
        print(f'\nAll {len(seqs)} sequence(s) completed successfully.')


if __name__ == '__main__':
    main()