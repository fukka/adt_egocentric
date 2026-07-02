"""
render_exocentric_benchmark_frames.py
======================================
Renders, for ALL eight exocentric cameras, only the subset of frames that
the depth benchmark actually evaluates (or would evaluate).

Frame selection uses a two-stage strategy with fallback:

  Stage 1 — depth_benchmark_results exists:
    Scan {base_dir}/depth_benchmark_results/ recursively for files matching
    `dav2_*_frame_<XXXXXX>_<timestamp>_cmp.png` and collect the unique set
    of frame indices.  Works regardless of whether the existing results came
    from an ego or exo eval run — only the frame index embedded in the
    filename is used, not the camera label.

  Stage 2 — fallback (no depth_benchmark_results found, or --force_fallback):
    Enumerate all entries in the trajectory CSV and subsample with np.linspace
    using --num_frames, replicating the logic in eval_depth_anything_v2.py.
    Pass --num_frames to match the exact count used in the eval run; omit it
    to render every frame in the trajectory.

Output layout (same as render_exocentric_blender.py, existing frames skipped):
  {output_dir}/{cam_name}/videos_rgb/frame_{idx:06d}_{ts}.png
  {output_dir}/{cam_name}/depth_maps/frame_{idx:06d}_{ts}.npy
  {output_dir}/{cam_name}/normal_maps/frame_{idx:06d}_{ts}.npy
  {output_dir}/{cam_name}/segmentation/frame_{idx:06d}_{ts}.npy

Usage:
    # Use depth_benchmark_results to drive the frame list (default):
    python render_exocentric_benchmark_frames.py --base_dir /path/to/seq131

    # Explicitly point at a benchmark results folder:
    python render_exocentric_benchmark_frames.py --base_dir /path/to/seq131 \\
        --benchmark_results_dir /path/to/seq131/depth_benchmark_results/dav2_large_exo-pinhole

    # Force fallback and sample N frames uniformly from the trajectory:
    python render_exocentric_benchmark_frames.py --base_dir /path/to/seq131 \\
        --force_fallback --num_frames 50
"""

import sys, os, csv, json, argparse, subprocess, re
sys.path.insert(0, '/user/f.zhang2/.local/lib/python3.10/site-packages')

import numpy as np
from PIL import Image
import os as _os; _os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2

from adt_render_utils import (
    pick_models_dir, resolve_gt_dir,
    load_all_object_poses, resolve_dynamic_poses,
    build_scene_lights, build_object_list,
    apply_aria_forward_isp, lookat_adt,
    colorize_seg, visualize_normal, visualize_depth,
)

# ── Fixed render config ────────────────────────────────────────────────────────
BLENDER_BIN  = '/user/f.zhang2/blender/blender'
BLEND_SCRIPT = '/user/f.zhang2/projects/adt_egocentric/render/blender_render_scene.py'

# ── Hard-coded exocentric cameras (identical to render_exocentric_blender.py) ─
_WORKSPACE = np.array([1.0, 0.9, 2.4])

EXOCENTRIC_CAMERAS = {
    'from_left': {
        'eye':    np.array([-2.0, 1.7, 2.4]),
        'target': _WORKSPACE,
        'desc':   'From -X (open plan side), looking +X toward cabinet wall  (pitch≈15°, 3.1 m)',
    },
    'from_right': {
        'eye':    np.array([1.8, 1.7, 2.4]),
        'target': np.array([-0.5, 0.9, 2.4]),
        'desc':   'From +X (in front of counter), looking -X toward open plan  (pitch≈19°, 2.4 m)',
    },
    'from_front': {
        'eye':    np.array([0.5, 1.7, 0.5]),
        'target': np.array([0.5, 0.9, 2.5]),
        'desc':   'From -Z (fridge end), looking +Z along kitchen  (pitch≈22°, 2.2 m)',
    },
    'from_back': {
        'eye':    np.array([0.5, 1.7, 4.2]),
        'target': np.array([0.5, 0.9, 2.0]),
        'desc':   'From +Z (back of kitchen), looking -Z toward fridge end  (pitch≈20°, 2.3 m)',
    },
    'from_front_left': {
        'eye':    np.array([-2.0, 1.7, 0.5]),
        'target': np.array([1.5, 0.9, 3.0]),
        'desc':   'From -X/-Z corner, diagonal toward back of counter  (pitch≈11°, 4.4 m)',
    },
    'from_back_left': {
        'eye':    np.array([-2.0, 1.7, 4.5]),
        'target': np.array([1.5, 0.9, 1.5]),
        'desc':   'From -X/+Z corner, diagonal toward fridge end  (pitch≈10°, 4.7 m)',
    },
    'from_front_right': {
        'eye':    np.array([1.5, 1.7, 0.5]),
        'target': np.array([-0.5, 0.9, 3.0]),
        'desc':   'From +X/-Z corner, diagonal toward back of open plan  (pitch≈14°, 3.3 m)',
    },
    'from_back_right': {
        'eye':    np.array([1.5, 1.7, 4.2]),
        'target': np.array([-0.5, 0.9, 2.0]),
        'desc':   'From +X/+Z corner, diagonal toward fridge/open-plan side  (pitch≈15°, 3.1 m)',
    },
}

# Pattern shared by all result files written by eval_depth_anything_v2.py:
#   dav2_<ckpt_tag>_frame_<XXXXXX>_<timestamp_ns>_cmp.png
_CMP_PATTERN = re.compile(r'frame_(\d{6})_\d+_cmp\.png$')


def _remove(path: str) -> None:
    """Delete a file if it exists; silently ignore if already gone."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# ── Frame index discovery ─────────────────────────────────────────────────────

def _collect_from_benchmark_results(bench_root: str) -> list | None:
    """
    Walk bench_root for *_cmp.png files written by eval_depth_anything_v2.py
    and return the sorted list of unique 6-digit frame indices found.
    Returns None when bench_root does not exist or contains no matching files.
    """
    if not os.path.isdir(bench_root):
        return None

    indices: set = set()
    for dirpath, _, filenames in os.walk(bench_root):
        for fname in filenames:
            m = _CMP_PATTERN.search(fname)
            if m:
                indices.add(int(m.group(1)))

    return sorted(indices) if indices else None


def _collect_from_trajectory(traj_len: int, num_frames) -> list:
    """
    Replicate eval_depth_anything_v2.py's np.linspace subsampling.
    Returns a sorted list of frame indices with no duplicates.
    """
    if num_frames is not None and traj_len > num_frames:
        return sorted(set(np.linspace(0, traj_len - 1, num_frames, dtype=int).tolist()))
    return list(range(traj_len))


# ── Per-camera render ─────────────────────────────────────────────────────────

def render_camera(cam_name: str, cam_cfg: dict,
                  frame_idx: int, frame_ts_ns: int,
                  all_objects: list, scene_lights: list,
                  args, tmp_dir: str):
    """Run Blender for one exocentric camera at one frame and save all outputs."""
    out_dir = f'{args.output_dir}/{cam_name}'
    os.makedirs(f'{out_dir}/videos_rgb', exist_ok=True)
    if args.segmentation:
        os.makedirs(f'{out_dir}/segmentation', exist_ok=True)
    if args.normals:
        os.makedirs(f'{out_dir}/normal_maps', exist_ok=True)
    if args.depth:
        os.makedirs(f'{out_dir}/depth_maps', exist_ok=True)

    tag     = f'frame_{frame_idx:06d}_{frame_ts_ns}'
    out_png = f'{out_dir}/videos_rgb/{tag}.png'

    if os.path.exists(out_png):
        print(f'  [{cam_name}] {tag} already exists — skipping')
        return

    T_WC    = lookat_adt(cam_cfg['eye'], cam_cfg['target'])
    cam_pos = np.asarray(cam_cfg['eye'])

    objs_with_dist = sorted(
        [(np.linalg.norm(np.array(o['T_WO']).reshape(4, 4)[:3, 3] - cam_pos), o)
         for o in all_objects],
        key=lambda x: x[0])
    visible_objects = [dict(o) for _, o in objs_with_dist]

    max_dist = objs_with_dist[min(79, len(objs_with_dist) - 1)][0]
    print(f'  [{cam_name}] {len(visible_objects)} objects (max dist {max_dist:.1f} m)')

    pass_idx_to_uid: dict = {}
    for i, obj in enumerate(visible_objects):
        obj['pass_index'] = i + 1
        try:    pass_idx_to_uid[i + 1] = int(obj.get('uid', 0))
        except: pass_idx_to_uid[i + 1] = 0

    fov_deg  = cam_cfg.get('fov_deg', 70.0)
    focal_px = (args.output_size / 2.0) / np.tan(np.radians(fov_deg / 2.0))

    frame_data = {
        'image_width':     args.output_size,
        'image_height':    args.output_size,
        'focal_px':        focal_px,
        'camera_pose':     T_WC.flatten().tolist(),
        'object_models':   visible_objects,
        'scene_lights':    scene_lights,
        'use_equirect':    False,
        'cycles_samples':  args.cycles_samples,
        'cycles_device':   args.cycles_device,
        'cycles_denoiser': args.cycles_denoiser,
        'pass_idx_to_uid': {str(k): v for k, v in pass_idx_to_uid.items()},
    }

    json_path      = f'{tmp_dir}/{cam_name}_{tag}.json'
    seg_exr_tmp    = f'{tmp_dir}/seg_{cam_name}_{tag}.exr'
    normal_exr_tmp = f'{tmp_dir}/normal_{cam_name}_{tag}.exr'
    depth_exr_tmp  = f'{tmp_dir}/depth_{cam_name}_{tag}.exr'

    with open(json_path, 'w') as f:
        json.dump(frame_data, f)

    cmd = [BLENDER_BIN, '--background', '--python', BLEND_SCRIPT,
           '--', '--frame_data', json_path, '--output', out_png]
    if args.segmentation:
        cmd += ['--seg_output',    seg_exr_tmp]
    if args.normals:
        cmd += ['--normal_output', normal_exr_tmp]
    if args.depth:
        cmd += ['--depth_output',  depth_exr_tmp]

    print(f'  [{cam_name}] Rendering {tag} — eye {cam_cfg["eye"].tolist()}')
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    _remove(json_path)   # no longer needed regardless of render outcome
    if not os.path.exists(out_png):
        print(f'  [{cam_name}] Output not created (returncode={result.returncode}).')
        if result.stderr:
            print(f'  stderr: {result.stderr[-800:]}')
        return
    if result.returncode != 0:
        print(f'  [{cam_name}] Blender returned {result.returncode} but output exists — continuing.')

    render_img = apply_aria_forward_isp(np.array(Image.open(out_png).convert('RGB')))
    Image.fromarray(render_img).save(out_png)

    # ── Segmentation ──────────────────────────────────────────────────────
    if args.segmentation:
        if os.path.exists(seg_exr_tmp):
            seg_raw3  = cv2.imread(seg_exr_tmp, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            seg_float = seg_raw3[..., 0] if seg_raw3 is not None else None
            if seg_float is not None:
                pass_arr = np.round(seg_float).astype(np.int32)
                seg_uid  = np.zeros(pass_arr.shape, dtype=np.int64)
                for pidx_s, uid_s in pass_idx_to_uid.items():
                    seg_uid[pass_arr == pidx_s] = uid_s
                np.save(f'{out_dir}/segmentation/{tag}.npy', seg_uid)
                Image.fromarray(colorize_seg(seg_uid)).save(
                    f'{out_dir}/segmentation/{tag}_vis.jpg')
                print(f'  [{cam_name}] Segmentation: {len(np.unique(seg_uid))-1} objects')
        else:
            print(f'  [{cam_name}] [seg] EXR not found at {seg_exr_tmp}')
        _remove(seg_exr_tmp)

    # ── Normals ───────────────────────────────────────────────────────────
    if args.normals:
        if os.path.exists(normal_exr_tmp):
            norm_bgr = cv2.imread(normal_exr_tmp, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if norm_bgr is None:
                print(f'  [{cam_name}] [normal] cv2 could not read EXR')
            else:
                norm_xyz = norm_bgr[..., ::-1].copy()   # BGR → XYZ
                np.save(f'{out_dir}/normal_maps/{tag}.npy', norm_xyz)
                Image.fromarray(visualize_normal(norm_xyz)).save(
                    f'{out_dir}/normal_maps/{tag}_vis.jpg')
                print(f'  [{cam_name}] Normal map saved')
        else:
            print(f'  [{cam_name}] [normal] EXR not found at {normal_exr_tmp}')
        _remove(normal_exr_tmp)

    # ── Depth ─────────────────────────────────────────────────────────────
    if args.depth:
        if os.path.exists(depth_exr_tmp):
            depth_bgr = cv2.imread(depth_exr_tmp, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if depth_bgr is None:
                print(f'  [{cam_name}] [depth] cv2 could not read EXR')
            else:
                depth = depth_bgr[..., 0].astype(np.float32)
                depth[depth > 1e6] = np.inf
                np.save(f'{out_dir}/depth_maps/{tag}.npy', depth)
                Image.fromarray(visualize_depth(depth)).save(
                    f'{out_dir}/depth_maps/{tag}_vis.jpg')
                finite_cnt = np.isfinite(depth).sum()
                median_d = (float(np.nanmedian(depth[np.isfinite(depth)]))
                            if finite_cnt > 0 else 0.0)
                print(f'  [{cam_name}] Depth: median {median_d:.2f}m saved')
        else:
            print(f'  [{cam_name}] [depth] EXR not found at {depth_exr_tmp}')
        _remove(depth_exr_tmp)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--base_dir', type=str,
                        # default=None,
                        default='/user/f.zhang2/Documents/projectaria_tools_adt_data_clean/Apartment_release_clean_seq131_M1292',
                        # default='/user/f.zhang2/Documents/projectaria_tools_adt_data_clean/Apartment_release_decoration_seq132_M1292',
                        help='Sequence base directory (default: seq131 path).')
    parser.add_argument('--benchmark_results_dir', type=str, default=None,
                        help='Root of depth_benchmark_results to read frame indices '
                             'from (default: {base_dir}/depth_benchmark_results). '
                             'Accepts a specific run subdirectory '
                             '(e.g. .../dav2_large_exo-pinhole) or the parent '
                             'depth_benchmark_results/ folder — both are scanned '
                             'recursively for *_cmp.png files.')
    parser.add_argument('--force_fallback', action='store_true',
                        help='Skip Stage 1 and always use trajectory subsampling.')
    parser.add_argument('--num_frames', type=int, default=None,
                        help='Fallback only: number of frames to sample uniformly '
                             'from the trajectory (mirrors eval --num_frames). '
                             'Omit to render every trajectory frame.')
    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument('--output_size',   type=int, default=1408)
    parser.add_argument('--output_dir',    type=str, default=None,
                        help='Output root (default: {base_dir}/exocentric_rendered).')
    parser.add_argument('--no_segmentation', action='store_true')
    parser.add_argument('--no_normals',      action='store_true')
    parser.add_argument('--no_depth',        action='store_true')
    # ── Render quality ────────────────────────────────────────────────────
    parser.add_argument('--cycles_samples',  type=int, default=32)
    parser.add_argument('--cycles_device',   type=str, default='OPTIX',
                        choices=['CPU', 'CUDA', 'OPTIX', 'HIP', 'METAL'])
    parser.add_argument('--cycles_denoiser', type=str, default='NONE',
                        choices=['NONE', 'OPENIMAGEDENOISE', 'OPTIX'])

    args = parser.parse_args()
    args.segmentation = not args.no_segmentation
    args.normals      = not args.no_normals
    args.depth        = not args.no_depth

    # ── Resolve paths ─────────────────────────────────────────────────────
    BASE = (args.base_dir or
            '/user/f.zhang2/Documents/projectaria_tools_adt_data_clean/'
            'Apartment_release_clean_seq131_M1292')
    _adt_root  = os.path.dirname(BASE)
    MODELS_DIR = pick_models_dir(os.path.join(_adt_root, 'object_models'),
                                 os.path.join(BASE, 'object_models'))
    GT_DIR     = resolve_gt_dir(BASE)
    if args.output_dir is None:
        args.output_dir = f'{BASE}/exocentric_rendered'
    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = f'{args.output_dir}/_tmp'
    os.makedirs(tmp_dir, exist_ok=True)

    # ── Load trajectory ───────────────────────────────────────────────────
    print('Loading ADT scene data...')
    with open(f'{GT_DIR}/instances.json') as f:
        instances = json.load(f)

    traj_rows = []
    with open(f'{GT_DIR}/aria_trajectory.csv') as f:
        for row in csv.DictReader(f):
            traj_rows.append(int(row['tracking_timestamp_us']))
    print(f'  Trajectory: {len(traj_rows)} frames')

    # ── Stage 1: derive frame list from depth_benchmark_results ──────────
    frame_indices = None

    if not args.force_fallback:
        bench_root = (args.benchmark_results_dir
                      or os.path.join(BASE, 'depth_benchmark_results'))
        frame_indices = _collect_from_benchmark_results(bench_root)
        if frame_indices is not None:
            print(f'  [Stage 1] {len(frame_indices)} unique frame indices '
                  f'from {bench_root}')
        else:
            print(f'  [Stage 1] No *_cmp.png files found under {bench_root} '
                  f'— falling back to trajectory sampling')

    # ── Stage 2 fallback: np.linspace over trajectory ────────────────────
    if frame_indices is None:
        frame_indices = _collect_from_trajectory(len(traj_rows), args.num_frames)
        print(f'  [Stage 2 fallback] {len(frame_indices)} frames sampled '
              f'(num_frames={args.num_frames})')

    # Drop indices that exceed trajectory length (guard against stale results)
    valid = [i for i in frame_indices if i < len(traj_rows)]
    if len(valid) < len(frame_indices):
        print(f'  [WARN] Dropped {len(frame_indices) - len(valid)} indices '
              f'that exceed trajectory length ({len(traj_rows)})')
    frame_indices = valid

    if not frame_indices:
        print('[ERROR] No valid frame indices to render.')
        sys.exit(1)

    n_cams   = len(EXOCENTRIC_CAMERAS)
    n_total  = len(frame_indices) * n_cams
    print(f'\n  Frames to render : {len(frame_indices)} '
          f'(indices {frame_indices[0]}–{frame_indices[-1]})')
    print(f'  Cameras          : {n_cams} (all exocentric)')
    print(f'  Total renders    : {n_total}')
    print(f'  Output dir       : {args.output_dir}')

    # ── Load static scene data (constant across frames) ───────────────────
    static_poses, dyn_poses = load_all_object_poses(f'{GT_DIR}/scene_objects.csv')
    static_obj_list = build_object_list(instances, static_poses, MODELS_DIR)
    print('Building scene lights...')
    scene_lights = build_scene_lights(f'{GT_DIR}/scene_objects.csv',
                                      f'{GT_DIR}/instances.json',
                                      lighting_mode='exocentric')
    print(f'  Models dir : {MODELS_DIR}')
    if not static_obj_list:
        print(f'  WARNING: No GLBs resolved. Expected: {MODELS_DIR}/{{name}}.glb')

    # ── Render loop ───────────────────────────────────────────────────────
    done = 0
    for frame_idx in frame_indices:
        frame_ts_ns     = traj_rows[frame_idx] * 1000   # µs → ns
        dyn_poses_frame = resolve_dynamic_poses(dyn_poses, frame_ts_ns)
        dyn_obj_list    = build_object_list(instances, dyn_poses_frame, MODELS_DIR)
        all_objects     = static_obj_list + dyn_obj_list

        print(f'\nFrame {frame_idx} (ts={frame_ts_ns} ns) — '
              f'static={len(static_obj_list)} dynamic={len(dyn_obj_list)}')

        for cam_name, cam_cfg in EXOCENTRIC_CAMERAS.items():
            done += 1
            eye = cam_cfg['eye'];  tgt = cam_cfg['target']
            dist  = float(np.linalg.norm(tgt - eye))
            pitch = np.degrees(np.arcsin(-(tgt - eye)[1] / dist))
            print(f'\n  [{done}/{n_total}] {cam_name}  '
                  f'eye=({eye[0]:.2f},{eye[1]:.2f},{eye[2]:.2f})  '
                  f'dist={dist:.1f}m  pitch={pitch:.1f}°')
            render_camera(cam_name, cam_cfg, frame_idx, frame_ts_ns,
                          all_objects, scene_lights, args, tmp_dir)

    # Remove _tmp if it is now empty (all per-render files were cleaned up inline)
    try:
        os.rmdir(tmp_dir)
        print(f'  Removed empty tmp dir: {tmp_dir}')
    except OSError:
        pass  # non-empty (e.g. a crashed render left files) or already gone

    print(f'\nAll done! Outputs in {args.output_dir}/')


if __name__ == '__main__':
    main()