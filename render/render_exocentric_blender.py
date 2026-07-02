"""
render_exocentric_blender.py
============================
Render the ADT Apartment scene from hard-coded exocentric (third-person)
camera poses using Blender Cycles + blender_render_scene.py.

Eight eye-level cameras cover the kitchen workspace:

Four orthogonal views (face one wall each):
  from_left        eye=(-2.0, 1.7, 2.4) → looks +X toward cabinet wall
  from_right       eye=(+1.8, 1.7, 2.4) → looks -X from in front of counter
  from_front       eye=(+0.5, 1.7, 0.5) → looks +Z from the fridge end
  from_back        eye=(+0.5, 1.7, 4.2) → looks -Z from back of kitchen

Four diagonal views (cover corners):
  from_front_left  eye=(-2.0, 1.7, 0.5) → diagonal toward back-right
  from_back_left   eye=(-2.0, 1.7, 4.5) → diagonal toward front-right
  from_front_right eye=(+1.5, 1.7, 0.5) → diagonal toward back-left
  from_back_right  eye=(+1.5, 1.7, 4.2) → diagonal toward front-left

Usage:
    python render_exocentric_blender.py --base_dir /path/to/sequence
    python render_exocentric_blender.py --base_dir /path/to/seq --camera from_left
    python render_exocentric_blender.py --base_dir /path/to/seq --camera all
    python render_exocentric_blender.py --base_dir /path/to/seq --no_segmentation

    Or call via run_sequences.py to render multiple sequences in one go.

Does NOT require the Aria VRS file — the scene timestamp for dynamic objects
is taken directly from the trajectory CSV.
"""

import sys, os, csv, json, argparse, subprocess
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


# ── Hard-coded exocentric cameras ─────────────────────────────────────────────
# All cameras observe the kitchen workspace (ego trajectory centroid ≈ (0.10, 1.57, 2.36)).
# Eye height Y=1.7 m, downward pitch 10–22°, FOV=70° (overridable via 'fov_deg').
# Positions verified clear of all static object centroids (±0.5 m cube).
# ADT Y-up: +X toward cabinet wall, +Z toward back of kitchen, +Y up.

_WORKSPACE = np.array([1.0, 0.9, 2.4])

EXOCENTRIC_CAMERAS = {
    # ── Orthogonal views ──────────────────────────────────────────────────
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
    # ── Diagonal / corner views ───────────────────────────────────────────
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


# ── Per-camera render ─────────────────────────────────────────────────────────

def render_camera(cam_name: str, cam_cfg: dict, frame_ts_ns: int,
                  all_objects: list, scene_lights: list,
                  args, tmp_dir: str):
    """Run Blender for one exocentric camera and save all requested outputs.

    Output layout mirrors render_from_poses_blender.py pinhole structure:
      {output_dir}/{cam_name}/videos_rgb/{tag}.png
      {output_dir}/{cam_name}/segmentation/{tag}.npy + {tag}_vis.png
      {output_dir}/{cam_name}/normal_maps/{tag}.npy  + {tag}_vis.png
      {output_dir}/{cam_name}/depth_maps/{tag}.npy   + {tag}_vis.png
    """
    out_dir = f'{args.output_dir}/{cam_name}'
    os.makedirs(f'{out_dir}/videos_rgb', exist_ok=True)
    if args.segmentation:
        os.makedirs(f'{out_dir}/segmentation', exist_ok=True)
    if args.normals:
        os.makedirs(f'{out_dir}/normal_maps', exist_ok=True)
    if args.depth:
        os.makedirs(f'{out_dir}/depth_maps', exist_ok=True)

    tag     = f'frame_{args.frame_idx:06d}_{frame_ts_ns}'
    out_png = f'{out_dir}/videos_rgb/{tag}.png'

    if os.path.exists(out_png):
        print(f'  [{cam_name}] {tag} already exists — skipping')
        return

    T_WC    = lookat_adt(cam_cfg['eye'], cam_cfg['target'])
    cam_pos = np.asarray(cam_cfg['eye'])

    # Sort objects by distance to camera
    objs_with_dist = sorted(
        [(np.linalg.norm(np.array(o['T_WO']).reshape(4,4)[:3,3] - cam_pos), o)
         for o in all_objects],
        key=lambda x: x[0])
    visible_objects = [dict(o) for _, o in objs_with_dist]   # shallow copy per object

    max_dist = objs_with_dist[min(79, len(objs_with_dist)-1)][0]
    print(f'  [{cam_name}] {len(visible_objects)} objects (max dist {max_dist:.1f} m)')

    # Assign sequential pass_index for segmentation
    pass_idx_to_uid: dict[int, int] = {}
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
    if not os.path.exists(out_png):
        print(f'  [{cam_name}] Output not created (returncode={result.returncode}).')
        if result.stderr:
            print(f'  stderr: {result.stderr[-800:]}')
        return
    if result.returncode != 0:
        print(f'  [{cam_name}] Blender returned {result.returncode} but output exists — continuing.')

    # RGB: apply Aria forward ISP for colour consistency
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
                seg_npy = f'{out_dir}/segmentation/{tag}.npy'
                seg_vis = f'{out_dir}/segmentation/{tag}_vis.png'
                np.save(seg_npy, seg_uid)
                Image.fromarray(colorize_seg(seg_uid)).save(seg_vis)
                print(f'  [{cam_name}] Segmentation: {len(np.unique(seg_uid))-1} objects')
        else:
            print(f'  [{cam_name}] [seg] EXR not found at {seg_exr_tmp}')

    # ── Normals ───────────────────────────────────────────────────────────
    # Exocentric uses the Blender Normal pass (world-space XYZ EXR) directly —
    # no fisheye remap needed.  BGR → XYZ flip applied on read.
    if args.normals:
        if os.path.exists(normal_exr_tmp):
            norm_bgr = cv2.imread(normal_exr_tmp, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if norm_bgr is None:
                print(f'  [{cam_name}] [normal] cv2 could not read EXR')
            else:
                norm_xyz     = norm_bgr[..., ::-1].copy()   # BGR → XYZ
                norm_npy_out = f'{out_dir}/normal_maps/{tag}.npy'
                norm_vis_out = f'{out_dir}/normal_maps/{tag}_vis.png'
                np.save(norm_npy_out, norm_xyz)
                Image.fromarray(visualize_normal(norm_xyz)).save(norm_vis_out)
                print(f'  [{cam_name}] Normal map → {norm_npy_out}')
        else:
            print(f'  [{cam_name}] [normal] EXR not found at {normal_exr_tmp}')

    # ── Depth ─────────────────────────────────────────────────────────────
    if args.depth:
        if os.path.exists(depth_exr_tmp):
            depth_bgr = cv2.imread(depth_exr_tmp, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if depth_bgr is None:
                print(f'  [{cam_name}] [depth] cv2 could not read EXR')
            else:
                depth = depth_bgr[..., 0].astype(np.float32)
                depth[depth > 1e6] = np.inf
                depth_npy_out = f'{out_dir}/depth_maps/{tag}.npy'
                depth_vis_out = f'{out_dir}/depth_maps/{tag}_vis.png'
                np.save(depth_npy_out, depth)
                Image.fromarray(visualize_depth(depth)).save(depth_vis_out)
                finite_cnt = np.isfinite(depth).sum()
                median_d = (float(np.nanmedian(depth[np.isfinite(depth)]))
                            if finite_cnt > 0 else 0.0)
                print(f'  [{cam_name}] Depth: median {median_d:.2f}m → {depth_npy_out}')
        else:
            print(f'  [{cam_name}] [depth] EXR not found at {depth_exr_tmp}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cam_names = list(EXOCENTRIC_CAMERAS.keys())

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # ── Sequence path ─────────────────────────────────────────────────────
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Sequence base directory. '
                             'Defaults to the hardcoded golden_skeleton_seq100 path.')
    # ── Camera selection ──────────────────────────────────────────────────
    parser.add_argument('--camera',      choices=cam_names + ['all'], default='from_right',
                        help='Which exocentric camera to render (default: from_right)')
    parser.add_argument('--frame_idx',   type=int, default=0,
                        help='Scene frame index to render (default: 0)')
    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument('--output_size', type=int, default=1408)
    parser.add_argument('--output_dir',  type=str, default=None,
                        help='Output directory (default: {base_dir}/exocentric_rendered)')
    parser.add_argument('--no_segmentation', action='store_true')
    parser.add_argument('--no_normals',      action='store_true')
    parser.add_argument('--no_depth',        action='store_true')
    # ── Render quality ────────────────────────────────────────────────────
    parser.add_argument('--cycles_samples',  type=int, default=32)
    parser.add_argument('--cycles_device',   type=str, default='OPTIX',
                        choices=['CPU', 'CUDA', 'OPTIX', 'HIP', 'METAL'])
    parser.add_argument('--cycles_denoiser', type=str, default='NONE',
                        choices=['NONE', 'OPENIMAGEDENOISE', 'OPTIX'])
    # ── Manual pose overrides (single-camera mode only) ───────────────────
    pose_grp = parser.add_argument_group(
        'manual pose overrides',
        'Override the hard-coded pose for the selected camera. '
        'Cannot be combined with --camera all.')
    pose_grp.add_argument('--eye',    type=float, nargs=3, metavar=('X','Y','Z'), default=None)
    pose_grp.add_argument('--target', type=float, nargs=3, metavar=('X','Y','Z'), default=None)
    pose_grp.add_argument('--fov',    type=float, metavar='DEGREES', default=None)

    args = parser.parse_args()
    args.segmentation = not args.no_segmentation
    args.normals      = not args.no_normals
    args.depth        = not args.no_depth

    has_override = args.eye is not None or args.target is not None or args.fov is not None
    if has_override and args.camera == 'all':
        parser.error('--eye / --target / --fov cannot be combined with --camera all.')

    # ── Resolve paths ─────────────────────────────────────────────────────
    BASE = (args.base_dir or
            '/user/f.zhang2/Documents/projectaria_tools_adt_data/'
            'Apartment_release_golden_skeleton_seq100_10s_sample_M1292')
    _adt_root  = os.path.dirname(BASE)
    MODELS_DIR = pick_models_dir(os.path.join(_adt_root, 'object_models'),
                                 os.path.join(BASE, 'object_models'))
    GT_DIR     = resolve_gt_dir(BASE)
    if args.output_dir is None:
        args.output_dir = f'{BASE}/exocentric_rendered'

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = f'{args.output_dir}/_tmp'
    os.makedirs(tmp_dir, exist_ok=True)

    # ── Load GT data ──────────────────────────────────────────────────────
    print('Loading ADT scene data...')
    with open(f'{GT_DIR}/instances.json') as f:
        instances = json.load(f)

    # Get frame timestamp from trajectory CSV (no VRS needed)
    traj_rows = []
    with open(f'{GT_DIR}/aria_trajectory.csv') as f:
        for row in csv.DictReader(f):
            traj_rows.append(int(row['tracking_timestamp_us']))
    if args.frame_idx >= len(traj_rows):
        print(f'Warning: frame_idx={args.frame_idx} > trajectory length '
              f'({len(traj_rows)}). Using last row.')
        args.frame_idx = len(traj_rows) - 1
    frame_ts_ns = traj_rows[args.frame_idx] * 1000   # µs → ns
    print(f'  Frame {args.frame_idx}: timestamp {frame_ts_ns} ns')
    print(f'  Models dir: {MODELS_DIR}')

    static_poses, dyn_poses = load_all_object_poses(f'{GT_DIR}/scene_objects.csv')
    dyn_poses_frame         = resolve_dynamic_poses(dyn_poses, frame_ts_ns)
    static_obj_list         = build_object_list(instances, static_poses,   MODELS_DIR)
    dyn_obj_list            = build_object_list(instances, dyn_poses_frame, MODELS_DIR)
    all_objects             = static_obj_list + dyn_obj_list
    print(f'  Static: {len(static_obj_list)}  Dynamic (at frame): {len(dyn_obj_list)}')
    if not static_obj_list:
        print(f'  WARNING: No GLBs resolved. Expected: {MODELS_DIR}/{{name}}.glb')

    print('Building scene lights...')
    scene_lights = build_scene_lights(f'{GT_DIR}/scene_objects.csv',
                                      f'{GT_DIR}/instances.json',
                                      lighting_mode='exocentric')

    # ── Camera selection + optional pose overrides ────────────────────────
    selected = (list(EXOCENTRIC_CAMERAS.items()) if args.camera == 'all'
                else [(args.camera, EXOCENTRIC_CAMERAS[args.camera])])

    if has_override:
        cam_name, cam_cfg = selected[0]
        cam_cfg = dict(cam_cfg)
        if args.eye    is not None: cam_cfg['eye']    = np.array(args.eye)
        if args.target is not None: cam_cfg['target'] = np.array(args.target)
        if args.fov    is not None: cam_cfg['fov_deg']= float(args.fov)
        cam_cfg['desc'] += ' [manually overridden]'
        selected = [(cam_name, cam_cfg)]

    print(f'\nRendering {len(selected)} camera(s)...')
    for cam_name, cam_cfg in selected:
        eye  = cam_cfg['eye'];  tgt = cam_cfg['target']
        dist = float(np.linalg.norm(tgt - eye))
        fwd  = (tgt - eye) / dist
        pitch = np.degrees(np.arcsin(-fwd[1]))
        print(f'\nCamera: {cam_name}')
        print(f'  {cam_cfg["desc"]}')
        print(f'  eye=({eye[0]:.3f}, {eye[1]:.3f}, {eye[2]:.3f})  '
              f'target=({tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f})  '
              f'dist={dist:.2f}m  pitch={pitch:.1f}°')
        render_camera(cam_name, cam_cfg, frame_ts_ns, all_objects,
                      scene_lights, args, tmp_dir)

    print(f'\nAll done! Outputs in {args.output_dir}/')


if __name__ == '__main__':
    main()