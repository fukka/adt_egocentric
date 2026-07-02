"""
ADT Object-Pose → Blender Rendering Pipeline
=============================================
Drives blender_render_scene.py for each frame:
  1. Reads ADT ground truth object 6DoF poses and camera trajectory
  2. Exports per-frame JSON (camera pose + object GLB paths + T_WO)
  3. Calls Blender headless to render each frame
Output labels:
  • Instance segmentation map  (object UIDs as int64 .npy + colourised PNG)
  • Surface normal map         (world-space XYZ float32 .npy + visualised PNG)
  • Depth map                  (camera-space distance float32 .npy + visualised PNG)

Usage:
    python render_from_poses_blender.py --base_dir /path/to/sequence \\
     [--start_frame N] [--end_frame N] [--frame_step K] [--fast_render] \\
     [--output_size S] [--focal F] \\
     [--fisheye] \\
     [--no_segmentation] [--no_normals] [--no_depth] \\
     [--cycles_device OPTIX] [--cycles_samples 32] \\
     [--output_dir /path/to/output]

    Or call via run_sequences.py to render multiple sequences in one go.

Normal map convention:
    Camera-space surface normals in OpenCV convention (+X right, +Y down, +Z forward).
    Saved as float32 (H, W, 3) in XYZ order, unit-length.
    Z < 0 for surfaces facing the camera (toward-camera convention).
    Background pixels are stored as (0, 0, 0).

Depth map convention:
    Camera-space distance in metres.  Background pixels = np.inf.
    Saved as float32 (H, W).
"""

import sys, os, json, argparse, subprocess
sys.path.insert(0, '/user/f.zhang2/.local/lib/python3.10/site-packages')

import numpy as np
from PIL import Image
import os as _os; _os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2

from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId

from adt_render_utils import (
    pick_models_dir, resolve_gt_dir,
    load_trajectory, nearest_pose,
    load_all_object_poses, resolve_dynamic_poses,
    build_scene_lights, build_object_list,
    apply_aria_forward_isp,
    colorize_seg, visualize_normal, visualize_depth,
)

# ── Fixed render config (override via CLI if needed) ──────────────────────────
BLENDER_BIN  = '/user/f.zhang2/blender/blender'
BLEND_SCRIPT = '/user/f.zhang2/projects/adt_egocentric/render/blender_render_scene.py'
RGB_STREAM   = StreamId('214-1')

# FRAME_STEP            = 30   # render every Kth frame (~1 fps at 30 Hz VRS)
FRAME_STEP            = 1   # render every Kth frame (~1 fps at 30 Hz VRS)
FAST_RENDER_N_OBJECTS = 5    # keep only N largest GLB objects in fast mode

FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])


# ── FISHEYE624 remap helpers ──────────────────────────────────────────────────

def build_fisheye624_remap(cam_calib, out_w, out_h, eq_w, eq_h):
    """Precompute a pixel-level lookup table: fisheye pixel (u,v) → equirect (x,y).

    The Aria FISHEYE624 model (Kannala-Brandt style):
        r = fx * θ * (1 + k0*θ² + k1*θ⁴ + k2*θ⁶ + k3*θ⁸ + k4*θ¹⁰ + k5*θ¹²)

    Steps per output pixel:
      1. Invert FISHEYE624 projection → ray in Aria camera space (+Z forward)
      2. Convert to Blender camera space via FLIP_YZ (+Y up, -Z forward)
      3. Convert to equirectangular azimuth/elevation → pixel in eq render

    Returns (map_x, map_y, valid_mask, ray_dirs) as float32 arrays (out_h, out_w).
    ray_dirs: unit ray directions in ADT camera space, used to convert panoramic
    ray-distance depth to perpendicular Z depth for normal computation.
    """
    params     = cam_calib.get_projection_params()
    fx         = params[0]
    cx, cy     = params[1], params[2]
    k0, k1, k2, k3, k4, k5 = params[3:9]

    # Scale calibration if output differs from native 1408 px
    native_size = float(cam_calib.get_image_size()[0])
    scale       = out_w / native_size
    fx_s        = fx * scale
    cx_s        = cx * scale + (scale - 1) * 0.5
    cy_s        = cy * scale + (scale - 1) * 0.5
    valid_r_s   = cam_calib.get_valid_radius() * scale

    u = np.arange(out_w, dtype=np.float64)
    v = np.arange(out_h, dtype=np.float64)
    UU, VV = np.meshgrid(u, v)
    u_hat = (UU - cx_s).ravel()
    v_hat = (VV - cy_s).ravel()
    r = np.sqrt(u_hat**2 + v_hat**2)

    # Newton-Raphson: solve r = fx_s * θ * poly(θ²) for θ
    theta = r / fx_s
    for _ in range(25):
        t2    = theta * theta
        poly  = 1.0 + t2 * (k0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * (k4 + t2 * k5)))))
        dpoly = t2 * (2*k0 + t2 * (4*k1 + t2 * (6*k2 + t2 * (8*k3 + t2 * (10*k4 + t2 * 12*k5)))))
        fval  = fx_s * theta * poly - r
        dfval = fx_s * (poly + dpoly)
        theta -= fval / (dfval + 1e-12)
        theta  = np.clip(theta, 0.0, np.pi)

    sin_t, cos_t = np.sin(theta), np.cos(theta)
    r_safe = np.maximum(r, 1e-9)
    # Ray in Aria camera space (+X right, +Y down, +Z forward)
    ray_x =  sin_t * (u_hat / r_safe)
    ray_y =  sin_t * (v_hat / r_safe)
    ray_z =  cos_t
    # Convert to Blender camera space (+X right, +Y up, -Z forward)
    rx_bl =  ray_x
    ry_bl = -ray_y
    rz_bl = -ray_z

    # Equirectangular convention (Blender EQUIRECTANGULAR panoramic):
    #   longitude = atan2(+X, -Z_local),  latitude = atan2(+Y, sqrt(X²+Z²))
    lon = np.arctan2(rx_bl, -rz_bl)
    lat = np.arctan2(ry_bl, np.sqrt(rx_bl**2 + rz_bl**2))
    map_x = ((lon + np.pi) / (2 * np.pi) * eq_w).astype(np.float32)
    map_y = ((np.pi / 2 - lat) / np.pi    * eq_h).astype(np.float32)

    valid    = (r < valid_r_s)
    ray_dirs = np.stack([ray_x, ray_y, ray_z],
                        axis=-1).reshape(out_h, out_w, 3).astype(np.float32)
    return (map_x.reshape(out_h, out_w),
            map_y.reshape(out_h, out_w),
            valid.reshape(out_h, out_w),
            ray_dirs)


def remap_equirect_to_fisheye(equirect_img, map_x, map_y, valid_mask):
    """Apply precomputed remap: equirectangular → fisheye numpy array."""
    if isinstance(equirect_img, Image.Image):
        equirect_arr = np.array(equirect_img.convert('RGB'), dtype=np.uint8)
    else:
        equirect_arr = np.asarray(equirect_img, dtype=np.uint8)
    out = cv2.remap(equirect_arr, map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    out[~valid_mask] = 0
    return out


# ── Normal computation from depth ─────────────────────────────────────────────

def _normals_from_depth_fisheye(depth: np.ndarray,
                                ray_dirs: np.ndarray,
                                valid_mask: np.ndarray) -> np.ndarray:
    """Compute surface normals from a fisheye depth map (perpendicular Z depth).

    Back-projects each pixel using pre-computed unit ray directions, then
    estimates normals via central-difference cross-products.

    depth      : (H, W) float32 — perpendicular Z depth in metres (inf/nan = bg)
    ray_dirs   : (H, W, 3) float32 — unit ray directions in ADT camera space
    valid_mask : (H, W) bool — pixels inside the fisheye circle

    Returns (H, W, 3) float32 unit normals in ADT camera space, Z < 0 for
    camera-facing surfaces; invalid pixels are NaN.
    """
    ray_z   = ray_dirs[..., 2].astype(np.float64)
    d       = depth.astype(np.float64)
    bad     = (~np.isfinite(d)) | (d <= 0) | (~valid_mask) | (ray_z < 1e-4)
    ray_z_s = np.where(bad, 1.0, ray_z)
    P = np.where(bad[..., None], np.nan,
                 (d / ray_z_s)[..., None] * ray_dirs.astype(np.float64))
    dPu = np.full_like(P, np.nan);  dPu[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dPv = np.full_like(P, np.nan);  dPv[1:-1, :] = P[2:, :] - P[:-2, :]
    normals = np.cross(dPu, dPv)
    normals[normals[..., 2] > 0] *= -1
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    valid = (norms[..., 0] > 1e-8) & np.isfinite(norms[..., 0])
    normals = np.where(valid[..., None], normals / (norms + 1e-8), np.nan)
    return normals.astype(np.float32)


def _normals_from_depth_pinhole(depth: np.ndarray,
                                fx: float, fy: float,
                                cx: float, cy: float) -> np.ndarray:
    """Compute surface normals from a pinhole depth map (perpendicular Z depth).

    Identical protocol to depth_to_normals() in eval_utils.py so that saved
    normals exactly match applying that function to the saved depth.

    Returns (H, W, 3) float32 unit normals in ADT camera space, Z < 0 for
    camera-facing surfaces; invalid pixels are NaN.
    """
    H, W = depth.shape
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    d   = depth.astype(np.float64)
    bad = ~np.isfinite(d) | (d <= 0)
    X   = np.where(bad, np.nan, (uu - cx) * d / fx)
    Y   = np.where(bad, np.nan, (vv - cy) * d / fy)
    Z   = np.where(bad, np.nan, d)
    P   = np.stack([X, Y, Z], axis=-1)
    dPu = np.full_like(P, np.nan);  dPu[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dPv = np.full_like(P, np.nan);  dPv[1:-1, :] = P[2:, :] - P[:-2, :]
    normals = np.cross(dPu, dPv)
    normals[normals[..., 2] > 0] *= -1
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    valid = (norms[..., 0] > 1e-8) & np.isfinite(norms[..., 0])
    normals = np.where(valid[..., None], normals / (norms + 1e-8), np.nan)
    return normals.astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # ── Sequence path ─────────────────────────────────────────────────────
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Sequence base directory '
                             '(e.g. .../Apartment_release_clean_seq131_M1292). '
                             'Defaults to the hardcoded seq131 path.')
    # ── Frame selection ───────────────────────────────────────────────────
    parser.add_argument('--start_frame', type=int, default=None,
                        help='First VRS frame index (default: auto from trajectory)')
    parser.add_argument('--end_frame',   type=int, default=None,
                        help='Last VRS frame index, exclusive (default: auto)')
    parser.add_argument('--frame_step',  type=int, default=FRAME_STEP,
                        help=f'Render every Kth frame (default: {FRAME_STEP})')
    parser.add_argument('--fast_render', action='store_true',
                        help=f'Keep only {FAST_RENDER_N_OBJECTS} largest GLB objects')
    parser.add_argument('--max_glb_mb', type=float, default=0,
                        help='Cap total GLB memory per frame in MB (0 = no cap)')
    # ── Render quality ────────────────────────────────────────────────────
    parser.add_argument('--cycles_samples', type=int, default=32,
                        help='Cycles samples per pixel (default: 32)')
    parser.add_argument('--cycles_device', type=str, default='OPTIX',
                        choices=['CPU', 'CUDA', 'OPTIX', 'HIP', 'METAL'],
                        help='Cycles compute device (default: CPU)')
    parser.add_argument('--cycles_denoiser', type=str, default='NONE',
                        choices=['NONE', 'OPENIMAGEDENOISE', 'OPTIX'],
                        help='Cycles denoiser — allows fewer samples (default: NONE)')
    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument('--output_size', type=int, default=1408)
    parser.add_argument('--focal',       type=float, default=None,
                        help='Focal length in pixels at output_size '
                             '(default: scaled from ADT calibration 611@1408)')
    parser.add_argument('--output_dir',  type=str, default=None,
                        help='Output directory (default: {base_dir}/blender_rendered_maps)')
    parser.add_argument('--fisheye',     action='store_true',
                        help='Render equirectangular and remap to Aria FISHEYE624')
    parser.add_argument('--equirect_scale', type=int, default=2,
                        help='Equirect width = output_size × this (default: 2). '
                             'Use 4 only if remap aliasing is visible.')
    # ── Auxiliary map toggles ─────────────────────────────────────────────
    parser.add_argument('--no_segmentation', action='store_true')
    parser.add_argument('--no_normals',      action='store_true')
    parser.add_argument('--no_depth',        action='store_true')
    parser.add_argument('--blender_normals', action='store_true',
                        help='Use Blender Normal pass instead of depth-derived normals '
                             '(debug; not consistent with depth_to_normals())')
    args = parser.parse_args()
    args.segmentation = not args.no_segmentation
    args.normals      = not args.no_normals
    args.depth        = not args.no_depth

    # ── Resolve paths from base_dir ───────────────────────────────────────
    # BASE = (args.base_dir or
    #         '/user/f.zhang2/Documents/projectaria_tools_adt_data/'
    #         'Apartment_release_clean_seq131_M1292')
    BASE = (args.base_dir or
            '/user/f.zhang2/Documents/projectaria_tools_adt_data_clean/'
            'Apartment_release_clean_seq131_M1292')
    EGO_VRS        = f'{BASE}/video.vrs'
    # _adt_root      = os.path.dirname(BASE)
    _adt_root = '/user/f.zhang2/Documents/projectaria_tools_adt_data'
    MODELS_DIR     = pick_models_dir(os.path.join(_adt_root, 'object_models'),
                                     os.path.join(BASE, 'object_models'))
    GT_DIR         = resolve_gt_dir(BASE)
    if args.output_dir is None:
        args.output_dir = f'{BASE}/blender_rendered_maps'

    # ── Output directory structure ────────────────────────────────────────
    if args.fisheye:
        os.makedirs(f'{args.output_dir}/equirect', exist_ok=True)
        os.makedirs(f'{args.output_dir}/fisheye/videos_rgb', exist_ok=True)
        if args.segmentation:
            os.makedirs(f'{args.output_dir}/fisheye/segmentation', exist_ok=True)
        if args.normals:
            os.makedirs(f'{args.output_dir}/fisheye/normal_maps', exist_ok=True)
        if args.depth:
            os.makedirs(f'{args.output_dir}/fisheye/depth_maps', exist_ok=True)
    else:
        os.makedirs(f'{args.output_dir}/pinhole/videos_rgb', exist_ok=True)
        if args.segmentation:
            os.makedirs(f'{args.output_dir}/pinhole/segmentation', exist_ok=True)
        if args.normals:
            os.makedirs(f'{args.output_dir}/pinhole/normal_maps', exist_ok=True)
        if args.depth:
            os.makedirs(f'{args.output_dir}/pinhole/depth_maps', exist_ok=True)
    tmp_dir = f'{args.output_dir}/_tmp'
    os.makedirs(tmp_dir, exist_ok=True)

    # ── Load VRS and calibration ──────────────────────────────────────────
    print('Loading VRS and ground truth...')
    p_ego        = data_provider.create_vrs_data_provider(EGO_VRS)
    calib        = p_ego.get_device_calibration()
    cam_calib    = calib.get_camera_calib('camera-rgb')
    T_DC         = calib.get_camera_calib('camera-rgb').get_transform_device_camera().to_matrix()
    n_frames     = p_ego.get_num_data(RGB_STREAM)

    if args.focal is None:
        calib_focal = cam_calib.get_focal_lengths()[0]
        calib_size  = cam_calib.get_image_size()[0]
        args.focal  = calib_focal * (args.output_size / calib_size)
        print(f'  Auto focal: {calib_focal:.1f}px @ {calib_size}px '
              f'→ {args.focal:.1f}px @ {args.output_size}px')

    # ── Precompute fisheye remap ──────────────────────────────────────────
    fisheye_remap = None
    EQ_W = args.output_size * args.equirect_scale
    EQ_H = EQ_W // 2
    if args.fisheye:
        remap_cache   = f'{args.output_dir}/fisheye/_remap_{args.output_size}.npz'
        need_recompute = True
        if os.path.exists(remap_cache):
            d = np.load(remap_cache)
            if 'ray_dirs' in d:
                print('  Loading cached fisheye remap...')
                fisheye_remap  = (d['map_x'], d['map_y'], d['valid'], d['ray_dirs'])
                need_recompute = False
            else:
                print('  Cached remap missing ray_dirs — recomputing...')
        if need_recompute:
            print(f'  Precomputing FISHEYE624 remap '
                  f'({args.output_size}×{args.output_size} → {EQ_W}×{EQ_H})...')
            map_x, map_y, valid, ray_dirs = build_fisheye624_remap(
                cam_calib, args.output_size, args.output_size, EQ_W, EQ_H)
            np.savez_compressed(remap_cache, map_x=map_x, map_y=map_y,
                                valid=valid, ray_dirs=ray_dirs)
            fisheye_remap = (map_x, map_y, valid, ray_dirs)
            print(f'    Done. {valid.sum()} valid pixels ({100*valid.mean():.1f}%)')

    # ── Load GT data ──────────────────────────────────────────────────────
    with open(f'{GT_DIR}/instances.json') as f:
        instances = json.load(f)
    traj, ts_arr             = load_trajectory(f'{GT_DIR}/aria_trajectory.csv')
    static_poses, dyn_poses  = load_all_object_poses(f'{GT_DIR}/scene_objects.csv')

    # ── Valid-frame boundary detection ────────────────────────────────────
    # VRS may start before the trajectory; skip those early frames.
    _traj_t0 = int(ts_arr[0]);  _traj_t1 = int(ts_arr[-1])

    def _vrs_ts_us(idx):
        return p_ego.get_image_data_by_index(RGB_STREAM, idx)[1].capture_timestamp_ns // 1000

    _lo, _hi = 0, n_frames - 1
    while _lo < _hi:
        _mid = (_lo + _hi) // 2
        if _vrs_ts_us(_mid) < _traj_t0: _lo = _mid + 1
        else: _hi = _mid
    first_valid_frame = _lo

    _lo, _hi = 0, n_frames - 1
    while _lo < _hi:
        _mid = (_lo + _hi + 1) // 2
        if _vrs_ts_us(_mid) > _traj_t1: _hi = _mid - 1
        else: _lo = _mid
    last_valid_frame = _lo

    print(f'  Trajectory range : {_traj_t0} … {_traj_t1} us')
    print(f'  Valid frame range: [{first_valid_frame}, {last_valid_frame}]  '
          f'({last_valid_frame - first_valid_frame + 1} frames)')
    if first_valid_frame > 0:
        _delta = abs(_vrs_ts_us(0) - _traj_t0)
        print(f'  NOTE: frames 0–{first_valid_frame-1} predate trajectory '
              f'(VRS starts {_delta/1e6:.2f}s early) — skipped')

    if args.start_frame is None: args.start_frame = first_valid_frame
    if args.end_frame   is None: args.end_frame   = last_valid_frame + 1

    if args.start_frame < first_valid_frame:
        raise ValueError(f'--start_frame {args.start_frame} is before first valid frame '
                         f'({first_valid_frame}). Use --start_frame {first_valid_frame}.')
    if args.end_frame - 1 > last_valid_frame:
        raise ValueError(f'--end_frame {args.end_frame} extends beyond last valid frame '
                         f'({last_valid_frame+1}). Use --end_frame {last_valid_frame+1}.')

    print('Building scene lights...')
    scene_lights = build_scene_lights(f'{GT_DIR}/scene_objects.csv',
                                      f'{GT_DIR}/instances.json',
                                      lighting_mode='egocentric')

    static_object_list = build_object_list(instances, static_poses, MODELS_DIR)
    print(f'  Models dir                     : {MODELS_DIR}')
    print(f'  Static objects with GLB models : {len(static_object_list)}')
    print(f'  Dynamic object UIDs            : {len(dyn_poses)}')
    if not static_object_list:
        print(f'  WARNING: No GLBs resolved. Expected: {MODELS_DIR}/{{name}}.glb')

    frames_idx = list(range(args.start_frame,
                            min(args.end_frame, last_valid_frame + 1),
                            args.frame_step))
    print(f'  Rendering {len(frames_idx)} frames '
          f'(start={args.start_frame}, end={min(args.end_frame, last_valid_frame+1)}, '
          f'step={args.frame_step})')
    print(f'  Outputs: videos_rgb'
          f'{" | segmentation" if args.segmentation else ""}'
          f'{" | normal_maps"  if args.normals      else ""}'
          f'{" | depth_maps"   if args.depth        else ""}')

    # ── Per-frame render loop ─────────────────────────────────────────────
    for count, idx in enumerate(frames_idx):
        img_data    = p_ego.get_image_data_by_index(RGB_STREAM, idx)
        ts_ns       = img_data[1].capture_timestamp_ns
        ego_rgb     = img_data[0].to_numpy_array()
        frame_stem  = f'frame_{idx:06d}_{ts_ns}'

        if args.fisheye:
            eq_out_png    = f'{args.output_dir}/equirect/{frame_stem}.png'
            rgb_out_png   = f'{args.output_dir}/fisheye/videos_rgb/{frame_stem}.png'
            seg_npy_out   = f'{args.output_dir}/fisheye/segmentation/{frame_stem}.npy'
            seg_vis_out   = f'{args.output_dir}/fisheye/segmentation/{frame_stem}_vis.png'
            norm_npy_out  = f'{args.output_dir}/fisheye/normal_maps/{frame_stem}.npy'
            norm_vis_out  = f'{args.output_dir}/fisheye/normal_maps/{frame_stem}_vis.png'
            depth_npy_out = f'{args.output_dir}/fisheye/depth_maps/{frame_stem}.npy'
            depth_vis_out = f'{args.output_dir}/fisheye/depth_maps/{frame_stem}_vis.png'
        else:
            rgb_out_png   = f'{args.output_dir}/pinhole/videos_rgb/{frame_stem}.png'
            seg_npy_out   = f'{args.output_dir}/pinhole/segmentation/{frame_stem}.npy'
            seg_vis_out   = f'{args.output_dir}/pinhole/segmentation/{frame_stem}_vis.png'
            norm_npy_out  = f'{args.output_dir}/pinhole/normal_maps/{frame_stem}.npy'
            norm_vis_out  = f'{args.output_dir}/pinhole/normal_maps/{frame_stem}_vis.png'
            depth_npy_out = f'{args.output_dir}/pinhole/depth_maps/{frame_stem}.npy'
            depth_vis_out = f'{args.output_dir}/pinhole/depth_maps/{frame_stem}_vis.png'

        if os.path.exists(rgb_out_png):
            print(f'  [{count+1}/{len(frames_idx)}] frame {idx:06d} already exists, skipping')
            continue

        # Camera pose
        T_WD = nearest_pose(traj, ts_arr, ts_ns // 1000)
        T_WC = T_WD @ T_DC
        cam_pos = T_WC[:3, 3]

        # Build visible object list
        dyn_poses_frame = resolve_dynamic_poses(dyn_poses, ts_ns)
        dyn_object_list = build_object_list(instances, dyn_poses_frame, MODELS_DIR)
        all_objects     = static_object_list + dyn_object_list

        # Sort by distance; optionally cap by GLB memory budget
        objs_with_dist = sorted(
            [(np.linalg.norm(np.array(o['T_WO']).reshape(4,4)[:3,3] - cam_pos), o)
             for o in all_objects],
            key=lambda x: x[0])
        visible_objects = [o for _, o in objs_with_dist]

        if args.max_glb_mb > 0:
            budget_bytes = args.max_glb_mb * 1024 * 1024
            env_objs  = [o for o in visible_objects if 'ApartmentEnv' in os.path.basename(o['glb_path'])]
            other_objs= [o for o in visible_objects if o not in env_objs]
            env_bytes = sum(os.path.getsize(o['glb_path']) for o in env_objs
                            if os.path.exists(o['glb_path']))
            kept, total_b = list(env_objs), env_bytes
            for o in other_objs:
                sz = os.path.getsize(o['glb_path']) if os.path.exists(o['glb_path']) else 0
                if total_b - env_bytes + sz > max(0, budget_bytes - env_bytes):
                    continue
                kept.append(o);  total_b += sz
            visible_objects = kept
            print(f'    --max_glb_mb {args.max_glb_mb:.0f}: kept {len(kept)} objects '
                  f'({total_b/1e6:.0f} MB)')

        if args.fast_render:
            visible_objects = sorted(visible_objects,
                                     key=lambda o: os.path.getsize(o['glb_path']),
                                     reverse=True)[:FAST_RENDER_N_OBJECTS]

        n_dyn  = sum(1 for o in visible_objects if any(o is x for x in dyn_object_list))
        max_d  = max((np.linalg.norm(np.array(o['T_WO']).reshape(4,4)[:3,3] - cam_pos)
                      for o in visible_objects), default=0.0)
        print(f'  [{count+1}/{len(frames_idx)}] frame {idx:06d} — '
              f'{len(visible_objects)} objects ({n_dyn} dynamic, max dist {max_d:.1f}m)')

        # Assign sequential pass_index for segmentation
        pass_idx_to_uid: dict[int, int] = {}
        for i, obj in enumerate(visible_objects):
            obj['pass_index'] = i + 1
            try:    pass_idx_to_uid[i + 1] = int(obj.get('uid', 0))
            except: pass_idx_to_uid[i + 1] = 0

        # Write per-frame JSON
        frame_data = {
            'image_width':     args.output_size,
            'image_height':    args.output_size,
            'focal_px':        args.focal,
            'camera_pose':     T_WC.flatten().tolist(),
            'object_models':   visible_objects,
            'scene_lights':    scene_lights,
            'use_equirect':    args.fisheye,
            'equirect_width':  EQ_W if args.fisheye else args.output_size,
            'equirect_height': EQ_H if args.fisheye else args.output_size,
            'cycles_samples':  args.cycles_samples,
            'cycles_device':   args.cycles_device,
            'cycles_denoiser': args.cycles_denoiser,
            'pass_idx_to_uid': {str(k): v for k, v in pass_idx_to_uid.items()},
        }
        json_path = f'{tmp_dir}/{frame_stem}.json'
        with open(json_path, 'w') as f:
            json.dump(frame_data, f)

        blender_out    = eq_out_png if args.fisheye else f'{tmp_dir}/{frame_stem}_raw.png'
        seg_exr_tmp    = f'{tmp_dir}/seg_{frame_stem}.exr'
        normal_exr_tmp = f'{tmp_dir}/normal_{frame_stem}.exr'
        depth_exr_tmp  = f'{tmp_dir}/depth_{frame_stem}.exr'

        cmd = [BLENDER_BIN, '--background', '--python', BLEND_SCRIPT,
               '--', '--frame_data', json_path, '--output', blender_out]
        if args.segmentation:
            cmd += ['--seg_output', seg_exr_tmp]
        if args.normals and args.blender_normals:
            cmd += ['--normal_output', normal_exr_tmp]
        if args.depth or (args.normals and not args.blender_normals):
            cmd += ['--depth_output', depth_exr_tmp]

        print(f'  Rendering frame {idx:06d} (ts={ts_ns})...')
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if not os.path.exists(blender_out):
            print(f'    Output not created (returncode={result.returncode}).')
            if result.stderr:
                print(f'    stderr: {result.stderr[-500:]}')
            continue
        if result.returncode != 0:
            print(f'    Blender returned {result.returncode} but output exists — continuing.')

        # ── RGB post-process ─────────────────────────────────────────────
        if args.fisheye and fisheye_remap is not None:
            map_x, map_y, valid, _ = fisheye_remap
            fish_arr   = remap_equirect_to_fisheye(Image.open(blender_out), map_x, map_y, valid)
            render_img = apply_aria_forward_isp(fish_arr)
        else:
            render_img = apply_aria_forward_isp(
                np.array(Image.open(blender_out).convert('RGB')))
        Image.fromarray(render_img).save(rgb_out_png)

        # ── Segmentation ──────────────────────────────────────────────────
        if args.segmentation and os.path.exists(seg_exr_tmp):
            seg_raw3  = cv2.imread(seg_exr_tmp, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            seg_float = seg_raw3[..., 0] if seg_raw3 is not None else None
            if seg_float is not None:
                if args.fisheye and fisheye_remap is not None:
                    map_x_s, map_y_s, valid_s, _ = fisheye_remap
                    seg_float = cv2.remap(seg_float, map_x_s, map_y_s,
                                          interpolation=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
                    seg_float[~valid_s] = 0.0
                pass_arr = np.round(seg_float).astype(np.int32)
                seg_uid  = np.zeros(pass_arr.shape, dtype=np.int64)
                for pidx_s, uid_s in pass_idx_to_uid.items():
                    seg_uid[pass_arr == pidx_s] = uid_s
                np.save(seg_npy_out, seg_uid)
                Image.fromarray(colorize_seg(seg_uid)).save(seg_vis_out)
                print(f'    Segmentation: {len(np.unique(seg_uid))-1} objects → {seg_npy_out}')
        elif args.segmentation:
            print(f'    [seg] EXR not found at {seg_exr_tmp}')

        # ── Depth ─────────────────────────────────────────────────────────
        depth_for_normals = None
        if args.depth or (args.normals and not args.blender_normals):
            if os.path.exists(depth_exr_tmp):
                depth_bgr = cv2.imread(depth_exr_tmp, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                if depth_bgr is None:
                    print(f'    [depth] cv2 could not read {depth_exr_tmp}')
                else:
                    depth = depth_bgr[..., 0].astype(np.float32)
                    depth[depth > 1e6] = np.inf
                    if args.fisheye and fisheye_remap is not None:
                        map_x_d, map_y_d, valid_d, ray_dirs_d = fisheye_remap
                        depth = cv2.remap(depth, map_x_d, map_y_d,
                                          interpolation=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=np.inf)
                        depth[~valid_d] = np.inf
                        inf_mask = ~np.isfinite(depth)
                        depth = depth * ray_dirs_d[..., 2]
                        depth[inf_mask] = np.inf
                    depth_for_normals = depth
                    if args.depth:
                        np.save(depth_npy_out, depth)
                        Image.fromarray(visualize_depth(depth)).save(depth_vis_out)
                        finite_cnt = np.isfinite(depth).sum()
                        median_d   = (float(np.nanmedian(depth[np.isfinite(depth)]))
                                      if finite_cnt > 0 else 0.0)
                        print(f'    Depth: median {median_d:.2f}m, {finite_cnt} valid px '
                              f'→ {depth_npy_out}')
            elif args.depth:
                print(f'    [depth] EXR not found at {depth_exr_tmp}')

        # ── Normals ───────────────────────────────────────────────────────
        if args.normals:
            norm_out = None
            if args.blender_normals:
                if os.path.exists(normal_exr_tmp):
                    norm_bgr = cv2.imread(normal_exr_tmp, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                    if norm_bgr is not None:
                        norm_world = norm_bgr[..., ::-1].copy()
                        R_CW = T_WC[:3, :3].T
                        norm_out = (norm_world.reshape(-1, 3) @ R_CW.T
                                    ).reshape(norm_world.shape).astype(np.float32)
                        if args.fisheye and fisheye_remap is not None:
                            map_x_n, map_y_n, valid_n, _ = fisheye_remap
                            remapped = np.stack([
                                cv2.remap(norm_out[..., c], map_x_n, map_y_n,
                                          interpolation=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
                                for c in range(3)], axis=-1)
                            remapped[~valid_n] = 0.0
                            norms = np.linalg.norm(remapped, axis=-1, keepdims=True)
                            norm_out = np.where(norms > 1e-4, remapped / (norms + 1e-8),
                                                0.0).astype(np.float32)
            else:
                if depth_for_normals is not None:
                    if args.fisheye and fisheye_remap is not None:
                        _, _, valid_n, ray_dirs_n = fisheye_remap
                        norm_out = _normals_from_depth_fisheye(
                            depth_for_normals, ray_dirs_n, valid_n)
                    else:
                        norm_out = _normals_from_depth_pinhole(
                            depth_for_normals, args.focal, args.focal,
                            args.output_size / 2.0, args.output_size / 2.0)
                else:
                    print('    [normal] depth not available — skipping normals')

            if norm_out is not None:
                np.save(norm_npy_out, norm_out)
                Image.fromarray(
                    visualize_normal(np.nan_to_num(norm_out, nan=0.0))).save(norm_vis_out)
                print(f'    Normal map → {norm_npy_out}')

        # ── Side-by-side comparison ───────────────────────────────────────
        os.makedirs(f'{args.output_dir}/comparison', exist_ok=True)
        ego_resized = np.array(
            Image.fromarray(ego_rgb).resize(
                (args.output_size, args.output_size), Image.LANCZOS))
        gap  = np.ones((args.output_size, 8, 3), dtype=np.uint8) * 60
        side = np.concatenate([ego_resized, gap, render_img], axis=1)
        Image.fromarray(side).save(f'{args.output_dir}/comparison/{frame_stem}.png')
        print('    Done.')

    print(f'\nAll done! Outputs in {args.output_dir}/')


if __name__ == '__main__':
    main()