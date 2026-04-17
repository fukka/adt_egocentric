"""
render_exocentric_blender.py
============================
Render the ADT Apartment scene from a hard-coded exocentric (third-person)
camera pose using Blender Cycles + blender_render_scene.py.

Three cameras are available (select with --camera):

  right_back  eye=(3.8, 2.8, 4.0) m — elevated diagonal, from behind-right
  left_side   eye=(-2.0, 2.8, 2.5) m — elevated side view from the left
  overhead    eye=(1.2, 3.8, 3.0) m — steep overhead (62° pitch)

All cameras look toward the kitchen counter area (≈1.0, 0.9, 1.5) which is
the region observed by the Aria ego camera at frame 0.

Camera construction (lookat_adt):
  Given eye position + target point, builds T_WC in ADT convention
  (col 2 = forward toward target, det = +1) via a Blender lookat formula
  back-converted through FLIP_YZ = diag([1,-1,-1]).

Usage:
    python render_exocentric_blender.py
    python render_exocentric_blender.py --camera left_side --output_size 1024
    python render_exocentric_blender.py --camera all --frame_idx 30
    python render_exocentric_blender.py --no_segmentation

Does NOT require the Aria VRS file — camera pose is hard-coded and the scene
timestamp for dynamic objects is taken from the trajectory CSV.
"""

import sys, os, csv, json, argparse, subprocess
sys.path.insert(0, '/sessions/dreamy-modest-brown/.local/lib/python3.10/site-packages')

import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import os as _os; _os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2

# ── Paths ──────────────────────────────────────────────────────────────────
BASE         = '/sessions/dreamy-modest-brown/mnt/ADT/Apartment_release_golden_skeleton_seq100_10s_sample_M1292'
GT_DIR       = f'{BASE}/groundtruth'
MODELS_DIR   = f'{BASE}/object_models'
BLENDER_BIN  = '/sessions/dreamy-modest-brown/blender/blender'
BLEND_SCRIPT = '/sessions/dreamy-modest-brown/mnt/ADT/blender_render_scene.py'

# ── Lookat helper ──────────────────────────────────────────────────────────

def lookat_adt(eye, target, world_up=None):
    """Build a 4×4 T_WC matrix in ADT convention from an eye + target point.

    ADT camera convention (same as used by projectaria_tools and the driver):
      col 0 = camera +X (right)
      col 1 = camera -Y_image (down — world space)
      col 2 = camera +Z (forward, toward target)
      det   = +1 (proper rotation)

    Strategy:
      1. Build T_WC_blender (Blender convention: col1=image-up, col2=backward)
         using a standard Blender camera lookat.
      2. Back-convert to T_WC_adt via FLIP_YZ = diag([1,-1,-1]):
           T_WC_adt = T_WC_blender @ FLIP_YZ
         → col1 flips sign (up→down), col2 flips sign (backward→forward)

    Args:
        eye:       3-vector camera position in ADT world coords (Y-up).
        target:    3-vector point the camera looks toward.
        world_up:  reference world-up (default [0,1,0]).

    Returns 4×4 float64 T_WC matrix (homogeneous).
    """
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0])
    eye    = np.asarray(eye,    dtype=float)
    target = np.asarray(target, dtype=float)
    world_up = np.asarray(world_up, dtype=float)

    fwd     = target - eye
    fwd     = fwd / np.linalg.norm(fwd)          # ADT forward (+Z)
    neg_fwd = -fwd                                # Blender local +Z (backward)

    right = np.cross(world_up, neg_fwd)
    if np.linalg.norm(right) < 1e-6:             # looking straight up/down
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(world_up, neg_fwd)
    right = right / np.linalg.norm(right)

    img_up = np.cross(neg_fwd, right)             # Blender image-up (col 1 of T_bl)
    img_up = img_up / np.linalg.norm(img_up)

    # Blender camera matrix: cols = [right, img_up, neg_fwd]
    # T_WC_adt = T_WC_bl @ diag([1,-1,-1]):
    #   col 0  = right              (unchanged)
    #   col 1  = -img_up            (down in world)
    #   col 2  = fwd                (forward, toward target)
    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = -img_up
    T[:3, 2] = fwd
    T[:3, 3] = eye
    return T


# ── Hard-coded exocentric cameras ──────────────────────────────────────────
# All look toward the kitchen counter area (approx. where the ego camera is
# focused at frame 0).  Pre-computed with lookat_adt; stored as flat 4×4 lists.

_SCENE_TARGET = np.array([1.0, 0.9, 1.5])   # kitchen counter look-at point

EXOCENTRIC_CAMERAS = {
    # Diagonal view from the rear-right corner, ~2.8 m elevation, ~27° down-pitch.
    # Similar to a ceiling-corner security camera. Sees the full kitchen counter,
    # the WhiteFlatwareTray and surrounding objects from the far side.
    'right_back': {
        'eye':    np.array([3.8, 2.8, 4.0]),
        'target': _SCENE_TARGET,
        'desc':   'Elevated diagonal from rear-right corner (≈27° pitch, 4.2 m from target)',
    },
    # Side view from the left, ~2.8 m elevation, ~31° down-pitch.
    # Perpendicular to the ego camera's viewing direction; shows counter depth.
    'left_side': {
        'eye':    np.array([-2.0, 2.8, 2.5]),
        'target': _SCENE_TARGET,
        'desc':   'Elevated from the left side (≈31° pitch, 3.7 m from target)',
    },
    # Near-overhead view, ~3.8 m elevation, ~62° down-pitch.
    # Emphasises top-down layout; good for object footprint comparison with GT.
    'overhead': {
        'eye':    np.array([1.2, 3.8, 3.0]),
        'target': _SCENE_TARGET,
        'desc':   'Steep overhead view (≈62° pitch, 3.3 m from target)',
    },
}


# ── Object rotation convention fix ─────────────────────────────────────────
# (Same as render_from_poses_blender.py — see that file for full derivation.)

R_x_neg90 = np.array([[1, 0,  0],
                       [0, 0,  1],
                       [0,-1,  0]], dtype=float)

_glb_baked_rotation_cache: dict = {}

def _read_glb_baked_rotation(glb_path: str) -> np.ndarray:
    if glb_path in _glb_baked_rotation_cache:
        return _glb_baked_rotation_cache[glb_path]
    R = np.eye(3)
    try:
        import struct as _struct
        with open(glb_path, 'rb') as f:
            f.read(12)
            chunk_len = _struct.unpack('<I', f.read(4))[0]
            f.read(4)
            gltf = json.loads(f.read(chunk_len))
        nodes = gltf.get('nodes', [])
        if nodes and 'rotation' in nodes[0]:
            q = nodes[0]['rotation']          # [x, y, z, w] glTF
            R = Rotation.from_quat(q).as_matrix()
    except Exception:
        pass
    _glb_baked_rotation_cache[glb_path] = R
    return R

def correct_object_rotation(T_WO, glb_path=None):
    R_baked = _read_glb_baked_rotation(glb_path) if glb_path else np.eye(3)
    T_c = T_WO.copy()
    T_c[:3, :3] = T_WO[:3, :3] @ R_baked @ R_x_neg90
    return T_c


# ── ADT data helpers ────────────────────────────────────────────────────────

def quat_to_matrix(tx, ty, tz, qx, qy, qz, qw):
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [tx, ty, tz]
    return T


def load_first_frame_timestamp_ns(traj_csv):
    """Return the trajectory timestamp of the first row, in nanoseconds.

    The trajectory CSV stores tracking_timestamp_us (microseconds).  Dynamic
    object CSVs store timestamps in nanoseconds.  This function returns the
    first trajectory timestamp converted to nanoseconds so that dynamic object
    poses can be resolved without requiring the VRS file.
    """
    with open(traj_csv) as f:
        row = next(csv.DictReader(f))
    return int(row['tracking_timestamp_us']) * 1000   # µs → ns


def load_all_object_poses(path):
    """Load static (timestamp=-1) and dynamic (timestamped) object poses.
    Returns static_poses {uid: T_WO_raw} and dynamic_poses {uid: [(ts_ns, T_WO_raw)]}.
    """
    static_poses  = {}
    dynamic_poses = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            uid   = row['object_uid']
            ts_ns = int(row['timestamp[ns]'])
            T = quat_to_matrix(
                float(row['t_wo_x[m]']), float(row['t_wo_y[m]']),
                float(row['t_wo_z[m]']),
                float(row['q_wo_x']),   float(row['q_wo_y']),
                float(row['q_wo_z']),   float(row['q_wo_w']))
            if ts_ns == -1:
                static_poses[uid] = T
            else:
                dynamic_poses.setdefault(uid, []).append((ts_ns, T))
    for uid in dynamic_poses:
        dynamic_poses[uid].sort(key=lambda x: x[0])
    return static_poses, dynamic_poses


def resolve_dynamic_poses(dynamic_poses, frame_ts_ns):
    resolved = {}
    for uid, entries in dynamic_poses.items():
        ts_arr = np.array([e[0] for e in entries], dtype=np.int64)
        idx = int(np.argmin(np.abs(ts_arr - frame_ts_ns)))
        resolved[uid] = entries[idx][1]
    return resolved


def build_object_list(instances, obj_poses, models_dir):
    uid_to_info = {str(v['instance_id']): v for v in instances.values()}
    result = []
    for uid, T_WO_raw in obj_poses.items():
        info = uid_to_info.get(uid)
        if info is None:
            continue
        instance_name  = info.get('instance_name', '')
        prototype_name = info.get('prototype_name', '')
        glb_path = None
        for candidate in [instance_name, prototype_name]:
            if candidate:
                p = os.path.join(models_dir, f'{candidate}.glb')
                if os.path.exists(p):
                    glb_path = p
                    break
        if glb_path:
            T_WO_corrected = correct_object_rotation(T_WO_raw, glb_path)
            result.append({'glb_path': glb_path,
                           'T_WO':     T_WO_corrected.flatten().tolist(),
                           'uid':      uid})
    return result


def build_scene_lights(scene_objects_csv, instances_json_path):
    """(Same as render_from_poses_blender.py — prop point lights + ceiling area lights.)"""
    LIGHT_PROPS = {
        'Lamp_1':        {'energy': 120.0, 'color': [1.0, 0.82, 0.60], 'radius': 0.15},
        'WhitTableLamp': {'energy':  80.0, 'color': [1.0, 0.85, 0.65], 'radius': 0.12},
        'NightLights':   {'energy':  30.0, 'color': [1.0, 0.90, 0.80], 'radius': 0.08},
        'Candles':       {'energy':  10.0, 'color': [1.0, 0.75, 0.40], 'radius': 0.04},
    }
    with open(instances_json_path) as f:
        instances_data = json.load(f)
    name_to_uid = {v.get('prototype_name', ''): str(v['instance_id'])
                   for v in instances_data.values()}
    lights = []
    with open(scene_objects_csv) as f:
        for row in csv.DictReader(f):
            uid = row['object_uid']
            if int(row['timestamp[ns]']) != -1:
                continue
            info = next((v for v in instances_data.values()
                         if str(v['instance_id']) == uid), None)
            if info is None:
                continue
            name = info.get('prototype_name', '')
            cfg  = next((c for k, c in LIGHT_PROPS.items() if k in name), None)
            if cfg is None:
                continue
            x, y, z = float(row['t_wo_x[m]']), float(row['t_wo_y[m]']), float(row['t_wo_z[m]'])
            y_offset = 0.30 if 'Lamp' in name else 0.10
            lights.append({
                'type':     'POINT',
                'location': [x, y + y_offset, z],
                'energy':   cfg['energy'],
                'color':    cfg['color'],
                'radius':   cfg['radius'],
            })
    # Ceiling area lights (same positions as main pipeline)
    CEIL_Y, CEIL_ENERGY = 2.35, 22.0
    CEIL_COLOR, CEIL_SIZE = [1.0, 0.90, 0.75], 1.8
    for (cx, cz) in [(0.5, 2.5), (0.5, 0.5), (-1.0, -1.5),
                     (-1.0, 4.5), (-2.5, -2.5), (-2.5, 1.5)]:
        lights.append({'type': 'AREA', 'location': [cx, CEIL_Y, cz],
                       'energy': CEIL_ENERGY, 'color': CEIL_COLOR, 'size': CEIL_SIZE})
    return lights


# ── Segmentation colouriser ─────────────────────────────────────────────────

def colorize_seg(seg_uid):
    import hashlib
    vis = np.zeros((*seg_uid.shape, 3), dtype=np.uint8)
    for uid in np.unique(seg_uid):
        if uid == 0:
            continue
        d = hashlib.md5(str(uid).encode()).digest()
        vis[seg_uid == uid] = (d[0], d[1], d[2])
    return vis


# ── Render one camera ──────────────────────────────────────────────────────

def render_camera(cam_name, cam_cfg, frame_ts_ns, all_objects, scene_lights,
                  args, tmp_dir):
    """Run Blender for one exocentric camera and save outputs."""
    out_dir = f'{args.output_dir}/{cam_name}'
    os.makedirs(f'{out_dir}/rgb',          exist_ok=True)
    if args.segmentation:
        os.makedirs(f'{out_dir}/segmentation', exist_ok=True)

    tag     = f'frame_{args.frame_idx:04d}'
    out_png = f'{out_dir}/rgb/{tag}.png'

    if os.path.exists(out_png):
        print(f'  [{cam_name}] {tag} already exists — skipping')
        return

    T_WC = lookat_adt(cam_cfg['eye'], cam_cfg['target'])

    # Distance cull: keep closest 80 objects to exocentric camera position
    cam_pos = np.asarray(cam_cfg['eye'])
    objs_with_dist = [
        (np.linalg.norm(np.array(obj['T_WO']).reshape(4, 4)[:3, 3] - cam_pos), obj)
        for obj in all_objects
    ]
    objs_with_dist.sort(key=lambda x: x[0])
    visible_objects = [obj for _, obj in objs_with_dist[:80]]
    max_dist = objs_with_dist[min(79, len(objs_with_dist) - 1)][0]
    print(f'  [{cam_name}] {len(visible_objects)} objects (max dist {max_dist:.1f} m)')

    # Assign sequential pass_index values for segmentation
    pass_idx_to_uid: dict[int, int] = {}
    for i, obj in enumerate(visible_objects):
        obj = dict(obj)   # don't mutate shared list
        obj['pass_index'] = i + 1
        try:
            pass_idx_to_uid[i + 1] = int(obj.get('uid', 0))
        except (ValueError, TypeError):
            pass_idx_to_uid[i + 1] = 0
        visible_objects[i] = obj

    # Focal length for standard wide-angle: VFOV ≈ 70° → f = (H/2) / tan(35°)
    focal_px = (args.output_size / 2.0) / np.tan(np.radians(35.0))

    frame_data = {
        'image_width':    args.output_size,
        'image_height':   args.output_size,
        'focal_px':       focal_px,
        'camera_pose':    T_WC.flatten().tolist(),
        'object_models':  visible_objects,
        'scene_lights':   scene_lights,
        'use_equirect':   False,
        'cycles_samples': 32,
        'pass_idx_to_uid': {str(k): v for k, v in pass_idx_to_uid.items()},
    }

    json_path = f'{tmp_dir}/{cam_name}_{tag}.json'
    with open(json_path, 'w') as f:
        json.dump(frame_data, f)

    seg_exr_tmp = f'{tmp_dir}/{cam_name}_{tag}_seg.exr'

    cmd = [
        BLENDER_BIN, '--background', '--python', BLEND_SCRIPT,
        '--', '--frame_data', json_path, '--output', out_png,
    ]
    if args.segmentation:
        cmd += ['--seg_output', seg_exr_tmp]

    print(f'  [{cam_name}] Rendering {tag} — eye {cam_cfg["eye"].tolist()}')
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    # Blender can return non-zero exit codes on certain import warnings while
    # still writing the output file successfully.  Check the file first.
    if not os.path.exists(out_png):
        print(f'  [{cam_name}] Output not created (returncode={result.returncode}).')
        if result.stderr:
            print(f'  stderr: {result.stderr[-800:]}')
        return
    if result.returncode != 0:
        print(f'  [{cam_name}] Blender returned {result.returncode} but output exists — continuing.')

    print(f'  [{cam_name}] Saved {out_png}')

    # Segmentation post-processing
    if args.segmentation and os.path.exists(seg_exr_tmp):
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
            n_objs = len(np.unique(seg_uid)) - 1
            print(f'  [{cam_name}] Segmentation: {n_objs} objects → {seg_npy}')


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    cam_names = list(EXOCENTRIC_CAMERAS.keys())

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera',      choices=cam_names + ['all'],
                        default='right_back',
                        help='Which exocentric camera to render (default: right_back)')
    parser.add_argument('--frame_idx',   type=int, default=0,
                        help='Scene frame index to render (default: 0)')
    parser.add_argument('--output_size', type=int, default=1024,
                        help='Output image size in pixels (default: 1024)')
    parser.add_argument('--output_dir',  type=str,
                        default=f'{BASE}/exocentric_rendered')
    parser.add_argument('--no_segmentation', action='store_true',
                        help='Skip instance segmentation output')
    args = parser.parse_args()
    args.segmentation = not args.no_segmentation

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = f'{args.output_dir}/_tmp'
    os.makedirs(tmp_dir, exist_ok=True)

    print('Loading ADT scene data...')
    with open(f'{GT_DIR}/instances.json') as f:
        instances = json.load(f)

    # Get frame timestamp without requiring VRS: use trajectory CSV row[frame_idx]
    traj_rows = []
    with open(f'{GT_DIR}/aria_trajectory.csv') as f:
        for row in csv.DictReader(f):
            traj_rows.append(int(row['tracking_timestamp_us']))
    if args.frame_idx >= len(traj_rows):
        print(f'Warning: frame_idx={args.frame_idx} exceeds trajectory '
              f'({len(traj_rows)} rows). Using last row.')
        args.frame_idx = len(traj_rows) - 1
    frame_ts_ns = traj_rows[args.frame_idx] * 1000   # µs → ns
    print(f'  Frame {args.frame_idx}: timestamp {frame_ts_ns} ns')

    static_poses, dynamic_poses = load_all_object_poses(f'{GT_DIR}/scene_objects.csv')
    dyn_poses_frame = resolve_dynamic_poses(dynamic_poses, frame_ts_ns)

    static_obj_list = build_object_list(instances, static_poses,    MODELS_DIR)
    dyn_obj_list    = build_object_list(instances, dyn_poses_frame,  MODELS_DIR)
    all_objects     = static_obj_list + dyn_obj_list
    print(f'  Static: {len(static_obj_list)}  Dynamic (at frame): {len(dyn_obj_list)}')

    print('Building scene lights...')
    scene_lights = build_scene_lights(f'{GT_DIR}/scene_objects.csv',
                                      f'{GT_DIR}/instances.json')

    # Select cameras to render
    if args.camera == 'all':
        selected = list(EXOCENTRIC_CAMERAS.items())
    else:
        selected = [(args.camera, EXOCENTRIC_CAMERAS[args.camera])]

    print(f'\nRendering {len(selected)} camera(s)...')
    for cam_name, cam_cfg in selected:
        print(f'\nCamera: {cam_name}')
        print(f'  {cam_cfg["desc"]}')
        render_camera(cam_name, cam_cfg, frame_ts_ns, all_objects,
                      scene_lights, args, tmp_dir)

    print(f'\nAll done! Outputs in {args.output_dir}/')


if __name__ == '__main__':
    main()
