"""
ADT Object-Pose → Blender Rendering Pipeline — Maps Extension
=============================================================
Extends render_from_poses_blender.py to also output per-frame:
  • Instance segmentation map  (object UIDs as int64 .npy + colorised PNG)
  • Surface normal map         (world-space XYZ float32 .npy + visualised PNG)
  • Depth map                  (camera-space distance float32 .npy + visualised PNG)

Drives blender_render_scene.py which must support --normal_output / --depth_output
(updated version).  All other rendering logic (camera, lights, fisheye remap,
Aria ISP colour correction) is identical to render_from_poses_blender.py.

Usage:
    python render_from_poses_blender_maps.py \
        [--num_frames N] [--frame_step K] \
        [--output_size S] [--focal F] \
        [--fisheye] \
        [--no_segmentation] [--no_normals] [--no_depth] \
        [--output_dir /path/to/output]

Output layout:
    <output_dir>/
        rgb/            frame_NNNN.png          Blender render (+ Aria ISP)
        comparison/     frame_NNNN.png          side-by-side ego vs render
        segmentation/   frame_NNNN.npy          int64 instance-UID per pixel
                        frame_NNNN_vis.png      hash-coloured seg visualisation
        normal_maps/    frame_NNNN.npy          float32 (H,W,3) XYZ in [-1,1]
                        frame_NNNN_vis.png      RGB visualisation (mapped to [0,255])
        depth_maps/     frame_NNNN.npy          float32 (H,W) metres (inf = bg)
                        frame_NNNN_vis.png      log-normalised grayscale

Normal map convention:
    World-space shading normals from Blender Cycles Normal pass.
    Saved as float32 (H, W, 3) in XYZ order (Blender: R=X, G=Y, B=Z).
    Values are in ADT Y-up world space, normalised to unit length.
    After fisheye remap the vectors are spatially correct but may not
    be exactly unit-length (linear interpolation artefact); normalise if needed.

Depth map convention:
    Camera-space distance in metres (Blender Z pass = distance from camera
    origin to surface along the ray, not projected Z).
    Background pixels = np.inf (no surface hit).
    Saved as float32 (H, W); clip to a finite max when loading for visualisation.
"""

import sys, os, csv, json, argparse, subprocess
sys.path.insert(0, '/sessions/dreamy-modest-brown/.local/lib/python3.10/site-packages')

import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import os as _os; _os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2

from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId

# ── Paths ──────────────────────────────────────────────────────────────────
BASE        = '/sessions/dreamy-modest-brown/mnt/ADT/Apartment_release_golden_skeleton_seq100_10s_sample_M1292'
EGO_VRS     = f'{BASE}/main_recording.vrs'
GT_DIR      = f'{BASE}/groundtruth'
MODELS_DIR  = f'{BASE}/object_models'
BLENDER_BIN = '/sessions/dreamy-modest-brown/blender/blender'
BLEND_SCRIPT= '/sessions/dreamy-modest-brown/mnt/ADT/blender_render_scene.py'
RGB_STREAM  = StreamId('214-1')

FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])

# ── Object rotation convention (see render_from_poses_blender.py for docs) ──
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
            q = nodes[0]['rotation']   # [x, y, z, w]
            R = Rotation.from_quat(q).as_matrix()
    except Exception:
        pass
    _glb_baked_rotation_cache[glb_path] = R
    return R


def correct_object_rotation(T_WO, glb_path: str | None = None):
    R_baked = _read_glb_baked_rotation(glb_path) if glb_path else np.eye(3)
    T_c = T_WO.copy()
    T_c[:3, :3] = T_WO[:3, :3] @ R_baked @ R_x_neg90
    return T_c


# ── FISHEYE624 remap helpers (identical to render_from_poses_blender.py) ────

def build_fisheye624_remap(cam_calib, out_w, out_h, eq_w, eq_h):
    params     = cam_calib.get_projection_params()
    fx         = params[0]
    cx, cy     = params[1], params[2]
    k0, k1, k2, k3, k4, k5 = params[3:9]

    native_size = float(cam_calib.get_image_size()[0])
    scale       = out_w / native_size
    fx_s        = fx * scale
    cx_s        = cx * scale + (scale - 1) * 0.5
    cy_s        = cy * scale + (scale - 1) * 0.5
    valid_r_s   = cam_calib.get_valid_radius() * scale

    u  = np.arange(out_w, dtype=np.float64)
    v  = np.arange(out_h, dtype=np.float64)
    UU, VV = np.meshgrid(u, v)
    u_hat = (UU - cx_s).ravel()
    v_hat = (VV - cy_s).ravel()
    r = np.sqrt(u_hat**2 + v_hat**2)

    theta = r / fx_s
    for _ in range(25):
        t2    = theta * theta
        poly  = 1.0 + t2 * (k0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * (k4 + t2 * k5)))))
        dpoly = t2 * (2*k0 + t2 * (4*k1 + t2 * (6*k2 + t2 * (8*k3 + t2 * (10*k4 + t2 * 12*k5)))))
        fval  = fx_s * theta * poly - r
        dfval = fx_s * (poly + dpoly)
        theta -= fval / (dfval + 1e-12)
        theta  = np.clip(theta, 0.0, np.pi)

    sin_t  = np.sin(theta)
    cos_t  = np.cos(theta)
    r_safe = np.maximum(r, 1e-9)

    ray_x =  sin_t * (u_hat / r_safe)
    ray_y =  sin_t * (v_hat / r_safe)
    ray_z =  cos_t

    rx_bl =  ray_x
    ry_bl = -ray_y
    rz_bl = -ray_z

    lon = np.arctan2(rx_bl, -rz_bl)
    lat = np.arctan2(ry_bl, np.sqrt(rx_bl**2 + rz_bl**2))

    map_x = ((lon + np.pi) / (2 * np.pi) * eq_w).astype(np.float32)
    map_y = ((np.pi / 2 - lat) / np.pi    * eq_h).astype(np.float32)
    valid = (r < valid_r_s)

    return (map_x.reshape(out_h, out_w),
            map_y.reshape(out_h, out_w),
            valid.reshape(out_h, out_w))


def remap_equirect_to_fisheye(equirect_img, map_x, map_y, valid_mask):
    if isinstance(equirect_img, Image.Image):
        equirect_arr = np.array(equirect_img.convert('RGB'), dtype=np.uint8)
    else:
        equirect_arr = np.asarray(equirect_img, dtype=np.uint8)
    out = cv2.remap(equirect_arr, map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    out[~valid_mask] = 0
    return out


def quat_to_matrix(tx, ty, tz, qx, qy, qz, qw):
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [tx, ty, tz]
    return T

def load_trajectory(path):
    traj = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ts = int(row['tracking_timestamp_us'])
            T  = quat_to_matrix(
                float(row['tx_world_device']), float(row['ty_world_device']),
                float(row['tz_world_device']),
                float(row['qx_world_device']), float(row['qy_world_device']),
                float(row['qz_world_device']), float(row['qw_world_device']))
            traj[ts] = T
    ts_arr = np.array(sorted(traj.keys()))
    return traj, ts_arr

def nearest_pose(traj, ts_arr, ts_us):
    idx = np.argmin(np.abs(ts_arr - ts_us))
    return traj[ts_arr[idx]]

def load_all_object_poses(path):
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

def build_scene_lights(scene_objects_csv, instances_json_path):
    with open(instances_json_path) as f:
        instances = json.load(f)
    uid_to_name = {info['instance_id']: info.get('instance_name', '')
                   for info in instances.values() if 'instance_id' in info}

    PROP_LIGHT_CFG = {
        'Lamp':          {'energy': 20.0, 'color': [1.0, 0.88, 0.70], 'radius': 0.15},
        'WhitTableLamp': {'energy': 15.0, 'color': [1.0, 0.90, 0.72], 'radius': 0.12},
        'NightLight':    {'energy':  4.0, 'color': [1.0, 0.80, 0.60], 'radius': 0.05},
        'Candle':        {'energy':  1.5, 'color': [1.0, 0.70, 0.40], 'radius': 0.02},
    }

    def cfg_for(name):
        for key, cfg in PROP_LIGHT_CFG.items():
            if key.lower() in name.lower():
                return cfg
        return None

    lights = []
    with open(scene_objects_csv) as f:
        for row in csv.DictReader(f):
            if row['timestamp[ns]'] != '-1':
                continue
            uid  = int(row['object_uid'])
            name = uid_to_name.get(uid, '')
            cfg  = cfg_for(name)
            if cfg is None:
                continue
            x = float(row['t_wo_x[m]'])
            y = float(row['t_wo_y[m]'])
            z = float(row['t_wo_z[m]'])
            y_offset = 0.30 if 'Lamp' in name else 0.10
            lights.append({
                'type':     'POINT',
                'location': [x, y + y_offset, z],
                'energy':   cfg['energy'],
                'color':    cfg['color'],
                'radius':   cfg['radius'],
            })

    CEIL_Y      = 2.35
    CEIL_ENERGY = 22.0
    CEIL_COLOR  = [1.0, 0.90, 0.75]
    CEIL_SIZE   = 1.8
    for (cx, cz) in [
        ( 0.5,  2.5),
        ( 0.5,  0.5),
        (-1.0, -1.5),
        (-1.0,  4.5),
        (-2.5, -2.5),
        (-2.5,  1.5),
    ]:
        lights.append({
            'type':     'AREA',
            'location': [cx, CEIL_Y, cz],
            'energy':   CEIL_ENERGY,
            'color':    CEIL_COLOR,
            'size':     CEIL_SIZE,
        })

    return lights


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


# ── Aria Gen1 colour pipeline ──────────────────────────────────────────────
_ARIA_INV_CRF = np.array([
    0.0, 0.0005489583158375797, 0.0011304615925053726, 0.0017206098925041386,
    0.002319584122130247, 0.0029275651876800664, 0.0035447339954499655, 0.004171271451736312,
    0.004807358462835479, 0.005453175935043828, 0.006108904774657734, 0.006774725887973564,
    0.007450820181287686, 0.008137368560896467, 0.00883455193309628, 0.009542551204183491,
    0.010261547280454473, 0.010991721068205586, 0.011733253473733206, 0.012486325403333701,
    0.013251117763303443, 0.01402781145993879, 0.01481658739953612, 0.015617626488391797,
    0.016431109632802192, 0.017257217739063677, 0.018096131713472616, 0.01894803246232538,
    0.019813100891918338, 0.020691517908547855, 0.0215834644185103, 0.02248912132810205,
    0.023408669543619462, 0.024342289971358917, 0.025290163517616784, 0.026252471088689406,
    0.027229393590873188, 0.02822111193046447, 0.029227807013759644, 0.030249659747055072,
    0.031286851036647106, 0.03233956178883213, 0.03340797290990651, 0.03449226530616662,
    0.03559261988390881, 0.036709217549429476, 0.03784223920902497, 0.03899186576899166,
    0.040158278135625926, 0.04134165721522412, 0.04254218391408263, 0.04376003913849781,
    0.04499540379476604, 0.046248458789183676, 0.0475193850280471, 0.04880836341765266,
    0.050115574864296755, 0.05144120027427573, 0.05278542055388596, 0.05414841660942382,
    0.055530369347185686, 0.05693145967346789, 0.05835186849456683, 0.05979177671677889,
    0.06125134402733013, 0.0627306452371655, 0.06422973393815963, 0.06574866372218725,
    0.06728748818112294, 0.06884626090684144, 0.07042503549121736, 0.07202386552612536,
    0.07364280460344011, 0.07528190631503627, 0.0769412242527885, 0.07862081200857149,
    0.08032072317425984, 0.08204101134172825, 0.08378173010285139, 0.0855429330495039,
    0.08732467377356042, 0.08912700586689566, 0.09094998292138422, 0.09279365852890081,
    0.09465808628132008, 0.09654331977051668, 0.09844941258836529, 0.10037641832674053,
    0.10232439057751708, 0.10429338293256958, 0.10628344898377279, 0.10829464232300128,
    0.11032701654212967, 0.11238062523303273, 0.11445552198758502, 0.11655176039766128,
    0.11866939405513612, 0.1208084765518842, 0.12296906147978025, 0.12515120243069883,
    0.12735495299651467, 0.1295803667691024, 0.13182749734033666, 0.13409639830209213,
    0.13638712324624355, 0.13869972576466547, 0.14103425944923256, 0.14339077789181956,
    0.14576933468430103, 0.14816998341855173, 0.15059277768644622, 0.15303777107985925,
    0.1555050171906654, 0.15799456961073938, 0.16050648193195585, 0.16304080774618943,
    0.16559760064531487, 0.16817691422120676, 0.17077880206573973, 0.17340331777078852,
    0.1760505149282277, 0.17872044712993204, 0.1814131679677761, 0.18412873103363459,
    0.18686718991938217, 0.18962859821689348, 0.19241322376272835, 0.19522219137218724,
    0.19805684010525568, 0.20091850902191935, 0.20380853718216377, 0.20672826364597457,
    0.2096790274733374, 0.21266216772423774, 0.21567902345866122, 0.21873093373659347,
    0.22181923761802008, 0.2249452741629267, 0.22811038243129877, 0.231315901483122,
    0.234563170378382, 0.2378535281770643, 0.2411883139391545, 0.2445688667246383,
    0.2479965255935011, 0.25147262960572864, 0.2549985178213065, 0.25857552930022026,
    0.2622050031024555, 0.26588827828799777, 0.2696266939168328, 0.2734215890489461,
    0.27727430274432324, 0.2811861740629498, 0.2851585420648115, 0.28919274580989385,
    0.29329012435818236, 0.2974520167696628, 0.30167954500568883, 0.30597296263308676,
    0.31033230612005097, 0.314757611934776, 0.31924891654545634, 0.3238062564202863,
    0.32842966802746065, 0.3331191878351734, 0.33787485231161934, 0.34269669792499285,
    0.3475847611434884, 0.35253907843530036, 0.35755968626862317, 0.36264662111165136,
    0.3677999194325794, 0.3730196176996018, 0.3783057523809129, 0.38365835994470704,
    0.3890774768591791, 0.3945631395925231, 0.4001153846129337, 0.4057342483886053,
    0.4114197673877324, 0.4171719780785095, 0.42299091692913093, 0.4288766204077912,
    0.4348291249826848, 0.44084846712200615, 0.44693468329394964, 0.4530878099667099,
    0.45930788360848124, 0.4655948393826846, 0.47194820723364705, 0.4783674158009217,
    0.48485189372406234, 0.49140106964262215, 0.49801437219615463, 0.5046912300242129,
    0.5114310717663507, 0.5182333260621212, 0.5250974215510781, 0.5320227868727746,
    0.5390088506667641, 0.5460550415725999, 0.5531607882298355, 0.5603255192780244,
    0.5675486633567199, 0.5748296491054756, 0.5821679051638445, 0.5895628601713804,
    0.5970139427676364, 0.6045205815921662, 0.612082205284523, 0.6196982424842601,
    0.6273681218309314, 0.6350912719640895, 0.6428671215232885, 0.6506950991480817,
    0.6585746334780221, 0.6665051531526636, 0.6744860868115592, 0.6825168630942625,
    0.6905969106403269, 0.6987256580893058, 0.7069025340807524, 0.7151269672542204,
    0.7233983862492631, 0.7317162197054341, 0.7400798962622864, 0.7484888445593735,
    0.7569424932362492, 0.7654402709324662, 0.7739816062875785, 0.7825659279411395,
    0.7911926645327022, 0.7998612447018203, 0.8085710970880473, 0.817321650330936,
    0.8261123330700405, 0.8349425739449137, 0.8438118015951095, 0.852719444660181,
    0.8616649317796815, 0.8706476915931647, 0.8796671527401837, 0.8887227438602919,
    0.897813893593043, 0.9069400305779904, 0.9161005834546873, 0.9252949808626869,
    0.9345226514415432, 0.943783023830809, 0.9530755266700381, 0.962399588598784,
    0.9717546382565994, 0.9811401042830384, 0.9905554153176543, 1.0
], dtype=np.float32)

_ARIA_CCM = np.array([
    [0.7436561584472656,  0.15223266184329987, -0.012550695799291134],
    [0.02287297323346138, 0.8269245028495789,  -0.004977707751095295],
    [-0.02940891683101654, -0.08261162042617798, 0.5401387810707092 ],
], dtype=np.float32)
_ARIA_CCM_INV = np.linalg.inv(_ARIA_CCM).astype(np.float32)
_ARIA_INV_CRF_X = np.linspace(0.0, 1.0, 256, dtype=np.float32)

CHANNEL_BALANCE_ADT_APARTMENT = np.array([0.925, 1.085, 0.883], dtype=np.float32)


def apply_aria_forward_isp(srgb_img: np.ndarray,
                           channel_balance: np.ndarray | None = CHANNEL_BALANCE_ADT_APARTMENT
                           ) -> np.ndarray:
    v = srgb_img.astype(np.float32) / 255.0
    linear = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    H, W, _ = linear.shape
    cam_lin = (linear.reshape(-1, 3) @ _ARIA_CCM_INV.T).reshape(H, W, 3)
    cam_lin = np.clip(cam_lin, 0.0, 1.0)
    out = np.empty_like(cam_lin)
    for c in range(3):
        out[..., c] = np.interp(cam_lin[..., c], _ARIA_INV_CRF, _ARIA_INV_CRF_X)
    out_uint8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    if channel_balance is not None:
        cb = np.asarray(channel_balance, dtype=np.float32)
        out_uint8 = np.clip(
            out_uint8.astype(np.float32) * cb[np.newaxis, np.newaxis, :], 0, 255
        ).astype(np.uint8)
    return out_uint8


def colorize_seg(seg_uid):
    import hashlib
    vis = np.zeros((*seg_uid.shape, 3), dtype=np.uint8)
    for uid in np.unique(seg_uid):
        if uid == 0:
            continue
        d = hashlib.md5(str(uid).encode()).digest()
        vis[seg_uid == uid] = (d[0], d[1], d[2])
    return vis


def visualize_normal(normal_xyz: np.ndarray) -> np.ndarray:
    """Convert float32 (H,W,3) normal map in [-1,1] to uint8 RGB image.

    Maps each component from [-1, +1] → [0, 255].
    Common convention: positive X/Y/Z → warm/green/blue tones.
    Invalid/zero normals (black in the input) remain near mid-grey (128).
    """
    vis = np.clip((normal_xyz + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    return vis


def visualize_depth(depth: np.ndarray,
                    near: float = 0.1,
                    far_pct: float = 99.0) -> np.ndarray:
    """Convert float32 (H,W) depth in metres to uint8 grayscale image.

    Uses log-scale normalisation between `near` and the `far_pct`-th
    percentile of finite depths, so highlights near-object details.
    Invalid (inf/nan) pixels are rendered as black.
    """
    valid = np.isfinite(depth) & (depth > near)
    if valid.sum() == 0:
        return np.zeros(depth.shape, dtype=np.uint8)
    far = float(np.percentile(depth[valid], far_pct))
    d_clip = np.clip(depth, near, far)
    log_d  = np.log(d_clip)
    log_near, log_far = np.log(near), np.log(far)
    norm   = (log_d - log_near) / (log_far - log_near + 1e-9)
    norm   = np.clip(norm, 0.0, 1.0)
    vis    = (norm * 255.0).astype(np.uint8)
    vis[~valid] = 0   # black for background
    return vis


def main():
    parser = argparse.ArgumentParser(
        description='Render ADT frames with segmentation, normal, and depth maps.')
    parser.add_argument('--num_frames',  type=int,   default=None)
    parser.add_argument('--frame_step',  type=int,   default=30,
                        help='Render every Nth frame (default: 30 ≈ 1 fps)')
    parser.add_argument('--output_size', type=int,   default=512)
    parser.add_argument('--focal',       type=float, default=None,
                        help='Focal length in pixels at output_size. '
                             'Default: scaled from ADT calibration (611@1408)')
    parser.add_argument('--output_dir',  type=str,
                        default=f'{BASE}/blender_rendered_maps')
    parser.add_argument('--fisheye',     action='store_true',
                        help='Render equirectangular panorama and remap to '
                             'Aria FISHEYE624 projection')
    # Auxiliary map toggles (all on by default; use --no_* to disable)
    parser.add_argument('--no_segmentation', action='store_true')
    parser.add_argument('--no_normals',      action='store_true',
                        help='Skip surface normal map output')
    parser.add_argument('--no_depth',        action='store_true',
                        help='Skip depth map output')
    args = parser.parse_args()
    args.segmentation = not args.no_segmentation
    args.normals      = not args.no_normals
    args.depth        = not args.no_depth

    # ── Output directories ─────────────────────────────────────────────────
    os.makedirs(f'{args.output_dir}/rgb',          exist_ok=True)
    os.makedirs(f'{args.output_dir}/comparison',   exist_ok=True)
    if args.segmentation:
        os.makedirs(f'{args.output_dir}/segmentation', exist_ok=True)
    if args.normals:
        os.makedirs(f'{args.output_dir}/normal_maps',  exist_ok=True)
    if args.depth:
        os.makedirs(f'{args.output_dir}/depth_maps',   exist_ok=True)
    tmp_dir = f'{args.output_dir}/_tmp'
    os.makedirs(tmp_dir, exist_ok=True)

    print('Loading VRS and ground truth...')
    p_ego  = data_provider.create_vrs_data_provider(EGO_VRS)
    calib  = p_ego.get_device_calibration()
    T_DC   = calib.get_camera_calib('camera-rgb').get_transform_device_camera().to_matrix()
    n_frames = p_ego.get_num_data(RGB_STREAM)

    cam_calib_rgb = calib.get_camera_calib('camera-rgb')

    if args.focal is None:
        calib_focal = cam_calib_rgb.get_focal_lengths()[0]
        calib_size  = cam_calib_rgb.get_image_size()[0]
        args.focal  = calib_focal * (args.output_size / calib_size)
        print(f'  Auto focal: {calib_focal:.1f}px @ {calib_size}px → '
              f'{args.focal:.1f}px @ {args.output_size}px')

    # ── Fisheye remap setup ────────────────────────────────────────────────
    fisheye_remap = None
    EQ_W = max(1024, args.output_size * 2)
    EQ_H = EQ_W // 2
    if args.fisheye:
        remap_cache = f'{args.output_dir}/_remap_{args.output_size}.npz'
        if os.path.exists(remap_cache):
            print('  Loading cached fisheye remap…')
            d = np.load(remap_cache)
            fisheye_remap = (d['map_x'], d['map_y'], d['valid'])
        else:
            print(f'  Precomputing FISHEYE624 remap '
                  f'({args.output_size}×{args.output_size} → {EQ_W}×{EQ_H})…')
            map_x, map_y, valid = build_fisheye624_remap(
                cam_calib_rgb, args.output_size, args.output_size, EQ_W, EQ_H)
            np.savez_compressed(remap_cache, map_x=map_x, map_y=map_y, valid=valid)
            fisheye_remap = (map_x, map_y, valid)
            print(f'    Done. {valid.sum()} valid pixels '
                  f'({100*valid.mean():.1f}% of image)')
        os.makedirs(f'{args.output_dir}/fisheye', exist_ok=True)

    with open(f'{GT_DIR}/instances.json') as f:
        instances = json.load(f)
    traj, ts_arr  = load_trajectory(f'{GT_DIR}/aria_trajectory.csv')
    static_poses, dynamic_poses = load_all_object_poses(f'{GT_DIR}/scene_objects.csv')

    print('Building scene lights...')
    scene_lights = build_scene_lights(f'{GT_DIR}/scene_objects.csv',
                                      f'{GT_DIR}/instances.json')

    static_object_list = build_object_list(instances, static_poses, MODELS_DIR)
    print(f'  Static objects with GLB models : {len(static_object_list)}')
    print(f'  Dynamic object UIDs            : {len(dynamic_poses)}')

    frames_idx = list(range(0, n_frames, args.frame_step))
    if args.num_frames is not None:
        frames_idx = frames_idx[:args.num_frames]
    print(f'  Rendering {len(frames_idx)} frames (step={args.frame_step})')
    print(f'  Outputs: rgb | comparison'
          f'{" | segmentation" if args.segmentation else ""}'
          f'{" | normal_maps"  if args.normals      else ""}'
          f'{" | depth_maps"   if args.depth        else ""}')

    for count, idx in enumerate(frames_idx):
        out_png = f'{args.output_dir}/rgb/frame_{idx:04d}.png'
        if os.path.exists(out_png):
            print(f'  [{count+1}/{len(frames_idx)}] frame {idx:04d} already exists, skipping')
            continue

        img_data = p_ego.get_image_data_by_index(RGB_STREAM, idx)
        ts_ns    = img_data[1].capture_timestamp_ns
        ego_rgb  = img_data[0].to_numpy_array()

        T_WD = nearest_pose(traj, ts_arr, ts_ns // 1000)
        T_WC = T_WD @ T_DC
        cam_pos = T_WC[:3, 3]

        dyn_poses_frame  = resolve_dynamic_poses(dynamic_poses, ts_ns)
        dyn_object_list  = build_object_list(instances, dyn_poses_frame, MODELS_DIR)
        all_objects      = static_object_list + dyn_object_list

        objs_with_dist = [
            (np.linalg.norm(np.array(obj['T_WO']).reshape(4,4)[:3,3] - cam_pos), obj)
            for obj in all_objects
        ]
        objs_with_dist.sort(key=lambda x: x[0])
        visible_objects = [obj for _, obj in objs_with_dist[:80]]
        n_dyn = sum(1 for _, obj in objs_with_dist[:80]
                    if any(obj is o for o in dyn_object_list))
        print(f'    Distance-culled to {len(visible_objects)} objects '
              f'({n_dyn} dynamic, '
              f'max dist: {objs_with_dist[min(79,len(objs_with_dist)-1)][0]:.1f}m)')

        pass_idx_to_uid: dict[int, int] = {}
        for i, obj in enumerate(visible_objects):
            obj['pass_index'] = i + 1
            try:
                pass_idx_to_uid[i + 1] = int(obj.get('uid', 0))
            except (ValueError, TypeError):
                pass_idx_to_uid[i + 1] = 0

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
            'cycles_samples':  16 if args.fisheye else 32,
            'pass_idx_to_uid': {str(k): v for k, v in pass_idx_to_uid.items()},
        }
        json_path = f'{tmp_dir}/frame_{idx:04d}.json'
        with open(json_path, 'w') as f:
            json.dump(frame_data, f)

        blender_out     = (f'{tmp_dir}/equirect_{idx:04d}.png'
                           if args.fisheye else out_png)
        seg_exr_tmp     = f'{tmp_dir}/seg_{idx:04d}.exr'
        normal_exr_tmp  = f'{tmp_dir}/normal_{idx:04d}.exr'
        depth_exr_tmp   = f'{tmp_dir}/depth_{idx:04d}.exr'

        # ── Blender subprocess ────────────────────────────────────────────
        print(f'  [{count+1}/{len(frames_idx)}] Rendering frame {idx:04d}...')
        cmd = [
            BLENDER_BIN, '--background', '--python', BLEND_SCRIPT,
            '--', '--frame_data', json_path, '--output', blender_out,
        ]
        if args.segmentation:
            cmd += ['--seg_output',    seg_exr_tmp]
        if args.normals:
            cmd += ['--normal_output', normal_exr_tmp]
        if args.depth:
            cmd += ['--depth_output',  depth_exr_tmp]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if not os.path.exists(blender_out):
            print(f'    Output not created (returncode={result.returncode}).')
            if result.stderr:
                print(f'    stderr: {result.stderr[-500:]}')
            continue
        if result.returncode != 0:
            print(f'    Blender returned {result.returncode} but output exists — continuing.')

        # ── Fisheye remap: RGB ────────────────────────────────────────────
        if args.fisheye and fisheye_remap is not None:
            eq_img   = Image.open(blender_out)
            map_x, map_y, valid = fisheye_remap
            fish_arr = remap_equirect_to_fisheye(eq_img, map_x, map_y, valid)
            fish_arr = apply_aria_forward_isp(fish_arr)
            Image.fromarray(fish_arr).save(out_png)
            Image.fromarray(fish_arr).save(
                f'{args.output_dir}/fisheye/frame_{idx:04d}.png')
            render_img = fish_arr
        else:
            raw_render = np.array(Image.open(out_png).convert('RGB'))
            render_img = apply_aria_forward_isp(raw_render)
            Image.fromarray(render_img).save(out_png)

        # ── Segmentation post-processing ──────────────────────────────────
        if args.segmentation and os.path.exists(seg_exr_tmp):
            seg_raw3  = cv2.imread(seg_exr_tmp, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            seg_float = seg_raw3[..., 0] if seg_raw3 is not None else None
            if seg_float is None:
                print(f'    [seg] cv2 could not read {seg_exr_tmp}; skipping')
            else:
                if args.fisheye and fisheye_remap is not None:
                    map_x_s, map_y_s, valid_s = fisheye_remap
                    seg_float = cv2.remap(seg_float, map_x_s, map_y_s,
                                          interpolation=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=0.0)
                    seg_float[~valid_s] = 0.0
                pass_arr = np.round(seg_float).astype(np.int32)
                seg_uid  = np.zeros(pass_arr.shape, dtype=np.int64)
                for pidx_s, uid_s in pass_idx_to_uid.items():
                    seg_uid[pass_arr == pidx_s] = uid_s
                seg_uid = np.rot90(seg_uid, k=-1).copy()
                seg_npy_out = f'{args.output_dir}/segmentation/frame_{idx:04d}.npy'
                seg_vis_out = f'{args.output_dir}/segmentation/frame_{idx:04d}_vis.png'
                np.save(seg_npy_out, seg_uid)
                Image.fromarray(colorize_seg(seg_uid)).save(seg_vis_out)
                n_objs = len(np.unique(seg_uid)) - 1
                print(f'    Segmentation: {n_objs} objects → {seg_npy_out}')
        elif args.segmentation:
            print(f'    [seg] EXR not found at {seg_exr_tmp}')

        # ── Normal map post-processing ────────────────────────────────────
        # Blender Normal pass: EXR RGB where R=X, G=Y, B=Z in world space.
        # OpenCV reads EXR as BGR, so channel 0 = B = Z, channel 2 = R = X.
        # After reversing: [:, :, ::-1] → (H, W, 3) in XYZ order.
        if args.normals and os.path.exists(normal_exr_tmp):
            norm_bgr = cv2.imread(normal_exr_tmp,
                                  cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if norm_bgr is None:
                print(f'    [normal] cv2 could not read {normal_exr_tmp}; skipping')
            else:
                # BGR → XYZ (R=X, G=Y, B=Z in Blender convention)
                norm_xyz = norm_bgr[..., ::-1].copy()   # float32 (H,W,3) in [-1,1]

                if args.fisheye and fisheye_remap is not None:
                    map_x_n, map_y_n, valid_n = fisheye_remap
                    # Remap each channel separately with linear interp
                    remapped = np.stack([
                        cv2.remap(norm_xyz[..., c], map_x_n, map_y_n,
                                  interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
                        for c in range(3)
                    ], axis=-1)
                    remapped[~valid_n] = 0.0
                    norm_xyz = remapped

                # Rotate to display orientation (matching RGB frames)
                norm_xyz = np.rot90(norm_xyz, k=-1).copy()

                norm_npy_out = f'{args.output_dir}/normal_maps/frame_{idx:04d}.npy'
                norm_vis_out = f'{args.output_dir}/normal_maps/frame_{idx:04d}_vis.png'
                np.save(norm_npy_out, norm_xyz)
                Image.fromarray(visualize_normal(norm_xyz)).save(norm_vis_out)
                print(f'    Normal map saved → {norm_npy_out}')
        elif args.normals:
            print(f'    [normal] EXR not found at {normal_exr_tmp}')

        # ── Depth map post-processing ─────────────────────────────────────
        # Blender Z pass broadcast to RGB EXR.  Read R channel for depth in metres.
        # Background / sky pixels have values ~1e10; replace with np.inf.
        if args.depth and os.path.exists(depth_exr_tmp):
            depth_bgr = cv2.imread(depth_exr_tmp,
                                   cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if depth_bgr is None:
                print(f'    [depth] cv2 could not read {depth_exr_tmp}; skipping')
            else:
                depth = depth_bgr[..., 0].astype(np.float32)  # R channel = depth
                # Mark sky/background (very large values) as inf
                depth[depth > 1e6] = np.inf

                if args.fisheye and fisheye_remap is not None:
                    map_x_d, map_y_d, valid_d = fisheye_remap
                    # Use linear remap for depth (continuous values)
                    depth = cv2.remap(depth, map_x_d, map_y_d,
                                      interpolation=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=np.inf)
                    depth[~valid_d] = np.inf

                # Rotate to display orientation (matching RGB frames)
                depth = np.rot90(depth, k=-1).copy()

                depth_npy_out = f'{args.output_dir}/depth_maps/frame_{idx:04d}.npy'
                depth_vis_out = f'{args.output_dir}/depth_maps/frame_{idx:04d}_vis.png'
                np.save(depth_npy_out, depth)
                Image.fromarray(visualize_depth(depth)).save(depth_vis_out)
                finite_cnt = np.isfinite(depth).sum()
                median_d   = float(np.nanmedian(depth[np.isfinite(depth)])) \
                             if finite_cnt > 0 else 0.0
                print(f'    Depth map saved → {depth_npy_out}  '
                      f'(median {median_d:.2f}m, {finite_cnt} valid px)')
        elif args.depth:
            print(f'    [depth] EXR not found at {depth_exr_tmp}')

        # ── Side-by-side comparison ───────────────────────────────────────
        ego_img = np.array(Image.fromarray(ego_rgb).resize(
                      (args.output_size, args.output_size), Image.LANCZOS))
        gap     = np.ones((args.output_size, 4, 3), dtype=np.uint8) * 40
        side    = np.concatenate([ego_img, gap, render_img], axis=1)
        Image.fromarray(side).save(f'{args.output_dir}/comparison/frame_{idx:04d}.png')
        print(f'    Done.')

    print(f'\nAll done! Outputs in {args.output_dir}/')


if __name__ == '__main__':
    main()
