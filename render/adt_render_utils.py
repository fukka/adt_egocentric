"""
adt_render_utils.py
===================
Shared utilities for render_from_poses_blender.py and render_exocentric_blender.py.

Exports
-------
Coordinate constants : FLIP_YZ, R_x_neg90
Path helpers         : pick_models_dir, resolve_gt_dir
GLB rotation         : correct_object_rotation
Data loaders         : quat_to_matrix, load_trajectory, nearest_pose,
                       load_all_object_poses, resolve_dynamic_poses
Scene                : build_scene_lights, resolve_glb, build_object_list
Camera               : lookat_adt
Aria ISP             : apply_aria_forward_isp, CHANNEL_BALANCE_ADT_APARTMENT
Visualisation        : colorize_seg, visualize_normal, visualize_depth
"""

import os, csv, json, struct
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import os as _os; _os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2

# ── Coordinate-system constants ────────────────────────────────────────────
# ADT camera: +X right, +Y down, +Z forward (OpenCV).
# Blender camera: +X right, +Y up, -Z forward.
# T_WC_blender = T_WC_adt @ FLIP_YZ
FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])

# Undoes Blender's glTF→native vertex-coord conversion (−90° around X).
R_x_neg90 = np.array([[1, 0,  0],
                       [0, 0,  1],
                       [0,-1,  0]], dtype=float)


# ── Path helpers ─────────────────────────────────────────────────────────────

def pick_models_dir(primary: str, fallback: str) -> str:
    """Return primary if it contains .glb files, otherwise fallback."""
    if os.path.isdir(primary) and any(f.endswith('.glb') for f in os.listdir(primary)):
        return primary
    print(f'  [models] {primary!r} has no .glb files — falling back to {fallback!r}')
    return fallback


def resolve_gt_dir(base: str) -> str:
    """Return the ground-truth directory for a sequence.

    Files may live at {base}/ or {base}/groundtruth/ depending on sequence.
    """
    for candidate in [base, os.path.join(base, 'groundtruth')]:
        if os.path.exists(os.path.join(candidate, 'instances.json')):
            if candidate != base:
                print(f'  GT files found in groundtruth/ subdirectory: {candidate}')
            return candidate
    return base   # let the caller report missing files


# ── GLB baked-rotation helpers ────────────────────────────────────────────────

_glb_baked_rotation_cache: dict = {}


def _read_glb_baked_rotation(glb_path: str) -> np.ndarray:
    """Return the root node's baked rotation as a 3×3 matrix (identity if none).

    glTF stores the root node's local transform in the JSON chunk.
    'rotation' is a quaternion [x, y, z, w].
    """
    if glb_path in _glb_baked_rotation_cache:
        return _glb_baked_rotation_cache[glb_path]
    R = np.eye(3)
    try:
        with open(glb_path, 'rb') as f:
            f.read(12)
            chunk_len = struct.unpack('<I', f.read(4))[0]
            f.read(4)
            gltf = json.loads(f.read(chunk_len))
        nodes = gltf.get('nodes', [])
        if nodes and 'rotation' in nodes[0]:
            R = Rotation.from_quat(nodes[0]['rotation']).as_matrix()
    except Exception:
        pass
    _glb_baked_rotation_cache[glb_path] = R
    return R


def correct_object_rotation(T_WO: np.ndarray, glb_path: str | None = None) -> np.ndarray:
    """Return corrected 4×4 world matrix: T_WO @ R_baked @ R_x(-90°).

    R_x(-90°) undoes Blender's glTF→native vertex coord conversion.
    R_baked   re-applies the root-node authored→canonical rotation stored
              in the GLB JSON chunk, which Blender's importer absorbs into
              matrix_local but loses when we override matrix_world directly.
    """
    R_baked = _read_glb_baked_rotation(glb_path) if glb_path else np.eye(3)
    T_c = T_WO.copy()
    T_c[:3, :3] = T_WO[:3, :3] @ R_baked @ R_x_neg90
    return T_c


# ── ADT data loaders ──────────────────────────────────────────────────────────

def quat_to_matrix(tx, ty, tz, qx, qy, qz, qw) -> np.ndarray:
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [tx, ty, tz]
    return T


def load_trajectory(path: str) -> tuple[dict, np.ndarray]:
    """Load aria_trajectory.csv → ({ts_us: T_WD}, sorted ts array)."""
    traj = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ts = int(row['tracking_timestamp_us'])
            traj[ts] = quat_to_matrix(
                float(row['tx_world_device']), float(row['ty_world_device']),
                float(row['tz_world_device']),
                float(row['qx_world_device']), float(row['qy_world_device']),
                float(row['qz_world_device']), float(row['qw_world_device']))
    ts_arr = np.array(sorted(traj.keys()))
    return traj, ts_arr


def nearest_pose(traj: dict, ts_arr: np.ndarray, ts_us: int) -> np.ndarray:
    return traj[ts_arr[int(np.argmin(np.abs(ts_arr - ts_us)))]]


def load_all_object_poses(path: str) -> tuple[dict, dict]:
    """Load scene_objects.csv.

    Poses are returned as raw ADT T_WO matrices (rotation correction is applied
    later in build_object_list once the GLB path is known).

    Returns:
        static_poses  : {uid: T_WO_raw}       — objects fixed in the scene
        dynamic_poses : {uid: [(ts_ns, T_WO_raw), ...]}  — moving objects
    """
    static_poses, dynamic_poses = {}, {}
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


def resolve_dynamic_poses(dynamic_poses: dict, frame_ts_ns: int) -> dict:
    """Pick nearest-timestamp pose for every dynamic object at frame_ts_ns."""
    resolved = {}
    for uid, entries in dynamic_poses.items():
        ts_arr = np.array([e[0] for e in entries], dtype=np.int64)
        resolved[uid] = entries[int(np.argmin(np.abs(ts_arr - frame_ts_ns)))][1]
    return resolved


# ── Scene lights ──────────────────────────────────────────────────────────────

_LIGHTING_CFG_PATH = os.path.join(os.path.dirname(__file__), 'lighting_config.json')


def build_scene_lights(scene_objects_csv: str,
                       instances_json_path: str,
                       lighting_mode: str = 'egocentric') -> list[dict]:
    """Build scene-fixed lights list for Blender.

    Sources:
      1. Physical lamp props in scene_objects.csv — placed as POINT lights.
      2. Ceiling AREA-light grid (5×6 = 30 panels) approximating Unreal GI.

    Parameters (energy, colour, grid layout) are loaded from lighting_config.json
    so they can be tuned without editing code.  lighting_mode selects the
    ceiling height/energy calibrated for 'egocentric' or 'exocentric' views.

    Returns list of dicts with keys: type, location, energy, color, radius/size.
    """
    with open(_LIGHTING_CFG_PATH) as f:
        cfg = json.load(f)
    prop_cfg  = cfg['prop_lights']
    ceil_cfg  = cfg['ceiling']
    ceil_mode = ceil_cfg['modes'][lighting_mode]

    with open(instances_json_path) as f:
        instances = json.load(f)
    uid_to_name = {info['instance_id']: info.get('instance_name', '')
                   for info in instances.values() if 'instance_id' in info}

    def _prop_cfg_for(name):
        for key, c in prop_cfg.items():
            if key.lower() in name.lower():
                return c
        return None

    lights = []
    with open(scene_objects_csv) as f:
        for row in csv.DictReader(f):
            if row['timestamp[ns]'] != '-1':
                continue
            uid  = int(row['object_uid'])
            name = uid_to_name.get(uid, '')
            c    = _prop_cfg_for(name)
            if c is None:
                continue
            x = float(row['t_wo_x[m]'])
            y = float(row['t_wo_y[m]'])
            z = float(row['t_wo_z[m]'])
            y_offset = 0.30 if 'Lamp' in name else 0.10
            lights.append({'type':     'POINT',
                           'location': [x, y + y_offset, z],
                           'energy':   c['energy'],
                           'color':    c['color'],
                           'radius':   c['radius']})
            print(f'    prop light: {name:<30} ({x:.2f}, {y:.2f}, {z:.2f})')

    # Ceiling area-light grid
    for cx in ceil_cfg['grid_x']:
        for cz in ceil_cfg['grid_z']:
            lights.append({'type':     'AREA',
                           'location': [cx, ceil_mode['y'], cz],
                           'energy':   ceil_mode['energy'],
                           'color':    ceil_cfg['color'],
                           'size':     ceil_cfg['size']})

    n_pt   = sum(1 for l in lights if l['type'] == 'POINT')
    n_area = sum(1 for l in lights if l['type'] == 'AREA')
    print(f'  Scene lights: {n_pt} prop points + {n_area} ceiling area  '
          f'(mode={lighting_mode}, ceil_y={ceil_mode["y"]}, energy={ceil_mode["energy"]}W)')
    return lights


# ── GLB resolution and object-list builder ────────────────────────────────────

def resolve_glb(models_dir: str, name: str) -> str | None:
    """Try flat and subdirectory GLB layouts; return first path that exists.

      1. {models_dir}/{name}.glb           ← download_adt_object_models.py (flat)
      2. {models_dir}/{name}/3d-asset.glb  ← per-object subdirectory layout
    """
    for p in [os.path.join(models_dir, f'{name}.glb'),
              os.path.join(models_dir, name, '3d-asset.glb')]:
        if os.path.exists(p):
            return p
    return None


def build_object_list(instances: dict, obj_poses: dict, models_dir: str) -> list[dict]:
    """Resolve GLB paths and apply rotation correction for each object pose.

    GLB lookup order per candidate name (instance_name first, then prototype_name):
      1. {models_dir}/{name}.glb
      2. {models_dir}/{name}/3d-asset.glb
    """
    uid_to_info = {str(v['instance_id']): v for v in instances.values()}
    result = []
    for uid, T_WO_raw in obj_poses.items():
        info = uid_to_info.get(uid)
        if info is None:
            continue
        glb_path = None
        for candidate in [info.get('instance_name', ''), info.get('prototype_name', '')]:
            if candidate:
                glb_path = resolve_glb(models_dir, candidate)
                if glb_path:
                    break
        if glb_path:
            result.append({'glb_path': glb_path,
                           'T_WO':     correct_object_rotation(T_WO_raw, glb_path).flatten().tolist(),
                           'uid':      uid})
    return result


# ── Camera lookat helper ──────────────────────────────────────────────────────

def lookat_adt(eye, target, world_up=None) -> np.ndarray:
    """Build a 4×4 T_WC matrix in ADT convention from an eye + target point.

    ADT camera convention: col0=right (+X), col1=down (-Y_image), col2=forward (+Z).

    Strategy: build T_WC_blender (col1=image-up, col2=backward) then back-convert
    via FLIP_YZ = diag([1,-1,-1]):  T_WC_adt = T_WC_blender @ FLIP_YZ.
    """
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0])
    eye, target, world_up = (np.asarray(x, dtype=float) for x in (eye, target, world_up))

    fwd     = target - eye;  fwd    /= np.linalg.norm(fwd)
    neg_fwd = -fwd                                   # Blender local +Z (backward)
    right   = np.cross(world_up, neg_fwd)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(np.array([0., 0., 1.]), neg_fwd)
    right  /= np.linalg.norm(right)
    img_up  = np.cross(neg_fwd, right);  img_up /= np.linalg.norm(img_up)

    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = -img_up   # ADT col1 = down = -Blender image-up
    T[:3, 2] = fwd
    T[:3, 3] = eye
    return T


# ── Aria Gen1 forward ISP ─────────────────────────────────────────────────────
# Source: projectaria_tools/core/image/utility/ColorCorrectData.h
# cameraInvCRFTableGen1: maps camera pixel index (0–255) → linear float.
# Used here in the FORWARD direction by inverting the LUT via np.interp.

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

# colorCorrectionMatrixDataGen1: standard linear RGB → camera linear RGB (row-major 3×3).
_ARIA_CCM = np.array([
    [ 0.7436561584472656,   0.15223266184329987, -0.012550695799291134],
    [ 0.02287297323346138,  0.8269245028495789,  -0.004977707751095295],
    [-0.02940891683101654, -0.08261162042617798,  0.5401387810707092  ],
], dtype=np.float32)
_ARIA_CCM_INV  = np.linalg.inv(_ARIA_CCM).astype(np.float32)
_ARIA_INV_CRF_X = np.linspace(0.0, 1.0, 256, dtype=np.float32)

# Per-channel balance calibrated for the ADT Apartment sequence after Aria ISP.
# Compensates for the residual tint between Blender Cycles and Omniverse rendering.
# Scales: R=60.4/65.3=0.925, G=52.7/48.6=1.085, B=48.3/54.7=0.883
CHANNEL_BALANCE_ADT_APARTMENT = np.array([0.925, 1.085, 0.883], dtype=np.float32)


def apply_aria_forward_isp(srgb_img: np.ndarray,
                           channel_balance: np.ndarray | None = CHANNEL_BALANCE_ADT_APARTMENT
                           ) -> np.ndarray:
    """Convert a Standard-sRGB uint8 image to Aria Gen1 camera ISP colour space.

    Pipeline:
      sRGB uint8 → linear float (inverse sRGB γ=2.4)
      → camera linear (inverse CCM: standard → sensor primaries)
      → camera pixel  (forward CRF: inverts cameraInvCRFTableGen1 LUT)
      → optional per-channel balance → uint8

    Args:
        srgb_img:        H×W×3 uint8 in Standard sRGB.
        channel_balance: Length-3 per-channel multipliers (camera pixel space).
                         Pass None to disable.  Default: CHANNEL_BALANCE_ADT_APARTMENT.
    Returns:
        H×W×3 uint8 in Aria Gen1 ISP colour space.
    """
    v      = srgb_img.astype(np.float32) / 255.0
    linear = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    H, W, _ = linear.shape
    cam_lin = np.clip((linear.reshape(-1, 3) @ _ARIA_CCM_INV.T).reshape(H, W, 3), 0.0, 1.0)
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


# ── Visualisation helpers ─────────────────────────────────────────────────────

def colorize_seg(seg_uid: np.ndarray) -> np.ndarray:
    """Map instance UIDs → stable hash-based RGB colours for visualisation."""
    import hashlib
    vis = np.zeros((*seg_uid.shape, 3), dtype=np.uint8)
    for uid in np.unique(seg_uid):
        if uid == 0:
            continue
        d = hashlib.md5(str(uid).encode()).digest()
        vis[seg_uid == uid] = (d[0], d[1], d[2])
    return vis


def visualize_normal(normal_xyz: np.ndarray) -> np.ndarray:
    """Float32 (H,W,3) normals in [-1,1] → uint8 RGB.
    Maps [-1,+1] → [0,255]; background (0,0,0) maps to mid-grey (128).
    """
    return np.clip((normal_xyz + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)


def visualize_depth(depth: np.ndarray,
                    near: float = 0.1,
                    far_pct: float = 99.0) -> np.ndarray:
    """Float32 (H,W) depth in metres → uint8 grayscale (log-scale).
    Uses log normalisation between near and the far_pct-th percentile of finite depths.
    Invalid (inf/nan) pixels are rendered black.
    """
    valid = np.isfinite(depth) & (depth > near)
    if valid.sum() == 0:
        return np.zeros(depth.shape, dtype=np.uint8)
    far      = float(np.percentile(depth[valid], far_pct))
    log_d    = np.log(np.clip(depth, near, far))
    log_near = np.log(near);  log_far = np.log(far)
    norm     = np.clip((log_d - log_near) / (log_far - log_near + 1e-9), 0.0, 1.0)
    vis      = (norm * 255.0).astype(np.uint8)
    vis[~valid] = 0
    return vis