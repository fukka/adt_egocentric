"""
ADT Object-Pose → Blender Rendering Pipeline
=============================================
Drives blender_render_scene.py for each frame:
  1. Reads ADT ground truth object 6DoF poses and camera trajectory
  2. Exports per-frame JSON (camera pose + object GLB paths + T_WO)
  3. Calls Blender headless to render each frame

Usage:
    python render_from_poses_blender.py [--num_frames N] [--frame_step K]
                                        [--output_size S] [--focal F]
"""

import sys, os, csv, json, argparse, subprocess
sys.path.insert(0, '/sessions/dreamy-modest-brown/.local/lib/python3.10/site-packages')

import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import os as _os; _os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')  # enable EXR read
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

# ── Object rotation convention fix ─────────────────────────────────────────
# ADT T_WO quaternions encode object orientation assuming the object's
# canonical "standing axis" is local +Z.  Most GLB object models are authored
# in the glTF 2.0 Y-up convention (standing axis = local +Z of the mesh data).
#
# Standard fix (no baked rotation in GLB root node):
#   T_corrected = T_WO_adt @ R_x(-90°)
#   R_x(-90°) = [[1, 0,  0],
#                [0, 0,  1],
#                [0,-1,  0]]
#   Effect: local +Z of mesh → world +Y (gravity-opposite = "up")  ✓
#
# Some GLB files have an additional rotation baked into the root node transform
# (visible in the glTF JSON as nodes[0].rotation).  Blender's GLTF importer
# sets matrix_world = baked_node_rotation, but our pipeline overwrites
# matrix_world entirely, so the baked rotation is silently discarded.
# To restore it we must compose it INTO the correction:
#
#   T_corrected = T_WO_adt @ R_x(-90°) @ R_baked
#
# Derivation:
#   We want: vertex_world = T_WO_adt @ R_x(-90°) @ (R_baked @ vertex_prebaked)
#   Blender applies: matrix_world @ vertex_prebaked
#   ∴ matrix_world = T_WO_adt @ R_x(-90°) @ R_baked
#
# For objects without a baked rotation R_baked = I → same as before.
#
# NOTE (diagnostic finding): Blender's glTF importer bakes any root-node
# rotation into vertex positions during import.  So after import_scene.gltf,
# the mesh vertices are already in "post-baked" space and only R_x(-90°) is
# needed — no additional R_baked term.  Applying R_baked again would
# double-rotate objects with a baked root-node TRS (verified on
# WhiteFlatwareTray.glb which has R_y(90°) baked in nodes[0].rotation).

R_x_neg90 = np.array([[1, 0,  0],
                       [0, 0,  1],
                       [0,-1,  0]], dtype=float)

def correct_object_rotation(T_WO, glb_path: str | None = None):
    """Post-multiply rotation by R_x(-90°) to produce the correct world matrix
    for a GLB object.

    Blender's glTF importer bakes any root-node rotation from the GLB's JSON
    chunk directly into vertex positions during import.  When we override
    matrix_world after import, the vertices are therefore already in the
    "post-baked" local space.  Applying R_baked a second time would
    double-rotate the object.  Consequently, we need only R_x(-90°) regardless
    of whether the GLB has a baked root-node rotation or not.

    R_x(-90°) maps the object's local +Z → world +Y (up in ADT Y-up).  This is
    the ADT object convention: T_WO represents the object pose assuming its
    canonical "standing axis" is +Z; R_x(-90°) re-orients that to world +Y.

    Args:
        T_WO:     4×4 ADT world-from-object transform (numpy).
        glb_path: Retained for API compatibility; no longer used.

    Returns corrected 4×4 matrix ready to pass as Blender matrix_world.
    """
    T_c = T_WO.copy()
    T_c[:3, :3] = T_WO[:3, :3] @ R_x_neg90
    return T_c


# ── FISHEYE624 remap helpers ────────────────────────────────────────────────

def build_fisheye624_remap(cam_calib, out_w, out_h, eq_w, eq_h):
    """
    Precompute a pixel-level lookup table that maps each output fisheye pixel
    (u, v) → (eq_x, eq_y) in an equirectangular panorama rendered by Blender.

    The Aria FISHEYE624 model (Kannala-Brandt style) uses:
        r = fx * θ * (1 + k0*θ² + k1*θ⁴ + k2*θ⁶ + k3*θ⁸ + k4*θ¹⁰ + k5*θ¹²)
    where θ is the angle from the optical axis.

    Steps for each output pixel:
      1. Invert the FISHEYE624 projection → get ray in Aria camera space (+Z forward)
      2. Convert to Blender camera space via FLIP_YZ (+Y up, -Z forward)
      3. Convert to equirectangular azimuth/elevation → pixel coords in eq render

    Returns (map_x, map_y, valid_mask) as numpy float32 arrays of shape (out_h, out_w).
    map_x / map_y are the equirectangular pixel coordinates to sample.
    """
    params     = cam_calib.get_projection_params()
    fx         = params[0]
    cx, cy     = params[1], params[2]
    k0, k1, k2, k3, k4, k5 = params[3:9]
    # tangential / thin-prism (params[9:15]) are < 1e-3 — ignored for remap

    # Scale calibration if output differs from native 1408px
    native_size = float(cam_calib.get_image_size()[0])
    scale       = out_w / native_size
    fx_s        = fx * scale
    cx_s        = cx * scale + (scale - 1) * 0.5  # shift for half-pixel origin
    cy_s        = cy * scale + (scale - 1) * 0.5
    valid_r_s   = cam_calib.get_valid_radius() * scale

    # Build pixel grid
    u = np.arange(out_w, dtype=np.float64)
    v = np.arange(out_h, dtype=np.float64)
    UU, VV = np.meshgrid(u, v)          # (out_h, out_w)
    u_hat = (UU - cx_s).ravel()
    v_hat = (VV - cy_s).ravel()
    r = np.sqrt(u_hat**2 + v_hat**2)

    # Newton-Raphson: solve  r = fx_s * θ * poly(θ²)  for θ
    theta = r / fx_s   # equidistant initialisation
    for _ in range(25):
        t2 = theta * theta
        poly  = 1.0 + t2 * (k0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * (k4 + t2 * k5)))))
        dpoly = t2 * (2*k0 + t2 * (4*k1 + t2 * (6*k2 + t2 * (8*k3 + t2 * (10*k4 + t2 * 12*k5)))))
        fval  = fx_s * theta * poly - r
        dfval = fx_s * (poly + dpoly)
        theta -= fval / (dfval + 1e-12)
        theta  = np.clip(theta, 0.0, np.pi)

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    r_safe = np.maximum(r, 1e-9)
    # Ray in Aria camera space: +X right, +Y down (OpenCV), +Z forward
    ray_x = sin_t * (u_hat / r_safe)
    ray_y = sin_t * (v_hat / r_safe)
    ray_z = cos_t

    # Convert to Blender camera space: FLIP_YZ  →  +X right, +Y up, -Z forward
    rx_bl =  ray_x
    ry_bl = -ray_y
    rz_bl = -ray_z      # Blender camera looks along local -Z

    # Equirectangular convention (Blender EQUIRECTANGULAR panoramic):
    #   longitude  = atan2(+X, -Z_local)  in [-π, π]   (0 = forward, +90° = right)
    #   latitude   = atan2(+Y, sqrt(X²+Z²))  in [-π/2, π/2]
    #   eq_u = (lon + π) / (2π)  →  [0, 1]
    #   eq_v = (π/2 - lat) / π   →  [0, 1]   (top of image = +lat = +Y = up)
    lon = np.arctan2(rx_bl, -rz_bl)
    lat = np.arctan2(ry_bl, np.sqrt(rx_bl**2 + rz_bl**2))

    map_x = ((lon + np.pi) / (2 * np.pi) * eq_w).astype(np.float32)
    map_y = ((np.pi / 2 - lat) / np.pi    * eq_h).astype(np.float32)

    valid = (r < valid_r_s)

    return (map_x.reshape(out_h, out_w),
            map_y.reshape(out_h, out_w),
            valid.reshape(out_h, out_w))


def remap_equirect_to_fisheye(equirect_img, map_x, map_y, valid_mask):
    """Apply precomputed remap: equirectangular (PIL/numpy) → fisheye numpy array."""
    if isinstance(equirect_img, Image.Image):
        equirect_arr = np.array(equirect_img.convert('RGB'), dtype=np.uint8)
    else:
        equirect_arr = np.asarray(equirect_img, dtype=np.uint8)
    out = cv2.remap(equirect_arr, map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    out[~valid_mask] = 0   # black outside fisheye circle
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
    """Load static (timestamp=-1) AND dynamic (timestamped) object poses.

    Poses are returned as RAW ADT T_WO matrices (no rotation correction
    applied yet).  Correction (R_x_neg90 @ R_baked) is applied per-object
    in build_object_list() once the GLB path is known, so that the baked
    node rotation in each GLB can be read and absorbed correctly.

    Returns:
        static_poses  : {uid: T_WO_raw}  — objects fixed in the scene
        dynamic_poses : {uid: [(ts_ns, T_WO_raw), ...]}  — moving objects
    """
    static_poses  = {}
    dynamic_poses = {}   # uid -> [(ts_ns, T_WO), ...]

    with open(path) as f:
        for row in csv.DictReader(f):
            uid   = row['object_uid']
            ts_ns = int(row['timestamp[ns]'])
            T = quat_to_matrix(
                float(row['t_wo_x[m]']), float(row['t_wo_y[m]']),
                float(row['t_wo_z[m]']),
                float(row['q_wo_x']),   float(row['q_wo_y']),
                float(row['q_wo_z']),   float(row['q_wo_w']))
            # NOTE: do NOT apply correct_object_rotation here — we need the
            # GLB path first (known only in build_object_list) to read the
            # baked node rotation before applying the full correction.
            if ts_ns == -1:
                static_poses[uid] = T
            else:
                dynamic_poses.setdefault(uid, []).append((ts_ns, T))

    for uid in dynamic_poses:
        dynamic_poses[uid].sort(key=lambda x: x[0])

    return static_poses, dynamic_poses


def resolve_dynamic_poses(dynamic_poses, frame_ts_ns):
    """Pick the nearest-timestamp pose for every dynamic object at frame_ts_ns."""
    resolved = {}
    for uid, entries in dynamic_poses.items():
        ts_arr = np.array([e[0] for e in entries], dtype=np.int64)
        idx = int(np.argmin(np.abs(ts_arr - frame_ts_ns)))
        resolved[uid] = entries[idx][1]
    return resolved

def build_scene_lights(scene_objects_csv, instances_json_path):
    """Build a list of scene-fixed lights for Blender from two sources:

    1. Physical lamp props in scene_objects.csv (Lamp_1, WhitTableLamp,
       NightLights, Candles) — placed as POINT lights at their world positions.

    2. Hardcoded ceiling AREA lights covering the apartment — approximating
       Unreal Engine's baked overhead room lighting which has no counterpart
       file in the ADT dataset.  Ceiling height ≈ 2.35m (SecurityCamera at
       2.41m is the tallest tracked object; apartment ceiling ≈ 2.5m).

    All positions are in ADT Y-up world coordinates (same as T_WO).
    The Blender script places lights directly at these world coordinates
    (no coordinate-system conversion needed — scene geometry is already
    imported in ADT coords).

    Returns a list of dicts with keys:
        type     : 'POINT' or 'AREA'
        location : [x, y, z]   in ADT world coords (Y-up)
        energy   : float        Watts (Cycles)
        color    : [r, g, b]   linear
        radius   : float        (POINT only) soft shadow radius in metres
        size     : float        (AREA only)  side length in metres
    """
    with open(instances_json_path) as f:
        instances = json.load(f)
    uid_to_name = {info['instance_id']: info.get('instance_name', '')
                   for info in instances.values() if 'instance_id' in info}

    # ── Prop lights: energy/colour per object type ────────────────────────
    # Night lights / table lamps: warm, moderate energy
    # Candles: very warm, low energy
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
            # Lamp_1 / WhitTableLamp: shift light slightly above the prop mesh
            y_offset = 0.30 if 'Lamp' in name else 0.10
            lights.append({
                'type':     'POINT',
                'location': [x, y + y_offset, z],
                'energy':   cfg['energy'],
                'color':    cfg['color'],
                'radius':   cfg['radius'],
            })
            print(f'    prop light: {name:<30} ({x:.2f}, {y:.2f}, {z:.2f})')

    # ── Ceiling AREA lights — approximate Unreal baked GI ─────────────────
    # The ADT Apartment spans roughly X∈[-4,3], Z∈[-4,9].
    # Two zones observed from scene geometry:
    #   Kitchen/entry  (X∈[-1,2], Z∈[-1,4]) — bowl and counters here
    #   Living/bedroom (X∈[-4,0], Z∈[-4,3]) — sofa, bed, shelves
    # Six 2×2m area lights cover the floor plan.  All at Y=2.35m (just below
    # ceiling).  Warm-white colour ≈ 3000 K (indoor tungsten/LED).
    CEIL_Y      = 2.35   # metres
    CEIL_ENERGY = 22.0   # Watts — calibrated to match ADT mean brightness ~66/255
    CEIL_COLOR  = [1.0, 0.90, 0.75]   # warm indoor white
    CEIL_SIZE   = 1.8    # metres — each tile covers a 1.8×1.8m patch
    for (cx, cz) in [
        ( 0.5,  2.5),   # above kitchen counter / WoodenBowl
        ( 0.5,  0.5),   # kitchen island / entry
        (-1.0, -1.5),   # living-room sofa area
        (-1.0,  4.5),   # dining / far kitchen
        (-2.5, -2.5),   # bedroom / hallway
        (-2.5,  1.5),   # living centre
    ]:
        lights.append({
            'type':     'AREA',
            'location': [cx, CEIL_Y, cz],
            'energy':   CEIL_ENERGY,
            'color':    CEIL_COLOR,
            'size':     CEIL_SIZE,
        })

    print(f'  Scene lights: {sum(1 for l in lights if l["type"]=="POINT")} prop points + '
          f'{sum(1 for l in lights if l["type"]=="AREA")} ceiling area lights')
    return lights


def build_object_list(instances, obj_poses, models_dir):
    """Return list of {glb_path, T_WO, uid} for objects that have a GLB model.

    Applies correct_object_rotation(T_WO, glb_path) here, after the GLB path
    is resolved, so that any baked node rotation stored in the GLB's JSON chunk
    is read and absorbed into the world matrix (R_x_neg90 @ R_baked).

    GLB lookup order:
      1. {instance_name}.glb  (DTC variant files, e.g. "Hook_4.glb")
      2. {prototype_name}.glb  (directly-named files, e.g. "KitchIsland.glb")
    """
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
                           'uid':      uid})   # ← instance UID for seg mapping
    return result


# ── Aria Gen1 colour pipeline ──────────────────────────────────────────────
# Source: projectaria_tools/core/image/utility/ColorCorrectData.h
# cameraInvCRFTableGen1: maps camera pixel index (0..255) → linear float.
# Used here in the FORWARD direction (linear → camera pixel) by inverting the LUT.
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

# colorCorrectionMatrixDataGen1: converts camera linear RGB → standard linear RGB.
# Rows = [R, G, B] output; columns = [R, G, B] input (row-major, 3×3).
_ARIA_CCM = np.array([
    [0.7436561584472656,  0.15223266184329987, -0.012550695799291134],
    [0.02287297323346138, 0.8269245028495789,  -0.004977707751095295],
    [-0.02940891683101654, -0.08261162042617798, 0.5401387810707092 ],
], dtype=np.float32)

# Inverse CCM: standard linear RGB → camera linear RGB
_ARIA_CCM_INV = np.linalg.inv(_ARIA_CCM).astype(np.float32)

# Pixel index axis for LUT interpolation (256 values mapping to [0, 1])
_ARIA_INV_CRF_X = np.linspace(0.0, 1.0, 256, dtype=np.float32)


# Scene-level colour balance correction calibrated to ADT Apartment sequence.
# Compensates for the residual per-channel mismatch between Blender Cycles
# (with warm ceiling lights) and Omniverse (ADT synthetic reference), AFTER
# the Aria Gen1 forward ISP has been applied.
# Calibrated from frame 0: target = ADT synthetic means (R=60.4, G=52.7, B=48.3)
# vs Blender Standard+ISP at EV=-0.8 (R=65.3, G=48.6, B=54.7).
# Scales: R=60.4/65.3=0.925, G=52.7/48.6=1.085, B=48.3/54.7=0.883
CHANNEL_BALANCE_ADT_APARTMENT = np.array([0.925, 1.085, 0.883], dtype=np.float32)


def apply_aria_forward_isp(srgb_img: np.ndarray,
                           channel_balance: np.ndarray | None = CHANNEL_BALANCE_ADT_APARTMENT
                           ) -> np.ndarray:
    """Convert a Standard-sRGB uint8 image to Aria Gen1 camera ISP colour space.

    ADT synthetic images are rendered with NVIDIA Omniverse + Aria ISP simulation,
    making them look like raw Aria camera output.  To match that colour space our
    Blender renders (rendered with Standard / sRGB colour management) must be passed
    through the same forward ISP:

      sRGB uint8
        → linear float  (inverse sRGB gamma, piecewise power γ=2.4)
        → camera linear  (inverse CCM: standard linear → sensor primaries)
        → camera pixel   (forward CRF: inverts the cameraInvCRFTableGen1 LUT)
        → optional per-channel balance scale  (corrects Blender-vs-Omniverse tint)
        → uint8 output   (Aria ISP colour space, matches ADT synthetic images)

    Args:
        srgb_img:        H×W×3 uint8 array in Standard sRGB colour space.
        channel_balance: Optional length-3 array of per-channel multipliers applied
                         after the ISP, in camera pixel space.  Compensates for the
                         residual tint from Blender's warm ceiling lights vs the ADT
                         Omniverse reference.  Pass None to disable.
                         Default: CHANNEL_BALANCE_ADT_APARTMENT (calibrated on
                         Apartment_release_golden_skeleton_seq100_10s_sample_M1292).

    Returns:
        H×W×3 uint8 array in Aria Gen1 camera ISP colour space.
    """
    # Step 1: sRGB → linear (inverse sRGB gamma)
    v = srgb_img.astype(np.float32) / 255.0
    linear = np.where(v <= 0.04045,
                      v / 12.92,
                      ((v + 0.055) / 1.055) ** 2.4)

    # Step 2: standard linear → camera linear (inverse CCM)
    H, W, _ = linear.shape
    cam_lin = (linear.reshape(-1, 3) @ _ARIA_CCM_INV.T).reshape(H, W, 3)
    cam_lin = np.clip(cam_lin, 0.0, 1.0)

    # Step 3: camera linear → camera pixel value (forward CRF)
    # _ARIA_INV_CRF maps pixel-index-fraction → linear; we invert via np.interp.
    # For each channel: pixel_fraction = interp(cam_lin_value, inv_crf_vals, x_axis)
    out = np.empty_like(cam_lin)
    for c in range(3):
        out[..., c] = np.interp(cam_lin[..., c], _ARIA_INV_CRF, _ARIA_INV_CRF_X)

    out_uint8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)

    # Step 4 (optional): scene-level channel balance correction
    if channel_balance is not None:
        cb = np.asarray(channel_balance, dtype=np.float32)
        out_uint8 = np.clip(
            out_uint8.astype(np.float32) * cb[np.newaxis, np.newaxis, :],
            0, 255
        ).astype(np.uint8)

    return out_uint8


def colorize_seg(seg_uid):
    """Map instance UIDs to stable RGB colours (hash-based) for visualisation."""
    import hashlib
    vis = np.zeros((*seg_uid.shape, 3), dtype=np.uint8)
    for uid in np.unique(seg_uid):
        if uid == 0:
            continue
        d = hashlib.md5(str(uid).encode()).digest()
        vis[seg_uid == uid] = (d[0], d[1], d[2])
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_frames',  type=int,   default=None)
    parser.add_argument('--frame_step',  type=int,   default=30,
                        help='Render every Nth frame (default: 30 = ~1fps)')
    parser.add_argument('--output_size', type=int,   default=512)
    # ADT camera-rgb: 611px focal at 1408px. Scaled to output_size: 611*(S/1408)
    parser.add_argument('--focal',       type=float, default=None,
                        help='Focal length in pixels at output_size. '
                             'Default: scale from ADT calibration (611@1408)')
    parser.add_argument('--output_dir',  type=str,
                        default=f'{BASE}/blender_rendered')
    parser.add_argument('--fisheye',     action='store_true',
                        help='Render equirectangular panorama and remap to '
                             'Aria FISHEYE624 projection (exact lens match)')
    args = parser.parse_args()

    os.makedirs(f'{args.output_dir}/rgb',          exist_ok=True)
    os.makedirs(f'{args.output_dir}/comparison',   exist_ok=True)
    os.makedirs(f'{args.output_dir}/segmentation', exist_ok=True)
    tmp_dir = f'{args.output_dir}/_tmp'
    os.makedirs(tmp_dir, exist_ok=True)

    print('Loading VRS and ground truth...')
    p_ego  = data_provider.create_vrs_data_provider(EGO_VRS)
    calib  = p_ego.get_device_calibration()
    T_DC   = calib.get_camera_calib('camera-rgb').get_transform_device_camera().to_matrix()
    n_frames = p_ego.get_num_data(RGB_STREAM)

    cam_calib_rgb = calib.get_camera_calib('camera-rgb')

    # Compute focal from calibration if not specified
    if args.focal is None:
        calib_focal = cam_calib_rgb.get_focal_lengths()[0]
        calib_size  = cam_calib_rgb.get_image_size()[0]
        args.focal  = calib_focal * (args.output_size / calib_size)
        print(f'  Auto focal: {calib_focal:.1f}px @ {calib_size}px → {args.focal:.1f}px @ {args.output_size}px')

    # Precompute fisheye remap if requested
    fisheye_remap = None
    # Equirect resolution: 2× the output size per axis gives good quality
    # while keeping Blender render time within the subprocess timeout.
    # (4× resolution is nicer but takes ~4× longer; 2× is a good trade-off.)
    EQ_W = max(1024, args.output_size * 2)   # equirect width  ≥ 1024
    EQ_H = EQ_W // 2                          # equirect height = EQ_W / 2
    if args.fisheye:
        remap_cache = f'{args.output_dir}/_remap_{args.output_size}.npz'
        if os.path.exists(remap_cache):
            print('  Loading cached fisheye remap map…')
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

    print('Building scene lights from prop positions + ceiling grid...')
    scene_lights = build_scene_lights(f'{GT_DIR}/scene_objects.csv',
                                      f'{GT_DIR}/instances.json')

    static_object_list = build_object_list(instances, static_poses, MODELS_DIR)
    print(f'  Static objects with GLB models : {len(static_object_list)}')
    print(f'  Static obj poses               : {len(static_poses)}')
    print(f'  Dynamic object UIDs            : {len(dynamic_poses)}')

    frames_idx = list(range(0, n_frames, args.frame_step))
    if args.num_frames is not None:
        frames_idx = frames_idx[:args.num_frames]
    print(f'  Rendering {len(frames_idx)} frames (step={args.frame_step})')

    for count, idx in enumerate(frames_idx):
        out_png = f'{args.output_dir}/rgb/frame_{idx:04d}.png'
        if os.path.exists(out_png):
            print(f'  [{count+1}/{len(frames_idx)}] frame {idx:04d} already exists, skipping')
            continue

        # Get ego RGB for comparison
        img_data = p_ego.get_image_data_by_index(RGB_STREAM, idx)
        ts_ns    = img_data[1].capture_timestamp_ns
        ego_rgb  = img_data[0].to_numpy_array()

        # Camera pose (pass raw T_WC in ADT coords; Blender script handles conversion)
        T_WD = nearest_pose(traj, ts_arr, ts_ns // 1000)
        T_WC = T_WD @ T_DC
        cam_pos = T_WC[:3, 3]

        # Resolve dynamic object poses at this frame's timestamp
        dyn_poses_frame = resolve_dynamic_poses(dynamic_poses, ts_ns)
        dyn_object_list = build_object_list(instances, dyn_poses_frame, MODELS_DIR)

        # Combine static + dynamic into one candidate list
        all_objects = static_object_list + dyn_object_list

        # Distance cull: sort by distance and take closest N objects to stay within memory.
        # With large GLB files (50-100MB each) and 3.8GB RAM, cap at ~60 objects.
        # Skip GLBs > MAX_GLB_MB — they carry enormous textures that don't improve
        # quality at 512px render resolution but spike memory by 3-4× on load.
        # Sort by distance; texture limit in Blender script keeps RAM usage manageable
        objs_with_dist = [
            (np.linalg.norm(np.array(obj['T_WO']).reshape(4,4)[:3,3] - cam_pos), obj)
            for obj in all_objects
        ]
        objs_with_dist.sort(key=lambda x: x[0])
        visible_objects = [obj for _, obj in objs_with_dist[:80]]
        n_dyn = sum(1 for _, obj in objs_with_dist[:80]
                    if any(obj is o for o in dyn_object_list))
        print(f'    Distance-culled to {len(visible_objects)} objects '
              f'({n_dyn} dynamic, max dist: {objs_with_dist[min(79,len(objs_with_dist)-1)][0]:.1f}m)')

        # Assign sequential pass_index values (1-based; 0 = background) for seg mask
        pass_idx_to_uid: dict[int, int] = {}
        for i, obj in enumerate(visible_objects):
            obj['pass_index'] = i + 1
            try:
                pass_idx_to_uid[i + 1] = int(obj.get('uid', 0))
            except (ValueError, TypeError):
                pass_idx_to_uid[i + 1] = 0

        # Write per-frame JSON
        frame_data = {
            'image_width':    args.output_size,
            'image_height':   args.output_size,
            'focal_px':       args.focal,
            'camera_pose':    T_WC.flatten().tolist(),
            'object_models':  visible_objects,
            'scene_lights':   scene_lights,
            'use_equirect':   args.fisheye,
            'equirect_width':  EQ_W if args.fisheye else args.output_size,
            'equirect_height': EQ_H if args.fisheye else args.output_size,
            # Equirect renders downsample via remap → fewer samples still look good
            'cycles_samples':  16 if args.fisheye else 32,
            'pass_idx_to_uid': {str(k): v for k, v in pass_idx_to_uid.items()},
        }
        json_path = f'{tmp_dir}/frame_{idx:04d}.json'
        with open(json_path, 'w') as f:
            json.dump(frame_data, f)

        # In fisheye mode Blender renders equirect at EQ_W×EQ_H then we remap.
        # Use a temp path for the Blender output so we don't overwrite out_png.
        if args.fisheye:
            blender_out = f'{tmp_dir}/equirect_{idx:04d}.png'
        else:
            blender_out = out_png

        # Segmentation EXR path (inside tmp; renamed to final location after render)
        seg_exr_tmp  = f'{tmp_dir}/seg_{idx:04d}.exr'
        seg_npy_out  = f'{args.output_dir}/segmentation/frame_{idx:04d}.npy'
        seg_vis_out  = f'{args.output_dir}/segmentation/frame_{idx:04d}_vis.png'

        # Call Blender
        print(f'  [{count+1}/{len(frames_idx)}] Rendering frame {idx:04d}...')
        cmd = [
            BLENDER_BIN, '--background', '--python', BLEND_SCRIPT,
            '--', '--frame_data', json_path, '--output', blender_out,
            '--seg_output', seg_exr_tmp,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f'    Blender error:\n{result.stderr[-500:]}')
            continue
        if not os.path.exists(blender_out):
            print(f'    Output file not created! Blender stdout: {result.stdout[-200:]}')
            continue

        # Fisheye remap: equirect → FISHEYE624 projection
        if args.fisheye and fisheye_remap is not None:
            eq_img   = Image.open(blender_out)
            map_x, map_y, valid = fisheye_remap
            fish_arr = remap_equirect_to_fisheye(eq_img, map_x, map_y, valid)
            # Apply Aria Gen1 forward ISP so colours match ADT synthetic images
            fish_arr = apply_aria_forward_isp(fish_arr)
            Image.fromarray(fish_arr).save(out_png)
            # Also save full fisheye copy in dedicated folder
            Image.fromarray(fish_arr).save(
                f'{args.output_dir}/fisheye/frame_{idx:04d}.png')
            render_img = fish_arr
        else:
            raw_render = np.array(Image.open(out_png).convert('RGB'))
            # Apply Aria Gen1 forward ISP so colours match ADT synthetic images
            render_img = apply_aria_forward_isp(raw_render)
            Image.fromarray(render_img).save(out_png)

        # ── Segmentation post-processing ────────────────────────────────────
        if os.path.exists(seg_exr_tmp):
            # Load float32 EXR (RGB; IndexOB value is broadcast to R=G=B channels)
            seg_raw3  = cv2.imread(seg_exr_tmp,
                                   cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            seg_float = seg_raw3[..., 0] if seg_raw3 is not None else None  # R channel
            if seg_float is None:
                print(f'    [seg] cv2 could not read {seg_exr_tmp}; skipping seg')
            else:
                if args.fisheye and fisheye_remap is not None:
                    # Remap equirect seg → fisheye using nearest-neighbour
                    # (must NOT interpolate class IDs)
                    map_x_s, map_y_s, valid_s = fisheye_remap
                    seg_float = cv2.remap(seg_float, map_x_s, map_y_s,
                                          interpolation=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=0.0)
                    seg_float[~valid_s] = 0.0

                # Convert float pass_index → int64 instance UID
                pass_arr = np.round(seg_float).astype(np.int32)
                seg_uid  = np.zeros(pass_arr.shape, dtype=np.int64)
                for pidx_s, uid_s in pass_idx_to_uid.items():
                    seg_uid[pass_arr == pidx_s] = uid_s

                # Rotate to display orientation (same rot90 CW as RGB frames)
                seg_uid = np.rot90(seg_uid, k=-1).copy()

                # Save NPY (int64 instance UIDs) + colourised PNG
                np.save(seg_npy_out, seg_uid)
                Image.fromarray(colorize_seg(seg_uid)).save(seg_vis_out)
                n_objs = len(np.unique(seg_uid)) - 1   # exclude background
                print(f'    Segmentation: {n_objs} objects → {seg_npy_out}')
        else:
            print(f'    [seg] EXR not found at {seg_exr_tmp}')

        # Side-by-side: ego (resized) | blender render
        ego_img = np.array(Image.fromarray(ego_rgb).resize(
                      (args.output_size, args.output_size), Image.LANCZOS))
        gap     = np.ones((args.output_size, 4, 3), dtype=np.uint8) * 40
        side    = np.concatenate([ego_img, gap, render_img], axis=1)
        Image.fromarray(side).save(f'{args.output_dir}/comparison/frame_{idx:04d}.png')
        print(f'    Done.')

    print(f'\nAll done! Outputs in {args.output_dir}/')


if __name__ == '__main__':
    main()
