"""
ADT Object-Pose → Synthetic Frame Rendering Pipeline
=====================================================
Reads object 6DoF poses and camera trajectory from ADT ground truth,
loads 3D GLB assets, builds a pyrender scene, and renders each frame
to produce synthetic RGB images matching the egocentric camera viewpoint.

Coordinate frames
-----------------
  World (W)    : ADT world frame  (Y-up, right-handed)
  Device (D)   : Aria glasses origin
  Camera (C)   : camera-rgb sensor frame  (Z forward, Y up in image)
  OpenGL (GL)  : pyrender convention      (Z backward, Y up)

  T_WD  from aria_trajectory.csv
  T_DC  from VRS device calibration (cam.get_transform_device_camera())
  T_WC  = T_WD @ T_DC
  T_GL  = T_WC @ FLIP_YZ          (flip Y and Z for OpenGL)

Usage
-----
  python render_from_poses.py [--num_frames N] [--output_size S] [--focal F]
"""

import sys, os, csv, json, argparse
sys.path.insert(0, '/sessions/dreamy-modest-brown/.local/lib/python3.10/site-packages')

import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import trimesh
import pyrender

from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = '/sessions/dreamy-modest-brown/mnt/ADT/Apartment_release_golden_skeleton_seq100_10s_sample_M1292'
EGO_VRS       = f'{BASE}/main_recording.vrs'
GT_DIR        = f'{BASE}/groundtruth'
MODELS_DIR    = f'{BASE}/object_models'
RGB_STREAM    = StreamId('214-1')

# Flip matrix: Aria camera (Z-forward) → OpenGL (−Z-forward, −Y down→up)
FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])

# Colours for objects without GLB (R,G,B as 0-1 floats)
FALLBACK_COLOR = np.array([0.6, 0.6, 0.65])


# ── Helpers ────────────────────────────────────────────────────────────────

def quat_to_matrix(tx, ty, tz, qx, qy, qz, qw):
    """Build a 4x4 SE3 matrix from translation + quaternion (scalar-last)."""
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [tx, ty, tz]
    return T


def load_trajectory(path):
    """Load aria_trajectory.csv → dict {timestamp_us: 4x4 T_WD}."""
    traj = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ts = int(row['tracking_timestamp_us'])
            T = quat_to_matrix(
                float(row['tx_world_device']), float(row['ty_world_device']),
                float(row['tz_world_device']),
                float(row['qx_world_device']), float(row['qy_world_device']),
                float(row['qz_world_device']), float(row['qw_world_device']))
            traj[ts] = T
    return traj


def nearest_pose(traj_dict, timestamp_us):
    """Return 4x4 T_WD for the trajectory entry nearest to timestamp_us."""
    ts_arr = np.array(list(traj_dict.keys()))
    idx = np.argmin(np.abs(ts_arr - timestamp_us))
    return traj_dict[ts_arr[idx]]


def load_static_object_poses(path):
    """Load scene_objects.csv → dict {uid_str: 4x4 T_WO} for static objects."""
    poses = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row['timestamp[ns]']) != -1:
                continue  # skip dynamic poses (person skeleton)
            uid = row['object_uid']
            # scene_objects uses q_wo_w first
            T = quat_to_matrix(
                float(row['t_wo_x[m]']), float(row['t_wo_y[m]']),
                float(row['t_wo_z[m]']),
                float(row['q_wo_x']),   float(row['q_wo_y']),
                float(row['q_wo_z']),   float(row['q_wo_w']))
            poses[uid] = T
    return poses


def load_bboxes(path):
    """Load 3d_bounding_box.csv → dict {uid_str: (xmin,xmax,ymin,ymax,zmin,zmax)}."""
    boxes = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            uid = row['object_uid']
            boxes[uid] = (
                float(row['p_local_obj_xmin[m]']), float(row['p_local_obj_xmax[m]']),
                float(row['p_local_obj_ymin[m]']), float(row['p_local_obj_ymax[m]']),
                float(row['p_local_obj_zmin[m]']), float(row['p_local_obj_zmax[m]']))
    return boxes


def make_box_mesh(xmin, xmax, ymin, ymax, zmin, zmax, color):
    """Create a solid box trimesh from AABB extents."""
    extents = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
    center  = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)
    box.visual.vertex_colors = (*[int(c * 255) for c in color], 180)
    return box


def load_glb_as_pyrender_mesh(glb_path, T_WO):
    """Load a GLB file and return a list of (pyrender.Mesh, pose_matrix) tuples."""
    try:
        scene = trimesh.load(glb_path, force='scene')
        meshes_out = []
        if isinstance(scene, trimesh.Scene):
            # scene.graph.nodes_geometry is a list of node names (not pairs)
            for node_name in scene.graph.nodes_geometry:
                # graph[node_name] returns (transform_4x4, geometry_name)
                node_transform, geom_name = scene.graph[node_name]
                geom = scene.geometry.get(geom_name)
                if geom is None or not isinstance(geom, trimesh.Trimesh):
                    continue
                if len(geom.faces) == 0:
                    continue
                combined = T_WO @ node_transform
                try:
                    m = pyrender.Mesh.from_trimesh(geom, smooth=False)
                    meshes_out.append((m, combined))
                except Exception:
                    pass
        elif isinstance(scene, trimesh.Trimesh):
            if len(scene.faces) > 0:
                m = pyrender.Mesh.from_trimesh(scene, smooth=False)
                meshes_out.append((m, T_WO))
        return meshes_out
    except Exception:
        return []


def build_static_scene(instances, obj_poses, bboxes, models_dir):
    """
    Build the static part of the pyrender scene (environment + all static objects).
    Returns list of (pyrender.Mesh, T_WO_4x4) tuples.
    """
    uid_to_proto = {str(v['instance_id']): v['prototype_name']
                    for v in instances.values()}

    mesh_items = []
    glb_loaded = 0
    box_fallback = 0
    missing = 0

    for uid, T_WO in obj_poses.items():
        proto = uid_to_proto.get(uid)
        if proto is None:
            missing += 1
            continue

        glb_path = os.path.join(models_dir, f'{proto}.glb')

        if os.path.exists(glb_path):
            items = load_glb_as_pyrender_mesh(glb_path, T_WO)
            if items:
                mesh_items.extend(items)
                glb_loaded += 1
                continue

        # (no GLB and no fallback box — skip to save GPU memory)

    print(f'  Scene: {glb_loaded} GLB objects loaded, '
          f'{missing} skipped (no instance info)')
    return mesh_items


def render_frame(pyrender_scene, camera_node, r, out_size):
    """Render a single frame and return (color_uint8, depth_float32)."""
    color, depth = r.render(pyrender_scene)
    return color, depth


def main():
    parser = argparse.ArgumentParser(description='Render ADT scene from object poses')
    parser.add_argument('--num_frames', type=int, default=None)
    parser.add_argument('--output_size', type=int, default=512)
    parser.add_argument('--focal', type=float, default=300.0,
                        help='Pinhole focal length in pixels (matches rectification)')
    parser.add_argument('--output_dir', type=str,
                        default=f'{BASE}/rendered_from_poses')
    parser.add_argument('--frame_step', type=int, default=1,
                        help='Only render every N-th frame (for speed)')
    args = parser.parse_args()

    os.makedirs(f'{args.output_dir}/rgb',          exist_ok=True)
    os.makedirs(f'{args.output_dir}/depth',        exist_ok=True)
    os.makedirs(f'{args.output_dir}/comparison',   exist_ok=True)

    print('Loading VRS provider...')
    p_ego = data_provider.create_vrs_data_provider(EGO_VRS)
    calib = p_ego.get_device_calibration()
    cam   = calib.get_camera_calib('camera-rgb')
    T_DC  = cam.get_transform_device_camera().to_matrix()  # device → camera
    n_frames = p_ego.get_num_data(RGB_STREAM)

    print('Loading ground truth...')
    with open(f'{GT_DIR}/instances.json') as f:
        instances = json.load(f)
    traj     = load_trajectory(f'{GT_DIR}/aria_trajectory.csv')
    obj_poses = load_static_object_poses(f'{GT_DIR}/scene_objects.csv')
    bboxes   = load_bboxes(f'{GT_DIR}/3d_bounding_box.csv')

    print(f'  Trajectory entries : {len(traj)}')
    print(f'  Static obj poses   : {len(obj_poses)}')
    print(f'  Bounding boxes     : {len(bboxes)}')

    # ── Camera intrinsics ─────────────────────────────────────────────────
    W = H = args.output_size
    f = args.focal
    # pyrender PerspectiveCamera uses yfov (full vertical angle)
    yfov = 2.0 * np.arctan(H / 2.0 / f)
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)

    # ── Lights ────────────────────────────────────────────────────────────
    # Ambient + directional from above
    ambient = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    ambient_pose = np.eye(4)
    ambient_pose[:3, :3] = Rotation.from_euler('x', -45, degrees=True).as_matrix()

    # ── Build static scene ────────────────────────────────────────────────
    print('\nBuilding static 3D scene...')
    mesh_items = build_static_scene(instances, obj_poses, bboxes, MODELS_DIR)
    print(f'  Total mesh objects: {len(mesh_items)}')

    # ── Set up offscreen renderer ─────────────────────────────────────────
    print(f'\nSetting up offscreen renderer ({W}x{H})...')
    r = pyrender.OffscreenRenderer(W, H)

    # Determine frames to render
    frames_to_render = range(0, n_frames, args.frame_step)
    if args.num_frames is not None:
        frames_to_render = list(frames_to_render)[:args.num_frames]
    print(f'Rendering {len(list(frames_to_render))} frames...')

    # ── Build STATIC pyrender scene once ─────────────────────────────────
    print('\nBuilding pyrender scene (static objects)...')
    scene = pyrender.Scene(bg_color=np.array([0.53, 0.81, 0.98, 1.0]),
                           ambient_light=np.array([0.3, 0.3, 0.3]))
    for mesh, T_WO in mesh_items:
        try:
            scene.add(mesh, pose=T_WO)
        except Exception:
            pass

    # Add static lights
    scene.add(ambient, pose=ambient_pose)
    fill_light = pyrender.PointLight(color=np.ones(3), intensity=8.0)
    fill_light_node = scene.add(fill_light, pose=np.eye(4))  # pose updated per frame

    # Add camera node (pose updated per frame)
    camera_node = scene.add(camera, pose=np.eye(4))
    print(f'  pyrender scene has {len(scene.mesh_nodes)} mesh nodes')

    for idx in frames_to_render:
        # ── Get camera pose for this frame ────────────────────────────────
        img_data = p_ego.get_image_data_by_index(RGB_STREAM, idx)
        ts_ns    = img_data[1].capture_timestamp_ns
        ts_us    = ts_ns // 1000
        ego_rgb  = img_data[0].to_numpy_array()

        T_WD = nearest_pose(traj, ts_us)
        T_WC = T_WD @ T_DC
        T_GL = T_WC @ FLIP_YZ

        # Update camera and fill-light poses in existing scene
        scene.set_pose(camera_node,    T_GL)
        fill_pose = T_GL.copy()
        fill_pose[:3, 3] += T_GL[:3, :3] @ np.array([0, 0.5, -1.0])
        scene.set_pose(fill_light_node, fill_pose)

        # ── Render ────────────────────────────────────────────────────────
        try:
            color, depth = r.render(scene)
        except Exception as e:
            print(f'  Frame {idx:04d}: render error: {e}')
            continue

        # ── Save outputs ──────────────────────────────────────────────────
        Image.fromarray(color).save(f'{args.output_dir}/rgb/frame_{idx:04d}.png')

        d_vis = np.zeros(depth.shape, dtype=np.uint8)
        mask  = depth > 0
        if mask.any():
            d_norm = (depth - depth[mask].min()) / (depth[mask].max() - depth[mask].min() + 1e-6)
            d_vis  = (d_norm * 255).astype(np.uint8)
        Image.fromarray(d_vis).save(f'{args.output_dir}/depth/frame_{idx:04d}.png')

        ego_resized = np.array(Image.fromarray(ego_rgb).resize((W, W), Image.LANCZOS))
        gap = np.ones((W, 4, 3), dtype=np.uint8) * 40
        side = np.concatenate([ego_resized, gap, color], axis=1)
        Image.fromarray(side).save(f'{args.output_dir}/comparison/frame_{idx:04d}.png')

        if (idx + 1) % 10 == 0 or idx == list(frames_to_render)[-1]:
            print(f'  Frame {idx:04d} done')

    r.delete()
    print(f'\nDone! Frames saved to {args.output_dir}/')
    print(f'  rgb/         — rendered synthetic RGB')
    print(f'  depth/       — rendered depth map')
    print(f'  comparison/  — ego RGB | rendered side-by-side')


if __name__ == '__main__':
    main()
