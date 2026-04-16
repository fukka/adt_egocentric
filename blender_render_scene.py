"""
Blender rendering script — run with:
  blender --background --python blender_render_scene.py -- \
      --frame_data /path/to/frame_data.json --output /path/to/out.png

Reads a JSON describing:
  - object_models: [{glb_path, T_WO (4x4 flat list)}]
  - camera_pose: T_GL (4x4 flat list, OpenGL convention)
  - focal_px, image_width, image_height
  - environment boxes (floor/ceiling/walls as simple primitives)
"""

import bpy, sys, os, json, math, mathutils
import numpy as np

# ADT uses Y-up (right-handed); Blender uses Z-up (right-handed).
# M4 rotates Y-up world → Z-up world:  x'=x, y'=-z, z'=y
M4     = np.array([[1, 0,  0, 0],
                   [0, 0, -1, 0],
                   [0, 1,  0, 0],
                   [0, 0,  0, 1]], dtype=float)
M4_inv = M4.T   # orthogonal, so inv = transpose

# Aria camera frame: +X right, +Y image-down, +Z forward (computer-vision convention).
# Blender camera frame: +X right, +Y image-up, -Z forward.
# Convert between them by flipping Y and Z of the LOCAL camera axes: FLIP_YZ.
# CONFIRMED: GLB vertex data is already in ADT Y-up world coords (importer keeps identity).
# Therefore: T_WC_bl = T_WC_adt @ FLIP_YZ  (no M4 — world stays in ADT coords)
# M4/M4_inv kept here for reference but NOT used.
FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])

# ── Parse args after "--" ─────────────────────────────────────────────────
argv = sys.argv
try:
    idx = argv.index('--') + 1
except ValueError:
    idx = len(argv)
user_args = argv[idx:]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--frame_data', required=True)
parser.add_argument('--output',     required=True)
args = parser.parse_args(user_args)

with open(args.frame_data) as f:
    data = json.load(f)

# ── Reset scene ───────────────────────────────────────────────────────────
bpy.ops.wm.read_homefile(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'NONE'
scene.cycles.device = 'CPU'
scene.cycles.samples = data.get('cycles_samples', 32)
scene.cycles.use_denoising = True
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.05

# Limit all textures to 1K max to stay within memory constraints.
# GLB files often embed 4K textures; at 512px render resolution 1K is plenty.
bpy.context.preferences.system.gl_texture_limit = 'CLAMP_1024'  # cap textures at 1K

W = data['image_width']
H = data['image_height']
scene.render.resolution_x = W
scene.render.resolution_y = H
scene.render.resolution_percentage = 100
scene.render.filepath = args.output
scene.render.image_settings.file_format = 'PNG'

# ── Lighting ──────────────────────────────────────────────────────────────
# Low ambient: the apartment roof mesh blocks outdoor light, so keep this dim.
# Warm slightly yellow-tinted ambient to approximate indoor tungsten lighting.
world = bpy.data.worlds.new('World')
world.use_nodes = True
bg = world.node_tree.nodes['Background']
bg.inputs[0].default_value = (0.85, 0.78, 0.65, 1.0)   # warm indoor tint
bg.inputs[1].default_value = 0.04                        # nearly dark — enclosed room
scene.world = world

# ── Scene-fixed lights from ADT prop positions + ceiling grid ─────────────
# Lights are supplied by render_from_poses_blender.py in scene_lights[]:
#   POINT — physical lamp/candle props at their world positions
#   AREA  — ceiling tiles approximating Unreal's baked indoor GI
#
# All coordinates are in ADT Y-up world space.  The scene geometry (GLBs)
# is also in ADT Y-up space (Blender GLTF importer kept identity world
# transform), so we place lights directly at the ADT coordinates without
# any extra conversion.
#
# Fallback: if scene_lights is absent (old JSON / manual run) we add a
# single soft overhead light above the camera as a safety default.
scene_lights = data.get('scene_lights', [])

if scene_lights:
    for ldef in scene_lights:
        loc   = tuple(ldef['location'])      # (x, y, z) in ADT Y-up world
        col   = tuple(ldef.get('color', [1.0, 0.90, 0.75]))
        eng   = ldef.get('energy', 10.0)
        ltype = ldef.get('type', 'POINT')

        if ltype == 'AREA':
            bpy.ops.object.light_add(type='AREA', location=loc)
            light = bpy.context.object.data
            light.energy = eng
            light.color  = col
            light.size   = ldef.get('size', 1.8)
            # Area light emits along local -Z.  In our world (ADT Y-up),
            # "down" = world -Y.  Rx(-90°) maps local -Z → world -Y:
            #   Rx(-90°) * [0,0,-1] = [0, -1, 0]  ✓
            bpy.context.object.rotation_euler = (-math.pi / 2, 0, 0)
        else:  # POINT
            bpy.ops.object.light_add(type='POINT', location=loc)
            light = bpy.context.object.data
            light.energy           = eng
            light.color            = col
            light.shadow_soft_size = ldef.get('radius', 0.10)
else:
    # Safety fallback — single warm overhead light 1.5m above camera
    T_WC_adt_early = np.array(data['camera_pose']).reshape(4, 4)
    cam_xyz        = (T_WC_adt_early @ FLIP_YZ)[:3, 3].tolist()
    bpy.ops.object.light_add(type='POINT',
                              location=(cam_xyz[0], cam_xyz[1] + 1.5, cam_xyz[2]))
    light = bpy.context.object.data
    light.energy           = 30.0
    light.color            = (1.0, 0.92, 0.78)
    light.shadow_soft_size = 1.0

# ── Colour management / exposure ─────────────────────────────────────────
# Match ADT synthetic: dark indoor exposure, Filmic tone-map.
scene.view_settings.view_transform = 'Filmic'
scene.view_settings.look            = 'Medium Low Contrast'
scene.view_settings.exposure        = -0.3   # adjusted to match ADT synthetic brightness
scene.view_settings.gamma           =  1.0

# ── Import GLB objects at world poses ─────────────────────────────────────
def mat4_to_blender(flat_list):
    """Convert flat 4x4 row-major list to Blender Matrix."""
    m = mathutils.Matrix(np.array(flat_list).reshape(4, 4).tolist())
    return m

for obj_info in data['object_models']:
    glb_path = obj_info['glb_path']
    T_WO_flat = obj_info['T_WO']

    if not os.path.exists(glb_path):
        continue

    before = set(bpy.data.objects.keys())
    try:
        bpy.ops.import_scene.gltf(filepath=glb_path,
                                   import_pack_images=False,
                                   merge_vertices=False)
    except Exception as e:
        continue

    after  = set(bpy.data.objects.keys())
    new_objs = [bpy.data.objects[n] for n in (after - before)]
    if not new_objs:
        continue

    # Find the root (object without a parent among the new ones)
    new_set = set(new_objs)
    roots = [o for o in new_objs if o.parent not in new_set]

    # CONFIRMED: The GLB vertex data is already in the ADT world coordinate frame (Y-up).
    # The Blender GLTF importer sets matrix_world = identity (no Y-up→Z-up correction).
    # Therefore we can set matrix_world = T_WO_adt directly — no M4 needed.
    # The scene geometry is already correctly expressed in ADT world coords.
    T_WO_adt = np.array(T_WO_flat).reshape(4, 4)
    T_bl = mathutils.Matrix(T_WO_adt.tolist())

    for root in roots:
        root.matrix_world = T_bl

# ── Downscale all textures to max 512px to limit Cycles memory usage ──────
MAX_TEX_PX = 512
for img in bpy.data.images:
    if img.size[0] > MAX_TEX_PX or img.size[1] > MAX_TEX_PX:
        try:
            img.scale(min(img.size[0], MAX_TEX_PX), min(img.size[1], MAX_TEX_PX))
        except Exception:
            pass

# ── Camera ────────────────────────────────────────────────────────────────
bpy.ops.object.camera_add()
cam_obj = bpy.context.object
cam = cam_obj.data
scene.camera = cam_obj

# Convert ADT camera pose to Blender camera pose:
#   T_WC_bl = T_WC_adt @ FLIP_YZ
# - No M4 needed: world geometry is already in ADT Y-up coords (importer kept identity)
# - FLIP_YZ converts camera-local convention: ADT(+Y=down,+Z=fwd) → Blender(+Y=up,-Z=fwd)
T_WC_adt = np.array(data['camera_pose']).reshape(4, 4)
T_WC_bl  = T_WC_adt @ FLIP_YZ
cam_obj.matrix_world = mathutils.Matrix(T_WC_bl.tolist())

# Equirectangular panoramic mode: render a full-sphere 360°×180° panorama.
# The driver script will remap this to the exact Aria FISHEYE624 projection.
use_equirect = data.get('use_equirect', False)

if use_equirect:
    eq_w = data.get('equirect_width',  4096)
    eq_h = data.get('equirect_height', 2048)
    cam.type = 'PANO'
    # In Blender 3.x Cycles the panoramic settings live on cam.cycles, not cam
    cyc = cam.cycles
    cyc.panorama_type = 'EQUIRECTANGULAR'
    # Full sphere — driver will crop/remap to fisheye FOV
    cyc.longitude_min = -math.pi
    cyc.longitude_max =  math.pi
    cyc.latitude_min  = -math.pi / 2
    cyc.latitude_max  =  math.pi / 2
    scene.render.resolution_x = eq_w
    scene.render.resolution_y = eq_h
    print(f'Equirectangular mode: {eq_w}x{eq_h}')
else:
    focal_px = data['focal_px']
    yfov_rad = 2.0 * math.atan(H / 2.0 / focal_px)
    cam.type  = 'PERSP'
    cam.angle = yfov_rad  # vertical FOV (since aspect=1)

# ── Render ────────────────────────────────────────────────────────────────
bpy.ops.render.render(write_still=True)
print(f'Rendered to {args.output}')
