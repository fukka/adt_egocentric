"""
Blender rendering script — run with:
  blender --background --python blender_render_scene.py -- \
      --frame_data /path/to/frame_data.json --output /path/to/out.png \
      [--seg_output /path/to/seg.exr]

Reads a JSON describing:
  - object_models: [{glb_path, T_WO (4x4 flat list), pass_index (int)}]
  - camera_pose: T_WC (4x4 flat list, ADT Y-up convention)
  - focal_px, image_width, image_height
  - scene_lights: [{type, location, energy, color, ...}]
  - use_equirect: bool

Segmentation output (--seg_output):
  Each object_model entry carries a pass_index (1-based int).  Blender's
  Object-Index render pass stores that value per pixel.  The compositor
  writes it to an EXR (float32 BW) at --seg_output so the driver can
  remap pass_index → instance_uid using the pass_idx_to_uid table in the
  JSON.
"""

import bpy, sys, os, json, math, mathutils, glob
import numpy as np

# ADT uses Y-up (right-handed); Blender uses Z-up (right-handed).
M4     = np.array([[1, 0,  0, 0],
                   [0, 0, -1, 0],
                   [0, 1,  0, 0],
                   [0, 0,  0, 1]], dtype=float)
M4_inv = M4.T

# Aria camera: +X right, +Y down, +Z forward (OpenCV).
# Blender camera: +X right, +Y up, -Z forward.
# T_WC_bl = T_WC_adt @ FLIP_YZ  (world stays in ADT coords — GLB importer keeps identity)
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
parser.add_argument('--seg_output', default=None,
                    help='Optional EXR path for per-pixel instance segmentation')
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

bpy.context.preferences.system.gl_texture_limit = 'CLAMP_1024'

W = data['image_width']
H = data['image_height']
scene.render.resolution_x = W
scene.render.resolution_y = H
scene.render.resolution_percentage = 100
scene.render.filepath = args.output
scene.render.image_settings.file_format = 'PNG'

# ── Lighting ──────────────────────────────────────────────────────────────
world = bpy.data.worlds.new('World')
world.use_nodes = True
bg = world.node_tree.nodes['Background']
bg.inputs[0].default_value = (0.85, 0.78, 0.65, 1.0)
bg.inputs[1].default_value = 0.04
scene.world = world

scene_lights = data.get('scene_lights', [])
if scene_lights:
    for ldef in scene_lights:
        loc   = tuple(ldef['location'])
        col   = tuple(ldef.get('color', [1.0, 0.90, 0.75]))
        eng   = ldef.get('energy', 10.0)
        ltype = ldef.get('type', 'POINT')
        if ltype == 'AREA':
            bpy.ops.object.light_add(type='AREA', location=loc)
            light = bpy.context.object.data
            light.energy = eng
            light.color  = col
            light.size   = ldef.get('size', 1.8)
            bpy.context.object.rotation_euler = (-math.pi / 2, 0, 0)
        else:
            bpy.ops.object.light_add(type='POINT', location=loc)
            light = bpy.context.object.data
            light.energy           = eng
            light.color            = col
            light.shadow_soft_size = ldef.get('radius', 0.10)
else:
    T_WC_adt_early = np.array(data['camera_pose']).reshape(4, 4)
    cam_xyz        = (T_WC_adt_early @ FLIP_YZ)[:3, 3].tolist()
    bpy.ops.object.light_add(type='POINT',
                              location=(cam_xyz[0], cam_xyz[1] + 1.5, cam_xyz[2]))
    light = bpy.context.object.data
    light.energy           = 30.0
    light.color            = (1.0, 0.92, 0.78)
    light.shadow_soft_size = 1.0

# ── Colour management ─────────────────────────────────────────────────────
scene.view_settings.view_transform = 'Filmic'
scene.view_settings.look            = 'Medium Low Contrast'
scene.view_settings.exposure        = -0.3
scene.view_settings.gamma           =  1.0

# ── Import GLB objects at world poses ─────────────────────────────────────
def mat4_to_blender(flat_list):
    return mathutils.Matrix(np.array(flat_list).reshape(4, 4).tolist())

for obj_info in data['object_models']:
    glb_path  = obj_info['glb_path']
    T_WO_flat = obj_info['T_WO']
    pass_idx  = obj_info.get('pass_index', 0)

    if not os.path.exists(glb_path):
        continue

    before = set(bpy.data.objects.keys())
    try:
        bpy.ops.import_scene.gltf(filepath=glb_path,
                                   import_pack_images=False,
                                   merge_vertices=False)
    except Exception:
        continue

    after    = set(bpy.data.objects.keys())
    new_objs = [bpy.data.objects[n] for n in (after - before)]
    if not new_objs:
        continue

    new_set = set(new_objs)
    roots   = [o for o in new_objs if o.parent not in new_set]

    T_WO_adt = np.array(T_WO_flat).reshape(4, 4)
    T_bl     = mathutils.Matrix(T_WO_adt.tolist())
    for root in roots:
        root.matrix_world = T_bl

    # Assign pass_index to every mesh in this GLB (used for segmentation pass)
    if pass_idx > 0:
        for o in new_objs:
            if o.type == 'MESH':
                o.pass_index = pass_idx

# ── Downscale textures ────────────────────────────────────────────────────
MAX_TEX_PX = 512
for img in bpy.data.images:
    if img.size[0] > MAX_TEX_PX or img.size[1] > MAX_TEX_PX:
        try:
            img.scale(min(img.size[0], MAX_TEX_PX), min(img.size[1], MAX_TEX_PX))
        except Exception:
            pass

# ── Compositor: segmentation Object-Index pass ────────────────────────────
# Only activated when --seg_output is given.
# Sets up:
#   RenderLayers.Image      → Composite  (drives write_still for RGB)
#   RenderLayers.IndexOB    → FileOutput → EXR float32 BW
# Blender appends a zero-padded frame number to the FileOutput path; the
# driver script renames the resulting file to --seg_output afterward.
if args.seg_output:
    vl = scene.view_layers[0]
    vl.use_pass_object_index = True

    scene.use_nodes = True
    scene.render.use_compositing = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl   = tree.nodes.new('CompositorNodeRLayers')
    comp = tree.nodes.new('CompositorNodeComposite')
    tree.links.new(rl.outputs['Image'], comp.inputs['Image'])

    seg_dir  = os.path.dirname(os.path.abspath(args.seg_output))
    seg_stem = os.path.splitext(os.path.basename(args.seg_output))[0]
    os.makedirs(seg_dir, exist_ok=True)

    out_seg = tree.nodes.new('CompositorNodeOutputFile')
    out_seg.base_path                 = seg_dir + os.sep
    out_seg.format.file_format        = 'OPEN_EXR'
    out_seg.format.color_mode         = 'RGB'   # value → R=G=B; read R channel in Python
    out_seg.format.color_depth        = '32'
    out_seg.format.exr_codec          = 'NONE'
    out_seg.file_slots[0].path        = seg_stem
    # IndexOB socket name varies by Blender version
    idx_socket = (rl.outputs.get('IndexOB') or rl.outputs.get('Object Index'))
    if idx_socket:
        tree.links.new(idx_socket, out_seg.inputs[0])
    else:
        print('[seg] WARNING: IndexOB socket not found — segmentation will be empty')

    scene.frame_current = 0   # → suffix "0000" on output filename

# ── Camera ────────────────────────────────────────────────────────────────
bpy.ops.object.camera_add()
cam_obj = bpy.context.object
cam     = cam_obj.data
scene.camera = cam_obj

T_WC_adt = np.array(data['camera_pose']).reshape(4, 4)
T_WC_bl  = T_WC_adt @ FLIP_YZ
cam_obj.matrix_world = mathutils.Matrix(T_WC_bl.tolist())

use_equirect = data.get('use_equirect', False)
if use_equirect:
    eq_w = data.get('equirect_width',  4096)
    eq_h = data.get('equirect_height', 2048)
    cam.type = 'PANO'
    cyc = cam.cycles
    cyc.panorama_type = 'EQUIRECTANGULAR'
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
    cam.angle = yfov_rad

# ── Render ────────────────────────────────────────────────────────────────
bpy.ops.render.render(write_still=True)

# Compositor wrote the segmentation EXR with a frame-number suffix.
# Rename it to exactly --seg_output so the driver can find it reliably.
if args.seg_output:
    seg_dir  = os.path.dirname(os.path.abspath(args.seg_output))
    seg_stem = os.path.splitext(os.path.basename(args.seg_output))[0]
    matches  = sorted(glob.glob(os.path.join(seg_dir, f'{seg_stem}????.exr')))
    if matches:
        os.replace(matches[0], args.seg_output)
        print(f'Segmentation EXR saved to {args.seg_output}')
    else:
        print(f'[seg] WARNING: no EXR found matching {seg_stem}????.exr in {seg_dir}')

print(f'Rendered to {args.output}')
