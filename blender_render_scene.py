"""
Blender rendering script — run with:
  blender --background --python blender_render_scene.py -- \
      --frame_data /path/to/frame_data.json --output /path/to/out.png \
      [--seg_output /path/to/seg.exr]
      [--normal_output /path/to/normal.exr]
      [--depth_output /path/to/depth.exr]

Reads a JSON describing:
  - object_models: [{glb_path, T_WO (4x4 flat list), pass_index (int)}]
  - camera_pose: T_WC (4x4 flat list, ADT Y-up convention)
  - focal_px, image_width, image_height
  - scene_lights: [{type, location, energy, color, ...}]
  - use_equirect: bool

Segmentation output (--seg_output):
  Each object_model entry carries a pass_index (1-based int).  Blender's
  Object-Index render pass stores that value per pixel.  The compositor
  writes it to an EXR (float32 RGB) at --seg_output so the driver can
  remap pass_index → instance_uid using the pass_idx_to_uid table in the
  JSON.

Normal output (--normal_output):
  World-space surface normals written as float32 RGB EXR.
  R = X, G = Y, B = Z components in ADT world space, values in [-1, 1].
  Read with cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR; note BGR channel order
  from OpenCV means B=Z, G=Y, R=X — flip channels to get (X,Y,Z).

Depth output (--depth_output):
  Camera-space distance (Z pass) written as float32 RGB EXR (depth value
  broadcast to R=G=B).  Read R channel for the depth float.
  Background pixels have very large values (~1e10); clip when visualising.
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
parser.add_argument('--normal_output', default=None,
                    help='Optional EXR path for per-pixel world-space surface normals '
                         '(float32 RGB; R=X G=Y B=Z in ADT world coords, range [-1,1])')
parser.add_argument('--depth_output', default=None,
                    help='Optional EXR path for per-pixel camera-space depth distance '
                         '(float32 RGB broadcast; read R channel for metres)')
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
# Use Standard (sRGB) output so render_from_poses_blender.py can apply the
# Aria Gen1 forward ISP in post-processing, matching ADT synthetic colour space.
# Filmic would bake a different tone-curve that fights the subsequent ISP transform.
# Exposure = -0.8 EV compensates for Standard mode being ~0.8 stops brighter than
# Filmic for this scene; the ISP and optional channel_balance correction in the
# driver script fine-tune the remaining per-channel mismatch vs ADT synthetic.
scene.view_settings.view_transform = 'Standard'
scene.view_settings.look            = 'None'
scene.view_settings.exposure        = -0.8
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

# ── Compositor: segmentation, normal, and depth passes ────────────────────
#
# Activated when any of --seg_output / --normal_output / --depth_output are given.
#
# ViewLayer design:
#   Main  : normal materials, glass transparent → drives RGB write_still output
#           Also provides Normal pass (world-space shading normals) and Z/Depth pass.
#   Seg   : material_override = fully opaque diffuse → IndexOB records glass pixels
#           correctly (avoids dotted pattern from Cycles HASHED transparency).
#
# Compositor wiring:
#   Main.Image  → Composite        (RGB output via write_still)
#   Main.Normal → FileOutput EXR   (--normal_output; float32 RGB, range [-1,1])
#   Main.Depth  → FileOutput EXR   (--depth_output;  float32 RGB broadcast)
#   Seg.IndexOB → FileOutput EXR   (--seg_output;    float32 RGB broadcast)
#
# All three EXR files use format RGB / 32-bit float / no compression.
# Driver reads R channel (or all 3 for normals) after renaming the
# Blender-appended frame-number suffix off the filename.

need_compositor = bool(args.seg_output or args.normal_output or args.depth_output)
if need_compositor:
    # ── Main ViewLayer ────────────────────────────────────────────────────
    vl_main = scene.view_layers[0]
    vl_main.name = 'Main'
    vl_main.use_pass_object_index = False   # not needed for RGB
    if args.normal_output:
        vl_main.use_pass_normal = True
    if args.depth_output:
        vl_main.use_pass_z = True

    # ── Seg ViewLayer (only when --seg_output given) ──────────────────────
    # Fully opaque diffuse override so Cycles rays always hit glass surfaces
    # and their pass_index values are recorded without transparency dropout.
    if args.seg_output:
        seg_override = bpy.data.materials.new('_SEG_OPAQUE_OVERRIDE')
        seg_override.use_nodes = True
        _sn = seg_override.node_tree.nodes
        _sn.clear()
        _so = _sn.new('ShaderNodeOutputMaterial')
        _sd = _sn.new('ShaderNodeBsdfDiffuse')
        _sd.inputs['Color'].default_value = (0.5, 0.5, 0.5, 1.0)
        seg_override.node_tree.links.new(_sd.outputs['BSDF'], _so.inputs['Surface'])
        seg_override.blend_method  = 'OPAQUE'
        seg_override.shadow_method = 'OPAQUE'

        vl_seg = scene.view_layers.new('Seg')
        vl_seg.use_pass_object_index = True
        vl_seg.material_override     = seg_override

    # ── Compositor ────────────────────────────────────────────────────────
    scene.use_nodes = True
    scene.render.use_compositing = True
    tree = scene.node_tree
    tree.nodes.clear()

    # Main layer node — source for RGB, Normal, Depth
    rl_main = tree.nodes.new('CompositorNodeRLayers')
    rl_main.layer = 'Main'

    # Main.Image → Composite (drives the write_still RGB output)
    comp = tree.nodes.new('CompositorNodeComposite')
    tree.links.new(rl_main.outputs['Image'], comp.inputs['Image'])

    # ── Normal output ─────────────────────────────────────────────────────
    # World-space shading normals: R=X, G=Y, B=Z, values in [-1, 1].
    # Note: OpenCV reads EXR as BGR, so caller must reverse channels to get (X,Y,Z).
    if args.normal_output:
        norm_dir  = os.path.dirname(os.path.abspath(args.normal_output))
        norm_stem = os.path.splitext(os.path.basename(args.normal_output))[0]
        os.makedirs(norm_dir, exist_ok=True)
        out_norm = tree.nodes.new('CompositorNodeOutputFile')
        out_norm.base_path          = norm_dir + os.sep
        out_norm.format.file_format = 'OPEN_EXR'
        out_norm.format.color_mode  = 'RGB'
        out_norm.format.color_depth = '32'
        out_norm.format.exr_codec   = 'NONE'
        out_norm.file_slots[0].path = norm_stem
        norm_socket = rl_main.outputs.get('Normal')
        if norm_socket:
            tree.links.new(norm_socket, out_norm.inputs[0])
        else:
            print('[normal] WARNING: Normal socket not found on Main layer — '
                  'enable use_pass_normal or check Blender version')

    # ── Depth output ──────────────────────────────────────────────────────
    # Camera-space distance in metres, broadcast to RGB channels.
    # Background pixels have very large values (~1e10); caller should clip.
    if args.depth_output:
        depth_dir  = os.path.dirname(os.path.abspath(args.depth_output))
        depth_stem = os.path.splitext(os.path.basename(args.depth_output))[0]
        os.makedirs(depth_dir, exist_ok=True)
        out_depth = tree.nodes.new('CompositorNodeOutputFile')
        out_depth.base_path          = depth_dir + os.sep
        out_depth.format.file_format = 'OPEN_EXR'
        out_depth.format.color_mode  = 'RGB'   # depth broadcast to R=G=B for easy read
        out_depth.format.color_depth = '32'
        out_depth.format.exr_codec   = 'NONE'
        out_depth.file_slots[0].path = depth_stem
        # Socket is 'Depth' in Blender 3.x+, 'Z' in older versions
        depth_socket = rl_main.outputs.get('Depth') or rl_main.outputs.get('Z')
        if depth_socket:
            tree.links.new(depth_socket, out_depth.inputs[0])
        else:
            print('[depth] WARNING: Depth/Z socket not found on Main layer — '
                  'enable use_pass_z or check Blender version')

    # ── Segmentation output ───────────────────────────────────────────────
    # IndexOB (pass_index per pixel) from opaque-override Seg layer.
    if args.seg_output:
        rl_seg = tree.nodes.new('CompositorNodeRLayers')
        rl_seg.layer = 'Seg'

        seg_dir  = os.path.dirname(os.path.abspath(args.seg_output))
        seg_stem = os.path.splitext(os.path.basename(args.seg_output))[0]
        os.makedirs(seg_dir, exist_ok=True)

        out_seg = tree.nodes.new('CompositorNodeOutputFile')
        out_seg.base_path          = seg_dir + os.sep
        out_seg.format.file_format = 'OPEN_EXR'
        out_seg.format.color_mode  = 'RGB'   # IndexOB broadcast to R=G=B; read R in Python
        out_seg.format.color_depth = '32'
        out_seg.format.exr_codec   = 'NONE'
        out_seg.file_slots[0].path = seg_stem

        # Socket name varies by Blender version
        idx_socket = (rl_seg.outputs.get('IndexOB') or rl_seg.outputs.get('Object Index'))
        if idx_socket:
            tree.links.new(idx_socket, out_seg.inputs[0])
        else:
            print('[seg] WARNING: IndexOB socket not found on Seg layer — '
                  'segmentation output will be empty')

    scene.frame_current = 0   # → suffix "0000" on all FileOutput filenames

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

# Compositor writes EXR files with a frame-number suffix (e.g. "seg0000.exr").
# Rename each to its exact target path so the driver can find them reliably.

def _rename_exr(target_path, tag):
    d    = os.path.dirname(os.path.abspath(target_path))
    stem = os.path.splitext(os.path.basename(target_path))[0]
    # Blender appends a 4-digit frame number before the extension
    hits = sorted(glob.glob(os.path.join(d, f'{stem}????.exr')))
    if hits:
        os.replace(hits[0], target_path)
        print(f'{tag} EXR saved to {target_path}')
    else:
        print(f'[{tag}] WARNING: no EXR found matching {stem}????.exr in {d}')

if args.seg_output:
    _rename_exr(args.seg_output, 'Segmentation')
if args.normal_output:
    _rename_exr(args.normal_output, 'Normal')
if args.depth_output:
    _rename_exr(args.depth_output, 'Depth')

print(f'Rendered to {args.output}')
