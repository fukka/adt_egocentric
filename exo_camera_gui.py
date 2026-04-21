#!/usr/bin/env python3
"""
exo_camera_gui.py
=================
Interactive GUI for hand-selecting exocentric camera poses for the ADT scene.

Left panel:  sliders for Eye (X/Y/Z), Target (X/Y/Z), and FOV.
             Preset buttons jump to the 3 hard-coded camera configurations.
             Live camera-info panel shows derived distance, pitch, yaw.
Right panel: rendered result image, updated after each Blender render.

Renders run in a background thread — the GUI stays responsive while Blender works.

Usage:
    python exo_camera_gui.py
    python exo_camera_gui.py --output_size 512    # faster previews
    python exo_camera_gui.py --no_segmentation    # skip seg pass (faster)

When happy with a pose, click "Copy command" to get the render_exocentric_blender.py
invocation with your current settings, ready to paste into a terminal.
"""

import sys, os, csv, json, argparse, threading, subprocess, time, hashlib, struct
sys.path.insert(0, os.path.expanduser('~/.local/lib/python3.10/site-packages'))

import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image

try:
    import tkinter as tk
    from tkinter import ttk, font as tkfont
except ImportError:
    print('tkinter is not available. Install it with: sudo apt install python3-tk')
    sys.exit(1)

try:
    from PIL import ImageTk
except ImportError:
    print('Pillow is required: pip install Pillow --break-system-packages')
    sys.exit(1)

# ── Paths — edit these to match your machine ──────────────────────────────────
BASE         = os.path.expanduser(
    '~/projectaria_tools_adt_data/'
    'Apartment_release_golden_skeleton_seq100_10s_sample_M1292')
GT_DIR       = f'{BASE}/groundtruth'
MODELS_DIR   = f'{BASE}/object_models'
BLENDER_BIN  = os.path.expanduser('~/blender/blender')
BLEND_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'blender_render_scene.py')

# Fall back to sandbox paths if the home-dir path doesn't exist
_SANDBOX_BASE = ('/sessions/dreamy-modest-brown/mnt/ADT/'
                 'Apartment_release_golden_skeleton_seq100_10s_sample_M1292')
if not os.path.isdir(BASE) and os.path.isdir(_SANDBOX_BASE):
    BASE       = _SANDBOX_BASE
    GT_DIR     = f'{BASE}/groundtruth'
    MODELS_DIR = f'{BASE}/object_models'
if not os.path.isfile(BLENDER_BIN):
    _SB_BLN = '/sessions/dreamy-modest-brown/blender/blender'
    if os.path.isfile(_SB_BLN):
        BLENDER_BIN = _SB_BLN

# ── Preset cameras (same as render_exocentric_blender.py) ─────────────────────
PRESETS = {
    'right_back': {
        'eye':    np.array([-2.0, 1.8, 1.7]),
        'target': np.array([2.55, 1.5, 1.7]),
        'fov':    70.0,
        'desc':   'Head-on cabinet view',
    },
    'left_side': {
        'eye':    np.array([-2.0, 2.8, 2.5]),
        'target': np.array([1.0, 0.9, 1.5]),
        'fov':    70.0,
        'desc':   'Elevated left-side view',
    },
    'overhead': {
        'eye':    np.array([1.2, 3.8, 3.0]),
        'target': np.array([1.0, 0.9, 1.5]),
        'fov':    70.0,
        'desc':   'Steep overhead view',
    },
}

# ── Lookat + rotation helpers (shared with render_exocentric_blender.py) ──────

FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])

def lookat_adt(eye, target, world_up=None):
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0])
    eye    = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    world_up = np.asarray(world_up, dtype=float)
    fwd     = target - eye; fwd /= np.linalg.norm(fwd)
    neg_fwd = -fwd
    right = np.cross(world_up, neg_fwd)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(np.array([0.0, 0.0, 1.0]), neg_fwd)
    right /= np.linalg.norm(right)
    img_up = np.cross(neg_fwd, right); img_up /= np.linalg.norm(img_up)
    T = np.eye(4)
    T[:3, 0] = right; T[:3, 1] = -img_up; T[:3, 2] = fwd; T[:3, 3] = eye
    return T

R_x_neg90 = np.array([[1, 0, 0], [0, 0, 1], [0,-1, 0]], dtype=float)
_glb_cache: dict = {}

def _read_glb_baked_rotation(glb_path):
    if glb_path in _glb_cache:
        return _glb_cache[glb_path]
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
    _glb_cache[glb_path] = R
    return R

def correct_object_rotation(T_WO, glb_path=None):
    R_baked = _read_glb_baked_rotation(glb_path) if glb_path else np.eye(3)
    T_c = T_WO.copy()
    T_c[:3, :3] = T_WO[:3, :3] @ R_baked @ R_x_neg90
    return T_c

def quat_to_matrix(tx, ty, tz, qx, qy, qz, qw):
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [tx, ty, tz]
    return T


# ── Scene loading ─────────────────────────────────────────────────────────────

def load_scene(frame_idx=0):
    """Load ADT scene objects and lights once at startup."""
    with open(f'{GT_DIR}/instances.json') as f:
        instances = json.load(f)

    # Resolve frame timestamp from trajectory CSV
    traj_rows = []
    with open(f'{GT_DIR}/aria_trajectory.csv') as f:
        for row in csv.DictReader(f):
            traj_rows.append(int(row['tracking_timestamp_us']))
    frame_idx  = min(frame_idx, len(traj_rows) - 1)
    frame_ts_ns = traj_rows[frame_idx] * 1000

    # Object poses
    static_poses, dynamic_poses = {}, {}
    with open(f'{GT_DIR}/scene_objects.csv') as f:
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

    # Resolve dynamic → nearest timestamp
    resolved_dyn = {}
    for uid, entries in dynamic_poses.items():
        ts_arr = np.array([e[0] for e in entries], dtype=np.int64)
        idx = int(np.argmin(np.abs(ts_arr - frame_ts_ns)))
        resolved_dyn[uid] = entries[idx][1]

    uid_to_info = {str(v['instance_id']): v for v in instances.values()}

    def _build(pose_dict):
        result = []
        for uid, T_WO_raw in pose_dict.items():
            info = uid_to_info.get(uid)
            if info is None:
                continue
            for candidate in [info.get('instance_name', ''), info.get('prototype_name', '')]:
                if candidate:
                    p = os.path.join(MODELS_DIR, f'{candidate}.glb')
                    if os.path.exists(p):
                        T_c = correct_object_rotation(T_WO_raw, p)
                        result.append({'glb_path': p, 'T_WO': T_c.flatten().tolist(),
                                       'uid': uid})
                        break
        return result

    all_objects = _build(static_poses) + _build(resolved_dyn)

    # Lights
    LIGHT_PROPS = {
        'Lamp_1':        {'energy': 120.0, 'color': [1.0, 0.82, 0.60], 'radius': 0.15},
        'WhitTableLamp': {'energy':  80.0, 'color': [1.0, 0.85, 0.65], 'radius': 0.12},
        'NightLights':   {'energy':  30.0, 'color': [1.0, 0.90, 0.80], 'radius': 0.08},
        'Candles':       {'energy':  10.0, 'color': [1.0, 0.75, 0.40], 'radius': 0.04},
    }
    lights = []
    with open(f'{GT_DIR}/scene_objects.csv') as f:
        for row in csv.DictReader(f):
            if int(row['timestamp[ns]']) != -1:
                continue
            uid  = row['object_uid']
            info = uid_to_info.get(uid)
            if info is None:
                continue
            name = info.get('prototype_name', '')
            cfg  = next((c for k, c in LIGHT_PROPS.items() if k in name), None)
            if cfg is None:
                continue
            x, y, z = float(row['t_wo_x[m]']), float(row['t_wo_y[m]']), float(row['t_wo_z[m]'])
            lights.append({'type': 'POINT',
                           'location': [x, y + (0.30 if 'Lamp' in name else 0.10), z],
                           **cfg})
    CEIL_Y, CEIL_E, CEIL_C, CEIL_S = 2.35, 22.0, [1.0, 0.90, 0.75], 1.8
    for cx, cz in [(0.5,2.5),(0.5,0.5),(-1.0,-1.5),(-1.0,4.5),(-2.5,-2.5),(-2.5,1.5)]:
        lights.append({'type':'AREA','location':[cx,CEIL_Y,cz],
                       'energy':CEIL_E,'color':CEIL_C,'size':CEIL_S})
    # Dedicated point light above the WoodenBowl (dynamic object, frame-resolved)
    WOODEN_BOWL_UID = '4508463855879675'
    bowl_T = resolved_dyn.get(WOODEN_BOWL_UID) or static_poses.get(WOODEN_BOWL_UID)
    if bowl_T is not None:
        bx, by, bz = float(bowl_T[0, 3]), float(bowl_T[1, 3]), float(bowl_T[2, 3])
        lights.append({
            'type':     'POINT',
            'location': [bx, by + 0.30, bz],
            'energy':   80.0,
            'color':    [1.0, 0.90, 0.75],
            'radius':   0.08,
        })

    return all_objects, lights


def make_frame_data(eye, target, fov_deg, output_size, all_objects, lights,
                    segmentation=False):
    """Build the JSON payload and visible-object list for one render."""
    T_WC     = lookat_adt(eye, target)
    cam_pos  = np.asarray(eye)
    with_dist = [(np.linalg.norm(np.array(o['T_WO']).reshape(4,4)[:3,3] - cam_pos), o)
                 for o in all_objects]
    with_dist.sort(key=lambda x: x[0])
    visible = [o for _, o in with_dist[:80]]

    pass_idx_to_uid = {}
    for i, obj in enumerate(visible):
        obj = dict(obj); obj['pass_index'] = i + 1
        try:
            pass_idx_to_uid[i + 1] = int(obj.get('uid', 0))
        except (ValueError, TypeError):
            pass_idx_to_uid[i + 1] = 0
        visible[i] = obj

    focal_px = (output_size / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    frame_data = {
        'image_width':     output_size,
        'image_height':    output_size,
        'focal_px':        focal_px,
        'camera_pose':     T_WC.flatten().tolist(),
        'object_models':   visible,
        'scene_lights':    lights,
        'use_equirect':    False,
        'cycles_samples':  32,
        'pass_idx_to_uid': {str(k): v for k, v in pass_idx_to_uid.items()},
    }
    return frame_data, pass_idx_to_uid


# ── Derived camera stats ───────────────────────────────────────────────────────

def camera_stats(eye, target):
    fwd  = np.asarray(target) - np.asarray(eye)
    dist = float(np.linalg.norm(fwd))
    if dist < 1e-6:
        return dist, 0.0, 0.0
    fwd /= dist
    pitch = float(np.degrees(np.arcsin(np.clip(-fwd[1], -1, 1))))
    yaw   = float(np.degrees(np.arctan2(fwd[0], fwd[2])))
    return dist, pitch, yaw


# ── GUI ───────────────────────────────────────────────────────────────────────

class SliderRow:
    """A labelled slider row: [label] [slider] [value-label]"""
    def __init__(self, parent, text, from_, to, resolution, initial, row,
                 on_change=None):
        self.var = tk.DoubleVar(value=initial)
        tk.Label(parent, text=text, anchor='w', width=9).grid(
            row=row, column=0, padx=(4, 2), pady=2, sticky='w')
        self.scale = ttk.Scale(parent, from_=from_, to=to,
                               variable=self.var, orient='horizontal',
                               length=200)
        self.scale.grid(row=row, column=1, padx=2, pady=2)
        self.val_lbl = tk.Label(parent, text=f'{initial:+.2f}', width=7,
                                anchor='e', font=('Courier', 10))
        self.val_lbl.grid(row=row, column=2, padx=(2, 6), pady=2)
        self.var.trace_add('write', self._update_label)
        if on_change:
            self.var.trace_add('write', lambda *_: on_change())

    def _update_label(self, *_):
        try:
            self.val_lbl.config(text=f'{self.var.get():+.2f}')
        except Exception:
            pass

    def get(self):
        return self.var.get()

    def set(self, v):
        self.var.set(v)


class App(tk.Tk):
    PLACEHOLDER_SIZE = (512, 512)

    def __init__(self, all_objects, lights, args):
        super().__init__()
        self.title('ADT Exocentric Camera Pose Selector')
        self.resizable(True, True)
        self.all_objects = all_objects
        self.lights      = lights
        self.args        = args
        self._render_thread  = None
        self._render_cancel  = threading.Event()
        self._last_render_ms = None
        self._photo          = None   # keep reference to prevent GC
        self._tmp_dir        = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '_gui_tmp')
        os.makedirs(self._tmp_dir, exist_ok=True)

        self._build_ui()
        self._refresh_info()
        self._show_placeholder()

    # ── UI layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Root grid: left = controls, right = image
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left  = tk.Frame(self, bd=1, relief='sunken', padx=4, pady=4)
        right = tk.Frame(self, bd=1, relief='sunken', padx=4, pady=4)
        left.grid(row=0, column=0, sticky='nsew', padx=(6,3), pady=6)
        right.grid(row=0, column=1, sticky='nsew', padx=(3,6), pady=6)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_controls(left)
        self._build_image_panel(right)

    def _section(self, parent, title, row):
        tk.Label(parent, text=title, font=('TkDefaultFont', 10, 'bold'),
                 anchor='w').grid(row=row, column=0, columnspan=3,
                                  sticky='w', padx=4, pady=(8,2))

    def _build_controls(self, f):
        preset = PRESETS['right_back']
        r = 0

        # ── Eye position ─────────────────────────────────────────────────────
        self._section(f, 'Eye position (m)', r); r += 1
        on = self._on_slider_change
        self.s_eye_x = SliderRow(f, 'Eye X', -5.0,  5.0, 0.05, preset['eye'][0], r, on); r += 1
        self.s_eye_y = SliderRow(f, 'Eye Y',  0.0,  4.0, 0.05, preset['eye'][1], r, on); r += 1
        self.s_eye_z = SliderRow(f, 'Eye Z', -5.0, 10.0, 0.05, preset['eye'][2], r, on); r += 1

        # ── Target ───────────────────────────────────────────────────────────
        self._section(f, 'Target point (m)', r); r += 1
        self.s_tgt_x = SliderRow(f, 'Tgt X', -5.0,  5.0, 0.05, preset['target'][0], r, on); r += 1
        self.s_tgt_y = SliderRow(f, 'Tgt Y',  0.0,  4.0, 0.05, preset['target'][1], r, on); r += 1
        self.s_tgt_z = SliderRow(f, 'Tgt Z', -5.0, 10.0, 0.05, preset['target'][2], r, on); r += 1

        # ── FOV ──────────────────────────────────────────────────────────────
        self._section(f, 'Field of view', r); r += 1
        self.s_fov = SliderRow(f, 'VFOV °', 20.0, 120.0, 1.0, preset['fov'], r, on); r += 1

        # ── Output size ──────────────────────────────────────────────────────
        self._section(f, 'Output size', r); r += 1
        size_frame = tk.Frame(f)
        size_frame.grid(row=r, column=0, columnspan=3, sticky='w', padx=4, pady=2); r += 1
        self.size_var = tk.IntVar(value=self.args.output_size)
        for sz in [256, 512, 1024]:
            ttk.Radiobutton(size_frame, text=f'{sz}px', value=sz,
                            variable=self.size_var).pack(side='left', padx=4)

        # ── Camera info ──────────────────────────────────────────────────────
        self._section(f, 'Camera info', r); r += 1
        info_frame = tk.Frame(f, bg='#1e1e2e', padx=6, pady=4)
        info_frame.grid(row=r, column=0, columnspan=3, sticky='ew', padx=4, pady=2); r += 1
        self._info_lbl = tk.Label(info_frame, text='', font=('Courier', 10),
                                  bg='#1e1e2e', fg='#cdd6f4',
                                  justify='left', anchor='w')
        self._info_lbl.pack(fill='x')

        # ── Presets ──────────────────────────────────────────────────────────
        self._section(f, 'Presets', r); r += 1
        for name, cfg in PRESETS.items():
            btn = ttk.Button(f, text=f'  {name}  — {cfg["desc"]}',
                             command=lambda c=cfg: self._load_preset(c))
            btn.grid(row=r, column=0, columnspan=3, sticky='ew', padx=4, pady=2); r += 1

        # ── Segmentation toggle ───────────────────────────────────────────────
        seg_frame = tk.Frame(f)
        seg_frame.grid(row=r, column=0, columnspan=3, sticky='w', padx=4, pady=(8,2)); r += 1
        self.seg_var = tk.BooleanVar(value=not self.args.no_segmentation)
        ttk.Checkbutton(seg_frame, text='Render segmentation mask',
                        variable=self.seg_var).pack(side='left')

        # ── Action buttons ────────────────────────────────────────────────────
        btn_frame = tk.Frame(f)
        btn_frame.grid(row=r, column=0, columnspan=3, sticky='ew',
                       padx=4, pady=(10, 4)); r += 1
        self.render_btn = ttk.Button(btn_frame, text='▶  Render',
                                     command=self._start_render)
        self.render_btn.pack(fill='x', pady=2)
        self.copy_btn = ttk.Button(btn_frame, text='📋  Copy command',
                                    command=self._copy_command)
        self.copy_btn.pack(fill='x', pady=2)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value='Ready')
        status_lbl = tk.Label(f, textvariable=self._status_var,
                               anchor='w', font=('TkDefaultFont', 9),
                               fg='#888')
        status_lbl.grid(row=r, column=0, columnspan=3, sticky='ew',
                        padx=4, pady=(6, 2)); r += 1

    def _build_image_panel(self, f):
        self._canvas = tk.Canvas(f, bg='#1e1e2e', cursor='crosshair')
        self._canvas.grid(row=0, column=0, sticky='nsew')
        self._canvas.bind('<Configure>', self._on_canvas_resize)

        # Overlay label shown while rendering
        self._overlay_var = tk.StringVar(value='')
        self._overlay_lbl = tk.Label(self._canvas,
                                     textvariable=self._overlay_var,
                                     font=('TkDefaultFont', 14, 'bold'),
                                     bg='#1e1e2e', fg='#f38ba8')
        # Bottom label for render time / path
        self._render_info_var = tk.StringVar(value='No render yet')
        tk.Label(f, textvariable=self._render_info_var,
                 anchor='w', font=('Courier', 9), fg='#666').grid(
            row=1, column=0, sticky='ew', padx=4, pady=(2, 4))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_params(self):
        eye    = np.array([self.s_eye_x.get(), self.s_eye_y.get(), self.s_eye_z.get()])
        target = np.array([self.s_tgt_x.get(), self.s_tgt_y.get(), self.s_tgt_z.get()])
        fov    = float(self.s_fov.get())
        size   = int(self.size_var.get())
        return eye, target, fov, size

    def _on_slider_change(self):
        self._refresh_info()

    def _refresh_info(self):
        eye, target, fov, _ = self._get_params()
        dist, pitch, yaw = camera_stats(eye, target)
        text = (f'Eye:    ({eye[0]:+.2f}, {eye[1]:+.2f}, {eye[2]:+.2f}) m\n'
                f'Target: ({target[0]:+.2f}, {target[1]:+.2f}, {target[2]:+.2f}) m\n'
                f'Dist:   {dist:.2f} m\n'
                f'Pitch:  {pitch:+.1f}°  (neg = looking down)\n'
                f'Yaw:    {yaw:+.1f}°  (0 = +Z, 90 = +X)\n'
                f'FOV:    {fov:.0f}°')
        self._info_lbl.config(text=text)

    def _load_preset(self, cfg):
        self.s_eye_x.set(float(cfg['eye'][0]))
        self.s_eye_y.set(float(cfg['eye'][1]))
        self.s_eye_z.set(float(cfg['eye'][2]))
        self.s_tgt_x.set(float(cfg['target'][0]))
        self.s_tgt_y.set(float(cfg['target'][1]))
        self.s_tgt_z.set(float(cfg['target'][2]))
        self.s_fov.set(float(cfg['fov']))
        self._refresh_info()

    def _show_placeholder(self):
        img = Image.new('RGB', self.PLACEHOLDER_SIZE, color=(30, 30, 46))
        self._display_image(img)

    def _display_image(self, pil_img):
        """Scale image to fit canvas while maintaining aspect ratio."""
        cw = max(self._canvas.winfo_width(),  self.PLACEHOLDER_SIZE[0])
        ch = max(self._canvas.winfo_height(), self.PLACEHOLDER_SIZE[1])
        scale = min(cw / pil_img.width, ch / pil_img.height, 1.0)
        # Allow upscaling up to fill the canvas
        scale = min(cw / pil_img.width, ch / pil_img.height)
        dw = max(1, int(pil_img.width  * scale))
        dh = max(1, int(pil_img.height * scale))
        display = pil_img.resize((dw, dh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(display)
        self._canvas.delete('all')
        self._canvas.create_image(cw // 2, ch // 2, anchor='center',
                                  image=self._photo)

    def _on_canvas_resize(self, event):
        if self._photo is not None:
            pass  # Image will redraw on next render; keep current for now

    # ── Render ────────────────────────────────────────────────────────────────

    def _start_render(self):
        if self._render_thread and self._render_thread.is_alive():
            self._status_var.set('⏳  Already rendering — please wait…')
            return
        eye, target, fov, size = self._get_params()
        seg = bool(self.seg_var.get())
        self.render_btn.config(state='disabled')
        self._status_var.set('⏳  Rendering…')
        self._overlay_var.set('Rendering…')
        self._overlay_lbl.place(relx=0.5, rely=0.5, anchor='center')

        self._render_thread = threading.Thread(
            target=self._render_worker,
            args=(eye, target, fov, size, seg),
            daemon=True)
        self._render_thread.start()
        self._poll_render()

    def _render_worker(self, eye, target, fov, size, segmentation):
        """Background thread: build JSON, call Blender, store result path."""
        self._render_result = None
        t0 = time.time()
        try:
            frame_data, _ = make_frame_data(
                eye, target, fov, size, self.all_objects, self.lights, segmentation)

            # Unique filename to avoid collisions
            tag = hashlib.md5(f'{eye}{target}{fov}{size}'.encode()).hexdigest()[:8]
            json_path = os.path.join(self._tmp_dir, f'gui_{tag}.json')
            out_png   = os.path.join(self._tmp_dir, f'gui_{tag}.png')

            with open(json_path, 'w') as f:
                json.dump(frame_data, f)

            cmd = [BLENDER_BIN, '--background', '--python', BLEND_SCRIPT,
                   '--', '--frame_data', json_path, '--output', out_png]
            if segmentation:
                cmd += ['--seg_output', os.path.join(self._tmp_dir, f'gui_{tag}_seg.exr')]

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            elapsed = time.time() - t0

            if os.path.exists(out_png):
                self._render_result = (out_png, elapsed)
            else:
                err = proc.stderr[-600:] if proc.stderr else '(no stderr)'
                self._render_result = ('error', elapsed, err)
        except Exception as exc:
            self._render_result = ('error', time.time() - t0, str(exc))

    def _poll_render(self):
        if self._render_thread and self._render_thread.is_alive():
            self.after(500, self._poll_render)
            return
        # Render finished
        self._overlay_lbl.place_forget()
        self._overlay_var.set('')
        result = getattr(self, '_render_result', None)
        if result is None:
            self._status_var.set('Render did not complete.')
        elif result[0] == 'error':
            elapsed, msg = result[1], result[2]
            self._status_var.set(f'Error after {elapsed:.1f}s — see terminal')
            print(f'[render error]\n{msg}')
        else:
            out_png, elapsed = result
            self._status_var.set(f'✓  Done in {elapsed:.1f}s')
            self._render_info_var.set(f'{out_png}  ({elapsed:.1f}s)')
            try:
                img = Image.open(out_png)
                self._display_image(img)
            except Exception as exc:
                self._status_var.set(f'Image load error: {exc}')
        self.render_btn.config(state='normal')

    # ── Copy command ──────────────────────────────────────────────────────────

    def _copy_command(self):
        eye, target, fov, size = self._get_params()
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'render_exocentric_blender.py')
        seg_flag = '' if self.seg_var.get() else ' --no_segmentation'
        cmd = (f'python {script} \\\n'
               f'    --camera right_back \\\n'
               f'    --eye {eye[0]:.3f} {eye[1]:.3f} {eye[2]:.3f} \\\n'
               f'    --target {target[0]:.3f} {target[1]:.3f} {target[2]:.3f} \\\n'
               f'    --fov {fov:.1f} \\\n'
               f'    --output_size {size}{seg_flag}')
        self.clipboard_clear()
        self.clipboard_append(cmd)
        self._status_var.set('✓  Command copied to clipboard!')
        # Also print to terminal
        print('\n── Copied render command ──────────────────────────────────')
        print(cmd)
        print('────────────────────────────────────────────────────────────\n')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output_size',     type=int, default=512)
    parser.add_argument('--frame_idx',       type=int, default=0)
    parser.add_argument('--no_segmentation', action='store_true')
    args = parser.parse_args()

    # Validate paths
    for label, path in [('GT_DIR', GT_DIR), ('MODELS_DIR', MODELS_DIR),
                        ('BLENDER_BIN', BLENDER_BIN), ('BLEND_SCRIPT', BLEND_SCRIPT)]:
        if not os.path.exists(path):
            print(f'[warn] {label} not found: {path}')
            print('       Edit the path constants at the top of this script.')

    print('Loading ADT scene data…', flush=True)
    try:
        all_objects, lights = load_scene(frame_idx=args.frame_idx)
    except FileNotFoundError as exc:
        print(f'Error loading scene: {exc}')
        print('Check that BASE / GT_DIR paths are correct in exo_camera_gui.py')
        sys.exit(1)
    print(f'  Loaded {len(all_objects)} objects, {len(lights)} lights.')

    app = App(all_objects, lights, args)
    app.mainloop()


if __name__ == '__main__':
    main()
