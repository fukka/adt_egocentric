"""
annotate_segmentation.py
========================
For a single rendered frame, overlays each visible object's name on:
  • the Blender render  (rgb/frame_NNNN.png)
  • the real ego-camera frame

Uses the instance segmentation map (segmentation/frame_NNNN.npy) to find
each object's pixel footprint and places a label at its centroid.

Usage:
    python annotate_segmentation.py \
        [--frame_idx 0] \
        [--render_dir /path/to/blender_rendered_maps] \
        [--output_dir /path/to/output] \
        [--min_px 200]          # ignore objects smaller than this (px)
        [--upscale 2.0]         # upscale image (not font) to reduce label overlap
        [--on_real]             # also annotate the real ego frame (needs VRS)

Output:
    <output_dir>/annotated_render_NNNN.png   — labels on Blender render
    <output_dir>/annotated_real_NNNN.png     — labels on real ego frame

Upscale note:
    --upscale enlarges the canvas so labels have more room without changing font
    size.  The image is resized with LANCZOS; the segmentation map is resized
    with NEAREST so integer UIDs are preserved exactly.  --min_px is
    automatically scaled by upscale² so it stays consistent in object-size terms.
"""

import sys, os, csv, json, argparse
sys.path.insert(0, '/sessions/dreamy-modest-brown/.local/lib/python3.10/site-packages')

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Paths ──────────────────────────────────────────────────────────────────
BASE        = ('/sessions/dreamy-modest-brown/mnt/ADT/'
               'Apartment_release_golden_skeleton_seq100_10s_sample_M1292')
GT_DIR      = f'{BASE}/groundtruth'
EGO_VRS     = f'{BASE}/main_recording.vrs'

# ── Helpers ────────────────────────────────────────────────────────────────

def load_font(size=14):
    """Return a PIL font, falling back to the built-in default."""
    for path in [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def upscale_inputs(img_arr: np.ndarray,
                   seg_uid: np.ndarray,
                   factor: float):
    """
    Upscale the image and segmentation map by *factor*.

    • Image   — LANCZOS (high-quality, anti-aliased)
    • Seg map — NEAREST  (preserves exact integer UIDs; no blending)

    Returns (img_arr_up, seg_uid_up) at the new resolution.
    """
    if factor == 1.0:
        return img_arr, seg_uid

    H, W = seg_uid.shape
    new_W = int(round(W * factor))
    new_H = int(round(H * factor))

    # ── Upscale image (LANCZOS) ────────────────────────────────────────────
    img_up = np.array(
        Image.fromarray(img_arr).resize((new_W, new_H), Image.LANCZOS)
    )

    # ── Upscale segmentation map (NEAREST, pure numpy index mapping) ────────
    # PIL's int32 mode would corrupt UIDs larger than 2^31-1.
    # Instead, build integer row/col lookup arrays and fancy-index directly —
    # int64 values pass through without any type conversion.
    row_idx = np.floor(np.arange(new_H) * H / new_H).astype(np.intp).clip(0, H - 1)
    col_idx = np.floor(np.arange(new_W) * W / new_W).astype(np.intp).clip(0, W - 1)
    seg_up  = seg_uid[np.ix_(row_idx, col_idx)]   # (new_H, new_W), still int64

    return img_up, seg_up


def uid_color(uid: int):
    """Stable distinct colour for a UID (same hash as colorize_seg)."""
    import hashlib
    d = hashlib.md5(str(uid).encode()).digest()
    return (d[0], d[1], d[2])


def annotate_image(img_arr: np.ndarray,
                   seg_uid: np.ndarray,
                   uid_to_name: dict,
                   min_px: int = 200,
                   font_size: int = 13,
                   outline: bool = True) -> Image.Image:
    """
    Draw a coloured label for every visible object onto img_arr.

    Parameters
    ----------
    img_arr   : H×W×3 uint8 RGB image to annotate
    seg_uid   : H×W int64 instance-UID map (0 = background)
    uid_to_name: {str(uid) → name}
    min_px    : skip objects whose mask is smaller than this many pixels
    font_size : point size of the label text
    outline   : draw a dark outline around the text for readability

    Returns annotated PIL Image.
    """
    img   = Image.fromarray(img_arr).convert('RGBA')
    draw  = ImageDraw.Draw(img)
    font  = load_font(font_size)

    H, W = seg_uid.shape
    unique_uids = [int(u) for u in np.unique(seg_uid) if u != 0]

    # ── Collect label positions, then resolve overlaps ─────────────────────
    labels = []   # (uid, name, col, cx, cy, tw, th)
    for uid in unique_uids:
        mask = (seg_uid == uid)
        if int(mask.sum()) < min_px:
            continue
        name = uid_to_name.get(str(uid), f'uid:{uid}')
        col  = uid_color(uid)
        ys, xs = np.where(mask)
        cx, cy = int(xs.mean()), int(ys.mean())
        try:
            bbox = font.getbbox(name)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        except AttributeError:
            tw, th = font.getsize(name)
        labels.append([uid, name, col, cx, cy, tw, th])

    # Sort largest-first so big objects claim space first
    labels.sort(key=lambda r: -(seg_uid == r[0]).sum())

    pad  = 3
    step = font_size + 2 * pad + 2

    def overlaps(tx, ty, tw, th, placed):
        """Return True if this label box overlaps any already-placed box."""
        for (px, py, pw, ph) in placed:
            if (tx < px + pw + pad and tx + tw + pad > px and
                    ty < py + ph + pad and ty + th + pad > py):
                return True
        return False

    placed = []   # list of (tx, ty, tw, th) of accepted labels

    for uid, name, col, cx, cy, tw, th in labels:
        # Start at centroid, then nudge downward until no overlap
        tx0 = max(2, min(cx - tw // 2, W - tw - 4))
        ty0 = max(2, min(cy - th // 2, H - th - 4))
        tx, ty = tx0, ty0
        for attempt in range(30):
            if not overlaps(tx, ty, tw, th, placed):
                break
            ty += step
            if ty + th + pad > H:
                ty = max(2, ty0 - step * (attempt + 1))
            tx = max(2, min(cx - tw // 2, W - tw - 4))
        placed.append((tx, ty, tw, th))

        # Semi-transparent background rectangle
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rectangle(
            [tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
            fill=(*col, 170)
        )
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)

        # Text with dark outline
        if outline:
            for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1),(0,-1),(0,1),(-1,0),(1,0)]:
                draw.text((tx+dx, ty+dy), name, fill=(0, 0, 0, 255), font=font)
        draw.text((tx, ty), name, fill=(255, 255, 255, 255), font=font)

        # Dot at centroid with line to label
        r = 3
        lx = tx + tw // 2
        ly = ty + th // 2
        if abs(lx - cx) + abs(ly - cy) > 20:
            draw.line([cx, cy, lx, ly], fill=(*col, 200), width=1)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r],
                     fill=(*col, 255), outline=(255, 255, 255, 200))

    return img.convert('RGB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_idx', type=int, default=0,
                        help='Frame index in the rendered output (default: 0)')
    parser.add_argument('--render_dir', type=str,
                        default=f'{BASE}/blender_rendered_maps',
                        help='Directory produced by render_from_poses_blender_maps.py')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save annotated images (default: render_dir/annotated)')
    parser.add_argument('--min_px', type=int, default=200,
                        help='Skip objects with fewer visible pixels than this (default: 200)')
    parser.add_argument('--font_size', type=int, default=13)
    parser.add_argument('--upscale', type=float, default=1.0,
                        help='Scale factor applied to the image canvas before labelling '
                             '(e.g. 2.0 → double resolution).  Font size is NOT scaled, '
                             'so labels stay the same size while objects spread further '
                             'apart, reducing overlap.  min_px is auto-scaled by '
                             'upscale^2 to keep the same object-size threshold.')
    parser.add_argument('--on_real', action='store_true',
                        help='Also annotate the real ego-camera frame from the VRS')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.render_dir, 'annotated')
    os.makedirs(args.output_dir, exist_ok=True)

    idx_str = f'{args.frame_idx:04d}'

    # ── Load segmentation map ───────────────────────────────────────────────
    seg_path = os.path.join(args.render_dir, 'segmentation', f'frame_{idx_str}.npy')
    if not os.path.exists(seg_path):
        sys.exit(f'ERROR: segmentation not found at {seg_path}\n'
                 f'Run render_from_poses_blender_maps.py first.')
    seg_uid = np.load(seg_path)   # (H, W) int64

    # ── Load instance names ─────────────────────────────────────────────────
    with open(f'{GT_DIR}/instances.json') as f:
        instances = json.load(f)

    uid_to_name = {}
    for v in instances.values():
        if 'instance_id' not in v:
            continue
        uid = str(v['instance_id'])
        # Prefer instance_name; fall back to prototype_name
        name = v.get('instance_name') or v.get('prototype_name', '?')
        # Strip trailing _N suffix (e.g. "Lamp_1" → "Lamp_1" kept; just clean up)
        uid_to_name[uid] = name

    unique_uids = [u for u in np.unique(seg_uid) if u != 0]
    print(f'Frame {idx_str}: {len(unique_uids)} visible objects '
          f'(≥{args.min_px}px threshold will reduce this)')

    # ── Effective min_px after upscaling (object pixel count grows by factor²) ─
    upscale      = max(args.upscale, 0.1)   # guard against zero/negative
    min_px_scaled = int(args.min_px * upscale ** 2)
    if upscale != 1.0:
        print(f'Upscale ×{upscale:.2f}: canvas {seg_uid.shape[1]}×{seg_uid.shape[0]}'
              f' → {int(seg_uid.shape[1]*upscale)}×{int(seg_uid.shape[0]*upscale)}, '
              f'min_px {args.min_px} → {min_px_scaled}')

    # ── Annotate Blender render ─────────────────────────────────────────────
    render_path = os.path.join(args.render_dir, 'rgb', f'frame_{idx_str}.png')
    if not os.path.exists(render_path):
        sys.exit(f'ERROR: render not found at {render_path}')

    render_arr = np.array(Image.open(render_path).convert('RGB'))
    render_arr_up, seg_uid_up = upscale_inputs(render_arr, seg_uid, upscale)
    ann_render  = annotate_image(render_arr_up, seg_uid_up, uid_to_name,
                                 min_px=min_px_scaled,
                                 font_size=args.font_size)

    out_render = os.path.join(args.output_dir, f'annotated_render_{idx_str}.png')
    ann_render.save(out_render)
    print(f'Saved → {out_render}')

    # ── Optionally annotate real ego frame ──────────────────────────────────
    if args.on_real:
        try:
            from projectaria_tools.core import data_provider
            from projectaria_tools.core.stream_id import StreamId
            p_ego    = data_provider.create_vrs_data_provider(EGO_VRS)
            img_data = p_ego.get_image_data_by_index(StreamId('214-1'), args.frame_idx)
            ego_arr  = img_data[0].to_numpy_array()
            # First resize to match the original (pre-upscale) seg map dimensions,
            # then upscale together so UID ↔ pixel correspondence is exact.
            H, W = seg_uid.shape
            ego_pil    = Image.fromarray(ego_arr).resize((W, H), Image.LANCZOS)
            ego_arr_rs = np.array(ego_pil.convert('RGB'))
            ego_arr_up, seg_uid_up_real = upscale_inputs(ego_arr_rs, seg_uid, upscale)
            ann_real = annotate_image(ego_arr_up, seg_uid_up_real, uid_to_name,
                                      min_px=min_px_scaled,
                                      font_size=args.font_size)
            out_real = os.path.join(args.output_dir, f'annotated_real_{idx_str}.png')
            ann_real.save(out_real)
            print(f'Saved → {out_real}')
        except Exception as e:
            print(f'[warning] Could not annotate real frame: {e}')

    # ── Print object list (pixel counts in original resolution) ────────────
    print(f'\nVisible objects (≥{args.min_px} px in original resolution):')
    rows = []
    for uid in sorted(unique_uids):
        mask = (seg_uid == uid)
        npx  = int(mask.sum())
        if npx < args.min_px:
            continue
        name = uid_to_name.get(str(uid), '?')
        ys, xs = np.where(mask)
        rows.append((npx, name, int(xs.mean()), int(ys.mean())))
    rows.sort(reverse=True)
    for npx, name, cx, cy in rows:
        print(f'  {name:<40s}  {npx:6d} px  centroid=({cx},{cy})')


if __name__ == '__main__':
    main()
