"""
visualize_depth_3d.py
=====================
3-D visualisation of a predicted depth map combined with its RGB input.

Inputs
------
  --depth       Path to a predicted depth .npy file
                (e.g. dav2_large_pred_aligned.npy saved by eval_depth_anything_v2.py)
  --rgb         Path to the matching RGB image (.png / .jpg)
  --intrinsics  fx fy cx cy  (estimated from image size when omitted)

Modes
-----
  --gui         Open an interactive Open3D window: drag/scroll to rotate & zoom.
                Falls back to matplotlib 3-D axes when open3d is not installed.

  --save_sideview PATH
                Save a 2-D projection image directly (no window needed).
                Use --view to choose the projection plane:
                  side  → X-Z  (horizontal extent vs depth)    [default]
                  top   → X-Y  (bird's-eye view)
                  front → Y-Z  (height vs depth)
                  iso   → isometric-ish 3-D scatter (matplotlib)
                  best  → auto-pick the plane with the largest point spread

Additional options
------------------
  --stride      Subsample stride (default 2). Increase for speed on large images.
  --max_depth   Clip depth beyond this value in metres (default 10.0).
  --mask_invalid_input
                Drop points whose input RGB is a black/blank border (e.g. outside
                the Aria fisheye circle).  Tune with --input_black_thresh and
                --input_valid_erode.
  --gt_depth PATH
                Optional GT depth .npy; also drop points where GT depth is invalid
                (non-finite, ≤0, or > --max_depth).  Scale it with --gt_depth_scale.
  --point_size  Point size in the Open3D window (default 2.0).
  --bg_color    Background colour for Open3D: dark | white (default dark).
  --variant     Label string shown in figure titles (default: pred_aligned).
  --dpi         DPI for saved side-view PNG (default 150).

Example usage
-------------
  # Interactive GUI
  python visualize_depth_3d.py \\
      --depth  /path/to/dav2_large_pred_aligned.npy \\
      --rgb    /path/to/frame_0000.png \\
      --gui

  # Save best side-view directly
  python visualize_depth_3d.py \\
      --depth  /path/to/dav2_large_pred_aligned.npy \\
      --rgb    /path/to/frame_0000.png \\
      --save_sideview /path/to/output.png \\
      --view best

  # Both at once
  python visualize_depth_3d.py \\
      --depth  /path/to/dav2_large_pred_aligned.npy \\
      --rgb    /path/to/frame_0000.png \\
      --gui \\
      --save_sideview /path/to/output.png \\
      --view side \\
      --intrinsics 611 611 704 704

Dependencies
------------
  pip install numpy pillow matplotlib
  pip install open3d          # optional but strongly recommended for GUI mode
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

# ─── matplotlib is always required ───────────────────────────────────────────
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D projection)

# ─── optional Open3D ─────────────────────────────────────────────────────────
try:
    import open3d as o3d
    _HAS_O3D = True
except ImportError:
    _HAS_O3D = False


# ─────────────────────────────────────────────────────────────────────────────
# Camera / back-projection utilities
# ─────────────────────────────────────────────────────────────────────────────

def estimate_intrinsics(h: int, w: int, diag_fov_deg: float = 55.0):
    """Estimate pinhole intrinsics from image size and a diagonal FoV."""
    diag_px  = np.sqrt(h ** 2 + w ** 2)
    diag_fov = np.radians(diag_fov_deg)
    f = diag_px / (2.0 * np.tan(diag_fov / 2.0))
    return f, f, w / 2.0, h / 2.0


def depth_to_pointcloud(depth: np.ndarray,
                         rgb:   np.ndarray,
                         fx: float, fy: float,
                         cx: float, cy: float,
                         max_depth: float = 10.0,
                         stride: int = 2,
                         extra_valid: np.ndarray = None):
    """
    Back-project a depth map to a coloured 3-D point cloud.

    Parameters
    ----------
    depth       : (H, W) float32 depth in metres
    rgb         : (H, W, 3) uint8 RGB image (must match depth H×W)
    fx,fy,cx,cy : pinhole camera intrinsics
    max_depth   : discard points beyond this distance
    stride      : pixel subsampling step (2 = every other pixel in each axis)
    extra_valid : optional (H, W) bool mask — additional per-pixel validity to
                  AND with the depth-range check (e.g. drop black input borders
                  or pixels where GT depth is invalid).  None keeps all pixels.

    Returns
    -------
    xyz    : (N, 3) float32 — 3-D point positions
    colors : (N, 3) float32 — RGB colours in [0, 1]
    """
    H, W = depth.shape
    # Resize RGB to match depth if needed
    if rgb.shape[:2] != (H, W):
        rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))

    u = np.arange(0, W, stride)
    v = np.arange(0, H, stride)
    uu, vv = np.meshgrid(u, v)

    d = depth[vv, uu]
    valid = np.isfinite(d) & (d > 0.01) & (d < max_depth)
    if extra_valid is not None:
        valid &= extra_valid[vv, uu]

    d  = d[valid]
    uu = uu[valid]
    vv = vv[valid]

    x = (uu - cx) * d / fx
    y = (vv - cy) * d / fy   # positive y = downward (image convention)
    z = d                    # positive z = into scene

    xyz    = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = (rgb[vv * stride // stride, uu * stride // stride].astype(np.float32)
              / 255.0)
    # Correct indexing: rgb is already at original res, uu/vv in original coords
    colors = rgb[vv, uu].astype(np.float32) / 255.0

    return xyz, colors


# ─────────────────────────────────────────────────────────────────────────────
# Open3D interactive viewer
# ─────────────────────────────────────────────────────────────────────────────

def show_open3d(xyz: np.ndarray, colors: np.ndarray,
                point_size: float = 2.0, bg_dark: bool = True,
                title: str = "Depth Point Cloud"):
    """Launch an interactive Open3D visualiser window."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title, width=1280, height=720)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = point_size
    if bg_dark:
        opt.background_color = np.array([0.1, 0.1, 0.15])
    else:
        opt.background_color = np.array([1.0, 1.0, 1.0])

    # Flip y so that "up" in the Open3D window is up in the scene
    ctr = vis.get_view_control()
    ctr.set_up([0, -1, 0])
    ctr.set_front([0, 0, -1])
    ctr.set_lookat(xyz.mean(axis=0).tolist())
    ctr.set_zoom(0.6)

    print(f"\n  [Open3D] Interactive window: drag to rotate, scroll to zoom.")
    print(f"           Press Q or Escape to close.\n")
    vis.run()
    vis.destroy_window()


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib fallback GUI (3-D scatter)
# ─────────────────────────────────────────────────────────────────────────────

def show_matplotlib_3d(xyz: np.ndarray, colors: np.ndarray,
                        title: str = "Depth Point Cloud",
                        max_pts: int = 80_000):
    """Interactive matplotlib 3-D scatter (drag to rotate in the window)."""
    if len(xyz) > max_pts:
        idx = np.random.choice(len(xyz), max_pts, replace=False)
        xyz    = xyz[idx]
        colors = colors[idx]

    matplotlib.use("TkAgg")   # interactive; switch to "Qt5Agg" / "MacOSX" if needed
    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#1a1a2e")

    ax.scatter(xyz[:, 0], xyz[:, 2], -xyz[:, 1],
               c=colors, s=0.5, depthshade=True)
    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Z depth (m)", color="white")
    ax.set_zlabel("Y up (m)", color="white")
    ax.tick_params(colors="white")
    ax.set_title(title, color="white")

    print("\n  [Matplotlib] Interactive 3-D window — drag to rotate.\n")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 2-D side-view projection helpers
# ─────────────────────────────────────────────────────────────────────────────

_VIEW_INFO = {
    # key: (horiz_axis_idx, vert_axis_idx, horiz_label, vert_label)
    "side":  (0, 1, "X (m)",      "Y (m)  [up=neg]"),
    "top":   (0, 2, "X (m)",      "Z depth (m)"),
    "front": (2, 1, "Z depth (m)","Y (m)  [up=neg]"),
}


def _pick_best_view(xyz: np.ndarray) -> str:
    """Pick the projection plane with the largest point-cloud spread."""
    spans = {
        "side":  xyz[:, 0].ptp() * xyz[:, 1].ptp(),
        "top":   xyz[:, 0].ptp() * xyz[:, 2].ptp(),
        "front": xyz[:, 2].ptp() * xyz[:, 1].ptp(),
    }
    return max(spans, key=spans.get)


def save_sideview(xyz: np.ndarray, colors: np.ndarray,
                   rgb_orig: np.ndarray, depth: np.ndarray,
                   view: str, output_path: str,
                   variant: str = "pred_aligned",
                   dpi: int = 150,
                   max_pts: int = 200_000):
    """
    Save a composite PNG:  [RGB input | depth map (coloured) | 2-D point projection].

    Parameters
    ----------
    view : 'side' | 'top' | 'front' | 'iso' | 'best'
    """
    if view == "best":
        view = _pick_best_view(xyz)
        print(f"  [sideview] Auto-selected view: '{view}'")

    # ── subsample for scatter speed ───────────────────────────────────────────
    if len(xyz) > max_pts:
        idx    = np.random.choice(len(xyz), max_pts, replace=False)
        xyz_s  = xyz[idx]
        col_s  = colors[idx]
    else:
        xyz_s, col_s = xyz, colors

    # ── build figure ──────────────────────────────────────────────────────────
    if view == "iso":
        fig = plt.figure(figsize=(18, 6), dpi=dpi)
        fig.patch.set_facecolor("#1a1a2e")
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.06)

        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2], projection="3d")

        _draw_rgb_panel(ax0, rgb_orig, "RGB Input")
        _draw_depth_panel(ax1, depth,  "Predicted Depth")
        _draw_iso_panel(ax2, xyz_s, col_s, "3-D Point Cloud (isometric)")

    else:
        h_idx, v_idx, h_label, v_label = _VIEW_INFO[view]
        fig = plt.figure(figsize=(18, 6), dpi=dpi)
        fig.patch.set_facecolor("#1a1a2e")
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.06)

        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])

        _draw_rgb_panel(ax0, rgb_orig,  "RGB Input")
        _draw_depth_panel(ax1, depth,   "Predicted Depth")
        _draw_proj_panel(ax2, xyz_s, col_s,
                         h_idx, v_idx, h_label, v_label,
                         f"3-D → 2-D projection  [{view} view]")

    # ── common footer ─────────────────────────────────────────────────────────
    fig.text(0.5, 0.005,
             f"Depth Anything V2  [{variant}]   —   {view} view",
             ha="center", va="bottom", color="white", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#2d2d44", alpha=0.85))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [sideview] Saved → {output_path}")


# ── panel helpers ─────────────────────────────────────────────────────────────

def _draw_rgb_panel(ax, rgb: np.ndarray, title: str):
    ax.imshow(rgb)
    ax.set_title(title, color="white", fontsize=11, pad=4)
    ax.axis("off")
    ax.set_facecolor("#1a1a2e")


def _draw_depth_panel(ax, depth: np.ndarray, title: str):
    d = depth.copy().astype(np.float32)
    vmin = float(np.nanpercentile(d, 2))
    vmax = float(np.nanpercentile(d, 98))
    d_norm = np.clip((d - vmin) / (vmax - vmin + 1e-8), 0, 1)
    d_norm[~np.isfinite(depth)] = 0.0
    img = plt.get_cmap("magma_r")(d_norm)[..., :3]
    ax.imshow(img)
    ax.set_title(title, color="white", fontsize=11, pad=4)
    ax.set_xlabel(f"[{vmin:.2f} – {vmax:.2f} m]", color="#aaaaaa", fontsize=8)
    ax.axis("off")
    ax.set_facecolor("#1a1a2e")


def _draw_proj_panel(ax, xyz, colors,
                      h_idx, v_idx, h_label, v_label, title):
    h = xyz[:, h_idx]
    v = -xyz[:, v_idx]   # flip so y=0 is at bottom (image y is downward)

    # Density-based alpha: thin out crowded regions visually
    alpha = np.clip(0.6 - 0.0001 * len(xyz) ** 0.5, 0.05, 0.6)

    ax.scatter(h, v, c=colors, s=0.3, alpha=alpha, linewidths=0)
    ax.set_facecolor("#0d0d1a")
    ax.set_title(title, color="white", fontsize=11, pad=4)
    ax.set_xlabel(h_label,  color="#aaaaaa", fontsize=8)
    ax.set_ylabel(v_label,  color="#aaaaaa", fontsize=8)
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    # Equal aspect so distances aren't distorted
    h_span = np.ptp(h) or 1.0
    v_span = np.ptp(v) or 1.0
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.02)


def _draw_iso_panel(ax, xyz, colors, title):
    ax.scatter(xyz[:, 0], xyz[:, 2], -xyz[:, 1],
               c=colors, s=0.3, depthshade=True, alpha=0.5)
    ax.set_facecolor("#0d0d1a")
    ax.set_title(title, color="white", fontsize=11, pad=4)
    ax.set_xlabel("X (m)",      color="#aaaaaa", fontsize=7)
    ax.set_ylabel("Z (depth m)",color="#aaaaaa", fontsize=7)
    ax.set_zlabel("Y up (m)",   color="#aaaaaa", fontsize=7)
    ax.tick_params(colors="#aaaaaa", labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="3-D visualisation of a predicted depth map + RGB input.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Required inputs ───────────────────────────────────────────────────────
    parser.add_argument("--depth", required=True,
                        help="Path to predicted depth .npy  "
                             "(e.g. dav2_large_pred_aligned.npy)")
    parser.add_argument("--rgb", required=True,
                        help="Path to matching RGB image (.png / .jpg)")

    # ── Camera ────────────────────────────────────────────────────────────────
    parser.add_argument("--intrinsics", type=float, nargs=4,
                        metavar=("fx", "fy", "cx", "cy"), default=None,
                        help="Pinhole intrinsics. Estimated from image size when omitted.")

    # ── Mode switches ─────────────────────────────────────────────────────────
    parser.add_argument("--gui", action="store_true",
                        help="Open an interactive 3-D viewer window "
                             "(Open3D if available, else matplotlib).")
    parser.add_argument("--save_sideview", default=None, metavar="PATH",
                        help="Save a 2-D projection PNG to this path.")
    parser.add_argument("--view", default="side",
                        choices=["side", "top", "front", "iso", "best"],
                        help="Projection plane for --save_sideview (default: side).")

    # ── Point cloud options ───────────────────────────────────────────────────
    parser.add_argument("--stride", type=int, default=2,
                        help="Pixel subsampling stride (default 2).")
    parser.add_argument("--max_depth", type=float, default=10.0,
                        help="Clip depth beyond this value in metres (default 10.0).")

    # ── Invalid-region masking ────────────────────────────────────────────────
    parser.add_argument("--mask_invalid_input", action="store_true", default=False,
                        help="Drop points whose input RGB is a black/blank border "
                             "(e.g. outside the Aria fisheye circle).")
    parser.add_argument("--input_black_thresh", type=int, default=8,
                        help="With --mask_invalid_input: a pixel is invalid input when "
                             "max(R,G,B) <= this value (default: 8).")
    parser.add_argument("--input_valid_erode", type=int, default=0,
                        help="With --mask_invalid_input: erode the valid-input region by "
                             "this many pixels (default: 0 = disabled).")
    parser.add_argument("--gt_depth", default=None, metavar="PATH",
                        help="Optional GT depth .npy. When given, points where GT depth "
                             "is invalid (non-finite, <=0, or > --max_depth) are dropped.")
    parser.add_argument("--gt_depth_scale", type=float, default=1.0,
                        help="Multiply GT depth (--gt_depth) by this factor before the "
                             "validity check (e.g. 0.001 for mm→m; default 1.0).")

    # ── Open3D rendering ──────────────────────────────────────────────────────
    parser.add_argument("--point_size", type=float, default=2.0,
                        help="Point size in Open3D window (default 2.0).")
    parser.add_argument("--bg_color", default="dark", choices=["dark", "white"],
                        help="Open3D background colour (default: dark).")

    # ── Display labels ────────────────────────────────────────────────────────
    parser.add_argument("--variant", default="pred_aligned",
                        help="Model variant label for figure titles.")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for saved side-view image (default 150).")

    args = parser.parse_args()

    if not args.gui and args.save_sideview is None:
        parser.error("At least one of --gui or --save_sideview must be specified.")

    # ── Load depth ────────────────────────────────────────────────────────────
    print(f"\n  Loading depth : {args.depth}")
    depth = np.load(args.depth).astype(np.float32)
    if depth.ndim == 3:
        depth = depth.squeeze(-1)
    print(f"    shape={depth.shape}  range=[{np.nanmin(depth):.3f}, {np.nanmax(depth):.3f}] m")

    # ── Load RGB ──────────────────────────────────────────────────────────────
    print(f"  Loading RGB   : {args.rgb}")
    rgb = np.array(Image.open(args.rgb).convert("RGB"))
    print(f"    shape={rgb.shape}")

    # Resize RGB to depth size if they differ (shouldn't happen with aligned files)
    H, W = depth.shape
    if rgb.shape[:2] != (H, W):
        print(f"    Resizing RGB from {rgb.shape[:2]} → {(H, W)}")
        rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))

    # ── Intrinsics ────────────────────────────────────────────────────────────
    if args.intrinsics is not None:
        fx, fy, cx, cy = args.intrinsics
        print(f"  Intrinsics    : fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")
    else:
        fx, fy, cx, cy = estimate_intrinsics(H, W)
        print(f"  Intrinsics    : estimated (55° diag FoV) "
              f"fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")

    # ── Build optional invalid-region mask ────────────────────────────────────
    # extra_valid is at depth resolution (H, W); rgb has already been resized to
    # (H, W) above so it can be used directly.
    extra_valid = None
    if args.mask_invalid_input:
        lum = rgb.max(axis=-1) if rgb.ndim == 3 else rgb
        iv = lum > args.input_black_thresh
        if args.input_valid_erode > 0:
            try:
                import cv2
                k = 2 * int(args.input_valid_erode) + 1
                iv = cv2.erode(iv.astype(np.uint8),
                               np.ones((k, k), np.uint8)).astype(bool)
            except ImportError:
                print("    [WARN] --input_valid_erode needs opencv (cv2); "
                      "skipping erosion.")
        extra_valid = iv
        print(f"  Input-valid mask : {iv.sum():,}/{iv.size:,} px "
              f"(black_thresh={args.input_black_thresh}, "
              f"erode={args.input_valid_erode})")

    if args.gt_depth is not None:
        print(f"  Loading GT depth : {args.gt_depth}")
        gt = np.load(args.gt_depth).astype(np.float32)
        if gt.ndim == 3:
            gt = gt.squeeze(-1)
        gt = gt * args.gt_depth_scale
        gt_valid = np.isfinite(gt) & (gt > 0) & (gt <= args.max_depth)
        if gt_valid.shape != (H, W):
            gt_valid = np.array(
                Image.fromarray((gt_valid.astype(np.uint8) * 255))
                .resize((W, H), Image.NEAREST)) > 127
        extra_valid = gt_valid if extra_valid is None else (extra_valid & gt_valid)
        print(f"  GT-valid mask    : {gt_valid.sum():,}/{gt_valid.size:,} px "
              f"(scale={args.gt_depth_scale}, max_depth={args.max_depth} m)")

    # ── Build point cloud ─────────────────────────────────────────────────────
    print(f"  Building point cloud (stride={args.stride}, max_depth={args.max_depth} m) …")
    xyz, colors = depth_to_pointcloud(
        depth, rgb, fx, fy, cx, cy,
        max_depth=args.max_depth,
        stride=args.stride,
        extra_valid=extra_valid,
    )
    print(f"    {len(xyz):,} points")

    title = f"Depth Point Cloud  [{args.variant}]"

    # ── Interactive GUI ───────────────────────────────────────────────────────
    if args.gui:
        if _HAS_O3D:
            print("\n  Launching Open3D interactive viewer …")
            show_open3d(xyz, colors,
                        point_size=args.point_size,
                        bg_dark=(args.bg_color == "dark"),
                        title=title)
        else:
            print("\n  open3d not found — falling back to matplotlib 3-D window.")
            print("  Install with:  pip install open3d")
            show_matplotlib_3d(xyz, colors, title=title)

    # ── Save side-view ────────────────────────────────────────────────────────
    if args.save_sideview is not None:
        print(f"\n  Saving side-view ({args.view}) …")
        save_sideview(
            xyz, colors,
            rgb_orig=rgb,
            depth=depth,
            view=args.view,
            output_path=args.save_sideview,
            variant=args.variant,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()