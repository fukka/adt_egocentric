"""
ADT Egocentric + Synthetic Rectification Pipeline
==================================================
Both the real egocentric and synthetic VRS files use FISHEYE624
(Kannala-Brandt-style) distortion. This script:
  1. Reads paired frames from both VRS files
  2. Rescales the synthetic calibration (stored at 2880x2880 but frames
     are 1408x1408) to match actual frame dimensions
  3. Undistorts both to the same pinhole (LINEAR) target calibration
  4. Saves rectified paired images for all frames (or a subset)

Usage:
    python rectify_pipeline.py [--num_frames N] [--output_size S] [--focal F]
"""

import sys, os, argparse
sys.path.insert(0, '/sessions/dreamy-modest-brown/.local/lib/python3.10/site-packages')

import numpy as np
from PIL import Image
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.stream_id import StreamId

BASE = '/sessions/dreamy-modest-brown/mnt/ADT/Apartment_release_golden_skeleton_seq100_10s_sample_M1292'
EGO_VRS  = os.path.join(BASE, 'main_recording.vrs')
SYN_VRS  = os.path.join(BASE, 'synthetic', 'synthetic_video.vrs')
RGB_STREAM = StreamId('214-1')


def rescale_calibration(cam_calib, actual_w, actual_h):
    """
    The synthetic VRS calibration is stored at render resolution (2880x2880)
    but frames are actually stored at 1408x1408. Rescale the calibration to
    match the actual stored frame size.
    Signature: cam.rescale(new_resolution [w,h], scale, origin_offset=[0,0])
    """
    import numpy as np
    calib_w, calib_h = cam_calib.get_image_size()
    if calib_w == actual_w and calib_h == actual_h:
        return cam_calib  # Already matches
    scale = actual_w / calib_w
    print(f'  Rescaling calibration from {calib_w}x{calib_h} -> {actual_w}x{actual_h} (factor={scale:.4f})')
    new_res = np.array([actual_w, actual_h], dtype=np.int32)
    return cam_calib.rescale(new_res, scale)


def build_linear_calibration(src_calib, out_size, focal_px):
    """Create a target pinhole calibration with square image and given focal length."""
    w, h = out_size, out_size
    T_device_camera = src_calib.get_transform_device_camera()
    return calibration.get_linear_camera_calibration(
        w, h, focal_px,
        'camera-rgb-linear',
        T_device_camera
    )


def rectify_frame(frame_np, src_calib, dst_calib):
    """Undistort a fisheye frame to pinhole using projectaria calibration swap."""
    # distort_by_calibration(src_image, dst_calib, src_calib) -> remapped image
    rectified = calibration.distort_by_calibration(frame_np, dst_calib, src_calib)
    return rectified.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Rectify ADT ego+synthetic frames')
    parser.add_argument('--num_frames', type=int, default=None,
                        help='Number of frames to process (default: all)')
    parser.add_argument('--output_size', type=int, default=512,
                        help='Output image width/height in pixels (default: 512)')
    parser.add_argument('--focal', type=float, default=300.0,
                        help='Target focal length in pixels for pinhole model (default: 300)')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(BASE, 'rectified'),
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, 'ego'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'synthetic'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'side_by_side'), exist_ok=True)

    # ── Load providers ──────────────────────────────────────────────────────
    print('Loading egocentric VRS...')
    p_ego = data_provider.create_vrs_data_provider(EGO_VRS)
    ego_calib = p_ego.get_device_calibration().get_camera_calib('camera-rgb')
    n_ego = p_ego.get_num_data(RGB_STREAM)

    print('Loading synthetic VRS...')
    p_syn = data_provider.create_vrs_data_provider(SYN_VRS)
    syn_calib_raw = p_syn.get_device_calibration().get_camera_calib('camera-rgb')
    n_syn = p_syn.get_num_data(RGB_STREAM)

    n_frames = min(n_ego, n_syn)
    if args.num_frames is not None:
        n_frames = min(n_frames, args.num_frames)
    print(f'\nTotal paired frames available: min({n_ego}, {n_syn}) = {min(n_ego,n_syn)}')
    print(f'Processing: {n_frames} frames')

    # ── Calibrations ────────────────────────────────────────────────────────
    print('\n─── Egocentric calibration ───')
    print(f'  model     : {ego_calib.get_model_name()}')
    print(f'  image_size: {ego_calib.get_image_size()}')
    print(f'  focal_px  : {ego_calib.get_focal_lengths()}')

    # Get actual frame size for synthetic to detect mismatch
    syn_frame0 = p_syn.get_image_data_by_index(RGB_STREAM, 0)[0].to_numpy_array()
    actual_h, actual_w = syn_frame0.shape[:2]
    print('\n─── Synthetic calibration ───')
    print(f'  model            : {syn_calib_raw.get_model_name()}')
    print(f'  calib image_size : {syn_calib_raw.get_image_size()}')
    print(f'  actual frame size: {actual_w}x{actual_h}')
    syn_calib = rescale_calibration(syn_calib_raw, actual_w, actual_h)
    print(f'  rescaled focal_px: {syn_calib.get_focal_lengths()}')

    # ── Build linear target calibration ─────────────────────────────────────
    print(f'\n─── Target LINEAR calibration ───')
    print(f'  output size : {args.output_size}x{args.output_size}')
    print(f'  focal_px    : {args.focal}')
    # Use ego_calib extrinsics as reference (T_device_camera)
    dst_calib = build_linear_calibration(ego_calib, args.output_size, args.focal)
    print(f'  model       : {dst_calib.get_model_name()}')

    # ── Process frames ───────────────────────────────────────────────────────
    print(f'\nRectifying {n_frames} frame pairs...')
    for i in range(n_frames):
        # Load raw frames
        ego_arr = p_ego.get_image_data_by_index(RGB_STREAM, i)[0].to_numpy_array()
        syn_arr = p_syn.get_image_data_by_index(RGB_STREAM, i)[0].to_numpy_array()

        # Rectify
        ego_rect = rectify_frame(ego_arr, ego_calib, dst_calib)
        syn_rect = rectify_frame(syn_arr, syn_calib, dst_calib)

        # Save individual frames
        Image.fromarray(ego_rect).save(os.path.join(args.output_dir, 'ego', f'frame_{i:04d}.png'))
        Image.fromarray(syn_rect).save(os.path.join(args.output_dir, 'synthetic', f'frame_{i:04d}.png'))

        # Save side-by-side comparison
        pad = 4
        combined = np.ones((args.output_size, args.output_size * 2 + pad, 3), dtype=np.uint8) * 40
        combined[:, :args.output_size] = ego_rect
        combined[:, args.output_size + pad:] = syn_rect
        Image.fromarray(combined).save(
            os.path.join(args.output_dir, 'side_by_side', f'frame_{i:04d}.png'))

        if (i + 1) % 10 == 0 or i == n_frames - 1:
            print(f'  [{i+1:3d}/{n_frames}] done')

    print(f'\nAll done! Rectified pairs saved to: {args.output_dir}')
    print(f'  {args.output_dir}/ego/         — rectified real egocentric frames')
    print(f'  {args.output_dir}/synthetic/   — rectified synthetic frames')
    print(f'  {args.output_dir}/side_by_side/ — paired comparisons')


if __name__ == '__main__':
    main()
