"""
ADTDepthDataset
===============
PyTorch Dataset for Aria Digital Twin (ADT) depth data.
Mirrors the frame-collection logic of eval_depth_anything_v2.py so that
training and evaluation operate on identical data splits.

Three splits are supported so that ego, exo, and real frames can be trained
on independently:

  'ego'  — egocentric Blender renders (fisheye + pinhole by default)
             {seq_dir}/blender_rendered_maps/{fisheye|pinhole}/videos_rgb/*.png
             {seq_dir}/blender_rendered_maps/{fisheye|pinhole}/depth_maps/*.npy

  'exo'  — exocentric Blender renders (all cameras found on disk)
             {seq_dir}/exocentric_rendered/{cam}/videos_rgb/*.png
             {seq_dir}/exocentric_rendered/{cam}/depth_maps/*.npy

  'real' — real Aria sensor frames from the standard ADT sequence layout:
             {seq_dir}/videos_rgb/*.jpg        (Aria RGB, .jpg or .png)
             {seq_dir}/depth_npy/*.npy         (paired depth maps)
           Rotation: 270° CCW is applied by default (corrects Aria sensor
           orientation); override with the ``rotation`` constructor argument.
           Note: depth_npy values are typically in millimetres (uint16).
           Pass depth_scale=0.001 to convert to metres.

Each __getitem__ returns:
  'image'      (3, H, W) float32 tensor — ImageNet-normalised
  'depth'      (H, W)    float32 tensor — metric depth in metres
  'valid_mask' (H, W)    float32 tensor — 1 where 0 < depth <= max_depth
"""

import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _collect_frames(rgb_dir: str, depth_dir: str) -> list:
    """
    Scan rgb_dir for image files (.png, .jpg, .jpeg); match each with a depth
    .npy of the same stem.  Only frames with a matching depth file are returned.
    Stems are deduplicated so that a frame present in both .png and .jpg form
    is only included once (the first alphabetically).
    """
    seen_stems = set()
    frames = []
    all_rgb = sorted(
        glob.glob(os.path.join(rgb_dir, '*.png'))
        + glob.glob(os.path.join(rgb_dir, '*.jpg'))
        + glob.glob(os.path.join(rgb_dir, '*.jpeg'))
    )
    for rgb_path in all_rgb:
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        if stem in seen_stems:
            continue
        depth_path = os.path.join(depth_dir, f'{stem}.npy')
        if os.path.exists(depth_path):
            seen_stems.add(stem)
            frames.append({'rgb': rgb_path, 'depth': depth_path})
    return frames


def _gather_ego(seq_dir: str, ego_subdirs: tuple) -> list:
    render_root = os.path.join(seq_dir, 'blender_rendered_maps')
    frames = []
    for sub in ego_subdirs:
        rgb_dir   = os.path.join(render_root, sub, 'videos_rgb')
        depth_dir = os.path.join(render_root, sub, 'depth_maps')
        if os.path.isdir(rgb_dir):
            found = _collect_frames(rgb_dir, depth_dir)
            frames.extend(found)
            print(f'  [ADT] ego/{sub}: {len(found)} frames — {seq_dir}')
        else:
            print(f'  [ADT] WARN ego/{sub} not found: {rgb_dir}')
    return frames


def _gather_exo(seq_dir: str, exo_cameras) -> list:
    render_root = os.path.join(seq_dir, 'exocentric_rendered')
    if not os.path.isdir(render_root):
        print(f'  [ADT] WARN exo root not found: {render_root}')
        return []
    cam_dirs = sorted([
        d for d in os.listdir(render_root)
        if os.path.isdir(os.path.join(render_root, d, 'videos_rgb'))
    ])
    if exo_cameras:
        wanted = set(exo_cameras)
        cam_dirs = [c for c in cam_dirs if c in wanted]
    frames = []
    for cam in cam_dirs:
        rgb_dir   = os.path.join(render_root, cam, 'videos_rgb')
        depth_dir = os.path.join(render_root, cam, 'depth_maps')
        found = _collect_frames(rgb_dir, depth_dir)
        frames.extend(found)
        print(f'  [ADT] exo/{cam}: {len(found)} frames — {seq_dir}')
    return frames


def _gather_real(seq_dir: str) -> list:
    """
    Collect real Aria sensor frames from the standard ADT sequence layout:

      {seq_dir}/videos_rgb/*.jpg   — Aria RGB frames (.jpg, .png accepted)
      {seq_dir}/depth_npy/*.npy    — paired depth maps

    Rotation (270° CCW) is handled by ADTDepthDataset.__getitem__, not here.
    """
    rgb_dir   = os.path.join(seq_dir, 'videos_rgb')
    depth_dir = os.path.join(seq_dir, 'depth_npy')
    if not os.path.isdir(rgb_dir):
        print(f'  [ADT] WARN real videos_rgb not found: {rgb_dir}')
        return []
    if not os.path.isdir(depth_dir):
        print(f'  [ADT] WARN real depth_npy not found: {depth_dir}')
        return []
    found = _collect_frames(rgb_dir, depth_dir)
    print(f'  [ADT] real: {len(found)} frames — {seq_dir}')
    return found


class ADTDepthDataset(Dataset):
    """
    Parameters
    ----------
    seq_dirs    : list of sequence root directories (same list as BATCH_SEQ_DIRS
                  in eval_depth_anything_v2.py)
    split       : 'ego' | 'exo' | 'real'
    size        : (H, W) spatial size to resize inputs (default 518×518)
    depth_scale : multiply raw depth values by this factor.
                  For ego/exo Blender renders: 1.0 (already in metres).
                  For real depth_npy (uint16 millimetres): use 0.001.
    max_depth   : depth ceiling used for the valid mask and loss clamping
                  (default 10.0 m, matches eval script default for ADT indoor)
    phase       : 'train' | 'val'
    ego_subdirs : tuple of render-mode subdirs under blender_rendered_maps/
                  to include for the 'ego' split (default ('fisheye', 'pinhole'))
    exo_cameras : optional list of camera names for the 'exo' split;
                  None = all cameras found on disk
    rotation    : CCW rotation in degrees applied to both RGB and depth before
                  resize.  None (default) = auto: 270 for 'real' (corrects Aria
                  sensor orientation), 0 for 'ego'/'exo'.
                  Explicit values: 0 | 90 | 180 | 270.
    """

    SPLITS = ('ego', 'exo', 'real')

    def __init__(
        self,
        seq_dirs: list,
        split: str,
        size: tuple = (518, 518),
        depth_scale: float = 1.0,
        max_depth: float = 10.0,
        phase: str = 'train',
        ego_subdirs: tuple = ('fisheye', 'pinhole'),
        exo_cameras: list = None,
        rotation: int = None,
    ):
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got '{split}'")

        self.split       = split
        self.size        = size
        self.depth_scale = depth_scale
        self.max_depth   = max_depth
        self.phase       = phase

        # Rotation: None → auto-select based on split
        if rotation is None:
            self.rotation = 270 if split == 'real' else 0
        else:
            if rotation not in (0, 90, 180, 270):
                raise ValueError(f'rotation must be 0/90/180/270, got {rotation}')
            self.rotation = rotation

        print(f'  [ADT] rotation={self.rotation}° CCW '
              f'({"auto" if rotation is None else "user-specified"})')

        self.samples: list = []
        for seq_dir in seq_dirs:
            if split == 'ego':
                self.samples.extend(_gather_ego(seq_dir, tuple(ego_subdirs)))
            elif split == 'exo':
                self.samples.extend(_gather_exo(seq_dir, exo_cameras))
            elif split == 'real':
                self.samples.extend(_gather_real(seq_dir))

        if not self.samples:
            raise RuntimeError(
                f"No samples found for split='{split}' in:\n"
                + '\n'.join(f'  {d}' for d in seq_dirs)
                + '\nCheck that the sequence directories exist and renders have been generated.'
            )

        print(f'  [ADT] Total {phase} samples for split="{split}": {len(self.samples)}')

        self._normalize = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # ── RGB ───────────────────────────────────────────────────────────────
        img = np.array(Image.open(sample['rgb']).convert('RGB'), dtype=np.float32) / 255.0

        # ── Depth ─────────────────────────────────────────────────────────────
        depth = np.load(sample['depth']).astype(np.float32)
        if depth.ndim == 3:
            depth = depth.squeeze(-1)   # (H, W, 1) → (H, W)
        depth = depth * self.depth_scale

        # ── Rotation (applied before resize so spatial dims stay consistent) ──
        # img:   (H, W, 3) numpy   — rot90 on axes 0 and 1
        # depth: (H, W)    numpy   — rot90 on axes 0 and 1
        if self.rotation != 0:
            k     = {90: 1, 180: 2, 270: 3}[self.rotation]
            img   = np.rot90(img,   k=k).copy()
            depth = np.rot90(depth, k=k).copy()

        # ── Convert to tensors ────────────────────────────────────────────────
        img   = torch.from_numpy(img).permute(2, 0, 1)   # (3, H, W)
        depth = torch.from_numpy(depth)                    # (H, W)

        # ── Resize to model input size ─────────────────────────────────────────
        # Image: bilinear (smooth interpolation for RGB)
        # Depth: nearest  (avoid creating spurious depth values at boundaries)
        h, w = self.size
        img = F.interpolate(
            img.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
        ).squeeze(0)
        depth = F.interpolate(
            depth.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest'
        ).squeeze(0).squeeze(0)

        # ── Normalise image ────────────────────────────────────────────────────
        img = self._normalize(img)

        # ── Valid mask: pixels with depth in (0, max_depth] ───────────────────
        valid_mask = ((depth > 0) & (depth <= self.max_depth)).float()

        # Zero invalid pixels so the DataLoader can stack without NaN issues;
        # the training loop re-applies the valid_mask before computing loss.
        depth = depth.clamp(min=0.0)

        return {'image': img, 'depth': depth, 'valid_mask': valid_mask}