# apt_egocentric

Render photorealistic egocentric fisheye images from [Aria Digital Twin (ADT)](https://www.projectaria.com/datasets/adt/) ground-truth poses using Blender 3.6 and the Aria FISHEYE624 camera model.

```
ADT ground-truth poses → Blender (Cycles) → equirectangular → FISHEYE624 remap → fisheye PNG
```

---

## Features

- Reads **6-DoF object poses** (`scene_objects.csv`) and **camera trajectory** (`aria_trajectory.csv`) directly from ADT ground truth
- Imports **400+ DTC GLB object models** and places them at their correct world poses
- Renders via **Blender 3.6 Cycles** (CPU or GPU) in equirectangular panoramic mode
- Remaps the panorama to the exact **Aria FISHEYE624** projection using Newton-Raphson inversion of the Kannala-Brandt polynomial
- Applies the **GLTF Y-up → ADT Z-up rotation fix** (`R_correct = R_adt @ R_x(−90°)`) so all objects render upright
- Distance-culls to the nearest 80 objects to stay within memory limits
- Caches the fisheye remap LUT (`.npz`) — computed once per output resolution, reused on subsequent runs

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.10+ |
| Blender | 3.6.x |
| projectaria-tools | 2.1.1 |
| numpy | any recent |
| scipy | any recent |
| Pillow | any recent |
| opencv-python | any recent |

---

## Installation

### 1. Python dependencies

```bash
pip install projectaria-tools==2.1.1 numpy scipy Pillow opencv-python --break-system-packages
```

### 2. Blender 3.6

```bash
wget https://download.blender.org/release/Blender3.6/blender-3.6.9-linux-x64.tar.xz
tar -xf blender-3.6.9-linux-x64.tar.xz -C ~
mv ~/blender-3.6.9-linux-x64 ~/blender
```

### 3. ADT dataset & DTC object models

Follow the [Project Aria download instructions](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/aria_digital_twin/aria_digital_twin) to obtain:

- `ADT_download_urls.json` — CDN file from the Aria website
- `DTC_objects_ADT_download_urls.json` — CDN file for the DTC object models

Then run:

```bash
# Main sequence (recording + ground truth + synthetic reference)
~/.local/bin/aria_dataset_downloader \
    -c ADT_download_urls.json \
    -o /path/to/ADT \
    -l Apartment_release_golden_skeleton_seq100_10s_sample_M1292 \
    -d main_recording groundtruth synthetic

# 3D object models (~400 GLB files)
~/.local/bin/dtc_object_downloader \
    -c DTC_objects_ADT_download_urls.json \
    -o /path/to/ADT/Apartment_release_golden_skeleton_seq100_10s_sample_M1292/object_models \
    -k 3d-asset_glb
```

---

## Configuration

Open `render_from_poses_blender.py` and update the path constants at the top:

```python
BASE        = '/path/to/ADT/Apartment_release_golden_skeleton_seq100_10s_sample_M1292'
BLENDER_BIN = '/path/to/blender/blender'
BLEND_SCRIPT= '/path/to/repo/blender_render_scene.py'
```

---

## Usage

```bash
cd /path/to/ADT

# Render frame 0 only (quick test)
python render_from_poses_blender.py \
    --num_frames 1 --frame_step 1 \
    --output_size 512 --fisheye

# Render at ~1 fps across the full 10-second clip
python render_from_poses_blender.py \
    --frame_step 30 --output_size 512 --fisheye

# High-res render of first 5 frames
python render_from_poses_blender.py \
    --num_frames 5 --frame_step 1 \
    --output_size 1408 --fisheye
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--num_frames N` | all | Render only the first N selected frames |
| `--frame_step K` | 30 | Render every Kth frame (~1 fps at K=30) |
| `--output_size S` | 512 | Output image size in pixels (square) |
| `--focal F` | auto | Override focal length (px at output_size). Default: scaled from ADT calibration |
| `--output_dir DIR` | `{BASE}/blender_rendered` | Output directory |
| `--fisheye` | off | Remap equirectangular → Aria FISHEYE624 projection |

### Output layout

```
{output_dir}/
    rgb/           ← final renders (fisheye if --fisheye, else perspective)
    fisheye/       ← (--fisheye only) fisheye-remapped frames
    comparison/    ← side-by-side: real ego-RGB | Blender render
    _remap_512.npz ← cached fisheye LUT (reused across runs)
```

---

## Key Technical Notes

### Coordinate systems

ADT uses a **Y-up right-handed** world frame (gravity = `[0, −9.81, 0]`). GLB models use the glTF 2.0 Y-up convention. The pipeline keeps world geometry in ADT/Y-up space — the Blender GLTF importer leaves `matrix_world = identity`, so no Y→Z world conversion is applied.

The camera pose is converted from ADT to Blender convention:

```python
FLIP_YZ = diag([1, -1, -1, 1])          # flip Y and Z of local camera axes
T_WC_blender = T_WC_adt @ FLIP_YZ
```

### Object rotation fix

ADT `T_WO` quaternions assume the canonical standing axis is local **+Z**. GLB models (glTF 2.0) have their standing axis as local **+Y**. Without correction every object renders tilted 90° on its side.

Fix — post-multiply by `R_x(−90°)` for every object:

```python
R_x_neg90 = np.array([[1, 0,  0],
                       [0, 0,  1],
                       [0,-1,  0]])

def correct_object_rotation(T_WO):
    T_c = T_WO.copy()
    T_c[:3, :3] = T_WO[:3, :3] @ R_x_neg90
    return T_c
```

Result: `R_correct[:,2] = R_adt[:,1]` — for an upright bowl, `R_adt[:,1] = [0,1,0]`, so the bowl opening faces world `+Y` ✓

### FISHEYE624 remap

Blender cannot render the Kannala-Brandt fisheye model natively. The pipeline uses a two-step approach:

1. Blender renders a full 360×180° equirectangular panorama.
2. A precomputed LUT maps each output fisheye pixel → equirectangular coordinate using **Newton-Raphson inversion** of the FISHEYE624 polynomial (25 iterations).

The LUT is cached as `_remap_{output_size}.npz` and reused on subsequent runs.

---

## File Reference

| File | Description |
|---|---|
| `render_from_poses_blender.py` | Driver script — reads ADT CSVs, exports per-frame JSON, calls Blender, applies fisheye remap |
| `blender_render_scene.py` | Blender headless script — imports GLB objects, sets camera pose, renders Cycles |

---

## Dataset

[Aria Digital Twin](https://www.projectaria.com/datasets/adt/) — © Meta Platforms, Inc.  
Object models from the [Digital Twin Catalog (DTC)](https://www.projectaria.com/datasets/dtc/).
