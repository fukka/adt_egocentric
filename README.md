# adt_egocentric

Render photorealistic egocentric fisheye images from [Aria Digital Twin (ADT)](https://www.projectaria.com/datasets/adt/) ground-truth poses using Blender 3.6 and the Aria FISHEYE624 camera model. Also includes a SAM 2.1 Large segmentation evaluation pipeline against ADT ground-truth instance segmentation.

```
ADT ground-truth poses → Blender (Cycles) → equirectangular → FISHEYE624 remap → fisheye PNG
ADT real RGB frame     → SAM 2.1 Large    → instance masks  → IoU vs GT segmentation
```

---

## Features

- Reads **6-DoF object poses** (`scene_objects.csv`) and **camera trajectory** (`aria_trajectory.csv`) directly from ADT ground truth
- Supports both **static** (`motion_type: static`) and **dynamic** (`motion_type: dynamic`) objects — dynamic objects use timestamp-resolved poses from per-object dynamic CSVs
- Imports **400+ DTC GLB object models** and places them at their correct world poses
- Renders via **Blender 3.6 Cycles** (CPU or GPU) in equirectangular panoramic mode
- Remaps the panorama to the exact **Aria FISHEYE624** projection using Newton-Raphson inversion of the Kannala-Brandt polynomial
- Applies the correct **GLB → ADT rotation fix** (`R_correct = T_WO @ R_baked @ R_x(−90°)`) so all objects render upright and correctly oriented
- Simulates **scene lighting** from ADT prop positions (19 point lights) plus 6 ceiling area lights
- Applies the **Aria Gen1 ISP colour pipeline** (sRGB → linear → camera linear via inverse CCM → camera pixel space via forward CRF + per-channel balance)
- Outputs **per-frame instance segmentation masks** (`.npy` int64 UIDs + colourised PNG)
- Distance-culls to the nearest 80 objects to stay within memory limits
- Caches the fisheye remap LUT (`.npz`) — computed once per output resolution, reused on subsequent runs
- Evaluates **SAM 2.1 Large** zero-shot instance segmentation on real egocentric frames against ADT GT, reporting per-instance IoU

---

## Requirements

| Dependency | Version | Used for |
|---|---|---|
| Python | 3.10+ | all scripts |
| Blender | 3.6.x | rendering |
| projectaria-tools | 2.1.1 | ADT data loading |
| numpy | any recent | all scripts |
| scipy | any recent | fisheye remap |
| Pillow | any recent | image I/O |
| opencv-python | any recent | image processing |
| sam2 | 2.1 (SAM 2.1) | segmentation |
| torch | 2.x | SAM 2 backend |

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

### 3. SAM 2.1

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e . --break-system-packages

# Download SAM 2.1 Large weights
mkdir -p sam2_weights
wget -O sam2_weights/sam2.1_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### 4. ADT dataset & DTC object models

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

# Segmentation ground truth (stream 400-1, 1408×1408 uint64 instance IDs)
~/.local/bin/aria_dataset_downloader \
    -c ADT_download_urls.json \
    -o /path/to/ADT \
    -l Apartment_release_golden_skeleton_seq100_10s_sample_M1292 \
    -d 1

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

For `run_sam2.py`, update:

```python
ADT   = '/path/to/ADT/Apartment_release_golden_skeleton_seq100_10s_sample_M1292'
SAM2  = '/path/to/sam2'          # repo root (for config resolution)
CKPT  = '/path/to/sam2_weights/sam2.1_hiera_large.pt'
```

---

## Usage

### Blender render pipeline

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

#### Arguments

| Argument | Default | Description |
|---|---|---|
| `--num_frames N` | all | Render only the first N selected frames |
| `--frame_step K` | 30 | Render every Kth frame (~1 fps at K=30) |
| `--output_size S` | 512 | Output image size in pixels (square) |
| `--focal F` | auto | Override focal length (px at output_size). Default: scaled from ADT calibration |
| `--output_dir DIR` | `{BASE}/blender_rendered` | Output directory |
| `--fisheye` | off | Remap equirectangular → Aria FISHEYE624 projection |

#### Output layout

```
{output_dir}/
    rgb/              ← final renders (fisheye if --fisheye, else equirectangular)
    fisheye/          ← (--fisheye only) fisheye-remapped frames
    comparison/       ← side-by-side: real ego-RGB | Blender render
    seg_masks/        ← per-frame instance segmentation (.npy int64 UIDs + colourised PNG)
    _remap_512.npz    ← cached fisheye LUT (reused across runs)
```

### SAM 2 segmentation evaluation

```bash
cd /path/to/sam2
python /path/to/repo/run_sam2.py
```

Outputs saved to `{ADT}/`:

| File | Description |
|---|---|
| `real_rot90_f0.png` | Real RGB frame rotated 90° CW to upright orientation |
| `gt_seg_rot_f0.npy` | GT instance segmentation (1408×1408 uint64), same rotation |
| `sam2_masks_f0.npy` | SAM 2 predicted masks rescaled to 1408×1408 (shape: N×H×W bool) |
| `sam2_iou_results.json` | Per-instance best-match IoU and summary metrics |
| `sam2_report_f0.png` | Visual report: real frame / GT / SAM 2 overlay + IoU table |

#### Results — Frame 0 (SAM 2.1 Large, zero-shot)

| Metric | Value |
|---|---|
| GT instances evaluated (≥500 px) | 68 |
| SAM 2 masks generated | 112 |
| Mean IoU | **0.501** |
| Median IoU | **0.555** |
| IoU ≥ 0.75 (Good) | 20 / 68 (29%) |
| IoU ≥ 0.50 (Partial) | 38 / 68 (56%) |
| IoU ≥ 0.25 (Weak) | 47 / 68 (69%) |

Strong matches: cabinet doors (0.92–0.95), WoodenBowl (0.905), ChoppingBoard (0.927).  
Failures: large background instances (ApartmentEnv 0.228, KitchIsland 0.094) where SAM splits the region, and small objects below ~700 px.

> **Note on image orientation:** The Aria RGB camera stream is rotated 90° CCW from natural upright. All scripts apply `np.rot90(arr, k=-1)` (90° CW) before processing or display.

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

GLB models require a two-part correction when placed at ADT world poses:

**Background:** Blender's glTF importer applies a gltf→Blender-native coordinate conversion (`R_x(+90°)`) by **baking it directly into vertex positions**. It also stores the root node's authored→canonical rotation (`R_baked`, read from `nodes[0].rotation` in the GLB JSON chunk) as the object's `matrix_local`. When our pipeline overrides `matrix_world` directly, `matrix_local` is silently discarded, so `R_baked` must be re-applied manually.

**Fix — post-multiply by `R_baked @ R_x(−90°)` for every object:**

```python
R_x_neg90 = np.array([[1, 0,  0],
                       [0, 0,  1],
                       [0,-1,  0]])

def _read_glb_baked_rotation(glb_path):
    """Read root node baked rotation quaternion from GLB JSON chunk."""
    import struct, json
    with open(glb_path, 'rb') as f:
        f.read(12)
        chunk_len = struct.unpack('<I', f.read(4))[0]
        f.read(4)
        gltf = json.loads(f.read(chunk_len))
    nodes = gltf.get('nodes', [])
    if nodes and 'rotation' in nodes[0]:
        q = nodes[0]['rotation']   # [x, y, z, w] glTF convention
        return Rotation.from_quat(q).as_matrix()
    return np.eye(3)

def correct_object_rotation(T_WO, glb_path=None):
    R_baked = _read_glb_baked_rotation(glb_path) if glb_path else np.eye(3)
    T_c = T_WO.copy()
    T_c[:3, :3] = T_WO[:3, :3] @ R_baked @ R_x_neg90
    return T_c
```

The multiplication order matters:
- `R_x_neg90` undoes the vertex-level gltf coord conversion baked by Blender's importer.
- `R_baked` (applied first, left-to-right in local frame) restores the authored→canonical mapping that was stored in `matrix_local` and lost when we override `matrix_world`.
- For objects with no root rotation (`R_baked = I`), this reduces to `T_WO @ R_x_neg90`.

### Dynamic object support

ADT objects are classified as `motion_type: static` or `motion_type: dynamic` in `instances.json`. Static objects have a single pose in `scene_objects.csv`. Dynamic objects (e.g. objects handled by the subject) have per-frame poses in `instances/<uid>/<uid>_<sequence_name>.csv`, resolved by nearest-timestamp lookup.

```python
# Pose resolution for dynamic objects
ts_col = dynamic_df['timestamp[ns]'].values.astype(np.int64)
idx = np.argmin(np.abs(ts_col - frame_timestamp_ns))
T_WO = dynamic_df.iloc[idx][['tx_world_object', ...]]
```

### Aria Gen1 ISP colour pipeline

To match the appearance of real Aria RGB frames, Blender renders (sRGB output) are converted through a forward approximation of the Aria Gen1 ISP:

```
sRGB → linear (gamma 2.2) → camera linear (× inv_CCM) → camera pixel (forward CRF) → channel balance
```

Key constants for the ADT Apartment sequence:

```python
CHANNEL_BALANCE_ADT_APARTMENT = np.array([0.925, 1.085, 0.883])  # R, G, B

ARIA_CCM = np.array([[ 1.7042, -0.4964, -0.2078],
                     [-0.3498,  1.5008, -0.1510],
                     [-0.0569, -0.5651,  1.6220]])
```

### Scene lighting

Lights are placed in ADT world space based on prop positions from `instances.json`:

- **Point lights**: 19 props tagged as light sources, placed at their ADT world positions (power ≈ 100 W each).
- **Ceiling area lights**: 6 large area lights (3 m × 3 m, 200 W each) distributed across the ceiling plane to simulate ambient fill.

### Blender colour management

```python
scene.view_settings.view_transform = 'Standard'   # not Filmic — avoids tone-mapping
scene.view_settings.exposure = -0.8               # compensate for Cycles HDR headroom
scene.view_settings.look = 'None'
scene.view_settings.gamma = 1.0
```

`Standard` mode is used (not Filmic) so that the rendered sRGB output can be passed through the Aria ISP pipeline without double tone-mapping.

### FISHEYE624 remap

Blender cannot render the Kannala-Brandt fisheye model natively. The pipeline uses a two-step approach:

1. Blender renders a full 360×180° equirectangular panorama.
2. A precomputed LUT maps each output fisheye pixel → equirectangular coordinate using **Newton-Raphson inversion** of the FISHEYE624 polynomial (25 iterations).

The LUT is cached as `_remap_{output_size}.npz` and reused on subsequent runs.

### SAM 2 memory considerations

SAM 2.1 Large requires significant RAM. On machines with limited memory (~4 GB):

- Resize input to 1024×1024 (SAM 2's native resolution) before inference
- Set `crop_n_layers=0` and `points_per_batch=32` to reduce peak memory
- Rescale predicted masks back to original resolution (1408×1408) using nearest-neighbour after inference

---

## File Reference

| File | Description |
|---|---|
| `render_from_poses_blender.py` | Driver script — reads ADT CSVs, exports per-frame JSON, calls Blender, applies fisheye remap + Aria ISP |
| `blender_render_scene.py` | Blender headless script — imports GLB objects, sets camera pose, renders Cycles, outputs segmentation EXR |
| `render_from_poses.py` | Lightweight CPU pose renderer (no Blender dependency) |
| `rectify_pipeline.py` | Equirectangular → FISHEYE624 remap via Newton-Raphson LUT |
| `run_sam2.py` | SAM 2.1 Large automatic mask generation + IoU evaluation against ADT GT |

---

## Dataset

[Aria Digital Twin](https://www.projectaria.com/datasets/adt/) — © Meta Platforms, Inc.  
Object models from the [Digital Twin Catalog (DTC)](https://www.projectaria.com/datasets/dtc/).
