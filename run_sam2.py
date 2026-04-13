import sys, os, json, gc
sys.path.insert(0, '/sessions/dreamy-modest-brown/.local/lib/python3.10/site-packages')
sys.path.insert(0, '/sessions/dreamy-modest-brown/sam2')
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

import numpy as np, torch
from PIL import Image

ADT = '/sessions/dreamy-modest-brown/mnt/ADT'

# Load and resize to 1024 (SAM2 native resolution — avoids double upsampling OOM)
real_full = np.array(Image.open(f'{ADT}/real_rot90_f0.png'))
real_img  = np.array(Image.fromarray(real_full).resize((1024, 1024), Image.LANCZOS))
print(f"Input (resized): {real_img.shape}")

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

cfg  = 'configs/sam2.1/sam2.1_hiera_l.yaml'
ckpt = '/sessions/dreamy-modest-brown/sam2_weights/sam2.1_hiera_large.pt'

print("Loading SAM 2.1 Large (CPU)...")
with torch.inference_mode():
    sam2_model = build_sam2(cfg, ckpt, device='cpu')

    generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=32,
        points_per_batch=32,       # smaller batches = less peak RAM
        pred_iou_thresh=0.70,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=0,           # no crops — saves significant RAM
        min_mask_region_area=100,
    )

    print("Running automatic mask generation...")
    masks = generator.generate(real_img)

print(f"  -> {len(masks)} masks generated")

# Scale masks back to original 1408x1408 for fair IoU comparison
print("Rescaling masks to 1408x1408...")
H, W = real_full.shape[:2]
masks_full = []
for m in masks:
    seg_small = m['segmentation'].astype(np.uint8)
    seg_full  = np.array(Image.fromarray(seg_small*255).resize((W, H), Image.NEAREST)) > 127
    masks_full.append(seg_full)

masks_arr = np.stack(masks_full, axis=0)  # (N,1408,1408) bool
np.save(f'{ADT}/sam2_masks_f0.npy', masks_arr)

mask_meta = [{
    'id': i,
    'area_1024': int(m['area']),
    'area_1408': int(masks_full[i].sum()),
    'predicted_iou': float(m['predicted_iou']),
    'stability_score': float(m['stability_score']),
    'bbox': [int(x) for x in m['bbox']],
} for i, m in enumerate(masks)]
with open(f'{ADT}/sam2_masks_f0.json', 'w') as f:
    json.dump(mask_meta, f, indent=2)

# Visualisation on 1024 image
np.random.seed(0)
overlay = np.zeros_like(real_img, dtype=float)
for m in masks:
    col = np.random.randint(60, 240, 3).astype(float)
    overlay[m['segmentation']] = col
blended = (real_img.astype(float)*0.45 + overlay*0.55).clip(0,255).astype(np.uint8)
Image.fromarray(blended).save(f'{ADT}/sam2_overlay_f0.png')

print(f"Saved sam2_masks_f0.npy ({masks_arr.shape}), .json, sam2_overlay_f0.png")
