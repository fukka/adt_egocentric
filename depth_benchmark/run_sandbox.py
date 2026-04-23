"""
run_sandbox.py — CPU-friendly benchmark runner for the ADT egocentric depth data.

Baselines:
  1. Depth Anything V2 Small  (transformers, CPU ~30s)
  2. Marigold LCM 1-step      (diffusers, CPU ~60s)
  3. Metric3D v2 Small        (torch.hub, CPU ~45s)
  4. MiDaS DPT-Hybrid         (torch.hub, CPU ~20s)  — classic baseline

All outputs saved to OUTPUT_DIR. Progress logged to stdout (redirect to file).
"""

import os, sys, time, traceback
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv

# ── Paths ─────────────────────────────────────────────────────────────────────
RGB_PATH   = "/sessions/brave-zen-hamilton/mnt/ADT/maps_opaque_test/rgb/frame_0000.png"
GT_PATH    = "/sessions/brave-zen-hamilton/mnt/ADT/maps_opaque_test/depth_maps/frame_0000.npy"
OUTPUT_DIR = "/sessions/brave-zen-hamilton/mnt/ADT/benchmark_results"
CSV_PATH   = os.path.join(OUTPUT_DIR, "results.csv")
ROTATION   = 0          # degrees CW; set to 90 if image needs rotating
MAX_DEPTH  = 3.0        # metres — cap GT depth

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_rgb(rotation=0):
    img = Image.open(RGB_PATH).convert("RGB")
    if rotation == 90:  img = img.transpose(Image.ROTATE_90)
    if rotation == 180: img = img.transpose(Image.ROTATE_180)
    if rotation == 270: img = img.transpose(Image.ROTATE_270)
    return np.array(img)

def load_gt():
    d = np.load(GT_PATH).astype(np.float32)
    if d.ndim == 3: d = d.squeeze(-1)
    invalid = ~np.isfinite(d) | (d <= 0.01) | (d >= MAX_DEPTH)
    d[invalid] = np.nan
    return d

def valid_mask(gt): return np.isfinite(gt) & (gt > 0)

def align_scale_shift(pred, gt, mask):
    p, g = pred[mask].astype(np.float64), gt[mask].astype(np.float64)
    A = np.stack([p, np.ones_like(p)], axis=1)
    x, _, _, _ = np.linalg.lstsq(A, g, rcond=None)
    return (x[0]*pred + x[1]).astype(np.float32)

def align_scale_only(pred, gt, mask):
    ratio = np.median(gt[mask] / (pred[mask]+1e-8))
    return (ratio*pred).astype(np.float32)

def compute_metrics(pred, gt, mask):
    p = np.clip(pred[mask].astype(np.float64), 1e-6, None)
    g = np.clip(gt[mask].astype(np.float64), 1e-6, None)
    abs_rel = float(np.mean(np.abs(p-g)/g))
    sq_rel  = float(np.mean((p-g)**2/g))
    rmse    = float(np.sqrt(np.mean((p-g)**2)))
    rmselog = float(np.sqrt(np.mean((np.log(p)-np.log(g))**2)))
    ratio   = np.maximum(p/g, g/p)
    return {
        "AbsRel": abs_rel, "SqRel": sq_rel,
        "RMSE": rmse, "RMSElog": rmselog,
        "delta1": float(np.mean(ratio<1.25)),
        "delta2": float(np.mean(ratio<1.25**2)),
        "delta3": float(np.mean(ratio<1.25**3)),
    }

CMAP_D = "magma_r"
CMAP_E = "hot"

def save_figure(rgb, gt, pred, metrics, model, variant, alignment, stem):
    mask = valid_mask(gt)
    vmin = float(np.nanpercentile(gt, 2))
    vmax = float(np.nanpercentile(gt, 98))
    def norm(d, lo=vmin, hi=vmax):
        d = np.clip(d.copy(), lo, hi)
        d = (d-lo)/(hi-lo+1e-8)
        return d

    gt_n   = norm(gt);   gt_n[~np.isfinite(gt)] = 0
    pr_n   = norm(pred); pr_n[~np.isfinite(pred)] = 0
    err    = np.abs(pred-gt); err[~mask] = np.nan
    emax   = float(np.nanpercentile(err, 95))
    err_n  = np.clip(err, 0, emax)/(emax+1e-8); err_n[~np.isfinite(err)] = 0

    cm_d = plt.get_cmap(CMAP_D)
    cm_e = plt.get_cmap(CMAP_E)

    fig = plt.figure(figsize=(20,5)); fig.patch.set_facecolor("#1a1a2e")
    gs = gridspec.GridSpec(1,4,figure=fig,wspace=0.04)
    panels = [
        (rgb,                      "Input RGB",           None),
        (cm_d(gt_n)[...,:3],       "GT Depth",            f"[{vmin:.2f}–{vmax:.2f} m]"),
        (cm_d(pr_n)[...,:3],       f"Pred ({model})",     f"align: {alignment}"),
        (cm_e(err_n)[...,:3],      "Abs Error",           f"0–{emax:.2f} m"),
    ]
    for i,(img_d,title,sub) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img_d); ax.set_title(title,color="white",fontsize=11,pad=4)
        if sub: ax.set_xlabel(sub,color="#aaa",fontsize=8)
        ax.axis("off")

    m = metrics
    txt = (f"{model} [{variant}]  |  align: {alignment}\n"
           f"AbsRel={m['AbsRel']:.4f}  SqRel={m['SqRel']:.4f}  "
           f"RMSE={m['RMSE']:.4f}  RMSElog={m['RMSElog']:.4f}\n"
           f"δ₁={m['delta1']*100:.1f}%  δ₂={m['delta2']*100:.1f}%  δ₃={m['delta3']*100:.1f}%")
    fig.text(0.5,0.01,txt,ha="center",va="bottom",color="white",fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4",facecolor="#2d2d44",alpha=0.85))

    path = os.path.join(OUTPUT_DIR, f"{stem}_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log(f"  Saved figure → {path}")
    return path

def append_csv(model, variant, alignment, metrics):
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH,"a",newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["model","variant","alignment","AbsRel","SqRel",
                        "RMSE","RMSElog","delta1","delta2","delta3"])
        w.writerow([model, variant, alignment,
                    f"{metrics['AbsRel']:.6f}", f"{metrics['SqRel']:.6f}",
                    f"{metrics['RMSE']:.6f}",   f"{metrics['RMSElog']:.6f}",
                    f"{metrics['delta1']:.6f}", f"{metrics['delta2']:.6f}",
                    f"{metrics['delta3']:.6f}"])

def resize_pred(pred_np, target_hw):
    t = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0).float()
    t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
    return t.squeeze().numpy()

# ── Load shared data ──────────────────────────────────────────────────────────
log("Loading RGB + GT …")
rgb = load_rgb(ROTATION)
gt  = load_gt()
mask = valid_mask(gt)
H, W = gt.shape
log(f"  RGB {rgb.shape}  GT {gt.shape}  valid={mask.sum()}/{mask.size} "
    f"range=[{np.nanmin(gt):.3f}, {np.nanmax(gt):.3f}] m")

results = {}   # model_key -> metrics dict

# ══════════════════════════════════════════════════════════════════════════════
# 1. Depth Anything V2 Small
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== [1/4] Depth Anything V2 Small ===")
t0 = time.time()
try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
    log(f"  Downloading/loading {MODEL_ID} …")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model_dav2 = AutoModelForDepthEstimation.from_pretrained(MODEL_ID,
                                                              torch_dtype=torch.float32)
    model_dav2.eval()
    log("  Model loaded. Running inference …")
    pil_img = Image.fromarray(rgb)
    inputs  = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        out = model_dav2(**inputs)
        pred_raw = F.interpolate(
            out.predicted_depth.unsqueeze(1),
            size=(H, W), mode="bilinear", align_corners=False
        ).squeeze().float().numpy()

    pred_aligned = align_scale_shift(pred_raw, gt, mask)
    alignment = "scale+shift (LS)"
    m = compute_metrics(pred_aligned, gt, mask)
    log(f"  AbsRel={m['AbsRel']:.4f}  RMSE={m['RMSE']:.4f}  δ₁={m['delta1']*100:.1f}%  [{time.time()-t0:.1f}s]")
    save_figure(rgb, gt, pred_aligned, m, "DAv2-Small", "small", alignment, "dav2_small")
    append_csv("Depth_Anything_V2", "small", alignment, m)
    np.save(os.path.join(OUTPUT_DIR,"dav2_small_aligned.npy"), pred_aligned)
    results["DAv2-Small"] = m
    del model_dav2
except Exception:
    log(f"  FAILED: {traceback.format_exc()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Marigold LCM (1 denoising step — fastest)
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== [2/4] Marigold LCM (1 step) ===")
t0 = time.time()
try:
    from diffusers import MarigoldDepthPipeline
    MODEL_ID = "prs-eth/marigold-depth-lcm-v1-0"
    log(f"  Downloading/loading {MODEL_ID} …")
    pipe = MarigoldDepthPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    log("  Model loaded. Running inference …")
    pil_img = Image.fromarray(rgb)
    with torch.no_grad():
        out = pipe(pil_img, denoising_steps=1, ensemble_size=1,
                   processing_res=512, match_input_res=True,
                   color_map=None, show_progress_bar=False)
    pred_disp = np.array(out.depth_np, dtype=np.float32)
    pred_pseudo = 1.0 - pred_disp        # flip disparity → pseudo-depth

    pred_aligned = align_scale_shift(pred_pseudo, gt, mask)
    alignment = "scale+shift (LS)"
    m = compute_metrics(pred_aligned, gt, mask)
    log(f"  AbsRel={m['AbsRel']:.4f}  RMSE={m['RMSE']:.4f}  δ₁={m['delta1']*100:.1f}%  [{time.time()-t0:.1f}s]")
    save_figure(rgb, gt, pred_aligned, m, "Marigold-LCM", "lcm-1step", alignment, "marigold_lcm")
    append_csv("Marigold", "lcm-1step", alignment, m)
    np.save(os.path.join(OUTPUT_DIR,"marigold_lcm_aligned.npy"), pred_aligned)
    results["Marigold-LCM"] = m
    del pipe
except Exception:
    log(f"  FAILED: {traceback.format_exc()}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. MiDaS DPT-Hybrid (torch.hub — classic strong baseline)
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== [3/4] MiDaS DPT-Hybrid ===")
t0 = time.time()
try:
    log("  Downloading MiDaS via torch.hub …")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.dpt_transform
    midas.eval()
    log("  Model loaded. Running inference …")
    pil_img = Image.fromarray(rgb)
    inp = transform(rgb).unsqueeze(0)   # transform expects np array
    with torch.no_grad():
        pred_raw = midas(inp).squeeze().float().numpy()
    pred_raw = resize_pred(pred_raw, (H, W))

    pred_aligned = align_scale_shift(pred_raw, gt, mask)
    alignment = "scale+shift (LS)"
    m = compute_metrics(pred_aligned, gt, mask)
    log(f"  AbsRel={m['AbsRel']:.4f}  RMSE={m['RMSE']:.4f}  δ₁={m['delta1']*100:.1f}%  [{time.time()-t0:.1f}s]")
    save_figure(rgb, gt, pred_aligned, m, "MiDaS-DPT", "dpt_hybrid", alignment, "midas_dpt")
    append_csv("MiDaS", "dpt_hybrid", alignment, m)
    np.save(os.path.join(OUTPUT_DIR,"midas_dpt_aligned.npy"), pred_aligned)
    results["MiDaS-DPT"] = m
    del midas
except Exception:
    log(f"  FAILED: {traceback.format_exc()}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Metric3D v2 Small (torch.hub)
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== [4/4] Metric3D v2 (vit_small) ===")
t0 = time.time()
try:
    log("  Downloading Metric3D v2 via torch.hub …")
    model_m3d = torch.hub.load("YvanYin/Metric3D", "metric3d_vit_small",
                                pretrain=True, trust_repo=True)
    model_m3d.eval()
    log("  Model loaded. Running inference …")

    # Estimate intrinsics (55° diag FoV heuristic)
    diag = np.sqrt(H**2 + W**2)
    f = (diag/2.0) / np.tan(np.radians(55.0)/2.0)
    cx, cy = W/2.0, H/2.0

    # Resize to canonical scale
    short = min(H, W)
    scale = 616.0 / short
    nH, nW = int(round(H*scale)), int(round(W*scale))
    pH = (14 - nH%14) % 14
    pW = (14 - nW%14) % 14

    img_r = np.array(Image.fromarray(rgb).resize((nW, nH), Image.BILINEAR))
    img_p = np.pad(img_r, ((0,pH),(0,pW),(0,0)), mode="reflect")

    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std  = np.array([0.229,0.224,0.225], dtype=np.float32)
    img_n = (img_p.astype(np.float32)/255.0 - mean) / std
    tensor = torch.from_numpy(img_n).permute(2,0,1).unsqueeze(0).float()

    fx_s, fy_s = f*scale, f*scale
    cx_s, cy_s = cx*scale, cy*scale
    K = torch.tensor([[fx_s,0,cx_s],[0,fy_s,cy_s],[0,0,1]],
                      dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_d, conf, _ = model_m3d.inference({"input": tensor, "intrinsic": K})

    pred_np = pred_d.squeeze().float().numpy()
    pred_np = pred_np[:nH, :nW]                    # crop padding
    pred_np = resize_pred(pred_np, (H, W))          # back to original

    # Direct metric comparison
    alignment_direct = "none (metric)"
    m_direct = compute_metrics(pred_np, gt, mask)
    log(f"  [metric]  AbsRel={m_direct['AbsRel']:.4f}  RMSE={m_direct['RMSE']:.4f}  δ₁={m_direct['delta1']*100:.1f}%")

    # Also scale+shift aligned
    pred_aligned = align_scale_shift(pred_np, gt, mask)
    alignment_aff = "scale+shift (LS)"
    m_aff = compute_metrics(pred_aligned, gt, mask)
    log(f"  [aligned] AbsRel={m_aff['AbsRel']:.4f}  RMSE={m_aff['RMSE']:.4f}  δ₁={m_aff['delta1']*100:.1f}%  [{time.time()-t0:.1f}s]")

    save_figure(rgb, gt, pred_aligned, m_aff, "Metric3D-v2", "vit_small", alignment_aff, "metric3dv2_small")
    append_csv("Metric3D_v2", "vit_small (metric)",  alignment_direct, m_direct)
    append_csv("Metric3D_v2", "vit_small (aligned)", alignment_aff,    m_aff)
    np.save(os.path.join(OUTPUT_DIR,"metric3dv2_metric.npy"), pred_np)
    np.save(os.path.join(OUTPUT_DIR,"metric3dv2_aligned.npy"), pred_aligned)
    results["Metric3D-v2"] = m_direct
    del model_m3d
except Exception:
    log(f"  FAILED: {traceback.format_exc()}")

# ══════════════════════════════════════════════════════════════════════════════
# Summary table figure
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== Summary ===")
if results:
    import csv as _csv
    rows = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH) as f:
            rows = list(_csv.DictReader(f))

    # Print table
    print(f"\n{'MODEL':<22} {'VARIANT':<18} {'ALIGN':<20} "
          f"{'AbsRel':>8} {'RMSE':>8} {'RMSElog':>8} {'δ₁%':>7} {'δ₂%':>7}")
    print("─"*100)
    primary = {"scale+shift (LS)", "none (metric)"}
    for r in rows:
        if r['alignment'] not in primary: continue
        print(f"  {r['model']:<20} {r['variant']:<18} {r['alignment']:<20} "
              f"{float(r['AbsRel']):>8.4f} {float(r['RMSE']):>8.4f} "
              f"{float(r['RMSElog']):>8.4f} {float(r['delta1'])*100:>6.1f}% "
              f"{float(r['delta2'])*100:>6.1f}%")
    print("─"*100)

    # Bar chart
    primary_rows = [r for r in rows if r['alignment'] in primary]
    if primary_rows:
        labels  = [f"{r['model']}\n[{r['variant']}]" for r in primary_rows]
        absrel  = [float(r["AbsRel"]) for r in primary_rows]
        delta1  = [float(r["delta1"])*100 for r in primary_rows]
        colors  = ["#2E75B6","#E8763A","#2ECC71","#9B59B6","#F39C12","#1ABC9C"]

        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(max(10,len(labels)*2.5),5))
        fig.patch.set_facecolor("#1a1a2e")
        x = np.arange(len(labels))
        for ax in (ax1,ax2):
            ax.set_facecolor("#2d2d44")
            ax.tick_params(colors="white")
            for sp in ("top","right"): ax.spines[sp].set_visible(False)
            for sp in ("bottom","left"): ax.spines[sp].set_color("#555")

        bars = ax1.bar(x, absrel, color=colors[:len(labels)], edgecolor="#1a1a2e")
        ax1.set_xticks(x); ax1.set_xticklabels(labels,color="white",fontsize=7)
        ax1.set_ylabel("AbsRel ↓", color="white",fontsize=9)
        ax1.set_title("Absolute Relative Error", color="white",fontsize=10)
        for b,v in zip(bars,absrel):
            ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.001,
                     f"{v:.4f}", ha="center",va="bottom",color="white",fontsize=7)

        bars = ax2.bar(x, delta1, color=colors[:len(labels)], edgecolor="#1a1a2e")
        ax2.set_xticks(x); ax2.set_xticklabels(labels,color="white",fontsize=7)
        ax2.set_ylabel("δ₁ (%) ↑", color="white",fontsize=9)
        ax2.set_title("δ₁ Accuracy (thr=1.25)", color="white",fontsize=10)
        ax2.set_ylim(0,105)
        for b,v in zip(bars,delta1):
            ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                     f"{v:.1f}%", ha="center",va="bottom",color="white",fontsize=7)

        fig.suptitle("Egocentric Depth Benchmark — ADT frame_0000", color="white",fontsize=13,y=1.02)
        fig.tight_layout()
        bar_path = os.path.join(OUTPUT_DIR,"summary_bar.png")
        plt.savefig(bar_path,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
        plt.close(fig)
        log(f"  Bar chart → {bar_path}")

log("\nDONE. All outputs in: " + OUTPUT_DIR)
