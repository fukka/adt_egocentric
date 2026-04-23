"""run_midas_metric3d.py — runs MiDaS DPT-Hybrid and Metric3D-v2-Small, then
produces per-model figures plus a final summary comparison grid."""

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

RGB_PATH   = "/sessions/brave-zen-hamilton/mnt/ADT/maps_opaque_test/rgb/frame_0000.png"
GT_PATH    = "/sessions/brave-zen-hamilton/mnt/ADT/maps_opaque_test/depth_maps/frame_0000.npy"
OUTPUT_DIR = "/sessions/brave-zen-hamilton/mnt/ADT/benchmark_results"
CSV_PATH   = os.path.join(OUTPUT_DIR, "results.csv")
MAX_DEPTH  = 3.0
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_gt():
    d = np.load(GT_PATH).astype(np.float32)
    invalid = ~np.isfinite(d) | (d <= 0.01) | (d >= MAX_DEPTH)
    d[invalid] = np.nan
    return d

def valid_mask(gt): return np.isfinite(gt) & (gt > 0)

def align_scale_shift(pred, gt, mask):
    p, g = pred[mask].astype(np.float64), gt[mask].astype(np.float64)
    A = np.stack([p, np.ones_like(p)], axis=1)
    x, _, _, _ = np.linalg.lstsq(A, g, rcond=None)
    return (x[0]*pred + x[1]).astype(np.float32), x[0], x[1]

def align_scale_only(pred, gt, mask):
    ratio = np.median(gt[mask] / (pred[mask]+1e-8))
    return (ratio*pred).astype(np.float32)

def compute_metrics(pred, gt, mask):
    p = np.clip(pred[mask].astype(np.float64), 1e-6, None)
    g = np.clip(gt[mask].astype(np.float64), 1e-6, None)
    ratio = np.maximum(p/g, g/p)
    return {
        "AbsRel":  float(np.mean(np.abs(p-g)/g)),
        "SqRel":   float(np.mean((p-g)**2/g)),
        "RMSE":    float(np.sqrt(np.mean((p-g)**2))),
        "RMSElog": float(np.sqrt(np.mean((np.log(p)-np.log(g))**2))),
        "delta1":  float(np.mean(ratio<1.25)),
        "delta2":  float(np.mean(ratio<1.25**2)),
        "delta3":  float(np.mean(ratio<1.25**3)),
        "pearson_r": float(np.corrcoef(p, g)[0,1]),
    }

def resize_pred(pred, hw):
    t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
    return F.interpolate(t, size=hw, mode="bilinear", align_corners=False).squeeze().numpy()

def save_figure(rgb, gt, pred_aligned, metrics, model, variant, alignment, stem):
    mask = valid_mask(gt)
    vmin = float(np.nanpercentile(gt, 2))
    vmax = float(np.nanpercentile(gt, 98))
    def norm(d):
        d = np.clip(d.copy(), vmin, vmax)
        d = (d-vmin)/(vmax-vmin+1e-8)
        return d
    gt_n = norm(gt);   gt_n[~np.isfinite(gt)] = 0
    pr_n = norm(pred_aligned); pr_n[~np.isfinite(pred_aligned)] = 0
    err = np.abs(pred_aligned-gt); err[~mask] = np.nan
    emax = float(np.nanpercentile(err, 95))
    err_n = np.clip(err, 0, emax)/(emax+1e-8); err_n[~np.isfinite(err)] = 0

    cm_d = plt.get_cmap("magma_r")
    cm_e = plt.get_cmap("hot")
    fig = plt.figure(figsize=(20,5)); fig.patch.set_facecolor("#1a1a2e")
    gs = gridspec.GridSpec(1,4,figure=fig,wspace=0.04)
    panels = [
        (rgb,                    "Input RGB",          None),
        (cm_d(gt_n)[...,:3],    "GT Depth",           f"[{vmin:.2f}–{vmax:.2f} m]"),
        (cm_d(pr_n)[...,:3],    f"Predicted ({model})", f"align: {alignment}"),
        (cm_e(err_n)[...,:3],   "Abs Error",          f"0–{emax:.2f} m"),
    ]
    for i,(img_d,title,sub) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img_d); ax.set_title(title,color="white",fontsize=11,pad=4)
        if sub: ax.set_xlabel(sub,color="#aaa",fontsize=8)
        ax.axis("off")
    m = metrics
    txt = (f"{model} [{variant}]  |  align: {alignment}\n"
           f"AbsRel={m['AbsRel']:.4f}  RMSE={m['RMSE']:.4f}  "
           f"RMSElog={m['RMSElog']:.4f}  r={m.get('pearson_r',0):.3f}\n"
           f"δ₁={m['delta1']*100:.1f}%  δ₂={m['delta2']*100:.1f}%  δ₃={m['delta3']*100:.1f}%")
    fig.text(0.5,0.01,txt,ha="center",va="bottom",color="white",fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4",facecolor="#2d2d44",alpha=0.85))
    path = os.path.join(OUTPUT_DIR, f"{stem}_comparison.png")
    plt.savefig(path,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig)
    log(f"  Saved → {path}")
    return path

def append_csv(model, variant, alignment, metrics):
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH,"a",newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["model","variant","alignment","AbsRel","SqRel","RMSE","RMSElog",
                        "delta1","delta2","delta3","pearson_r"])
        w.writerow([model, variant, alignment,
                    f"{metrics['AbsRel']:.6f}", f"{metrics['SqRel']:.6f}",
                    f"{metrics['RMSE']:.6f}",   f"{metrics['RMSElog']:.6f}",
                    f"{metrics['delta1']:.6f}", f"{metrics['delta2']:.6f}",
                    f"{metrics['delta3']:.6f}", f"{metrics.get('pearson_r',0):.6f}"])

# ── Load shared data ──────────────────────────────────────────────────────────
rgb  = np.array(Image.open(RGB_PATH).convert("RGB"))
gt   = load_gt()
mask = valid_mask(gt)
H, W = gt.shape
log(f"RGB={rgb.shape}  GT={gt.shape}  valid={mask.sum()}  "
    f"range=[{np.nanmin(gt):.3f},{np.nanmax(gt):.3f}]m")

results = {}

# ══════════════════════════════════════════════════════════════════════════════
# 1. MiDaS DPT-Hybrid
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== [1/2] MiDaS DPT-Hybrid ===")
t0 = time.time()
try:
    log("  Loading MiDaS via torch.hub …")
    midas = torch.hub.load("intel-isl/MiDaS","DPT_Hybrid",trust_repo=True)
    tforms = torch.hub.load("intel-isl/MiDaS","transforms",trust_repo=True)
    transform = tforms.dpt_transform
    midas.eval()
    log("  Running inference …")
    inp = transform(rgb).unsqueeze(0)
    with torch.no_grad():
        raw = midas(inp).squeeze().float().numpy()
    raw = resize_pred(raw,(H,W))

    # MiDaS outputs inverse-depth (disparity): larger = closer
    # Convert to depth: depth = 1/disp (then affine-align)
    depth_from_disp = 1.0 / (raw + 1e-6)
    r_direct = np.corrcoef(raw[mask], gt[mask])[0,1]
    r_inv    = np.corrcoef(depth_from_disp[mask], gt[mask])[0,1]
    log(f"  Pearson r (raw):    {r_direct:.4f}")
    log(f"  Pearson r (1/raw):  {r_inv:.4f}")

    # Use whichever has better positive correlation
    use_pred = depth_from_disp if r_inv > r_direct else raw
    pred_al, sc, sh = align_scale_shift(use_pred, gt, mask)
    log(f"  LS scale={sc:.4f}  shift={sh:.4f}")
    pred_al = np.clip(pred_al, 1e-3, None)
    alignment = "scale+shift (LS, inv-depth)"
    m = compute_metrics(pred_al, gt, mask)
    log(f"  AbsRel={m['AbsRel']:.4f}  RMSE={m['RMSE']:.4f}  "
        f"δ₁={m['delta1']*100:.1f}%  r={m['pearson_r']:.3f}  [{time.time()-t0:.1f}s]")
    save_figure(rgb, gt, pred_al, m, "MiDaS-DPT", "dpt_hybrid", alignment, "midas_dpt")
    append_csv("MiDaS", "dpt_hybrid", alignment, m)
    np.save(os.path.join(OUTPUT_DIR,"midas_dpt_aligned.npy"), pred_al)
    results["MiDaS-DPT"] = m
    del midas
except Exception:
    log(f"  FAILED:\n{traceback.format_exc()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Metric3D v2 Small
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== [2/2] Metric3D v2 (vit_small) ===")
t0 = time.time()
try:
    log("  Loading Metric3D v2 via torch.hub …")
    m3d = torch.hub.load("YvanYin/Metric3D","metric3d_vit_small",
                          pretrain=True,trust_repo=True)
    m3d.eval()
    log("  Running inference …")

    # Intrinsics estimate (55° diag FoV heuristic for Aria)
    diag = np.sqrt(H**2+W**2)
    f    = (diag/2.0)/np.tan(np.radians(55.0)/2.0)
    cx, cy = W/2.0, H/2.0

    # Resize to canonical scale
    scale = 616.0/min(H,W)
    nH, nW = int(round(H*scale)), int(round(W*scale))
    pH = (14-nH%14)%14; pW = (14-nW%14)%14

    img_r = np.array(Image.fromarray(rgb).resize((nW,nH),Image.BILINEAR))
    img_p = np.pad(img_r,((0,pH),(0,pW),(0,0)),mode="reflect")
    mean_ = np.array([0.485,0.456,0.406],dtype=np.float32)
    std_  = np.array([0.229,0.224,0.225],dtype=np.float32)
    img_n = (img_p.astype(np.float32)/255.0 - mean_)/std_
    tensor = torch.from_numpy(img_n).permute(2,0,1).unsqueeze(0).float()

    fx_s,fy_s,cx_s,cy_s = f*scale, f*scale, cx*scale, cy*scale
    K = torch.tensor([[fx_s,0,cx_s],[0,fy_s,cy_s],[0,0,1]],dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_d, conf, _ = m3d.inference({"input":tensor,"intrinsic":K})

    pred_np = pred_d.squeeze().float().numpy()[:nH,:nW]
    pred_np = resize_pred(pred_np,(H,W))

    r_metric = np.corrcoef(pred_np[mask], gt[mask])[0,1]
    log(f"  Pearson r (metric):  {r_metric:.4f}")

    # Direct metric eval
    m_direct = compute_metrics(pred_np, gt, mask)
    log(f"  [metric]  AbsRel={m_direct['AbsRel']:.4f}  RMSE={m_direct['RMSE']:.4f}  "
        f"δ₁={m_direct['delta1']*100:.1f}%  r={m_direct['pearson_r']:.3f}")

    # Affine-aligned eval
    pred_al, sc, sh = align_scale_shift(pred_np, gt, mask)
    pred_al = np.clip(pred_al, 1e-3, None)
    log(f"  LS scale={sc:.4f}  shift={sh:.4f}")
    m_aff = compute_metrics(pred_al, gt, mask)
    log(f"  [aligned] AbsRel={m_aff['AbsRel']:.4f}  RMSE={m_aff['RMSE']:.4f}  "
        f"δ₁={m_aff['delta1']*100:.1f}%  r={m_aff['pearson_r']:.3f}  [{time.time()-t0:.1f}s]")

    best_m   = m_aff if m_aff['AbsRel'] < m_direct['AbsRel'] else m_direct
    best_al  = pred_al if m_aff['AbsRel'] < m_direct['AbsRel'] else pred_np
    best_aln = "scale+shift (LS)" if m_aff['AbsRel'] < m_direct['AbsRel'] else "none (metric)"

    save_figure(rgb, gt, best_al, best_m, "Metric3D-v2", "vit_small", best_aln, "metric3dv2_small")
    append_csv("Metric3D_v2","vit_small (metric)", "none (metric)", m_direct)
    append_csv("Metric3D_v2","vit_small (aligned)","scale+shift (LS)", m_aff)
    np.save(os.path.join(OUTPUT_DIR,"metric3dv2_metric.npy"), pred_np)
    np.save(os.path.join(OUTPUT_DIR,"metric3dv2_aligned.npy"), pred_al)
    results["Metric3D-v2"] = m_direct
    del m3d
except Exception:
    log(f"  FAILED:\n{traceback.format_exc()}")

# ══════════════════════════════════════════════════════════════════════════════
# Print summary table
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== Summary (all runs) ===")
if os.path.exists(CSV_PATH):
    with open(CSV_PATH) as f:
        all_rows = list(csv.DictReader(f))
    primary = {"scale+shift (LS)","scale+shift (LS, inv-depth)","none (metric)"}
    print(f"\n{'MODEL':<22} {'VARIANT':<24} {'AbsRel':>8} {'RMSE':>8} "
          f"{'RMSElog':>8} {'δ₁%':>7} {'δ₂%':>7} {'r':>7}")
    print("─"*100)
    for r in all_rows:
        if r.get('alignment','') not in primary: continue
        print(f"  {r['model']:<20} {r['variant']:<24} "
              f"{float(r['AbsRel']):>8.4f} {float(r['RMSE']):>8.4f} "
              f"{float(r['RMSElog']):>8.4f} {float(r['delta1'])*100:>6.1f}% "
              f"{float(r['delta2'])*100:>6.1f}% "
              f"{float(r.get('pearson_r',0)):>7.3f}")
    print("─"*100)

log("DONE → " + OUTPUT_DIR)
