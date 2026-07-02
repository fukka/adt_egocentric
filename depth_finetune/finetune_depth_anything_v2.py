"""
finetune_depth_anything_v2.py
=============================
Fine-tune Depth Anything V2 for metric depth estimation on ADT data.

Directly adapted from the official metric_depth/train.py in the
Depth-Anything-V2 repo (same loss, same optimizer, same LR schedule).

Three mutually-exclusive training splits (train on each independently):
  --split ego   egocentric Blender renders (fisheye + pinhole)
  --split exo   exocentric Blender renders (all cameras)
  --split real  real Aria sensor frames
                  RGB:   {seq_dir}/videos_rgb/*.jpg
                  Depth: {seq_dir}/depth_npy/*.npy  (uint16, millimetres)
                  → pass --depth_scale 0.001 to convert mm → metres
                  Rotation 270° CCW applied automatically (corrects Aria
                  sensor orientation); override with --rotation.

The data loader mirrors eval_depth_anything_v2.py's _collect_frames() so that
the same frames that are evaluated can also be used for training.

Usage — single GPU:
  torchrun --standalone --nproc_per_node=1 finetune_depth_anything_v2.py \\
      --split real \\
      --depth_scale 0.001 \\
      --repo_dir /path/to/Depth-Anything-V2 \\
      --pretrained_from /path/to/depth_anything_v2_vitl.pth \\
      --save_path /path/to/checkpoints/real

Usage — multi-GPU (e.g. 4 GPUs):
  torchrun --standalone --nproc_per_node=4 finetune_depth_anything_v2.py \\
      --split exo \\
      --repo_dir /path/to/Depth-Anything-V2 \\
      --pretrained_from /path/to/depth_anything_v2_vitl.pth \\
      --save_path /path/to/checkpoints/exo

Checkpoints saved to --save_path:
  latest.pth        — full training state (resume-able)
  best_<split>.pth  — model weights only, best δ₁ on validation set
"""

import argparse
import logging
import os
import pprint
import random
import sys
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader


# ── Sequence list — mirrors BATCH_SEQ_DIRS in eval_depth_anything_v2.py ──────
_ADT_ROOT = '/Users/fengjiazhang/Documents/projectaria_tools_adt_data'
if not os.path.exists(_ADT_ROOT):
    _ADT_ROOT = '/user/f.zhang2/Documents/projectaria_tools_adt_data_clean'
    if not os.path.exists(_ADT_ROOT):
        _ADT_ROOT = '/group-volume/Fengjia/data/projectaria_tools_adt_data_clean'
assert os.path.exists(_ADT_ROOT), _ADT_ROOT
BATCH_SEQ_DIRS = [
    f'{_ADT_ROOT}/Apartment_release_clean_seq133_M1292',
    f'{_ADT_ROOT}/Apartment_release_clean_seq134_M1292',
    f'{_ADT_ROOT}/Apartment_release_clean_seq135_M1292',
    f'{_ADT_ROOT}/Apartment_release_clean_seq136_M1292',
    f'{_ADT_ROOT}/Apartment_release_clean_seq137_M1292',
    f'{_ADT_ROOT}/Apartment_release_clean_seq138_M1292',
    f'{_ADT_ROOT}/Apartment_release_clean_seq140_M1292',
    f'{_ADT_ROOT}/Apartment_release_clean_seq141_M1292',
    f'{_ADT_ROOT}/Apartment_release_clean_seq142_M1292',
    f'{_ADT_ROOT}/Apartment_release_clean_seq143_M1292',
    f'{_ADT_ROOT}/Apartment_release_clean_seq144_M1292',
    f'{_ADT_ROOT}/Apartment_release_clean_seq145_M1292',
    f'{_ADT_ROOT}/Apartment_release_clean_seq146_M1292',
    f'{_ADT_ROOT}/Apartment_release_clean_seq147_M1292',
    f'{_ADT_ROOT}/Apartment_release_clean_seq148_M1292',
]
BATCH_SEQ_DIRS_VAL = [
    f'{_ADT_ROOT}/Apartment_release_clean_seq131_M1292',
]

# ── Model configs — mirrors MODEL_CONFIGS in eval_depth_anything_v2.py ───────
MODEL_CONFIGS = {
    'small': {'encoder': 'vits', 'features': 64,  'out_channels': [48,  96,  192,  384]},
    'base':  {'encoder': 'vitb', 'features': 128, 'out_channels': [96,  192, 384,  768]},
    'large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Fine-tune Depth Anything V2 on ADT depth data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── data ──────────────────────────────────────────────────────────────────
    p.add_argument('--split', required=True, choices=['ego', 'exo', 'real'],
                   help='Which ADT data split to train on')
    p.add_argument('--seq_dirs', nargs='+', default=None,
                   help='Training sequence directories. Default: BATCH_SEQ_DIRS in script.')
    p.add_argument('--val_seq_dirs', nargs='+', default=None,
                   help='Validation sequence directories. Default: same as --seq_dirs.')
    p.add_argument('--val_fraction', type=float, default=None,
                   help='Hold-out fraction of training frames for validation '
                        '(e.g. 0.1 = last 10%%). Only used when --val_seq_dirs is not set. '
                        'Default: use all frames for both train and val.')

    # ego-specific
    p.add_argument('--ego_subdirs', nargs='+', default=['fisheye', 'pinhole'],
                   help="Render subdirs under blender_rendered_maps/ to include for 'ego' split.")
    # exo-specific
    p.add_argument('--exo_cameras', nargs='+', default=None,
                   help="Camera names for 'exo' split. Default: all cameras on disk.")
    # rotation
    p.add_argument('--rotation', type=int, default=None, choices=[0, 90, 180, 270],
                   help="CCW rotation applied to RGB and depth before training. "
                        "Default: auto (270 for 'real' to correct Aria sensor orientation, "
                        "0 for 'ego'/'exo').")

    # ── model ─────────────────────────────────────────────────────────────────
    p.add_argument('--variant', default='large', choices=['small', 'base', 'large'],
                   help='DepthAnythingV2 encoder size.')
    p.add_argument('--pretrained_from', type=str, required=True,
                   help='Path to pretrained relative-depth .pth checkpoint. '
                        'Only encoder (ViT backbone) weights are loaded; '
                        'the metric head is trained from random init.')
    p.add_argument('--repo_dir', type=str, required=True,
                   help='Path to cloned Depth-Anything-V2 repo root. '
                        'Must contain depth_anything_v2/ and metric_depth/util/.')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to a latest.pth checkpoint to resume training from.')
    p.add_argument('--baseline_ckpt', type=str, default=None,
                   help='Path to the pretrained relative-depth .pth (same file as '
                        '--pretrained_from, or any other relative DAv2 checkpoint). '
                        'When set, the full relative model is loaded on rank 0 and '
                        'run on val vis samples each vis epoch. Its output is '
                        'aligned to GT (least-squares disparity) and shown as a '
                        '4th column: [RGB | finetuned metric | relative baseline | GT]. '
                        'This directly visualises what fine-tuning adds over the '
                        'pre-trained model. For full offline baseline numbers use '
                        'eval_depth_anything_v2.py separately.')

    # ── training hyperparameters ───────────────────────────────────────────────
    p.add_argument('--img_size',   default=518,   type=int)
    p.add_argument('--min_depth',  default=0.001, type=float,
                   help='Minimum valid depth in metres.')
    p.add_argument('--max_depth',  default=10.0,  type=float,
                   help='Maximum valid depth in metres. ADT indoor default: 10.0.')
    p.add_argument('--depth_scale', default=1.0,  type=float,
                   help='Multiply raw .npy depth values by this factor. '
                        'Blender renders are already in metres (default 1.0). '
                        'Real depth_npy files are uint16 millimetres: use 0.001.')
    p.add_argument('--epochs',     default=40,    type=int)
    p.add_argument('--bs',         default=4,     type=int,
                   help='Batch size per GPU.')
    p.add_argument('--lr',         default=5e-6,  type=float,
                   help='Base learning rate for the encoder. '
                        'The metric decoder head uses lr * 10.')
    p.add_argument('--num_workers', default=4,    type=int)

    # ── output ────────────────────────────────────────────────────────────────
    p.add_argument('--save_path', type=str, required=True,
                   help='Directory to write checkpoints, TensorBoard logs, and metrics CSV.')
    p.add_argument('--dataset_dir', default=None,
                   help='Directory containing dataset/adt.py. '
                        'Default: directory of this script.')
    p.add_argument('--port', default=None, type=int,
                   help='Master port for distributed training (auto if None).')

    # ── visualisation ─────────────────────────────────────────────────────────
    p.add_argument('--vis_interval', default=5, type=int,
                   help='Log image grids to TensorBoard every N epochs. '
                        'Set 0 to disable image logging.')
    p.add_argument('--n_vis', default=4, type=int,
                   help='Number of samples shown per image grid (train and val).')

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    # np.RankWarning was removed in NumPy 1.24; suppress it only if present
    rank_warning = getattr(np, 'RankWarning', None) or getattr(
        getattr(np, 'exceptions', None), 'RankWarning', None
    )
    if rank_warning is not None:
        warnings.simplefilter('ignore', rank_warning)

    # ── Import shared training helpers ───────────────────────────────────────
    _script_dir = args.dataset_dir or os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _script_dir)
    from training_utils import align_depth_pred, append_metrics_csv, log_image_grid

    # ── Import from Depth-Anything-V2 repo ────────────────────────────────────
    for path in (args.repo_dir, os.path.join(args.repo_dir, 'metric_depth')):
        if path not in sys.path:
            sys.path.insert(0, path)

    from depth_anything_v2.dpt import DepthAnythingV2
    from util.dist_helper import setup_distributed
    from util.loss import SiLogLoss
    from util.metric import eval_depth
    from util.utils import init_log

    # ── Import ADT dataset by absolute path ───────────────────────────────────
    # metric_depth/ in the DAv2 repo also contains a dataset/ package, so a
    # plain "from dataset.adt import ..." would be shadowed by that package.
    # Use importlib to load our module directly from its file path instead.
    import importlib.util as _ilu
    _dataset_dir = args.dataset_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _spec = _ilu.spec_from_file_location(
        'dataset_adt',
        os.path.join(_dataset_dir, 'dataset', 'adt.py'),
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    ADTDepthDataset = _mod.ADTDepthDataset

    # ── Distributed setup ─────────────────────────────────────────────────────
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Auto-apply depth_scale for real split: depth_npy files are uint16 mm.
    # Only kicks in when the user leaves --depth_scale at its default of 1.0,
    # so an explicit --depth_scale still takes precedence.
    if args.split == 'real' and args.depth_scale == 1.0:
        args.depth_scale = 0.001
        if rank == 0:
            logger.info(
                'Auto-set --depth_scale=0.001 for split=real '
                '(depth_npy is uint16 mm → m). '
                'Pass --depth_scale explicitly to override.'
            )

    seq_dirs     = args.seq_dirs     or BATCH_SEQ_DIRS
    val_seq_dirs = args.val_seq_dirs or BATCH_SEQ_DIRS_VAL

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
        logger.info(pprint.pformat({
            **vars(args),
            'ngpus':        world_size,
            'seq_dirs':     seq_dirs,
            'val_seq_dirs': val_seq_dirs,
        }))
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(args.save_path)
        except ImportError:
            writer = None
            logger.warning('TensorBoard not available; skipping scalar logging.')

    cudnn.enabled   = True
    cudnn.benchmark = True

    # ── Build datasets ────────────────────────────────────────────────────────
    # rotation=None lets ADTDepthDataset auto-select:
    #   real  → 270° CCW (corrects Aria sensor orientation)
    #   ego/exo → 0°
    # Pass args.rotation (may be None) to allow explicit override.
    dataset_kwargs = dict(
        split=args.split,
        size=(args.img_size, args.img_size),
        depth_scale=args.depth_scale,
        max_depth=args.max_depth,
        ego_subdirs=tuple(args.ego_subdirs),
        exo_cameras=args.exo_cameras,
        rotation=args.rotation,
    )

    trainset = ADTDepthDataset(seq_dirs=seq_dirs,     phase='train', **dataset_kwargs)
    valset   = ADTDepthDataset(seq_dirs=val_seq_dirs, phase='val',   **dataset_kwargs)

    # Optional: carve val frames off the end of the training set when the user
    # has not pointed to a separate val sequence directory.
    if args.val_fraction is not None and args.val_seq_dirs is None:
        n_total  = len(trainset.samples)
        n_val    = max(1, int(n_total * args.val_fraction))
        n_train  = n_total - n_val
        val_samples   = trainset.samples[n_train:]
        train_samples = trainset.samples[:n_train]
        trainset.samples = train_samples
        valset.samples   = val_samples
        if rank == 0:
            logger.info(f'val_fraction={args.val_fraction}: '
                        f'train={len(trainset)} val={len(valset)} frames')

    if rank == 0:
        logger.info(f'Train samples : {len(trainset)}')
        logger.info(f'Val   samples : {len(valset)}')

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader  = DataLoader(
        trainset, batch_size=args.bs, pin_memory=True,
        num_workers=args.num_workers, drop_last=True, sampler=trainsampler,
    )

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader  = DataLoader(
        valset, batch_size=1, pin_memory=True,
        num_workers=args.num_workers, drop_last=True, sampler=valsampler,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    # max_depth is passed to the constructor: it switches the model into metric
    # mode (final head outputs sigmoid(x) * max_depth instead of raw disparity).
    model_cfg = {**MODEL_CONFIGS[args.variant], 'max_depth': args.max_depth}
    model = DepthAnythingV2(**model_cfg)

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        # resume checkpoint contains the full DDP state_dict
        state = ckpt['model'] if 'model' in ckpt else ckpt
        # strip the "module." prefix added by DDP when saved
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        if rank == 0:
            logger.info(f'Resumed from {args.resume} (epoch {ckpt.get("epoch", "?")})')
    else:
        # Load only the encoder (ViT backbone) from the relative-depth checkpoint.
        # Parameter names containing 'pretrained' belong to the DINOv2 backbone;
        # everything else is the DPT decoder + metric head (trained from scratch).
        ckpt = torch.load(args.pretrained_from, map_location='cpu', weights_only=True)
        encoder_weights = {k: v for k, v in ckpt.items() if 'pretrained' in k}
        missing, unexpected = model.load_state_dict(encoder_weights, strict=False)
        if rank == 0:
            logger.info(f'Loaded {len(encoder_weights)} encoder tensors '
                        f'from {args.pretrained_from}')
            logger.info(f'  Missing keys (metric head — expected): {len(missing)}')
            if unexpected:
                logger.warning(f'  Unexpected keys: {unexpected}')

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True,
    )

    criterion = SiLogLoss().cuda(local_rank)

    # ── Optional relative baseline model (rank 0 only, for vis) ──────────────
    # Load the full pretrained relative-depth model (all weights, no max_depth).
    # Used only for comparison visualisation — never trained, never distributed.
    # Confirmed standard: the official metric_depth/train.py uses the same
    # 'pretrained'-key-only loading strategy as this script, so the decoder is
    # always trained from random init.  This baseline shows what fine-tuning adds.
    rel_model = None
    if rank == 0 and args.baseline_ckpt:
        rel_cfg   = MODEL_CONFIGS[args.variant]            # no max_depth → relative mode
        rel_model = DepthAnythingV2(**rel_cfg)
        rel_ckpt  = torch.load(args.baseline_ckpt, map_location='cpu', weights_only=True)
        missing_rel, unexpected_rel = rel_model.load_state_dict(rel_ckpt, strict=False)
        rel_model.cuda(local_rank).eval()
        logger.info(
            f'Relative baseline loaded from {args.baseline_ckpt} '
            f'({len(rel_ckpt)} keys, {len(missing_rel)} missing, '
            f'{len(unexpected_rel)} unexpected)'
        )
        if missing_rel:
            logger.warning(f'  Relative model missing keys: {missing_rel}')

    # Encoder gets the base LR; the metric decoder head gets 10× to compensate
    # for random init (mirrors the official train.py strategy).
    encoder_params = [p for n, p in model.named_parameters() if 'pretrained' in n]
    decoder_params = [p for n, p in model.named_parameters() if 'pretrained' not in n]
    optimizer = AdamW(
        [
            {'params': encoder_params, 'lr': args.lr},
            {'params': decoder_params, 'lr': args.lr * 10.0},
        ],
        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01,
    )

    start_epoch = 0
    previous_best = {
        'd1': 0, 'd2': 0, 'd3': 0,
        'abs_rel': 100, 'sq_rel': 100, 'rmse': 100,
        'rmse_log': 100, 'log10': 100, 'silog': 100,
    }

    if args.resume and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch    = ckpt.get('epoch', 0) + 1
        previous_best  = ckpt.get('previous_best', previous_best)

    total_iters = args.epochs * len(trainloader)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        if rank == 0:
            logger.info(
                f'Epoch {epoch}/{args.epochs}  |  split={args.split}  '
                f'variant={args.variant}  |  '
                f'd1={previous_best["d1"]:.3f}  '
                f'abs_rel={previous_best["abs_rel"]:.3f}  '
                f'rmse={previous_best["rmse"]:.3f}  '
                f'silog={previous_best["silog"]:.3f}'
            )

        trainloader.sampler.set_epoch(epoch + 1)
        model.train()

        do_vis = (
            args.vis_interval > 0
            and rank == 0
            and writer is not None
            and (epoch % args.vis_interval == 0)
        )
        train_vis = {'rgb': [], 'pred': [], 'gt': []}

        _nan_warned = False   # emit at most one depth-scale warning per epoch

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()

            img        = sample['image'].cuda()
            depth      = sample['depth'].cuda()
            valid_mask = sample['valid_mask'].cuda()

            # Horizontal flip augmentation applied after batching, same as official train.py.
            # Both image and depth are flipped together to maintain spatial correspondence.
            if random.random() < 0.5:
                img        = img.flip(-1)
                depth      = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            pred = model(img)

            loss_mask = (
                (valid_mask == 1)
                & (depth >= args.min_depth)
                & (depth <= args.max_depth)
            )

            # Skip batches with too few valid pixels — SiLog on an empty / near-empty
            # tensor produces NaN.  Most common cause: depth_scale not set correctly
            # (real depth_npy files are uint16 mm; pass --depth_scale 0.001).
            if loss_mask.sum() < 10:
                if rank == 0 and not _nan_warned:
                    logger.warning(
                        f'  [{epoch}] iter {i}: only {loss_mask.sum().item()} valid '
                        f'depth pixels in batch — skipping loss. '
                        f'depth min={depth.min().item():.4f} '
                        f'max={depth.max().item():.4f} '
                        f'(check --depth_scale; real depth_npy is uint16 mm → use 0.001)'
                    )
                    _nan_warned = True
                continue

            # Clamp predictions to a safe positive minimum before SiLog's log().
            # pred is sigmoid(x)*max_depth so it is theoretically >0, but floating-
            # point underflow can reach exactly 0 on some hardware/compiler configs.
            loss = criterion(pred.clamp(min=1e-3), depth, loss_mask)

            if not torch.isfinite(loss):
                if rank == 0 and not _nan_warned:
                    logger.warning(
                        f'  [{epoch}] iter {i}: non-finite loss ({loss.item()}) — '
                        f'skipping update. pred range '
                        f'[{pred.min().item():.4f}, {pred.max().item():.4f}]'
                    )
                    _nan_warned = True
                continue

            loss.backward()
            optimizer.step()

            iters = epoch * len(trainloader) + i
            # Poly LR decay: lr * (1 - iter/total)^0.9  (official schedule)
            lr = args.lr * (1.0 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * 10.0

            if rank == 0:
                if writer:
                    writer.add_scalar('train/loss', loss.item(), iters)
                    writer.add_scalar('train/lr',   lr,          iters)
                if i % 100 == 0:
                    logger.info(
                        f'  [{epoch}] iter {i}/{len(trainloader)}  '
                        f'lr={lr:.2e}  loss={loss.item():.4f}'
                    )

            # Collect samples for the training image grid from the last batch of the
            # epoch so the snapshot reflects the fully-trained end-of-epoch state,
            # not the randomly-initialised beginning (which produced uniform output).
            if do_vis and i == len(trainloader) - 1 and len(train_vis['rgb']) < args.n_vis:
                with torch.no_grad():
                    pred_vis = F.interpolate(
                        pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True
                    )[:, 0]   # (B, H, W)
                for b in range(min(args.n_vis, img.size(0))):
                    train_vis['rgb'].append(img[b].cpu())
                    train_vis['pred'].append(pred_vis[b].cpu())
                    train_vis['gt'].append(depth[b].cpu())

        if do_vis:
            log_image_grid(
                writer, 'train/vis',
                train_vis['rgb'], train_vis['pred'], train_vis['gt'],
                step=epoch, n=args.n_vis,
            )

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        result_keys = ('d1', 'd2', 'd3', 'abs_rel', 'sq_rel',
                        'rmse', 'rmse_log', 'log10', 'silog')
        results  = {k: torch.tensor([0.0]).cuda() for k in result_keys}
        nsamples = torch.tensor([0.0]).cuda()

        val_vis = {'rgb': [], 'pred': [], 'gt': [], 'rel': []}

        for sample in valloader:
            img        = sample['image'].cuda().float()
            depth      = sample['depth'].cuda()[0]        # (H, W)
            valid_mask = sample['valid_mask'].cuda()[0]   # (H, W)

            with torch.no_grad():
                pred = model(img)
                # model output is (B, H, W); interpolate to GT resolution
                pred = F.interpolate(
                    pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True
                )[0, 0]   # → (H, W)

            val_mask = (
                (valid_mask == 1)
                & (depth >= args.min_depth)
                & (depth <= args.max_depth)
            )
            if val_mask.sum() < 10:
                continue

            cur = eval_depth(pred[val_mask], depth[val_mask])
            for k in result_keys:
                results[k] += cur[k]
            nsamples += 1

            # Collect for image grid (rank 0, first n_vis valid samples)
            if do_vis and rank == 0 and len(val_vis['rgb']) < args.n_vis:
                val_vis['rgb'].append(img[0].cpu())
                val_vis['pred'].append(pred.cpu())
                val_vis['gt'].append(depth.cpu())

                # Run relative baseline on the same sample and align to GT.
                # Same disparity-space alignment as eval_depth_anything_v2.py.
                if rel_model is not None:
                    with torch.no_grad():
                        rel_raw = rel_model(img)   # (1, H', W') relative depth
                        rel_raw = F.interpolate(
                            rel_raw[:, None], depth.shape[-2:],
                            mode='bilinear', align_corners=True,
                        )[0, 0]                    # (H, W)
                    rel_aligned = align_depth_pred(
                        rel_raw.cpu().numpy(),
                        depth.cpu().numpy(),
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                    )
                    val_vis['rel'].append(torch.from_numpy(rel_aligned))

        torch.distributed.barrier()
        for k in result_keys:
            dist.reduce(results[k], dst=0)
        dist.reduce(nsamples, dst=0)

        if rank == 0:
            n = max(nsamples.item(), 1)
            means = {k: (results[k] / n).item() for k in result_keys}
            logger.info('=' * 90)
            logger.info('  ' + '  '.join(f'{k:>8}' for k in result_keys))
            logger.info('  ' + '  '.join(f'{means[k]:8.3f}' for k in result_keys))
            logger.info('=' * 90)

            if writer:
                for k, v in means.items():
                    writer.add_scalar(f'val/{k}', v, epoch)

            # Image grids: val visualization.
            # Passes relative baseline preds when available → 4-column grid
            # [RGB | finetuned metric | relative baseline (aligned) | GT]
            # which directly shows what fine-tuning adds over the pretrained model.
            if do_vis:
                log_image_grid(
                    writer, 'val/vis',
                    val_vis['rgb'], val_vis['pred'], val_vis['gt'],
                    step=epoch, n=args.n_vis,
                    rel_list=val_vis['rel'] if val_vis['rel'] else None,
                )

            # Append epoch metrics to CSV
            append_metrics_csv(
                os.path.join(args.save_path, 'metrics.csv'),
                {'split': args.split, 'variant': args.variant, 'epoch': epoch, **means},
            )

            # Update best tracker and save best checkpoint (by δ₁)
            is_best_d1 = means['d1'] > previous_best['d1']
            for k in result_keys:
                if k in ('d1', 'd2', 'd3'):
                    previous_best[k] = max(previous_best[k], means[k])
                else:
                    previous_best[k] = min(previous_best[k], means[k])

            # Save full training state (always)
            torch.save(
                {
                    'model':         model.state_dict(),
                    'optimizer':     optimizer.state_dict(),
                    'epoch':         epoch,
                    'previous_best': previous_best,
                },
                os.path.join(args.save_path, 'latest.pth'),
            )

            # Save best model weights only
            if is_best_d1:
                best_path = os.path.join(
                    args.save_path, f'best_{args.split}_{args.variant}.pth'
                )
                torch.save(
                    {k.replace('module.', '', 1): v
                     for k, v in model.state_dict().items()},
                    best_path,
                )
                logger.info(f'  New best δ₁={previous_best["d1"]:.4f} → saved {best_path}')


if __name__ == '__main__':
    main()