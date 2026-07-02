"""
finetune_adt.py
===============
Fine-tune Depth Anything V2 for metric depth estimation on ADT rendered data.

Directly adapted from the official metric_depth/train.py in the
Depth-Anything-V2 repo (same loss, same optimizer, same LR schedule).

Three mutually-exclusive training splits (train on each independently):
  --split ego   egocentric Blender renders (fisheye + pinhole)
  --split exo   exocentric Blender renders (all cameras)
  --split real  real-captured frames with depth maps

The data loader mirrors eval_depth_anything_v2.py's _collect_frames() so that
the same frames that are evaluated can also be used for training.

Usage — single GPU:
  torchrun --standalone --nproc_per_node=1 finetune_adt.py \\
      --split ego \\
      --repo_dir /path/to/Depth-Anything-V2 \\
      --pretrained_from /path/to/depth_anything_v2_vitl.pth \\
      --save_path /path/to/checkpoints/ego

Usage — multi-GPU (e.g. 4 GPUs):
  torchrun --standalone --nproc_per_node=4 finetune_adt.py \\
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
_ADT_ROOT = '/user/f.zhang2/Documents/projectaria_tools_adt_data'
BATCH_SEQ_DIRS = [
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
        description='Fine-tune Depth Anything V2 on ADT rendered depth data',
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
    # real-specific
    p.add_argument('--real_subdir', default='real_captures',
                   help="Subdirectory name under each seq_dir for 'real' split.")

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

    # ── training hyperparameters ───────────────────────────────────────────────
    p.add_argument('--img_size',   default=518,   type=int)
    p.add_argument('--min_depth',  default=0.001, type=float,
                   help='Minimum valid depth in metres.')
    p.add_argument('--max_depth',  default=10.0,  type=float,
                   help='Maximum valid depth in metres. ADT indoor default: 10.0.')
    p.add_argument('--depth_scale', default=1.0,  type=float,
                   help='Multiply raw .npy depth values by this factor. '
                        'ADT .npy files are already in metres so default is 1.0.')
    p.add_argument('--epochs',     default=40,    type=int)
    p.add_argument('--bs',         default=4,     type=int,
                   help='Batch size per GPU.')
    p.add_argument('--lr',         default=5e-6,  type=float,
                   help='Base learning rate for the encoder. '
                        'The metric decoder head uses lr * 10.')
    p.add_argument('--num_workers', default=4,    type=int)

    # ── output ────────────────────────────────────────────────────────────────
    p.add_argument('--save_path', type=str, required=True,
                   help='Directory to write checkpoints and TensorBoard logs.')
    p.add_argument('--port', default=None, type=int,
                   help='Master port for distributed training (auto if None).')

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    warnings.simplefilter('ignore', np.RankWarning)

    # ── Import from Depth-Anything-V2 repo ────────────────────────────────────
    for path in (args.repo_dir, os.path.join(args.repo_dir, 'metric_depth')):
        if path not in sys.path:
            sys.path.insert(0, path)

    from depth_anything_v2.dpt import DepthAnythingV2
    from util.dist_helper import setup_distributed
    from util.loss import SiLogLoss
    from util.metric import eval_depth
    from util.utils import init_log

    # ── Import ADT dataset (from same directory as this script) ───────────────
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset.adt_depth import ADTDepthDataset

    # ── Distributed setup ─────────────────────────────────────────────────────
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    seq_dirs     = args.seq_dirs     or BATCH_SEQ_DIRS
    val_seq_dirs = args.val_seq_dirs or seq_dirs

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
    dataset_kwargs = dict(
        split=args.split,
        size=(args.img_size, args.img_size),
        depth_scale=args.depth_scale,
        max_depth=args.max_depth,
        real_subdir=args.real_subdir,
        ego_subdirs=tuple(args.ego_subdirs),
        exo_cameras=args.exo_cameras,
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
            loss = criterion(pred, depth, loss_mask)
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

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        result_keys = ('d1', 'd2', 'd3', 'abs_rel', 'sq_rel',
                        'rmse', 'rmse_log', 'log10', 'silog')
        results  = {k: torch.tensor([0.0]).cuda() for k in result_keys}
        nsamples = torch.tensor([0.0]).cuda()

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