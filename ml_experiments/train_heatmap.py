#!/usr/bin/env python3
"""Standalone CLI entrypoint for training the heatmap-regression detector (see
heatmap_model.py / heatmap_dataset.py / heatmap_targets.py for why this exists instead of
the patch classifier in model.py -- it predicts a per-pixel map and finds peaks in it,
instead of classifying fixed-size windows on a coarse grid, which removes the
stride/NMS localization ceiling the classifier approach is capped at).

Example (pool mode, cross-folder, species-agnostic -- the large bucket):
    python ml_experiments/train_heatmap.py --mode pool \\
        --sources-config ml_experiments/bucket_configs/large.json \\
        --bucket large --num-stages 3 --sigma 2.0 \\
        --out-dir ml_experiments/runs/large_bucket_heatmap

Example (same-image spatial split -- the small, densely-packed bucket, only one photo
available, all classes on it pooled since --class-name is omitted):
    python ml_experiments/train_heatmap.py --mode same-image \\
        --pnt /Users/giacomo/Documents/Coscoroba/202607270736/S377018913.pnt \\
        --image-dir /Users/giacomo/Documents/Coscoroba/202607270736 \\
        --image-name IMG_4153.jpeg --bucket small --num-stages 2 --sigma 1.0 \\
        --out-dir ml_experiments/runs/small_bucket_heatmap
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_experiments.heatmap_dataset import HeatmapTileDataset, compute_mean_std_from_images
from ml_experiments.heatmap_model import HeatmapCNN
from ml_experiments.patches import load_image_array
from ml_experiments.pnt_io import all_points_with_class_for_image, load_pnt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sources_pooled_classes(sources):
    classes = set()
    for s in sources:
        pnt_data = load_pnt(s['pnt'])
        for _, _, c in all_points_with_class_for_image(pnt_data, s['image']):
            classes.add(c)
    return sorted(classes)


def weighted_mse_loss(pred_logits, target, pos_weight):
    pred = torch.sigmoid(pred_logits)
    weight = 1.0 + pos_weight * target
    return (weight * (pred - target) ** 2).mean()


def run_epoch(model, loader, optimizer, device, train, pos_weight):
    model.train(mode=train)
    total_loss, total_count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = weighted_mse_loss(logits, y, pos_weight)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)
    return total_loss / total_count


def build_same_image_ranges(pnt_data, image_name, tile_size, train_frac=0.8):
    xs = np.array([p[0] for p in
                   [(x, y) for x, y, _ in all_points_with_class_for_image(pnt_data, image_name)]])
    boundary = float(np.percentile(xs, train_frac * 100))
    half = tile_size / 2
    return boundary, (0, boundary - half), (boundary + half, None)  # val hi filled by caller (image width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pool', 'same-image'], required=True)
    parser.add_argument('--sources-config', help='pool mode: JSON with train_sources/val_sources')
    parser.add_argument('--pnt', help='same-image mode')
    parser.add_argument('--image-dir', help='same-image mode')
    parser.add_argument('--image-name', help='same-image mode')
    parser.add_argument('--train-frac', type=float, default=0.8)
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--tile-size', type=int, default=256)
    parser.add_argument('--sigma', type=float, required=True,
                         help='Gaussian stddev at TARGET (post-output_stride) resolution')
    parser.add_argument('--num-stages', type=int, choices=[2, 3], default=3)
    parser.add_argument('--samples-per-epoch', type=int, default=1500)
    parser.add_argument('--val-samples', type=int, default=400)
    parser.add_argument('--bg-fraction', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--pos-weight', type=float, default=10.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f'device: {device}')
    output_stride = 2 ** args.num_stages

    split_meta = {}
    if args.mode == 'pool':
        assert args.sources_config, '--mode pool requires --sources-config'
        with open(args.sources_config) as f:
            sources_config = json.load(f)
        train_sources = sources_config['train_sources']
        val_sources = sources_config['val_sources']
        train_x_ranges = None
        val_x_ranges = None
        split_meta = {'sources_config': args.sources_config,
                       'train_sources': train_sources, 'val_sources': val_sources}
    else:
        assert args.pnt and args.image_dir and args.image_name, \
            '--mode same-image requires --pnt, --image-dir, --image-name'
        source = {'pnt': args.pnt, 'image_dir': args.image_dir, 'image': args.image_name}
        pnt_data = load_pnt(args.pnt)
        img_probe = load_image_array(args.image_dir, args.image_name)
        boundary, train_range, val_range_partial = build_same_image_ranges(
            pnt_data, args.image_name, args.tile_size, args.train_frac)
        val_range = (val_range_partial[0], img_probe.shape[1])
        train_sources = [source]
        val_sources = [source]
        train_x_ranges = [train_range]
        val_x_ranges = [val_range]
        split_meta = {'image_name': args.image_name, 'boundary': boundary,
                       'train_x_range': train_range, 'val_x_range': val_range}

    train_images = [load_image_array(s['image_dir'], s['image']) for s in train_sources]
    mean, std = compute_mean_std_from_images(train_images)
    del train_images
    print(f'mean={mean.tolist()} std={std.tolist()}')

    train_ds = HeatmapTileDataset(train_sources, args.tile_size, args.sigma, output_stride,
                                   args.samples_per_epoch, bg_fraction=args.bg_fraction,
                                   augment=True, mean=mean, std=std, rng_seed=args.seed,
                                   x_ranges=train_x_ranges)
    val_ds = HeatmapTileDataset(val_sources, args.tile_size, args.sigma, output_stride,
                                 args.val_samples, bg_fraction=args.bg_fraction,
                                 augment=False, mean=mean, std=std, rng_seed=args.seed + 1000,
                                 x_ranges=val_x_ranges)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = HeatmapCNN(num_stages=args.num_stages).to(device)
    print(f'model parameters: {model.num_parameters()} output_stride: {output_stride}')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    start = time.time()
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device, True, args.pos_weight)
        val_loss = run_epoch(model, val_loader, optimizer, device, False, args.pos_weight)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f'epoch {epoch:3d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}')
    elapsed = time.time() - start
    print(f'training finished in {elapsed:.1f}s')

    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pt'))
    config = {
        'args': vars(args),
        'split_meta': split_meta,
        'bucket': args.bucket,
        'output_stride': output_stride,
        'pooled_classes': sources_pooled_classes(train_sources),
        'mean': mean.tolist(),
        'std': std.tolist(),
        'model_parameters': model.num_parameters(),
        'elapsed_seconds': elapsed,
    }
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.out_dir, 'train_log.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f'saved model + config + train_log to {args.out_dir}')


if __name__ == '__main__':
    main()
