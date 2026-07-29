#!/usr/bin/env python3
"""Standalone CLI entrypoint for training the patch classifier experiment.

Example (Pair A, cross-image, single species):
    python ml_experiments/train.py \\
        --pnt /Users/giacomo/Documents/Coscoroba/202607270736/S377018913.pnt \\
        --image-dir /Users/giacomo/Documents/Coscoroba/202607270736 \\
        --pair a --class-name "Plegadis chihi" \\
        --train-image IMG_4147.jpeg --eval-image IMG_4148.jpeg \\
        --out-dir ml_experiments/runs/pair_a_plegadis

Example (Pair B, same-image spatial holdout, all classes on the image pooled as
generic "bird" -- omit --class-name):
    python ml_experiments/train.py \\
        --pnt /Users/giacomo/Documents/Coscoroba/202607270736/S377018913.pnt \\
        --image-dir /Users/giacomo/Documents/Coscoroba/202607270736 \\
        --pair b --image-name IMG_4153.jpeg --patch-size 32 \\
        --out-dir ml_experiments/runs/small_bucket

Example (pooled, cross-folder, species-agnostic, size-bucketed):
    python ml_experiments/train.py --pair pool \\
        --sources-config ml_experiments/bucket_configs/large.json \\
        --patch-size 96 --out-dir ml_experiments/runs/large_bucket
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

from ml_experiments.dataset import PatchDataset, compute_mean_std
from ml_experiments.model import SmallPatchCNN
from ml_experiments.pnt_io import load_pnt
from ml_experiments.splits import multi_source_split, pair_a_split, pair_b_split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(records):
    counts = [0, 0]
    for r in records:
        counts[r.label] += 1
    total = sum(counts)
    weights = [total / (2 * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def pooled_classes(records):
    """Distinct species names among positive records -- recorded in config.json so
    ddg/ml_detector.py can route a given class_name to whichever bucket's model was
    trained on it, without needing an exact single-class match."""
    return sorted({r.class_name for r in records if r.label == 1 and r.class_name})


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(mode=train)
    total_loss, total_correct, total_count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += x.size(0)
    return total_loss / total_count, total_correct / total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pnt', help='Required for --pair a/b')
    parser.add_argument('--image-dir', help='Required for --pair a/b')
    parser.add_argument('--pair', choices=['a', 'b', 'pool'], required=True)
    parser.add_argument('--class-name', default=None,
                         help='Omit to pool every annotated class on the image(s) as a '
                              'single generic "bird" positive set (species-agnostic).')
    parser.add_argument('--train-image', help='Pair A: image to train on')
    parser.add_argument('--eval-image', help='Pair A: image to evaluate on')
    parser.add_argument('--image-name', help='Pair B: single image to spatially split')
    parser.add_argument('--train-frac', type=float, default=0.8, help='Pair B split fraction')
    parser.add_argument('--sources-config', help='Pair pool: JSON with train_sources/val_sources')
    parser.add_argument('--bucket', default=None, help='Optional size-bucket label, e.g. small/large')
    parser.add_argument('--patch-size', type=int, default=96)
    parser.add_argument('--neg-ratio', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--jitter-px', type=int, default=16,
                         help='Random crop-center jitter for training positives/negatives, '
                              'simulating sliding-window off-centering at eval time.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cpu')

    split_meta = {}
    if args.pair == 'a':
        assert args.pnt and args.image_dir and args.train_image and args.eval_image, \
            '--pair a requires --pnt, --image-dir, --train-image, --eval-image'
        pnt_data = load_pnt(args.pnt)
        train_records, eval_records = pair_a_split(
            pnt_data, args.image_dir, args.train_image, args.eval_image, args.class_name,
            args.patch_size, args.neg_ratio, args.seed)
        split_meta = {'train_image': args.train_image, 'eval_image': args.eval_image}
    elif args.pair == 'b':
        assert args.pnt and args.image_dir and args.image_name, \
            '--pair b requires --pnt, --image-dir, --image-name'
        pnt_data = load_pnt(args.pnt)
        train_records, eval_records, meta = pair_b_split(
            pnt_data, args.image_dir, args.image_name, args.class_name,
            args.patch_size, args.neg_ratio, args.seed, train_frac=args.train_frac)
        split_meta = {'image_name': args.image_name, **meta}
    else:
        assert args.sources_config, '--pair pool requires --sources-config'
        with open(args.sources_config) as f:
            sources_config = json.load(f)
        train_records, eval_records = multi_source_split(
            sources_config['train_sources'], sources_config['val_sources'],
            args.patch_size, args.neg_ratio, args.seed)
        split_meta = {'sources_config': args.sources_config,
                       'train_sources': sources_config['train_sources'],
                       'val_sources': sources_config['val_sources']}

    print(f'train: {len(train_records)} patches ({sum(r.label for r in train_records)} positive)')
    print(f'eval: {len(eval_records)} patches ({sum(r.label for r in eval_records)} positive)')
    train_classes = pooled_classes(train_records)
    print(f'pooled classes in training positives: {train_classes}')

    mean, std = compute_mean_std(train_records, args.patch_size)
    train_ds = PatchDataset(train_records, args.patch_size, augment=True,
                             mean=mean, std=std, rng_seed=args.seed, jitter_px=args.jitter_px)
    eval_ds = PatchDataset(eval_records, args.patch_size, augment=False,
                            mean=mean, std=std, rng_seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SmallPatchCNN().to(device)
    print(f'model parameters: {model.num_parameters()}')
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_records).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    start = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        eval_loss, eval_acc = run_epoch(model, eval_loader, criterion, optimizer, device, train=False)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                         'eval_loss': eval_loss, 'eval_acc': eval_acc})
        print(f'epoch {epoch:3d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  '
              f'eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f}')
    elapsed = time.time() - start
    print(f'training finished in {elapsed:.1f}s')

    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pt'))
    config = {
        'args': vars(args),
        'split_meta': split_meta,
        'bucket': args.bucket,
        'pooled_classes': train_classes,
        'mean': mean.tolist(),
        'std': std.tolist(),
        'model_parameters': model.num_parameters(),
        'train_size': len(train_records),
        'eval_size': len(eval_records),
        'elapsed_seconds': elapsed,
    }
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(args.out_dir, 'train_log.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f'saved model + config + train_log to {args.out_dir}')


if __name__ == '__main__':
    main()
