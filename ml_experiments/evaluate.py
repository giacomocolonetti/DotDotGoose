#!/usr/bin/env python3
"""Evaluation: patch-level metrics, point-level sliding-window detection metrics, and a
classical-CV-detector baseline comparison -- run through the identical match_points
function so the two are directly comparable."""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddg.detector import ClassicCVDetector
from ml_experiments.dataset import PatchDataset
from ml_experiments.model import SmallPatchCNN
from ml_experiments.patches import extract_patch, load_image_array
from ml_experiments.pnt_io import load_pnt, points_for_image
from ml_experiments.splits import pair_a_split, pair_b_split


def roc_auc(labels, scores):
    """Tie-aware AUC via the Mann-Whitney U rank-sum formula (avoids an sklearn dependency)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    rank = 1
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (rank + rank + (j - i)) / 2
        ranks[order[i:j + 1]] = avg_rank
        rank += (j - i + 1)
        i = j + 1
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def patch_level_metrics(model, dataset, device='cpu', batch_size=64):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y.numpy().tolist())
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= 0.5).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc(labels, probs)}


def sliding_window_detect(model, img, mean, std, patch_size=96, stride=32, score_threshold=0.5,
                           x_range=None, batch_size=256, device='cpu'):
    model.eval()
    h, w = img.shape[0], img.shape[1]
    half = patch_size // 2
    x_lo, x_hi = (half, w - half) if x_range is None else x_range
    xs = list(range(int(x_lo), int(x_hi), stride))
    ys = list(range(half, h - half, stride))
    if not xs or not ys:
        return []
    centers = [(x, y) for y in ys for x in xs]

    detections = []
    with torch.no_grad():
        for i in range(0, len(centers), batch_size):
            batch_centers = centers[i:i + batch_size]
            chips = np.stack([extract_patch(img, cx, cy, patch_size).astype(np.float32)
                               for cx, cy in batch_centers])
            chips = (chips - mean) / std
            chips = np.ascontiguousarray(np.transpose(chips, (0, 3, 1, 2)))
            tensor = torch.from_numpy(chips).to(device)
            probs = torch.softmax(model(tensor), dim=1)[:, 1].cpu().numpy()
            for (cx, cy), score in zip(batch_centers, probs):
                if score >= score_threshold:
                    detections.append((float(cx), float(cy), float(score)))
    return detections


def nms_points(detections, dedup_radius):
    """Greedy NMS: keep the highest-scoring detection, drop others within dedup_radius, repeat."""
    if not detections:
        return []
    ordered = sorted(detections, key=lambda d: -d[2])
    kept = []
    kept_xy = []
    for x, y, score in ordered:
        if kept_xy:
            arr = np.array(kept_xy)
            dist2 = np.sum((arr - np.array([x, y])) ** 2, axis=1)
            if dist2.min() <= dedup_radius ** 2:
                continue
        kept.append((x, y, score))
        kept_xy.append((x, y))
    return kept


def match_points(pred_xy, gt_xy, match_radius):
    """Greedy nearest-first matching within match_radius."""
    n_pred, n_gt = len(pred_xy), len(gt_xy)
    if n_pred == 0 or n_gt == 0:
        tp = 0
    else:
        pred_arr = np.array(pred_xy, dtype=np.float64)
        gt_arr = np.array(gt_xy, dtype=np.float64)
        dist = np.sqrt(((pred_arr[:, None, :] - gt_arr[None, :, :]) ** 2).sum(axis=2))
        pairs = [(dist[i, j], i, j) for i in range(n_pred) for j in range(n_gt)
                 if dist[i, j] <= match_radius]
        pairs.sort(key=lambda p: p[0])
        matched_pred, matched_gt = set(), set()
        for _, i, j in pairs:
            if i in matched_pred or j in matched_gt:
                continue
            matched_pred.add(i)
            matched_gt.add(j)
        tp = len(matched_pred)
    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}


def median_nn_distance(points_xy):
    xy = np.array(points_xy, dtype=np.float64)
    if len(xy) < 2:
        return None
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    return float(np.median(dist.min(axis=1)))


def best_classical_detections(image_array, gt_points_xy, match_radius, x_range=None,
                               sensitivities=(30, 50, 70)):
    """Sweep the shipped ClassicCVDetector over sensitivity and polarity, keep the
    operating point with the best F1 at `match_radius`, and return its raw predicted
    points (so callers can re-score the same fixed prediction set at other tolerances --
    an apples-to-apples comparison against the CNN's one prediction set)."""
    from PyQt6 import QtCore
    detector = ClassicCVDetector()
    region = None
    if x_range is not None:
        region = QtCore.QRectF(x_range[0], 0, x_range[1] - x_range[0], image_array.shape[0])
    best_f1 = -1.0
    best_pred, best_params = [], {}
    for polarity in ('bright', 'dark'):
        for sensitivity in sensitivities:
            result = detector.detect(image_array, region=region, sensitivity=sensitivity, polarity=polarity)
            pred_xy = [(p.x(), p.y()) for p in result.points]
            metrics = match_points(pred_xy, gt_points_xy, match_radius)
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_pred = pred_xy
                best_params = {'sensitivity': sensitivity, 'polarity': polarity}
    return best_pred, best_params


def evaluate_run(run_dir, stride=32, score_threshold=0.5, dedup_radius=None):
    with open(os.path.join(run_dir, 'config.json')) as f:
        config = json.load(f)
    args = config['args']
    split_meta = config['split_meta']
    pnt_data = load_pnt(args['pnt'])
    mean = np.array(config['mean'], dtype=np.float32)
    std = np.array(config['std'], dtype=np.float32)
    patch_size = args['patch_size']

    model = SmallPatchCNN()
    model.load_state_dict(torch.load(os.path.join(run_dir, 'model.pt'), map_location='cpu'))
    model.eval()

    if args['pair'] == 'a':
        eval_image = args['eval_image']
        gt_points = points_for_image(pnt_data, eval_image, args['class_name'])
        x_range = None
        _, eval_records = pair_a_split(pnt_data, args['image_dir'], args['train_image'],
                                        args['eval_image'], args['class_name'], patch_size,
                                        args['neg_ratio'], args['seed'])
    else:
        eval_image = args['image_name']
        boundary = split_meta['boundary']
        half = patch_size // 2
        all_pts = points_for_image(pnt_data, eval_image, args['class_name'])
        gt_points = [(x, y) for x, y in all_pts if x - half >= boundary]
        img_probe = load_image_array(args['image_dir'], eval_image)
        x_range = (max(boundary + half, half), img_probe.shape[1] - half)
        _, eval_records, _ = pair_b_split(pnt_data, args['image_dir'], args['image_name'],
                                           args['class_name'], patch_size, args['neg_ratio'],
                                           args['seed'], train_frac=args.get('train_frac', 0.8))

    img = load_image_array(args['image_dir'], eval_image)
    eval_dataset = PatchDataset(eval_records, args['image_dir'], patch_size, augment=False,
                                 mean=mean, std=std)
    patch_metrics = patch_level_metrics(model, eval_dataset)
    if dedup_radius is None:
        dedup_radius = patch_size // 2
    precise_radius = median_nn_distance(gt_points) / 2 if len(gt_points) >= 2 else patch_size / 4
    # Three tolerances: (1) precise -- matches the app's own point-placement precision need
    # (half the natural inter-bird spacing), (2) grid stride granularity, (3) NMS dedup
    # radius -- an upper bound on what a stride/NMS sliding-window pipeline can resolve at
    # all. A classifier-only detector fundamentally can't beat its own stride/NMS
    # resolution, so precise_radius alone can be misleadingly pessimistic; reporting all
    # three shows the real precision/recall tradeoff instead of picking one number.
    radii = sorted(set([round(precise_radius, 1), float(stride), float(dedup_radius)]))

    detections = sliding_window_detect(model, img, mean, std, patch_size=patch_size, stride=stride,
                                        score_threshold=score_threshold, x_range=x_range)
    cnn_pred = [(x, y) for x, y, _ in nms_points(detections, dedup_radius)]
    cnn_by_radius = {r: match_points(cnn_pred, gt_points, r) for r in radii}

    classical_pred, classical_params = best_classical_detections(img, gt_points, precise_radius, x_range=x_range)
    classical_by_radius = {r: match_points(classical_pred, gt_points, r) for r in radii}

    return {
        'run_dir': run_dir,
        'pair': args['pair'],
        'class_name': args['class_name'],
        'eval_image': eval_image,
        'n_gt': len(gt_points),
        'cnn_patch_level': patch_metrics,
        'radii': radii,
        'cnn_point_level_by_radius': cnn_by_radius,
        'classical_point_level_by_radius': classical_by_radius,
        'classical_params': classical_params,
        'stride': stride,
        'dedup_radius': dedup_radius,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--score-threshold', type=float, default=0.5)
    args = parser.parse_args()
    results = evaluate_run(args.run_dir, stride=args.stride, score_threshold=args.score_threshold)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
