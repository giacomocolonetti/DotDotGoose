#!/usr/bin/env python3
"""Evaluation for the heatmap-regression detector: whole-image tiled inference, local-peak
finding at native heatmap resolution (not a coarse stride grid), and the same point-level
matching + classical-CV baseline comparison used by evaluate.py, so the two approaches are
directly comparable."""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddg.detector import ClassicCVDetector
from ml_experiments.evaluate import match_points, median_nn_distance
from ml_experiments.heatmap_model import HeatmapCNN
from ml_experiments.patches import load_image_array
from ml_experiments.pnt_io import load_pnt, points_for_image


def predict_heatmap(model, img, mean, std, output_stride, tile=1536, overlap=64):
    """Run the fully-convolutional model over the whole image via large overlapping
    tiles (needed for the huge stitched panoramas), stitching results with a max-blend so
    overlap seams don't create duplicate/split peaks."""
    h, w = img.shape[0], img.shape[1]
    out_h, out_w = h // output_stride, w // output_stride
    heatmap = np.zeros((out_h, out_w), dtype=np.float32)
    step = tile - overlap

    model.eval()
    with torch.no_grad():
        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                if y1 - y0 < output_stride or x1 - x0 < output_stride:
                    continue
                chunk = img[y0:y1, x0:x1].astype(np.float32)
                chunk = (chunk - mean) / std
                chunk = np.ascontiguousarray(np.transpose(chunk, (2, 0, 1)))
                tensor = torch.from_numpy(chunk).unsqueeze(0)
                probs = torch.sigmoid(model(tensor))[0, 0].numpy()
                oy0, ox0 = y0 // output_stride, x0 // output_stride
                oh, ow = probs.shape
                heatmap[oy0:oy0 + oh, ox0:ox0 + ow] = np.maximum(heatmap[oy0:oy0 + oh, ox0:ox0 + ow], probs)
    return heatmap


def find_peaks(heatmap, threshold, min_distance):
    """Greedy NMS over candidate pixels above threshold -- at native heatmap resolution,
    not constrained to a stride-spaced grid like the classifier's sliding window."""
    ys, xs = np.nonzero(heatmap >= threshold)
    if len(ys) == 0:
        return []
    scores = heatmap[ys, xs]
    order = np.argsort(-scores)
    kept, kept_yx = [], []
    for idx in order:
        y, x, s = int(ys[idx]), int(xs[idx]), float(scores[idx])
        if kept_yx:
            arr = np.array(kept_yx, dtype=np.float64)
            dist2 = np.sum((arr - np.array([y, x])) ** 2, axis=1)
            if dist2.min() < min_distance ** 2:
                continue
        kept.append((y, x, s))
        kept_yx.append((y, x))
    return kept


def load_heatmap_run(run_dir):
    with open(os.path.join(run_dir, 'config.json')) as f:
        config = json.load(f)
    model = HeatmapCNN(num_stages=config['args']['num_stages'])
    model.load_state_dict(torch.load(os.path.join(run_dir, 'model.pt'), map_location='cpu'))
    model.eval()
    mean = np.array(config['mean'], dtype=np.float32)
    std = np.array(config['std'], dtype=np.float32)
    return model, mean, std, config


def evaluate_heatmap_run(run_dir, threshold=0.5, min_distance_native=None,
                          classical_sensitivities=(30, 50, 70)):
    model, mean, std, config = load_heatmap_run(run_dir)
    output_stride = config['output_stride']
    split_meta = config['split_meta']
    args = config['args']

    if args['mode'] == 'pool':
        val_source = split_meta['val_sources'][0]
        gt_points = None  # filled below once we know the class(es) to compare against
        img = load_image_array(val_source['image_dir'], val_source['image'])
        pnt_data = load_pnt(val_source['pnt'])
        eval_image, eval_pnt, eval_dir = val_source['image'], val_source['pnt'], val_source['image_dir']
        x_range = None
    else:
        eval_pnt, eval_dir, eval_image = args['pnt'], args['image_dir'], args['image_name']
        pnt_data = load_pnt(eval_pnt)
        img = load_image_array(eval_dir, eval_image)
        x_range = split_meta['val_x_range']

    # ground truth: all classes annotated on the eval image within (optionally) the val x_range
    from ml_experiments.pnt_io import all_points_with_class_for_image
    all_pts = all_points_with_class_for_image(pnt_data, eval_image)
    if x_range is not None:
        gt_points = [(x, y) for x, y, _ in all_pts if x_range[0] <= x < x_range[1]]
    else:
        gt_points = [(x, y) for x, y, _ in all_pts]

    if min_distance_native is None:
        min_distance_native = max(8, output_stride * 2)
    min_distance_target = min_distance_native / output_stride

    region_img = img
    offset_x = 0
    if x_range is not None:
        x0 = max(0, int(x_range[0]))
        x1 = min(img.shape[1], int(np.ceil(x_range[1])))
        region_img = img[:, x0:x1]
        offset_x = x0

    heatmap = predict_heatmap(model, region_img, mean, std, output_stride)
    peaks = find_peaks(heatmap, threshold, min_distance_target)
    pred_points = [(x * output_stride + offset_x, y * output_stride) for y, x, _ in peaks]

    match_radius = median_nn_distance(gt_points) / 2 if len(gt_points) >= 2 else 10.0
    heatmap_metrics = match_points(pred_points, gt_points, match_radius)
    heatmap_metrics.update({'match_radius': match_radius, 'threshold': threshold,
                             'min_distance_native': min_distance_native, 'n_gt': len(gt_points),
                             'n_pred': len(pred_points)})

    # classical baseline, same region + same match_radius
    from PyQt6 import QtCore
    detector = ClassicCVDetector()
    region = None
    if x_range is not None:
        region = QtCore.QRectF(x_range[0], 0, x_range[1] - x_range[0], img.shape[0])
    best_f1, best_metrics = -1.0, None
    for polarity in ('bright', 'dark'):
        for sensitivity in classical_sensitivities:
            result = detector.detect(img, region=region, sensitivity=sensitivity, polarity=polarity)
            pred_xy = [(p.x(), p.y()) for p in result.points]
            m = match_points(pred_xy, gt_points, match_radius)
            if m['f1'] > best_f1:
                best_f1, best_metrics = m['f1'], {**m, 'sensitivity': sensitivity, 'polarity': polarity}

    return {
        'run_dir': run_dir,
        'bucket': config.get('bucket'),
        'eval_image': eval_image,
        'pooled_classes': config.get('pooled_classes'),
        'heatmap_point_level': heatmap_metrics,
        'classical_point_level': best_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    results = evaluate_heatmap_run(args.run_dir, threshold=args.threshold)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
