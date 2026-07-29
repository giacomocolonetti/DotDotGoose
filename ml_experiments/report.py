#!/usr/bin/env python3
"""Assemble evaluate_run() results for one or more runs into a markdown report with
metrics tables and a few example true/false positive/negative crops for qualitative
inspection."""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_experiments.evaluate import evaluate_run
from ml_experiments.patches import extract_patch, load_image_array


def _radius_table(cnn_by_radius, classical_by_radius, radii):
    lines = ['| radius (px) | CNN P | CNN R | CNN F1 | Classical P | Classical R | Classical F1 |',
             '|---|---|---|---|---|---|---|']
    for r in radii:
        c = cnn_by_radius[r]
        b = classical_by_radius[r]
        lines.append(f"| {r} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} "
                      f"| {b['precision']:.3f} | {b['recall']:.3f} | {b['f1']:.3f} |")
    return '\n'.join(lines)


def write_report(run_dirs, out_path, caveats=None):
    sections = []
    for run_dir, label in run_dirs:
        results = evaluate_run(run_dir)
        radii = results['radii']
        patch = results['cnn_patch_level']
        section = [
            f"## {label}",
            '',
            f"Eval image: `{results['eval_image']}`, class `{results['class_name']}`, "
            f"n_gt={results['n_gt']}",
            '',
            f"CNN patch-level (on held-out patches): precision={patch['precision']:.3f} "
            f"recall={patch['recall']:.3f} F1={patch['f1']:.3f} ROC-AUC={patch['roc_auc']:.3f}",
            '',
            "Point-level detection at three tolerances (radius = how close a predicted point "
            "must be to a ground-truth point to count as found): "
            f"precise (~half median inter-bird spacing), stride ({results['stride']}px, the "
            f"sliding-window grid spacing), dedup ({results['dedup_radius']}px, the NMS "
            "collapse radius -- an upper bound on what this pipeline can resolve at all).",
            '',
            _radius_table(results['cnn_point_level_by_radius'], results['classical_point_level_by_radius'], radii),
            '',
            f"Classical baseline's best operating point: sensitivity={results['classical_params']['sensitivity']}, "
            f"polarity={results['classical_params']['polarity']}.",
        ]
        if caveats and label in caveats:
            section += ['', f"**Caveat:** {caveats[label]}"]
        sections.append('\n'.join(section))

    report = '\n\n'.join(sections)
    with open(out_path, 'w') as f:
        f.write(report)
    print(f'wrote {out_path}')
    return report


def save_example_crops(run_dir, out_dir, patch_size=96, n_examples=3):
    """Save a few true-positive / false-positive / false-negative crops for a run, for a
    qualitative sanity check alongside the metrics table."""
    import torch
    from ml_experiments.evaluate import nms_points, sliding_window_detect
    from ml_experiments.model import SmallPatchCNN
    from ml_experiments.pnt_io import load_pnt, points_for_image

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json')) as f:
        config = json.load(f)
    args = config['args']
    mean = np.array(config['mean'], dtype=np.float32)
    std = np.array(config['std'], dtype=np.float32)
    model = SmallPatchCNN()
    model.load_state_dict(torch.load(os.path.join(run_dir, 'model.pt'), map_location='cpu'))
    model.eval()

    pnt_data = load_pnt(args['pnt'])
    eval_image = args['eval_image'] if args['pair'] == 'a' else args['image_name']
    gt_points = np.array(points_for_image(pnt_data, eval_image, args['class_name']))
    img = load_image_array(args['image_dir'], eval_image)

    dets = sliding_window_detect(model, img, mean, std, patch_size=patch_size, stride=32, score_threshold=0.5)
    pred = np.array([(x, y) for x, y, _ in nms_points(dets, dedup_radius=48)]) if dets else np.zeros((0, 2))

    def nearest_dist(a, b):
        if len(a) == 0 or len(b) == 0:
            return np.full(len(a), np.inf)
        diff = a[:, None, :] - b[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)

    tp_mask = nearest_dist(pred, gt_points) <= 32 if len(pred) else np.array([])
    fp_mask = ~tp_mask if len(pred) else np.array([])
    fn_mask = nearest_dist(gt_points, pred) > 32 if len(gt_points) else np.array([])

    def save(points, mask, name):
        pts = points[mask] if len(points) else points
        for i, (x, y) in enumerate(pts[:n_examples]):
            chip = extract_patch(img, x, y, patch_size)
            Image.fromarray(chip).save(os.path.join(out_dir, f'{name}_{i}.png'))

    save(pred, tp_mask, 'true_positive')
    save(pred, fp_mask, 'false_positive')
    save(gt_points, fn_mask, 'false_negative')
    print(f'saved example crops to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', action='append', required=True, dest='run_dirs')
    parser.add_argument('--label', action='append', required=True, dest='labels')
    parser.add_argument('--out', required=True)
    parsed = parser.parse_args()
    write_report(list(zip(parsed.run_dirs, parsed.labels)), parsed.out)
