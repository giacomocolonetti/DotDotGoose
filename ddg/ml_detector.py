# -*- coding: utf-8 -*-
#
# DotDotGoose
#
# --------------------------------------------------------------------------
#
# This file is part of the DotDotGoose application.
# DotDotGoose was forked from the Neural Network Image Classifier (Nenetic).
#
# DotDotGoose is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DotDotGoose is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with with this software.  If not, see <http://www.gnu.org/licenses/>.
#
# --------------------------------------------------------------------------
"""Experimental ML detector: wraps a heatmap-regression model trained by
ml_experiments/train_heatmap.py (see ml_experiments/runs/report.md and the size-bucket
follow-up notes for what it can/can't do) behind the same PointDetector interface as
ClassicCVDetector.

This predicts a per-pixel "birdness" map and finds local peaks in it, rather than
classifying fixed-size windows on a coarse grid (the earlier classifier approach in
ml_experiments/model.py) -- peaks can land anywhere at the map's native resolution, which
is what actually fixed the sliding-window/NMS localization ceiling diagnosed earlier.

Models are species-agnostic within a size "bucket" (small/large -- see
ml_experiments/bucket_configs/): each trained run records which species contributed to its
training pool (`pooled_classes` in config.json), and detect() routes a requested
class_name to whichever bucket's model was trained on it, rather than requiring an exact
per-species checkpoint. Unmatched classes raise ModelUnavailableError with the available
list, same contract as before.

torch and ml_experiments are imported lazily (inside methods, not at module load time) so
the rest of the app never needs them unless a caller actually picks the ML detector.
"""
import json
import os

import numpy as np
from PyQt6 import QtCore

from .detector import DetectionResult, PointDetector, dedupe_points

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUNS_DIR = os.path.join(REPO_ROOT, 'ml_experiments', 'runs')


class ModelUnavailableError(Exception):
    """Raised when detect() is asked for a class with no trained checkpoint, or when
    torch/ml_experiments aren't importable."""


class MLPointDetector(PointDetector):

    def __init__(self, runs_dir=DEFAULT_RUNS_DIR):
        self.runs_dir = runs_dir
        self._registry = None  # class_name -> run_dir, built lazily, no torch import needed
        self._loaded = {}      # run_dir -> (model, mean, std, output_stride)

    def available_classes(self):
        self._build_registry()
        return sorted(self._registry.keys())

    def _build_registry(self):
        if self._registry is not None:
            return
        registry = {}
        if os.path.isdir(self.runs_dir):
            for entry in sorted(os.listdir(self.runs_dir)):
                config_path = os.path.join(self.runs_dir, entry, 'config.json')
                if not os.path.isfile(config_path):
                    continue
                with open(config_path) as f:
                    config = json.load(f)
                if 'output_stride' not in config:
                    continue  # skip older/non-heatmap runs (e.g. superseded classifier checkpoints)
                for class_name in config.get('pooled_classes', []):
                    registry[class_name] = os.path.join(self.runs_dir, entry)
        self._registry = registry

    def _load(self, run_dir):
        if run_dir in self._loaded:
            return self._loaded[run_dir]
        try:
            import torch
            from ml_experiments.heatmap_model import HeatmapCNN
        except ImportError as e:
            raise ModelUnavailableError(
                'ML detector requires torch and the ml_experiments package '
                '(pip install torch; only present on the ml-detector-experiment branch).') from e

        with open(os.path.join(run_dir, 'config.json')) as f:
            config = json.load(f)
        model = HeatmapCNN(num_stages=config['args']['num_stages'])
        model.load_state_dict(torch.load(os.path.join(run_dir, 'model.pt'), map_location='cpu'))
        model.eval()
        mean = np.array(config['mean'], dtype=np.float32)
        std = np.array(config['std'], dtype=np.float32)
        output_stride = config['output_stride']
        self._loaded[run_dir] = (model, mean, std, output_stride)
        return self._loaded[run_dir]

    def detect(self, image_array, region=None, sensitivity=50, polarity='bright',
               existing_points=None, dedup_radius=None, class_name=None):
        # polarity is unused: the heatmap model learns its own appearance model rather
        # than a bright/dark heuristic.
        self._build_registry()
        if class_name not in self._registry:
            available = ', '.join(self.available_classes()) or '(none trained)'
            raise ModelUnavailableError(
                "No trained ML model for class '{}'. Available: {}".format(class_name, available))
        model, mean, std, output_stride = self._load(self._registry[class_name])

        existing_points = existing_points or []
        min_distance_native = max(8, output_stride * 2)
        # Never dedup less aggressively than the model's own spatial resolution, same
        # reasoning as the earlier classifier detector's dedup_radius floor.
        dedup_radius = max(dedup_radius or 0, min_distance_native)

        offset_x, offset_y = 0, 0
        array = image_array
        if region is not None:
            x0 = max(0, int(region.left()))
            y0 = max(0, int(region.top()))
            x1 = min(image_array.shape[1], int(np.ceil(region.right())))
            y1 = min(image_array.shape[0], int(np.ceil(region.bottom())))
            array = image_array[y0:y1, x0:x1]
            offset_x, offset_y = x0, y0

        if array.size == 0 or array.shape[0] < output_stride or array.shape[1] < output_stride:
            return DetectionResult([], meta={'count': 0})

        # sensitivity (0-100, shared UI control with ClassicCVDetector) maps to a peak
        # probability threshold: higher sensitivity -> lower threshold -> more detections.
        sensitivity = max(0, min(100, int(sensitivity)))
        threshold = max(0.01, 1.0 - sensitivity / 100.0)
        min_distance_target = min_distance_native / output_stride

        heatmap = self._predict_heatmap(model, array, mean, std, output_stride)
        peaks = self._find_peaks(heatmap, threshold, min_distance_target)
        candidates_yx = (np.array([[y * output_stride, x * output_stride] for y, x, _ in peaks], dtype=np.float64)
                         if peaks else np.zeros((0, 2)))
        candidates_yx = dedupe_points(candidates_yx, existing_points, dedup_radius)

        points = [QtCore.QPointF(x + offset_x, y + offset_y) for y, x in candidates_yx]
        return DetectionResult(points, meta={'count': len(points)})

    def _predict_heatmap(self, model, img, mean, std, output_stride, tile=1536, overlap=64):
        """Whole-image inference via large overlapping tiles (fully-convolutional model,
        so any input size works), stitched with a max-blend so overlap seams don't create
        duplicate/split peaks. Needed for the huge stitched-panorama images in this app's
        real datasets, which won't fit through the model in a single forward pass."""
        import torch

        h, w = img.shape[0], img.shape[1]
        out_h, out_w = h // output_stride, w // output_stride
        heatmap = np.zeros((out_h, out_w), dtype=np.float32)
        step = tile - overlap

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

    def _find_peaks(self, heatmap, threshold, min_distance):
        """Greedy NMS over candidate pixels above threshold, at the heatmap's native
        resolution -- not constrained to a stride-spaced grid like the classifier's
        sliding window."""
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
