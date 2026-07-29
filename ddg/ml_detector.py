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
"""Experimental ML detector: wraps a patch classifier trained by ml_experiments/train.py
(see ml_experiments/runs/report.md for what it can/can't do yet) behind the same
PointDetector interface as ClassicCVDetector, so it's a drop-in alternative.

Unlike ClassicCVDetector, each trained checkpoint is bound to one species (the class it
was trained on), so detect() needs `class_name` to pick the right one -- this is why
PointDetector.detect() has an (optional, unused-by-Classical) class_name parameter.

This module intentionally imports torch and ml_experiments lazily (inside methods, not at
module load time) so the rest of the app never needs torch installed unless a caller
actually picks the ML detector. Import this module itself is cheap; constructing
MLPointDetector and scanning for available checkpoints is cheap; only detect() pulls in
torch/ml_experiments, and only for a class that has a matching trained checkpoint.
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
        self._loaded = {}      # run_dir -> (model, mean, std, patch_size)

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
                if os.path.isfile(config_path):
                    with open(config_path) as f:
                        config = json.load(f)
                    class_name = config['args']['class_name']
                    registry[class_name] = os.path.join(self.runs_dir, entry)
        self._registry = registry

    def _load(self, run_dir):
        if run_dir in self._loaded:
            return self._loaded[run_dir]
        try:
            import torch
            from ml_experiments.model import SmallPatchCNN
        except ImportError as e:
            raise ModelUnavailableError(
                'ML detector requires torch and the ml_experiments package '
                '(pip install torch; only present on the ml-detector-experiment branch).') from e

        with open(os.path.join(run_dir, 'config.json')) as f:
            config = json.load(f)
        model = SmallPatchCNN()
        model.load_state_dict(torch.load(os.path.join(run_dir, 'model.pt'), map_location='cpu'))
        model.eval()
        mean = np.array(config['mean'], dtype=np.float32)
        std = np.array(config['std'], dtype=np.float32)
        patch_size = config['args']['patch_size']
        self._loaded[run_dir] = (model, mean, std, patch_size)
        return self._loaded[run_dir]

    def detect(self, image_array, region=None, sensitivity=50, polarity='bright',
               existing_points=None, dedup_radius=None, class_name=None):
        # polarity is unused: the ML classifier learns its own appearance model rather than
        # a bright/dark heuristic.
        self._build_registry()
        if class_name not in self._registry:
            available = ', '.join(self.available_classes()) or '(none trained)'
            raise ModelUnavailableError(
                "No trained ML model for class '{}'. Available: {}".format(class_name, available))
        model, mean, std, patch_size = self._load(self._registry[class_name])

        existing_points = existing_points or []
        # Never dedup less aggressively than the model's own spatial resolution (half the
        # patch size): the UI's configurable point radius defaults to 25px, well below the
        # ~48px granularity a stride/NMS sliding-window pipeline actually resolves at, which
        # would otherwise flood the result with near-duplicate detections of the same bird.
        dedup_radius = max(dedup_radius or 0, patch_size // 2)

        offset_x, offset_y = 0, 0
        array = image_array
        if region is not None:
            x0 = max(0, int(region.left()))
            y0 = max(0, int(region.top()))
            x1 = min(image_array.shape[1], int(np.ceil(region.right())))
            y1 = min(image_array.shape[0], int(np.ceil(region.bottom())))
            array = image_array[y0:y1, x0:x1]
            offset_x, offset_y = x0, y0

        if array.size == 0 or array.shape[0] < patch_size or array.shape[1] < patch_size:
            return DetectionResult([], meta={'count': 0})

        # sensitivity (0-100, shared UI control with ClassicCVDetector) maps to a confidence
        # threshold: higher sensitivity -> lower threshold -> more detections.
        sensitivity = max(0, min(100, int(sensitivity)))
        score_threshold = max(0.01, 1.0 - sensitivity / 100.0)
        stride = max(8, patch_size // 3)

        detections = self._sliding_window_detect(model, array, mean, std, patch_size, stride, score_threshold)
        kept = self._nms(detections, dedup_radius)
        candidates_yx = np.array([[y, x] for x, y, _ in kept], dtype=np.float64) if kept else np.zeros((0, 2))
        candidates_yx = dedupe_points(candidates_yx, existing_points, dedup_radius)

        points = [QtCore.QPointF(x + offset_x, y + offset_y) for y, x in candidates_yx]
        return DetectionResult(points, meta={'count': len(points)})

    def _sliding_window_detect(self, model, img, mean, std, patch_size, stride, score_threshold, batch_size=256):
        import torch
        from ml_experiments.patches import extract_patch

        h, w = img.shape[0], img.shape[1]
        half = patch_size // 2
        xs = list(range(half, w - half, stride)) or [w // 2]
        ys = list(range(half, h - half, stride)) or [h // 2]
        centers = [(x, y) for y in ys for x in xs]

        detections = []
        with torch.no_grad():
            for i in range(0, len(centers), batch_size):
                batch_centers = centers[i:i + batch_size]
                chips = np.stack([extract_patch(img, cx, cy, patch_size).astype(np.float32)
                                   for cx, cy in batch_centers])
                chips = (chips - mean) / std
                chips = np.ascontiguousarray(np.transpose(chips, (0, 3, 1, 2)))
                tensor = torch.from_numpy(chips)
                probs = torch.softmax(model(tensor), dim=1)[:, 1].numpy()
                for (cx, cy), score in zip(batch_centers, probs):
                    if score >= score_threshold:
                        detections.append((float(cx), float(cy), float(score)))
        return detections

    def _nms(self, detections, dedup_radius):
        """Greedy NMS: keep the highest-scoring detection, drop others within dedup_radius."""
        if not detections:
            return []
        ordered = sorted(detections, key=lambda d: -d[2])
        kept, kept_xy = [], []
        for x, y, score in ordered:
            if kept_xy:
                arr = np.array(kept_xy)
                dist2 = np.sum((arr - np.array([x, y])) ** 2, axis=1)
                if dist2.min() <= dedup_radius ** 2:
                    continue
            kept.append((x, y, score))
            kept_xy.append((x, y))
        return kept
