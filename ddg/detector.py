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
import numpy as np

from PyQt6 import QtCore


class DetectionResult:
    def __init__(self, points, meta=None):
        self.points = points
        self.meta = meta or {}


def dedupe_points(candidates_yx, existing_points, radius):
    """Remove candidate (y, x) rows that fall within radius of an already placed QPointF."""
    if len(candidates_yx) == 0 or not existing_points:
        return candidates_yx
    existing = np.array([[p.y(), p.x()] for p in existing_points], dtype=np.float64)
    diff = candidates_yx[:, None, :] - existing[None, :, :]
    dist2 = np.sum(diff ** 2, axis=2)
    keep = dist2.min(axis=1) > (radius * radius)
    return candidates_yx[keep]


class PointDetector:
    """Common interface for automatic point placement. Implementations take plain numpy
    arrays/Qt value types (no Canvas reference) so they stay swappable and testable in
    isolation. A future ML-based detector implements the same signature."""

    def detect(self, image_array, region=None, sensitivity=50, polarity='bright',
               existing_points=None, dedup_radius=None):
        raise NotImplementedError


class ClassicCVDetector(PointDetector):
    """Dependency-free blob-centroid detector: grayscale -> box blur -> Otsu threshold ->
    connected components -> centroids -> de-duplication. Suited to well-contrasted subjects
    (e.g. birds on water/mud/snow); not a substitute for a trained model on cluttered scenes."""

    STRIP_HEIGHT = 2000

    def detect(self, image_array, region=None, sensitivity=50, polarity='bright',
               existing_points=None, dedup_radius=None):
        existing_points = existing_points or []
        if dedup_radius is None:
            dedup_radius = 12.0

        offset_x, offset_y = 0, 0
        array = image_array
        if region is not None:
            x0 = max(0, int(region.left()))
            y0 = max(0, int(region.top()))
            x1 = min(image_array.shape[1], int(np.ceil(region.right())))
            y1 = min(image_array.shape[0], int(np.ceil(region.bottom())))
            array = image_array[y0:y1, x0:x1]
            offset_x, offset_y = x0, y0

        if array.size == 0 or array.shape[0] == 0 or array.shape[1] == 0:
            return DetectionResult([], meta={'count': 0})

        blur_radius, min_area, max_area, threshold_bias = self._map_sensitivity(sensitivity)
        candidates_yx = self._detect_chunked(array, blur_radius, polarity, threshold_bias, min_area, max_area)
        candidates_yx = self._dedupe_self(candidates_yx, dedup_radius)
        candidates_yx = dedupe_points(candidates_yx, existing_points, dedup_radius)

        points = [QtCore.QPointF(x + offset_x, y + offset_y) for y, x in candidates_yx]
        return DetectionResult(points, meta={'count': len(points)})

    def _detect_chunked(self, array, blur_radius, polarity, threshold_bias, min_area, max_area):
        h = array.shape[0]
        if h <= self.STRIP_HEIGHT:
            return self._detect_in_block(array, blur_radius, polarity, threshold_bias, min_area, max_area)

        overlap = max(2 * blur_radius, 8)
        chunks = []
        for y_start in range(0, h, self.STRIP_HEIGHT):
            y_end = min(h, y_start + self.STRIP_HEIGHT)
            pad_top = min(overlap, y_start)
            pad_bottom = min(overlap, h - y_end)
            block = array[y_start - pad_top:y_end + pad_bottom]
            block_points = self._detect_in_block(block, blur_radius, polarity, threshold_bias, min_area, max_area)
            if len(block_points):
                block_points = block_points.copy()
                block_points[:, 0] += (y_start - pad_top)
                chunks.append(block_points)
        if not chunks:
            return np.zeros((0, 2))
        return np.concatenate(chunks, axis=0)

    def _detect_in_block(self, array, blur_radius, polarity, threshold_bias, min_area, max_area):
        gray = self._to_grayscale(array)
        blurred = self._box_blur(gray, blur_radius)
        threshold = self._otsu_threshold(blurred) + threshold_bias
        mask = self._binarize(blurred, threshold, polarity)
        labels = self._label_components(mask)
        return self._extract_points(labels, min_area, max_area)

    def _map_sensitivity(self, sensitivity):
        sensitivity = max(0, min(100, int(sensitivity)))
        blur_radius = max(1, int(round(3 + (100 - sensitivity) / 20)))
        min_area = max(4, int(round((100 - sensitivity) * 0.8 + 4)))
        max_area = min_area * 50
        threshold_bias = (sensitivity - 50) / 50 * 20
        return blur_radius, min_area, max_area, threshold_bias

    def _to_grayscale(self, array):
        if array.ndim == 2:
            return array.astype(np.float64)
        channels = array.shape[2]
        if channels == 1:
            return array[:, :, 0].astype(np.float64)
        r = array[:, :, 0].astype(np.float64)
        g = array[:, :, 1].astype(np.float64)
        b = array[:, :, 2].astype(np.float64)
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _box_blur(self, gray, radius):
        radius = int(max(0, radius))
        if radius == 0:
            return gray.astype(np.float64)
        h, w = gray.shape
        padded = np.pad(gray, radius, mode='edge')
        integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
        sat = np.zeros((integral.shape[0] + 1, integral.shape[1] + 1), dtype=np.float64)
        sat[1:, 1:] = integral
        size = 2 * radius + 1
        term1 = sat[size:size + h, size:size + w]
        term2 = sat[0:h, size:size + w]
        term3 = sat[size:size + h, 0:w]
        term4 = sat[0:h, 0:w]
        return (term1 - term2 - term3 + term4) / (size * size)

    def _otsu_threshold(self, gray):
        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
        total = hist.sum()
        if total == 0:
            return 128.0
        levels = np.arange(256)
        sum_all = float(np.dot(levels, hist))
        sum_bg = 0.0
        weight_bg = 0.0
        best_variance = -1.0
        threshold = 0
        for level in range(256):
            weight_bg += hist[level]
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            sum_bg += level * hist[level]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_all - sum_bg) / weight_fg
            variance_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if variance_between > best_variance:
                best_variance = variance_between
                threshold = level
        return float(threshold)

    def _binarize(self, gray, threshold, polarity):
        if polarity == 'dark':
            return gray < threshold
        return gray > threshold

    def _label_components(self, mask):
        """8-connectivity labeling via iterative min-neighbor propagation. Converges in a
        number of passes proportional to blob diameter (not image size), which is fine for
        small, point-like targets; a hard iteration cap guards against a pathological single
        giant component (e.g. a failed threshold) spinning forever -- oversized components are
        rejected by the area filter regardless."""
        h, w = mask.shape
        labels = np.where(mask, np.arange(1, h * w + 1).reshape(h, w), 0).astype(np.int64)
        if not mask.any():
            return labels

        max_iterations = min(500, h + w)
        max_value = np.iinfo(np.int64).max
        for _ in range(max_iterations):
            candidates = [labels]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    shifted = np.zeros_like(labels)
                    src_y0, src_y1 = max(0, dy), h + min(0, dy)
                    src_x0, src_x1 = max(0, dx), w + min(0, dx)
                    dst_y0, dst_y1 = max(0, -dy), h + min(0, -dy)
                    dst_x0, dst_x1 = max(0, -dx), w + min(0, -dx)
                    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = labels[src_y0:src_y1, src_x0:src_x1]
                    candidates.append(np.where(shifted > 0, shifted, max_value))
            new_labels = np.minimum.reduce(candidates)
            new_labels = np.where(mask, new_labels, 0)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
        return labels

    def _component_stats(self, labels):
        mask = labels > 0
        if not mask.any():
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), np.zeros((0, 2))
        ys, xs = np.nonzero(mask)
        flat_labels = labels[mask]
        unique_labels, inverse = np.unique(flat_labels, return_inverse=True)
        counts = np.bincount(inverse)
        sum_y = np.bincount(inverse, weights=ys.astype(np.float64))
        sum_x = np.bincount(inverse, weights=xs.astype(np.float64))
        centroids = np.stack([sum_y / counts, sum_x / counts], axis=1)
        return unique_labels, counts, centroids

    def _extract_points(self, labels, min_area, max_area):
        _, counts, centroids = self._component_stats(labels)
        if len(counts) == 0:
            return np.zeros((0, 2))
        keep = (counts >= min_area) & (counts <= max_area)
        return centroids[keep]

    def _dedupe_self(self, points_yx, radius):
        if len(points_yx) <= 1:
            return points_yx
        keep_mask = np.ones(len(points_yx), dtype=bool)
        for i in range(len(points_yx)):
            if not keep_mask[i]:
                continue
            diff = points_yx[i + 1:] - points_yx[i]
            dist2 = np.sum(diff ** 2, axis=1)
            close = dist2 <= (radius * radius)
            keep_mask[i + 1:][close] = False
        return points_yx[keep_mask]
