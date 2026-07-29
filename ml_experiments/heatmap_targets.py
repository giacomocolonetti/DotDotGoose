"""Gaussian-blob target heatmap rendering for the density/heatmap-regression detector.

Rather than classifying whether a fixed-size patch is centered on a bird (the
ml_experiments/model.py approach, whose sliding-window+NMS conversion is capped at
roughly stride/NMS-radius localization resolution -- see runs/report.md), this predicts a
per-pixel "birdness" map directly and finds local peaks in it. Peaks can land anywhere at
the map's native resolution, not just at grid points spaced `stride` pixels apart.
"""
import numpy as np


def render_gaussian_heatmap(shape, points_xy, sigma):
    """shape: (h, w) of the *target* map (already at output resolution -- points_xy must
    already be in that same coordinate space, i.e. pre-divided by output_stride).
    Overlapping Gaussians take the max, not the sum, so nearby birds don't create an
    artificially brighter blob that biases peak-finding."""
    h, w = shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    if not points_xy:
        return heatmap
    radius = max(1, int(np.ceil(3 * sigma)))
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)).astype(np.float32)
    ksize = 2 * radius + 1

    for x, y in points_xy:
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        kx0, kx1 = x0 - (cx - radius), ksize - ((cx + radius + 1) - x1)
        ky0, ky1 = y0 - (cy - radius), ksize - ((cy + radius + 1) - y1)
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], kernel[ky0:ky1, kx0:kx1])
    return heatmap
