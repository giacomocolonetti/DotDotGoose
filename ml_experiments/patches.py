"""Positive/negative patch extraction for the ML detector experiment.

Positive-patch cropping mirrors ddg/exporter.py's Exporter.run() crop-and-zero-pad
convention exactly, so results are consistent with what the app's existing
"Export > Chips" feature already produces.
"""
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ml_experiments.pnt_io import (all_points_for_image, all_points_with_class_for_image,
                                    load_pnt, points_for_image)

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 1000000000


@dataclass
class PatchRecord:
    image_dir: str
    image_name: str
    x: float
    y: float
    label: int  # 1 positive, 0 negative
    class_name: str = None


def load_image_array(image_dir, image_name):
    path = f'{image_dir}/{image_name}'
    with Image.open(path) as img:
        return np.array(img.convert('RGB'))


def extract_patch(img, x, y, patch_size):
    """Crop a fixed-size chip centered on (x, y), zero-padding at image edges.
    Mirrors ddg/exporter.py Exporter.run()'s crop formula exactly."""
    half = patch_size // 2
    x0 = max(0, int(x) - half)
    y0 = max(0, int(y) - half)
    x1 = min((int(x) - half) + 2 * half, img.shape[1])
    y1 = min((int(y) - half) + 2 * half, img.shape[0])
    window = img[y0:y1, x0:x1]
    chip = np.zeros((2 * half, 2 * half, img.shape[2]), dtype=img.dtype)
    chip[0:window.shape[0], 0:window.shape[1]] = window
    return chip


def sample_negative_centers(img_shape, exclude_points_xy, n, patch_size, exclusion_radius,
                             rng, x_range=None, max_attempts_factor=20):
    half = patch_size // 2
    h, w = img_shape[0], img_shape[1]
    x_lo, x_hi = (half, w - half) if x_range is None else x_range
    y_lo, y_hi = half, h - half
    if x_hi <= x_lo or y_hi <= y_lo:
        return []

    exclude = np.array(exclude_points_xy, dtype=np.float64) if len(exclude_points_xy) else np.zeros((0, 2))
    centers = []
    max_attempts = n * max_attempts_factor
    attempts = 0
    while len(centers) < n and attempts < max_attempts:
        batch = min(n - len(centers), 256)
        xs = rng.uniform(x_lo, x_hi, size=batch)
        ys = rng.uniform(y_lo, y_hi, size=batch)
        candidates = np.stack([xs, ys], axis=1)
        attempts += batch

        if len(exclude) > 0:
            diff = candidates[:, None, :] - exclude[None, :, :]
            dist2 = np.sum(diff ** 2, axis=2)
            ok = dist2.min(axis=1) > (exclusion_radius ** 2)
        else:
            ok = np.ones(batch, dtype=bool)

        for cx, cy in candidates[ok]:
            centers.append((cx, cy))
            if len(centers) >= n:
                break

    if len(centers) < n:
        logger.warning('sample_negative_centers: requested %d, only found %d after %d attempts',
                        n, len(centers), attempts)
    return centers


def extract_dataset(pnt_data, image_dir, image_name, class_name, patch_size, neg_ratio,
                     rng_seed, exclusion_radius=None):
    """Build PatchRecord list (metadata only, no pixel data materialized) for one image:
    all annotated points as positives, plus sampled negatives. class_name=None pools every
    annotated class on this image as a single generic "bird" positive set -- this is what
    makes a detector species-agnostic instead of tied to one species' checkpoint."""
    if exclusion_radius is None:
        exclusion_radius = patch_size // 2

    if class_name is None:
        # Pool every class on this image as generic "bird" positives, but keep each
        # point's own species tag around (for pooled_classes() reporting) rather than
        # collapsing it to None.
        positives = all_points_with_class_for_image(pnt_data, image_name)
    else:
        positives = [(x, y, class_name) for x, y in points_for_image(pnt_data, image_name, class_name)]
    all_points_xy = all_points_for_image(pnt_data, image_name)

    img = load_image_array(image_dir, image_name)
    rng = np.random.default_rng(rng_seed)
    n_negatives = len(positives) * neg_ratio
    negative_centers = sample_negative_centers(img.shape, all_points_xy, n_negatives,
                                                patch_size, exclusion_radius, rng)

    records = []
    for x, y, point_class in positives:
        records.append(PatchRecord(image_dir=image_dir, image_name=image_name, x=x, y=y,
                                    label=1, class_name=point_class))
    for x, y in negative_centers:
        records.append(PatchRecord(image_dir=image_dir, image_name=image_name, x=x, y=y,
                                    label=0, class_name=None))
    return records


def extract_pooled_dataset(sources, patch_size, neg_ratio, rng_seed, exclusion_radius=None):
    """Like extract_dataset, but pooled across multiple (possibly cross-folder) sources,
    each contributing every annotated class on its image as generic "bird" positives.
    `sources` is a list of dicts: {'pnt': path, 'image_dir': path, 'image': image_name}.
    Used to build a single species-agnostic, size-bucketed detector from several photos."""
    all_records = []
    for i, source in enumerate(sources):
        pnt_data = load_pnt(source['pnt'])
        records = extract_dataset(pnt_data, source['image_dir'], source['image'], None,
                                   patch_size, neg_ratio, rng_seed + i, exclusion_radius)
        all_records.extend(records)
    return all_records
