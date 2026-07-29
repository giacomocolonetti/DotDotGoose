"""Tile-based Dataset for heatmap-regression training: samples random square tiles from
(preloaded) source images and renders Gaussian-blob targets on the fly, instead of
extracting fixed small patches with a binary label like ml_experiments/dataset.py."""
import numpy as np
import torch
from torch.utils.data import Dataset

from ml_experiments.heatmap_targets import render_gaussian_heatmap
from ml_experiments.patches import load_image_array
from ml_experiments.pnt_io import all_points_for_image, load_pnt


def compute_mean_std_from_images(images):
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    count = 0
    for img in images:
        arr = img.astype(np.float64)
        pixel_sum += arr.sum(axis=(0, 1))
        pixel_sq_sum += (arr ** 2).sum(axis=(0, 1))
        count += arr.shape[0] * arr.shape[1]
    mean = pixel_sum / count
    var = np.maximum(pixel_sq_sum / count - mean ** 2, 1e-6)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def crop_tile(img, cx, cy, tile_size):
    """Like patches.extract_patch but square tile_size x tile_size, zero-padded at image
    edges, also returning the tile's image-space origin so point coordinates can be
    translated into tile-local space."""
    half = tile_size // 2
    origin_x = int(cx) - half
    origin_y = int(cy) - half
    x0, y0 = max(0, origin_x), max(0, origin_y)
    x1 = min(origin_x + tile_size, img.shape[1])
    y1 = min(origin_y + tile_size, img.shape[0])
    window = img[y0:y1, x0:x1]
    tile = np.zeros((tile_size, tile_size, img.shape[2]), dtype=img.dtype)
    tile[y0 - origin_y:y1 - origin_y, x0 - origin_x:x1 - origin_x] = window
    return tile, origin_x, origin_y


class HeatmapTileDataset(Dataset):
    def __init__(self, sources, tile_size, sigma, output_stride, samples_per_epoch,
                 bg_fraction=0.2, augment=True, mean=None, std=None, rng_seed=0, x_ranges=None):
        """x_ranges: optional list (parallel to `sources`) of (lo, hi) tuples restricting
        where tile centers may be sampled from that source -- used for a same-image
        train/val spatial split (one source, two dataset instances with disjoint ranges,
        each with at least tile_size/2 margin from the split boundary so tiles never
        straddle it) when only one photo exists for a bucket."""
        self.tile_size = tile_size
        self.sigma = sigma
        self.output_stride = output_stride
        self.samples_per_epoch = samples_per_epoch
        self.bg_fraction = bg_fraction
        self.augment = augment
        self.mean = mean
        self.std = std
        self.rng = np.random.default_rng(rng_seed)
        self.x_ranges = x_ranges or [None] * len(sources)

        self.images = []
        self.points = []  # list of (n, 2) float arrays, one per source image
        for source, x_range in zip(sources, self.x_ranges):
            pnt_data = load_pnt(source['pnt'])
            img = load_image_array(source['image_dir'], source['image'])
            pts = np.array(all_points_for_image(pnt_data, source['image']), dtype=np.float64)
            if pts.size == 0:
                pts = pts.reshape(0, 2)
            elif x_range is not None:
                pts = pts[(pts[:, 0] >= x_range[0]) & (pts[:, 0] < x_range[1])]
            self.images.append(img)
            self.points.append(pts)

    def __len__(self):
        return self.samples_per_epoch

    def _sample_center(self):
        src_idx = int(self.rng.integers(0, len(self.images)))
        img = self.images[src_idx]
        pts = self.points[src_idx]
        x_range = self.x_ranges[src_idx]
        h, w = img.shape[0], img.shape[1]
        x_lo, x_hi = (0, w) if x_range is None else x_range
        if len(pts) > 0 and self.rng.random() > self.bg_fraction:
            p = pts[self.rng.integers(0, len(pts))]
            jitter = self.tile_size / 4
            cx = p[0] + self.rng.uniform(-jitter, jitter)
            cy = p[1] + self.rng.uniform(-jitter, jitter)
        else:
            cx = self.rng.uniform(x_lo, x_hi)
            cy = self.rng.uniform(0, h)
        return src_idx, cx, cy

    def __getitem__(self, idx):
        src_idx, cx, cy = self._sample_center()
        img = self.images[src_idx]
        pts = self.points[src_idx]

        tile, origin_x, origin_y = crop_tile(img, cx, cy, self.tile_size)
        if len(pts) > 0:
            local = pts - np.array([origin_x, origin_y])
            mask = ((local[:, 0] >= 0) & (local[:, 0] < self.tile_size) &
                    (local[:, 1] >= 0) & (local[:, 1] < self.tile_size))
            local_points = local[mask]
        else:
            local_points = np.zeros((0, 2))

        tile = tile.astype(np.float32)
        target_size = self.tile_size // self.output_stride
        target_points = [(x / self.output_stride, y / self.output_stride) for x, y in local_points]

        if self.augment:
            # Coordinate transforms must match np.rot90/flip's *discrete* array indexing
            # exactly: (size - 1) - coord, not size - coord (verified empirically -- the
            # latter is off by one pixel at the target resolution).
            last = target_size - 1
            if self.rng.random() < 0.5:
                tile = tile[:, ::-1, :]
                target_points = [(last - x, y) for x, y in target_points]
            if self.rng.random() < 0.5:
                tile = tile[::-1, :, :]
                target_points = [(x, last - y) for x, y in target_points]
            k = int(self.rng.integers(0, 4))
            if k:
                tile = np.rot90(tile, k, axes=(0, 1))
                for _ in range(k):
                    target_points = [(y, last - x) for x, y in target_points]
            contrast = self.rng.uniform(0.8, 1.2)
            brightness = self.rng.uniform(-20, 20)
            tile = np.clip(tile * contrast + brightness, 0, 255)

        heatmap = render_gaussian_heatmap((target_size, target_size), target_points, self.sigma)

        tile = (tile - self.mean) / self.std
        tile = np.ascontiguousarray(np.transpose(tile, (2, 0, 1)))
        return torch.from_numpy(tile), torch.from_numpy(heatmap).unsqueeze(0)
