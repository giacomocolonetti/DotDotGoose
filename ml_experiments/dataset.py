"""torch Dataset wrapping patches.py's PatchRecord list, with lazy per-record cropping,
hand-rolled augmentation (no torchvision dependency), and dataset-specific normalization."""
import numpy as np
import torch
from torch.utils.data import Dataset

from ml_experiments.patches import extract_patch, load_image_array


def compute_mean_std(records, patch_size):
    """Per-channel mean/std over all patches in `records`, used to normalize instead of
    ImageNet stats (not a natural-scene domain). Each record carries its own image_dir, so
    this works across pooled records from different source folders."""
    cache_key, cache_array = None, None
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    count = 0
    for r in records:
        key = (r.image_dir, r.image_name)
        if cache_key != key:
            cache_array = load_image_array(r.image_dir, r.image_name)
            cache_key = key
        chip = extract_patch(cache_array, r.x, r.y, patch_size).astype(np.float64)
        pixel_sum += chip.sum(axis=(0, 1))
        pixel_sq_sum += (chip ** 2).sum(axis=(0, 1))
        count += chip.shape[0] * chip.shape[1]
    mean = pixel_sum / count
    var = np.maximum(pixel_sq_sum / count - mean ** 2, 1e-6)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


class PatchDataset(Dataset):
    def __init__(self, records, patch_size, augment, mean, std, rng_seed=0, jitter_px=0):
        """jitter_px: max +/- pixel offset applied to the crop center during training.
        A sliding-window detector at inference time almost never lands exactly on an
        object's true center (it samples a fixed grid), so a classifier trained only on
        exactly-centered positive crops learns to fire only for near-perfect centering and
        fails as a detector. Jittering the crop center during training teaches the model
        "object somewhere in this patch" instead, matching the offsets a grid search will
        actually present."""
        self.records = records
        self.patch_size = patch_size
        self.augment = augment
        self.mean = mean
        self.std = std
        self.jitter_px = jitter_px
        self.rng = np.random.default_rng(rng_seed)
        self._cache_key = None
        self._cache_array = None

    def __len__(self):
        return len(self.records)

    def _get_image(self, image_dir, image_name):
        key = (image_dir, image_name)
        if self._cache_key != key:
            self._cache_array = load_image_array(image_dir, image_name)
            self._cache_key = key
        return self._cache_array

    def _apply_augmentation(self, chip):
        if self.rng.random() < 0.5:
            chip = chip[:, ::-1, :]
        if self.rng.random() < 0.5:
            chip = chip[::-1, :, :]
        k = int(self.rng.integers(0, 4))
        if k:
            chip = np.rot90(chip, k, axes=(0, 1))
        contrast = self.rng.uniform(0.8, 1.2)
        brightness = self.rng.uniform(-20, 20)
        chip = np.clip(chip * contrast + brightness, 0, 255)
        return chip

    def __getitem__(self, idx):
        record = self.records[idx]
        img = self._get_image(record.image_dir, record.image_name)
        x, y = record.x, record.y
        if self.augment and self.jitter_px > 0:
            x = x + self.rng.uniform(-self.jitter_px, self.jitter_px)
            y = y + self.rng.uniform(-self.jitter_px, self.jitter_px)
        chip = extract_patch(img, x, y, self.patch_size).astype(np.float32)
        if self.augment:
            chip = self._apply_augmentation(chip)
        chip = (chip - self.mean) / self.std
        chip = np.ascontiguousarray(np.transpose(chip, (2, 0, 1)))  # HWC -> CHW
        return torch.from_numpy(chip), record.label
