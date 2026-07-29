"""Train/eval split strategies.

Pair A (cross-image): train on one photo, evaluate on a wholly separate photo of the
same species/session. This is the trustworthy generalization signal.

Pair B (same-image spatial holdout): split by the 80th percentile of the annotated
points' own x-coordinates (not "80% of image width" -- points may occupy only a narrow
band of the frame). Patches whose window straddles the boundary are dropped from both
sides so train/val never share overlapping pixels. Still optimistic vs Pair A since both
sides share the same photo's lighting/sensor-noise/background statistics.
"""
import numpy as np

from ml_experiments.patches import (PatchRecord, all_points_for_image, load_image_array,
                                     points_for_image, sample_negative_centers)


def pair_a_split(pnt_data, image_dir, train_image, eval_image, class_name, patch_size,
                  neg_ratio, rng_seed, exclusion_radius=None):
    from ml_experiments.patches import extract_dataset
    train_records = extract_dataset(pnt_data, image_dir, train_image, class_name,
                                     patch_size, neg_ratio, rng_seed, exclusion_radius)
    eval_records = extract_dataset(pnt_data, image_dir, eval_image, class_name,
                                    patch_size, neg_ratio, rng_seed + 1, exclusion_radius)
    return train_records, eval_records


def pair_b_split(pnt_data, image_dir, image_name, class_name, patch_size, neg_ratio,
                  rng_seed, train_frac=0.8, exclusion_radius=None):
    if exclusion_radius is None:
        exclusion_radius = patch_size // 2
    half = patch_size // 2

    positives_xy = points_for_image(pnt_data, image_name, class_name)
    all_points_xy = all_points_for_image(pnt_data, image_name)
    xs = np.array([p[0] for p in positives_xy], dtype=np.float64)
    boundary = float(np.percentile(xs, train_frac * 100))

    train_pos = [(x, y) for x, y in positives_xy if x + half < boundary]
    val_pos = [(x, y) for x, y in positives_xy if x - half >= boundary]
    dropped = len(positives_xy) - len(train_pos) - len(val_pos)

    img = load_image_array(image_dir, image_name)
    rng = np.random.default_rng(rng_seed)
    w = img.shape[1]

    train_x_range = (half, min(boundary - half, w - half))
    val_x_range = (max(boundary + half, half), w - half)

    n_train_neg = len(train_pos) * neg_ratio
    n_val_neg = len(val_pos) * neg_ratio
    train_neg = sample_negative_centers(img.shape, all_points_xy, n_train_neg, patch_size,
                                         exclusion_radius, rng, x_range=train_x_range)
    val_neg = sample_negative_centers(img.shape, all_points_xy, n_val_neg, patch_size,
                                       exclusion_radius, rng, x_range=val_x_range)

    train_records = (
        [PatchRecord(image_name=image_name, x=x, y=y, label=1, class_name=class_name) for x, y in train_pos] +
        [PatchRecord(image_name=image_name, x=x, y=y, label=0, class_name=None) for x, y in train_neg]
    )
    val_records = (
        [PatchRecord(image_name=image_name, x=x, y=y, label=1, class_name=class_name) for x, y in val_pos] +
        [PatchRecord(image_name=image_name, x=x, y=y, label=0, class_name=None) for x, y in val_neg]
    )
    meta = {'boundary': boundary, 'train_points': len(train_pos), 'val_points': len(val_pos),
            'dropped_points': dropped}
    return train_records, val_records, meta
