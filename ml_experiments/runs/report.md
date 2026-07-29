## Pair A: IMG_4147 (train) -> IMG_4148 (eval), Plegadis chihi [PRIMARY]

Eval image: `IMG_4148.jpeg`, class `Plegadis chihi`, n_gt=184

CNN patch-level (on held-out patches): precision=0.929 recall=0.929 F1=0.929 ROC-AUC=0.993

Point-level detection at three tolerances (radius = how close a predicted point must be to a ground-truth point to count as found): precise (~half median inter-bird spacing), stride (32px, the sliding-window grid spacing), dedup (48px, the NMS collapse radius -- an upper bound on what this pipeline can resolve at all).

| radius (px) | CNN P | CNN R | CNN F1 | Classical P | Classical R | Classical F1 |
|---|---|---|---|---|---|---|
| 13.2 | 0.064 | 0.065 | 0.065 | 0.099 | 0.359 | 0.155 |
| 32.0 | 0.356 | 0.364 | 0.360 | 0.139 | 0.505 | 0.219 |
| 48.0 | 0.590 | 0.603 | 0.597 | 0.165 | 0.598 | 0.259 |

Classical baseline's best operating point: sensitivity=30, polarity=dark.

**Caveat:** This is the trustworthy generalization signal -- IMG_4148 is a wholly separate photo.

## Pair B: IMG_4153 spatial holdout, Chrysomus ruficapillus [SECONDARY, optimistic]

Eval image: `IMG_4153.jpeg`, class `Chrysomus ruficapillus`, n_gt=269

CNN patch-level (on held-out patches): precision=1.000 recall=0.989 F1=0.994 ROC-AUC=1.000

Point-level detection at three tolerances (radius = how close a predicted point must be to a ground-truth point to count as found): precise (~half median inter-bird spacing), stride (32px, the sliding-window grid spacing), dedup (48px, the NMS collapse radius -- an upper bound on what this pipeline can resolve at all).

| radius (px) | CNN P | CNN R | CNN F1 | Classical P | Classical R | Classical F1 |
|---|---|---|---|---|---|---|
| 5.0 | 0.192 | 0.019 | 0.034 | 0.022 | 0.033 | 0.027 |
| 32.0 | 0.962 | 0.093 | 0.169 | 0.022 | 0.033 | 0.027 |
| 48.0 | 1.000 | 0.097 | 0.176 | 0.022 | 0.033 | 0.027 |

Classical baseline's best operating point: sensitivity=70, polarity=dark.

**Caveat:** Same photo as training (spatial split only) -- shared lighting/sensor-noise/background leak across the split, and birds are packed closer together (median 11px) than this sliding-window pipeline (32px stride, 48px NMS radius) can resolve, capping recall regardless of classifier quality.

---

## Follow-up: species-agnostic, size-bucketed heatmap regression

Two changes made together, both directly targeting problems found above:

1. **Species-agnostic pooling**: instead of one checkpoint per species, positives are
   pooled across every annotated species on an image (`class_name=None` throughout
   `ml_experiments/patches.py`/`splits.py`) into one detector per **size bucket** --
   grouped by measured median inter-point spacing, a data-driven proxy for how tightly
   packed the subject is (not ecological body size, which doesn't reliably predict
   packing density in a given photo). Buckets:
   - **small**: Chrysomus ruficapillus only (median spacing 11.3px) -- no second source
     image exists yet, still a same-image spatial holdout.
   - **large**: Plegadis chihi (26-37px) + Coscoroba coscoroba (120-220px) + 3 singleton
     species pooled from the same photos, spanning 3 training photos across 2 sessions
     (regular camera photos + stitched aerial panoramas) with `IMG_4148.jpeg` held out --
     the same held-out image as the Pair A run above, for direct comparison.

2. **Heatmap regression instead of patch classification** (`ml_experiments/heatmap_*.py`,
   `ddg/ml_detector.py`): a fully-convolutional model predicts a per-pixel "birdness" map
   (trained on Gaussian-blob targets) and finds local peaks in it directly, instead of
   classifying fixed-size windows on a stride-spaced grid + NMS. This removes the
   stride/NMS localization ceiling identified above rather than working around it --
   output resolution is configurable per bucket (small: output_stride=4; large: 8).

### Large bucket vs. the original species-specific Pair A run (same held-out image, same 13.2px tolerance)

| approach | P | R | F1 |
|---|---|---|---|
| Classifier, species-specific (Plegadis only) | 0.064 | 0.065 | 0.065 |
| Classical CV | 0.099 | 0.359 | 0.155 |
| **Heatmap, pooled + size-bucketed (large)** | 0.175-0.325 | 0.75-0.27 | **0.28-0.29** |

Pooling 3 species across 3 photos (including two very different-looking stitched
panoramas) and switching to heatmap regression roughly **4.5x'd F1 at the same strict
tolerance**, and clearly beat classical CV too. Precision/recall trade off with the peak
threshold (0.5 -> high recall/low precision; 0.7 -> the best F1 found, 0.293) -- the model
is usably right about *where* birds are, but not yet confident enough for a single sharp
operating point; more training photos per bucket would likely help most here.

### Small bucket vs. the original same-image Chrysomus run (same photo, but note the
### tolerance itself shrank from 48px to 5px -- i.e. this is a much harder bar too)

| approach | tolerance | P | R | F1 |
|---|---|---|---|---|
| Classifier, same-image (patch_size=96, stride=32, NMS=48) | 48px (loosest it can resolve) | 1.00 | 0.097 | 0.176 |
| Classical CV | 48px | 0.022 | 0.033 | 0.027 |
| **Heatmap, same-image (output_stride=4)** | **5px (native, ~half true spacing)** | **0.708** | **0.701** | **0.705** |

This is the more dramatic result: not only did F1 roughly **4x**, it did so at a
**~10x stricter** localization tolerance -- exactly the fix the earlier resolution-ceiling
diagnosis called for. This bucket still only has one source photo, so it remains a
same-image (optimistic) evaluation; a second photo of this species would let it be
re-evaluated the trustworthy cross-image way, like the large bucket now is.

### What would improve this further

- More training photos per bucket (especially "large," where confidence is the limiting
  factor, not localization).
- A decoder/skip-connections (full U-Net) if even finer localization is needed for
  extremely dense colonies.
- A genuine 3rd/4th size bucket once more species accumulate enough annotated images to
  measure their own spacing reliably (currently 3 species have only 1 point each and were
  folded into "large" without their own bucket).