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