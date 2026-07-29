"""Fully-convolutional heatmap-regression model: outputs a per-pixel "birdness" map at
1/output_stride input resolution instead of a single classification score for a
fixed-size patch. No dense/FC layers, so it runs on any input size, including whole
images tiled in large chunks at inference time.

Encoder-only (no upsampling decoder back to full resolution) is a deliberate simplicity
tradeoff: even output_stride=8 is already a 4x localization improvement over the
classifier's sliding-window stride (32px), for much less code than a full U-Net.
num_stages is configurable (2 -> stride 4, 3 -> stride 8) because how fine a resolution is
actually needed depends on the size bucket: the small/densely-packed bucket (~11px native
spacing) needs stride 4, while the large/sparser bucket is fine with stride 8 (cheaper,
especially over the huge stitched panorama images in that bucket's training data).
"""
import torch.nn as nn


def _conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class HeatmapCNN(nn.Module):

    def __init__(self, in_channels=3, num_stages=3, base_channels=16):
        super().__init__()
        assert num_stages in (2, 3)
        self.num_stages = num_stages
        self.output_stride = 2 ** num_stages

        channels = [base_channels * (2 ** i) for i in range(num_stages)]  # e.g. [16, 32, 64]
        stages = []
        in_c = in_channels
        for out_c in channels:
            stages.append(nn.Sequential(_conv_block(in_c, out_c), nn.MaxPool2d(2)))
            in_c = out_c
        self.stages = nn.ModuleList(stages)
        self.refine = _conv_block(in_c, in_c)
        self.head = nn.Conv2d(in_c, 1, kernel_size=1)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        x = self.refine(x)
        return self.head(x)  # (N, 1, H/output_stride, W/output_stride) logits

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
