"""Small CNN patch classifier, sized for a few hundred to ~1500 positive examples.

Global-average-pooling before the linear head (rather than a flattened dense head) is
deliberate: with this little data, a head over a flattened 12x12x64 feature map (9216
inputs) would have far more parameters than training examples and overfit immediately.
"""
import torch
import torch.nn as nn


class SmallPatchCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
