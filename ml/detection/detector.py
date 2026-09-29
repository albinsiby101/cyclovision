"""
Cyclone Detection Model (Binary: Non-Cyclonic vs Cyclonic Pattern)
Utilizes a streamlined ResNet-based spatial feature extractor with channel adaptation
for multi-spectral satellite imagery (IR + Water Vapour).
"""

import torch
import torch.nn as nn
import torchvision.models as models

class CycloneDetector(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        # Use resnet18 as spatial backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        
        # Adapt first convolution to accept in_channels (e.g. 2 for IR + WV, or 1 for IR)
        if in_channels != 3:
            orig_conv = base.conv1
            self.conv1 = nn.Conv2d(
                in_channels,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False
            )
            # Initialize weights
            if orig_conv.weight is not None:
                with torch.no_grad():
                    if in_channels == 2:
                        self.conv1.weight.copy_(orig_conv.weight[:, :2, :, :])
                    elif in_channels == 1:
                        self.conv1.weight.copy_(orig_conv.weight[:, :1, :, :])
        else:
            self.conv1 = base.conv1
            
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4  # Target layer for Grad-CAM
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, H, W]
        features = self.extract_features(x)
        pooled = self.avgpool(features)
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)
        return logits