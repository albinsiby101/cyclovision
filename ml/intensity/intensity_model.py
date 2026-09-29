"""
CNN + ConvLSTM Temporal Cyclone Intensity Classification Model
Models spatio-temporal dynamics from satellite image sequences (6-12 frames)
to classify intensity across 7 standard IMD/WMO North Indian Ocean stages.
"""

import torch
import torch.nn as nn
from typing import Tuple

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        # Gates: input, forget, cell candidate, output
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_cur, c_cur = state
        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size: int, image_size: Tuple[int, int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = image_size
        return (
            torch.zeros(batch_size, self.hidden_channels, h, w, device=device),
            torch.zeros(batch_size, self.hidden_channels, h, w, device=device)
        )

class SpatialFeatureEncoder(nn.Module):
    """
    Per-frame spatial CNN feature extractor
    """
    def __init__(self, in_channels: int = 2, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 128 -> 64
            
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 64 -> 32
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CycloneIntensityModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 7,
        spatial_features: int = 64,
        convlstm_hidden: int = 64
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = SpatialFeatureEncoder(in_channels=in_channels, out_channels=spatial_features)
        self.convlstm = ConvLSTMCell(in_channels=spatial_features, hidden_channels=convlstm_hidden, kernel_size=3)
        
        # Spatial pooling + Temporal classification head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(convlstm_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq shape: [Batch, Time, Channels, Height, Width]
        Returns logits for intensity category: [Batch, num_classes]
        """
        b, t, c, h, w = seq.size()
        
        # Determine spatial feature dimensions after encoder
        sample_feat = self.encoder(seq[:, 0])
        feat_h, feat_w = sample_feat.size(2), sample_feat.size(3)
        
        h_state, c_state = self.convlstm.init_hidden(b, (feat_h, feat_w), seq.device)
        
        for step in range(t):
            frame = seq[:, step] # [B, C, H, W]
            spatial_feat = self.encoder(frame) # [B, spatial_features, H', W']
            h_state, c_state = self.convlstm(spatial_feat, (h_state, c_state))
            
        # Final hidden state represents temporal sequence accumulation
        pooled = self.pool(h_state) # [B, convlstm_hidden, 1, 1]
        flat = torch.flatten(pooled, 1) # [B, convlstm_hidden]
        logits = self.classifier(flat)
        return logits