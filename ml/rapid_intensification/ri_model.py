"""
Rapid Intensification (RI) Early-Warning Module
Estimates whether a cyclonic system will undergo Rapid Intensification
(defined as >= 30 knots sustained wind increase within 24 hours).
Integrates temporal CNN+ConvLSTM spatio-temporal features with environmental/intensity history,
using Focal Loss and positive class weighting for extreme event imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing severe class imbalance in rare Rapid Intensification events (~7-10%).
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class RapidIntensificationModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        spatial_features: int = 48,
        convlstm_hidden: int = 48,
        tabular_dim: int = 4 # e.g. current_wind, 12h_wind_trend, lat, pressure
    ):
        super().__init__()
        # Visual temporal branch
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 128 -> 64
            
            nn.Conv2d(24, spatial_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(spatial_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 64 -> 32
        )
        
        from ml.intensity.intensity_model import ConvLSTMCell
        self.convlstm = ConvLSTMCell(in_channels=spatial_features, hidden_channels=convlstm_hidden, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Environmental / Tabular branch
        self.tab_fc = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16)
        )
        
        # Fusion Classifier
        self.fusion = nn.Sequential(
            nn.Linear(convlstm_hidden + 16, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 1) # Single logit for RI probability
        )

    def forward(self, seq: torch.Tensor, tab_features: torch.Tensor) -> torch.Tensor:
        """
        seq: [B, T, C, H, W]
        tab_features: [B, tabular_dim]
        Returns: [B, 1] RI raw logit
        """
        b, t, c, h, w = seq.size()
        sample_feat = self.encoder(seq[:, 0])
        fh, fw = sample_feat.size(2), sample_feat.size(3)
        
        h_state, c_state = self.convlstm.init_hidden(b, (fh, fw), seq.device)
        for step in range(t):
            f = self.encoder(seq[:, step])
            h_state, c_state = self.convlstm(f, (h_state, c_state))
            
        vis_emb = torch.flatten(self.pool(h_state), 1) # [B, convlstm_hidden]
        tab_emb = self.tab_fc(tab_features)           # [B, 16]
        
        fused = torch.cat([vis_emb, tab_emb], dim=1)
        logit = self.fusion(fused)
        return logit

    def predict_risk(self, seq: torch.Tensor, tab_features: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            logit = self.forward(seq, tab_features)
            prob = torch.sigmoid(logit).item()
            
        if prob < 0.35:
            level = "LOW"
        elif prob < 0.65:
            level = "MODERATE"
        else:
            level = "HIGH"
            
        return {
            "rapid_intensification_risk": round(prob, 4),
            "risk_level": level,
            "forecast_window_hours": 24,
            "threshold_definition": ">= 30 knots sustained wind increase within 24 hours"
        }