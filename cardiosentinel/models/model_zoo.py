"""
CardioSentinel Architectural Model Zoo & Ablation Framework
Provides comparable architectural variants (CNN-only, CNN-GRU, CNN-Transformer)
coupled with Dirichlet Evidential Uncertainty Heads for empirical ablation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
from cardiosentinel.models.evidential import EvidentialHead


class MultiScaleConvBlock(nn.Module):
    """Parallel multi-scale 1D convolutions capturing micro-notches, QRS, and ST waves."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_channels = out_channels // 3
        remainder = out_channels - (branch_channels * 2)

        self.conv_s = nn.Conv1d(in_channels, branch_channels, kernel_size=5, padding=2)
        self.conv_m = nn.Conv1d(in_channels, branch_channels, kernel_size=15, padding=7)
        self.conv_l = nn.Conv1d(in_channels, remainder, kernel_size=31, padding=15)

        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_s = self.conv_s(x)
        out_m = self.conv_m(x)
        out_l = self.conv_l(x)
        merged = torch.cat([out_s, out_m, out_l], dim=1)
        return self.act(self.bn(merged))


# -------------------------------------------------------------
# 1. BASELINE ARCHITECTURE: 1D-CNN Only (Morphological)
# -------------------------------------------------------------
class BaselineCNN(nn.Module):
    """
    Pure 1D-CNN baseline: Focuses strictly on local beat morphology
    without recurrent or attention sequence modeling. Ultra-low latency for MCU.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6):
        super().__init__()
        self.name = "Baseline 1D-CNN"
        self.ms1 = MultiScaleConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4) # 2500 -> 625
        self.ms2 = MultiScaleConvBlock(32, 64)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=5) # 625 -> 125
        self.ms3 = MultiScaleConvBlock(64, 128)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=5) # 125 -> 25

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 64)
        self.act = nn.GELU()
        self.evidential_head = EvidentialHead(64, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.pool1(self.ms1(x))
        h = self.pool2(self.ms2(h))
        h = self.pool3(self.ms3(h))
        h = self.gap(h).squeeze(-1)
        feat = self.act(self.fc(h))
        return self.evidential_head(feat)


# -------------------------------------------------------------
# 2. HYBRID ARCHITECTURE: 1D-CNN + Temporal GRU (Proposed Standard)
# -------------------------------------------------------------
class CNN_GRU(nn.Module):
    """
    1D-CNN + Bidirectional GRU:
    Extracts spatial wave morphology followed by bidirectional sequence tracking
    of beat-to-beat interval dynamics across the 10-second rhythm strip.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6):
        super().__init__()
        self.name = "1D-CNN + Temporal GRU"
        self.ms1 = MultiScaleConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.ms2 = MultiScaleConvBlock(32, 64)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.ms3 = MultiScaleConvBlock(64, 128)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=5)

        self.temporal_gru = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.fc = nn.Linear(128, 64)
        self.act = nn.GELU()
        self.evidential_head = EvidentialHead(64, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.pool1(self.ms1(x))
        h = self.pool2(self.ms2(h))
        h = self.pool3(self.ms3(h)) # [B, 128, 25]

        h = h.permute(0, 2, 1) # [B, 25, 128]
        gru_out, _ = self.temporal_gru(h)
        pooled, _ = torch.max(gru_out, dim=1) # [B, 128]

        feat = self.act(self.fc(pooled))
        return self.evidential_head(feat)


# -------------------------------------------------------------
# 3. ADVANCED ARCHITECTURE: 1D-CNN + Temporal Transformer
# -------------------------------------------------------------
class CNN_TemporalTransformer(nn.Module):
    """
    1D-CNN + Multi-Head Self-Attention Transformer:
    Models long-range global self-attention across R-R intervals with positional encoding.
    Evaluated directly in the ablation study to justify architectural trade-offs.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6, n_heads: int = 4, d_model: int = 128):
        super().__init__()
        self.name = "1D-CNN + Temporal Transformer"
        self.ms1 = MultiScaleConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.ms2 = MultiScaleConvBlock(32, 64)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.ms3 = MultiScaleConvBlock(64, d_model)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=5)

        # Learnable positional embedding for sequence length 25
        self.pos_embedding = nn.Parameter(torch.randn(1, 25, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(d_model, 64)
        self.act = nn.GELU()
        self.evidential_head = EvidentialHead(64, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.pool1(self.ms1(x))
        h = self.pool2(self.ms2(h))
        h = self.pool3(self.ms3(h)) # [B, 128, 25]

        h = h.permute(0, 2, 1) # [B, 25, 128]
        h = h + self.pos_embedding
        tf_out = self.transformer_encoder(h)

        # Global average pool over sequence length
        pooled = torch.mean(tf_out, dim=1) # [B, 128]
        feat = self.act(self.fc(pooled))
        return self.evidential_head(feat)
