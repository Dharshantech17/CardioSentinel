"""
CardioSentinel Hybrid Multi-Scale 1D-CNN + Temporal GRU Architecture
Combines morphological wavelet-like spatial receptive fields with sequential
beat-to-beat interval dynamics and evidential Dirichlet uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
from cardiosentinel.models.evidential import EvidentialHead
from cardiosentinel.config import ARRHYTHMIA_CLASSES, ARRHYTHMIA_SEVERITY, EvidentialConfig


class MultiScaleConvBlock(nn.Module):
    """Parallel multi-scale 1D convolutions capturing micro-notches, QRS, and ST waves."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_channels = out_channels // 3
        remainder = out_channels - (branch_channels * 2)

        # Micro-scale (sharp spikes / notches)
        self.conv_s = nn.Conv1d(in_channels, branch_channels, kernel_size=5, padding=2)
        # Meso-scale (QRS duration & P/T waves)
        self.conv_m = nn.Conv1d(in_channels, branch_channels, kernel_size=15, padding=7)
        # Macro-scale (ST elevation / broad ectopic waves)
        self.conv_l = nn.Conv1d(in_channels, remainder, kernel_size=31, padding=15)

        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_s = self.conv_s(x)
        out_m = self.conv_m(x)
        out_l = self.conv_l(x)
        merged = torch.cat([out_s, out_m, out_l], dim=1)
        return self.act(self.bn(merged))


class CardioSentinelClassifier(nn.Module):
    """
    End-to-End Deep Arrhythmia Classifier with Dirichlet Evidential Uncertainty Head.
    Input shape: [Batch, 1, 2500] (10-second rhythm strip at 250 Hz)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes

        # Spatial Morphological Feature Extractor
        self.ms_block1 = MultiScaleConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4) # 2500 -> 625

        self.ms_block2 = MultiScaleConvBlock(32, 64)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=5) # 625 -> 125

        self.ms_block3 = MultiScaleConvBlock(64, 128)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=5) # 125 -> 25

        # Temporal Sequence Modeling (Bidirectional GRU)
        self.temporal_gru = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        # Feature Pooling and Dense Head
        self.fc_dense = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # Evidential Dirichlet Head
        self.evidential_head = EvidentialHead(in_features=64, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            alpha: Dirichlet parameters [B, K]
            prob: Expected probabilities [B, K]
            uncertainty: Epistemic uncertainty [B]
        """
        # Morphological extraction
        h = self.ms_block1(x)
        h = self.pool1(h)
        h = self.ms_block2(h)
        h = self.pool2(h)
        h = self.ms_block3(h)
        h = self.pool3(h) # [B, 128, 25]

        # Reshape for GRU: [B, Seq_len=25, Features=128]
        h = h.permute(0, 2, 1)
        gru_out, _ = self.temporal_gru(h) # [B, 25, 128]

        # Temporal attention / max pooling over sequence length
        pooled, _ = torch.max(gru_out, dim=1) # [B, 128]

        features = self.fc_dense(pooled)
        alpha, prob, uncertainty = self.evidential_head(features)
        return alpha, prob, uncertainty

    def predict_window(
        self,
        ecg_window: np.ndarray,
        uncertainty_threshold: float = 0.38
    ) -> Dict[str, Any]:
        """
        Evaluates a single 10-second ECG window.
        Returns detailed clinical triage diagnostic package.
        """
        self.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(ecg_window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            alpha, prob, uncertainty = self.forward(tensor_x)

            alpha_arr = alpha.squeeze(0).cpu().numpy()
            prob_arr = prob.squeeze(0).cpu().numpy()
            u_val = float(uncertainty.squeeze(0).cpu().numpy())

            pred_class_idx = int(np.argmax(prob_arr))
            pred_prob = float(prob_arr[pred_class_idx])
            class_name = ARRHYTHMIA_CLASSES.get(pred_class_idx, "Unknown")
            severity = ARRHYTHMIA_SEVERITY.get(pred_class_idx, "UNKNOWN")

            # Abstention logic: If epistemic uncertainty is excessive
            is_ambiguous = u_val > uncertainty_threshold
            triage_status = "CONFIRMED_ALGORITHM_SCREENING"
            if is_ambiguous:
                triage_status = "ABSTAIN_FLAGGED_FOR_CARDIOLOGIST"

            prob_distribution = {
                ARRHYTHMIA_CLASSES[i]: round(float(prob_arr[i]), 4)
                for i in range(self.num_classes)
            }

            return {
                "predicted_class_id": pred_class_idx,
                "rhythm_diagnosis": class_name,
                "confidence_score": round(pred_prob, 4),
                "epistemic_uncertainty": round(u_val, 4),
                "severity": severity,
                "triage_status": triage_status,
                "is_ambiguous": is_ambiguous,
                "probabilities": prob_distribution,
                "dirichlet_alphas": [round(float(a), 3) for a in alpha_arr]
            }

