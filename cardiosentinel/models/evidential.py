"""
CardioSentinel Evidential Deep Learning & Uncertainty Calibration Engine
Implements Subjective Logic, Dirichlet Prior Evidential Neural Networks (Sensoy et al., NeurIPS 2018),
Three-Zone Clinical Decision Logic, Expected Calibration Error (ECE), and Brier Score evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Any, List


class EvidentialHead(nn.Module):
    """
    Transforms neural latent representations into non-negative evidence vectors
    and calculates Dirichlet distribution parameters (alpha, expected probability, epistemic uncertainty).
    """

    def __init__(self, in_features: int, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            alpha: Dirichlet concentration parameters [B, K] where alpha_k >= 1
            prob: Expected class probabilities [B, K]
            uncertainty: Total epistemic uncertainty score [B] in [0, 1]
        """
        logits = self.fc(x)
        # Non-negative evidence via Softplus activation
        evidence = F.softplus(logits)
        # Dirichlet parameters alpha_k = evidence_k + 1
        alpha = evidence + 1.0

        # Total Dirichlet strength S = sum(alpha_k)
        S = torch.sum(alpha, dim=1, keepdim=True)

        # Expected class probability E[p_k] = alpha_k / S
        prob = alpha / S

        # Epistemic uncertainty u = K / S
        uncertainty = (self.num_classes / S).squeeze(-1)

        return alpha, prob, uncertainty


class EvidentialLoss(nn.Module):
    """
    Evidential Loss Function with KL-Divergence Regularization:
    Penalizes misclassifications and prevents misleading high evidence on OOD samples.
    """

    def __init__(self, num_classes: int = 6, kl_weight: float = 0.005):
        super().__init__()
        self.num_classes = num_classes
        self.kl_weight = kl_weight

    def forward(
        self,
        alpha: torch.Tensor,
        target_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes expected cross-entropy under Dirichlet prior + KL divergence.
        """
        y_one_hot = F.one_hot(target_indices, num_classes=self.num_classes).float()
        S = torch.sum(alpha, dim=1, keepdim=True)

        # Expected Cross Entropy: sum(y_k * (psi(S) - psi(alpha_k)))
        psi_S = torch.digamma(S)
        psi_alpha = torch.digamma(alpha)
        loss_ce = torch.sum(y_one_hot * (psi_S - psi_alpha), dim=1)

        # KL Divergence to Uniform Dirichlet Prior Dir(1, ..., 1) for incorrect classes
        alpha_tilde = y_one_hot + (1.0 - y_one_hot) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)

        first_term = (
            torch.lgamma(S_tilde.squeeze(-1))
            - torch.lgamma(torch.tensor(float(self.num_classes), device=alpha.device))
            - torch.sum(torch.lgamma(alpha_tilde), dim=1)
        )
        second_term = torch.sum(
            (alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)),
            dim=1
        )
        kl_div = first_term + second_term

        total_loss = torch.mean(loss_ce + self.kl_weight * kl_div)
        return total_loss


class ThreeZoneDecisionEngine:
    """
    Measurable Clinical Decision Matrix bridging Signal Quality and Evidential Uncertainty:
    - High-Quality + High Confidence (Low Uncertainty) -> ACCEPT (Display for automated screening)
    - High-Quality + High Uncertainty                  -> REVIEW (Flag episode for priority cardiologist assessment)
    - Low-Quality / Severe Noise                       -> ABSTAIN (Request sensor reposition / re-recording)
    """

    def __init__(self, uncertainty_threshold: float = 0.380, sqi_accept_threshold: float = 0.650):
        self.uncertainty_threshold = uncertainty_threshold
        self.sqi_accept_threshold = sqi_accept_threshold

    def evaluate_decision(
        self,
        overall_sqi: float,
        epistemic_uncertainty: float,
        passed_sqi_gate: bool,
        is_pathological_morphology: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluates the 3-zone clinical triage decision.
        """
        # If genuine cardiac abnormality is preserved, bypass SQI rejection
        effective_sqi_pass = passed_sqi_gate or is_pathological_morphology
        is_high_uncertainty = epistemic_uncertainty > self.uncertainty_threshold

        if not effective_sqi_pass:
            action = "ABSTAIN"
            clinical_guidance = "Low-Quality ECG: Motion or contact artifact degraded signal beyond diagnostic reliability. Prompting re-recording."
            badge_color = "#991B1B" # Red
        elif is_high_uncertainty:
            action = "REVIEW"
            clinical_guidance = "Ambiguous Episode: High-quality trace, but prediction ambiguity exceeds safety bounds. Flagged for priority cardiologist overread."
            badge_color = "#D97706" # Amber
        else:
            action = "ACCEPT"
            clinical_guidance = "Validated Screening: High signal quality and calibrated model confidence. Displaying screening impression."
            badge_color = "#16A34A" # Green

        return {
            "decision_zone": action,
            "clinical_guidance": clinical_guidance,
            "badge_color": badge_color,
            "is_automated_accept": action == "ACCEPT",
            "is_clinician_review": action == "REVIEW",
            "is_sensor_abstain": action == "ABSTAIN"
        }


class UncertaintyCalibrationEvaluator:
    """
    Computes Expected Calibration Error (ECE), Brier Score, and Reliability Curves
    to evaluate whether evidential uncertainty truly identifies misclassifications.
    """

    @staticmethod
    def compute_brier_score(probs: np.ndarray, targets: np.ndarray, num_classes: int = 6) -> float:
        """Computes multiclass Brier Score: 1/N * sum(||p - y||^2)."""
        one_hot = np.zeros((len(targets), num_classes))
        one_hot[np.arange(len(targets)), targets] = 1.0
        return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    @staticmethod
    def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        """
        Calculates Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
        """
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == targets)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
        bin_data = []

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                abs_diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += abs_diff * prop_in_bin
                mce = max(mce, abs_diff)

                bin_data.append({
                    "bin_lower": round(bin_lower, 2),
                    "bin_upper": round(bin_upper, 2),
                    "accuracy": round(float(accuracy_in_bin), 4),
                    "confidence": round(float(avg_confidence_in_bin), 4),
                    "samples_count": int(np.sum(in_bin))
                })

        return {
            "ece": round(float(ece), 4),
            "mce": round(float(mce), 4),
            "brier_score": round(UncertaintyCalibrationEvaluator.compute_brier_score(probs, targets), 4),
            "bin_details": bin_data
        }
