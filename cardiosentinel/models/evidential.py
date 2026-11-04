"""
CardioSentinel Evidential Deep Learning & Dirichlet Uncertainty Engine
Implements Subjective Logic and Dirichlet Prior Evidential Neural Networks
for uncertainty quantification and clinical abstention (Sensoy et al., NeurIPS 2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any


class EvidentialHead(nn.Module):
    """
    Transforms neural latent representations into non-negative evidence vectors
    and calculates Dirichlet distribution parameters (alpha, belief, uncertainty).
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
