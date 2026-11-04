"""Unit tests for CardioSentinel Evidential Classifier & Dirichlet formulation."""

import sys
import unittest
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from cardiosentinel.models.evidential import EvidentialHead, EvidentialLoss
from cardiosentinel.models.hybrid_classifier import CardioSentinelClassifier


class TestModels(unittest.TestCase):
    def test_evidential_head_properties(self):
        batch_size = 4
        in_features = 32
        num_classes = 6

        head = EvidentialHead(in_features=in_features, num_classes=num_classes)
        x = torch.randn(batch_size, in_features)

        alpha, prob, uncertainty = head(x)

        # 1. Dimensions
        self.assertEqual(alpha.shape, (batch_size, num_classes))
        self.assertEqual(prob.shape, (batch_size, num_classes))
        self.assertEqual(uncertainty.shape, (batch_size,))

        # 2. Dirichlet constraints: alpha >= 1.0
        self.assertTrue((alpha >= 1.0).all().item())

        # 3. Probability sums to 1.0
        prob_sums = torch.sum(prob, dim=1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-5))

        # 4. Uncertainty bounded in [0, 1]
        self.assertTrue(((uncertainty >= 0.0) & (uncertainty <= 1.0)).all().item())

    def test_evidential_loss_gradient_flow(self):
        num_classes = 6
        loss_fn = EvidentialLoss(num_classes=num_classes)
        raw_logits = torch.randn(4, num_classes, requires_grad=True)
        evidence = torch.nn.functional.softplus(raw_logits)
        alpha = evidence + 1.0
        target = torch.tensor([0, 1, 2, 3])

        loss = loss_fn(alpha, target)
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(raw_logits.grad)

    def test_hybrid_classifier_forward_pass(self):
        model = CardioSentinelClassifier(in_channels=1, num_classes=6)
        model.eval()

        # Input: 2 windows of 2500 samples
        x = torch.randn(2, 1, 2500)
        with torch.no_grad():
            alpha, prob, uncertainty = model(x)

        self.assertEqual(alpha.shape, (2, 6))
        self.assertEqual(prob.shape, (2, 6))
        self.assertEqual(uncertainty.shape, (2,))


if __name__ == "__main__":
    unittest.main()
