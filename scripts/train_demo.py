"""
CardioSentinel Pretraining & Weight Calibration Script
Trains the Multi-Scale 1D-CNN + GRU with Evidential Dirichlet Loss on diverse
cardiac rhythms with physiological ambulatory noise augmentation.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from cardiosentinel.models.hybrid_classifier import CardioSentinelClassifier
from cardiosentinel.models.evidential import EvidentialLoss
from cardiosentinel.data.synthetic_ecg import ECGSynthesizer
from cardiosentinel.preprocessing.filters import ECGPreprocessor


class ECGAugmentedDataset(Dataset):
    """Generates synthetic multi-rhythm batches with dynamic wearable noise."""

    def __init__(self, samples_per_class: int = 50, fs: int = 250):
        self.synthesizer = ECGSynthesizer(sampling_rate=fs)
        self.preprocessor = ECGPreprocessor(fs=fs)
        self.fs = fs
        self.data = []
        self.labels = []

        rhythms = [
            ("NSR", 0),
            ("AFib", 1),
            ("PVC", 2),
            ("PAC", 3),
            ("VT", 4),
            ("Bradycardia", 5)
        ]

        print(f"Generating calibration dataset ({samples_per_class * len(rhythms)} total rhythm windows)...")
        for rhythm_name, label_id in rhythms:
            for s in range(samples_per_class):
                sig, _, _ = self.synthesizer.generate_rhythm(
                    rhythm_type=rhythm_name,
                    duration_sec=10.0,
                    random_seed=(label_id * 1000) + s
                )
                # Inject mild to moderate noise
                noisy, _ = self.synthesizer.inject_ambulatory_noise(
                    sig,
                    baseline_wander_level=np.random.uniform(0.02, 0.12),
                    emg_tremor_level=np.random.uniform(0.01, 0.05),
                    motion_artifact_level=np.random.uniform(0.0, 0.1)
                )
                _, norm = self.preprocessor.process(noisy)
                target_len = int(10.0 * fs)
                if len(norm) < target_len:
                    norm = np.pad(norm, (0, target_len - len(norm)), mode='edge')
                else:
                    norm = norm[:target_len]

                self.data.append(norm.astype(np.float32))
                self.labels.append(label_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx]).unsqueeze(0) # [1, 2500]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def train_and_save_weights(epochs: int = 15, batch_size: int = 16):
    dataset = ECGAugmentedDataset(samples_per_class=40)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CardioSentinel on device: {device}")

    model = CardioSentinelClassifier(in_channels=1, num_classes=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = EvidentialLoss(num_classes=6, kl_weight=0.005)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            alpha, prob, uncertainty = model(x)
            loss = criterion(alpha, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            preds = torch.argmax(prob, dim=1)
            correct += (preds == y).sum().item()
            total += len(y)

        train_acc = (correct / total) * 100.0
        avg_loss = total_loss / total
        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} | Accuracy: {train_acc:.1f}%")

    # Save weights
    weights_path = PROJECT_ROOT / "cardiosentinel" / "models" / "pretrained_weights.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"\nModel weights saved successfully to: {weights_path}")


if __name__ == "__main__":
    train_and_save_weights(epochs=15)
