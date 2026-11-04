"""Unit tests for CardioSentinel ECG Preprocessing & Filters."""

import sys
import unittest
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from cardiosentinel.preprocessing.filters import ECGPreprocessor


class TestFilters(unittest.TestCase):
    def test_notch_filter_attenuates_50hz(self):
        fs = 250
        t = np.linspace(0, 2.0, int(fs * 2.0), endpoint=False)
        signal_50hz = np.sin(2 * np.pi * 50.0 * t)

        preprocessor = ECGPreprocessor(fs=fs, notch_freq=50.0)
        filtered = preprocessor.apply_notch(signal_50hz)

        # 50Hz amplitude should be significantly attenuated (> 70% reduction)
        self.assertLess(np.max(np.abs(filtered[50:-50])), 0.30)

    def test_bandpass_preserves_qrs_frequencies(self):
        fs = 250
        t = np.linspace(0, 2.0, int(fs * 2.0), endpoint=False)
        signal_10hz = np.sin(2 * np.pi * 10.0 * t)

        preprocessor = ECGPreprocessor(fs=fs, lowcut=0.5, highcut=45.0)
        filtered = preprocessor.apply_bandpass(signal_10hz)

        # 10Hz should pass through with high gain (> 0.85)
        self.assertGreater(np.max(np.abs(filtered[50:-50])), 0.85)

    def test_robust_normalization(self):
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        preprocessor = ECGPreprocessor(fs=250)
        norm = preprocessor.normalize(sig, method="robust")
        self.assertTrue(np.isfinite(norm).all())
        self.assertLess(norm[0], norm[-1])


if __name__ == "__main__":
    unittest.main()
