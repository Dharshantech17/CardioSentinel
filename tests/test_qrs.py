"""Unit tests for CardioSentinel Pan-Tompkins & Hamilton QRS detectors."""

import sys
import unittest
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from cardiosentinel.data.synthetic_ecg import ECGSynthesizer
from cardiosentinel.preprocessing.qrs_detector import PanTompkinsDetector, HamiltonDetector


class TestQRSDetectors(unittest.TestCase):
    def setUp(self):
        self.fs = 250
        self.syn = ECGSynthesizer(sampling_rate=self.fs)

    def test_pan_tompkins_detects_regular_beats(self):
        sig, true_peaks, _ = self.syn.generate_rhythm("NSR", duration_sec=5.0, target_hr=60.0, random_seed=42)
        detector = PanTompkinsDetector(fs=self.fs)
        detected = detector.detect(sig)

        self.assertGreaterEqual(len(detected), 3)
        self.assertLessEqual(abs(len(detected) - len(true_peaks)), 1)

    def test_hamilton_detects_beats(self):
        sig, true_peaks, _ = self.syn.generate_rhythm("NSR", duration_sec=5.0, target_hr=60.0, random_seed=42)
        detector = HamiltonDetector(fs=self.fs)
        detected = detector.detect(sig)

        self.assertGreaterEqual(len(detected), 3)
        self.assertLessEqual(abs(len(detected) - len(true_peaks)), 1)


if __name__ == "__main__":
    unittest.main()
