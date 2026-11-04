"""Unit tests for CardioSentinel Hierarchical SQI Engine."""

import sys
import unittest
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from cardiosentinel.sqi.sqi_engine import SQIEngine
from cardiosentinel.data.synthetic_ecg import ECGSynthesizer


class TestSQIEngine(unittest.TestCase):
    def setUp(self):
        self.sqi = SQIEngine(fs=250)

    def test_bsqi_identical_peaks(self):
        peaks = np.array([250, 500, 750, 1000])
        score = self.sqi.compute_b_sqi(peaks, peaks)
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_bsqi_disjoint_peaks(self):
        peaks1 = np.array([100, 300, 500])
        peaks2 = np.array([200, 400, 600])
        score = self.sqi.compute_b_sqi(peaks1, peaks2, tolerance_ms=50.0)
        self.assertEqual(score, 0.0)

    def test_lead_off_detection(self):
        flatline = np.zeros(2500)
        self.assertTrue(self.sqi.detect_lead_off(flatline))

        normal_signal = np.sin(np.linspace(0, 10, 2500))
        self.assertFalse(self.sqi.detect_lead_off(normal_signal))

    def test_gatekeeper_accepts_clean_ecg(self):
        syn = ECGSynthesizer(sampling_rate=250)
        sig, peaks, _ = syn.generate_rhythm("NSR", duration_sec=10.0, random_seed=42)

        res = self.sqi.evaluate_gatekeeper(
            raw_signal=sig,
            filtered_signal=sig,
            peaks_pantomptkins=peaks,
            peaks_secondary=peaks
        )
        self.assertTrue(res["passed_gate"])
        self.assertEqual(res["decision"], "ACCEPT")
        self.assertGreaterEqual(res["overall_sqi"], 0.65)


if __name__ == "__main__":
    unittest.main()
