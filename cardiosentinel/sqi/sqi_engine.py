"""
CardioSentinel Hierarchical Signal Quality Indexing (SQI) Engine
Multi-tier gatekeeper that inspects beat-agreement, spectral distribution,
kurtosis, and baseline wander to reject noisy ambulatory artifacts before classification.
"""

import numpy as np
from scipy import signal, stats
from typing import Dict, Any, Tuple
from cardiosentinel.config import SQIConfig


class SQIEngine:
    """
    Evaluates physiological validity of single-lead ambulatory ECG signals.
    Employs benchmarked PhysioNet SQI formulations (bSQI, sSQI, kSQI, basSQI).
    """

    def __init__(self, fs: int = 250, config: SQIConfig = SQIConfig()):
        self.fs = fs
        self.config = config

    def compute_b_sqi(
        self,
        peaks_primary: np.ndarray,
        peaks_secondary: np.ndarray,
        tolerance_ms: float = 50.0
    ) -> float:
        """
        Beat Signal Quality Index (bSQI):
        Measures the F1 agreement between two independent QRS detectors.
        Tolerance window accounts for slight differences in fiducial center definitions.
        """
        n1 = len(peaks_primary)
        n2 = len(peaks_secondary)

        if n1 == 0 and n2 == 0:
            return 0.0
        if n1 == 0 or n2 == 0:
            return 0.0

        tol_samples = int((tolerance_ms / 1000.0) * self.fs)
        matched_count = 0

        # Match each primary peak to closest secondary peak within tolerance
        matched_secondary = set()
        for p1 in peaks_primary:
            diffs = np.abs(peaks_secondary - p1)
            min_idx = np.argmin(diffs)
            if diffs[min_idx] <= tol_samples and min_idx not in matched_secondary:
                matched_count += 1
                matched_secondary.add(min_idx)

        # Harmonic F1 agreement
        bsqi = (2.0 * matched_count) / (n1 + n2)
        return float(np.clip(bsqi, 0.0, 1.0))

    def compute_s_sqi(self, ecg_signal: np.ndarray) -> float:
        """
        Spectral Signal Quality Index (sSQI):
        Evaluates the ratio of power concentrated in the physiological QRS band (5 - 15 Hz)
        versus the extended ambulatory frequency band (5 - 40 Hz).
        """
        if len(ecg_signal) < int(self.fs * 0.5):
            return 0.0

        freqs, psd = signal.welch(ecg_signal, fs=self.fs, nperseg=min(len(ecg_signal), self.fs * 2))

        # Band 5-15 Hz
        idx_qrs = np.logical_and(freqs >= 5.0, freqs <= 15.0)
        power_qrs = np.sum(psd[idx_qrs])

        # Band 5-40 Hz
        idx_full = np.logical_and(freqs >= 5.0, freqs <= 40.0)
        power_full = np.sum(psd[idx_full])

        if power_full <= 1e-12:
            return 0.0

        ssqi = power_qrs / power_full
        return float(np.clip(ssqi, 0.0, 1.0))

    def compute_k_sqi(self, ecg_signal: np.ndarray) -> float:
        """
        Kurtosis Signal Quality Index (kSQI):
        Clean ECG exhibits high kurtosis (> 4.5) due to sparse, steep QRS spikes
        relative to the isoelectric baseline. White Gaussian noise has kurtosis ~ 3.0.
        """
        if len(ecg_signal) < 10:
            return 0.0

        k = float(stats.kurtosis(ecg_signal, fisher=False)) # Pearson's kurtosis (normal distribution = 3.0)
        # Scale to a [0, 1] confidence index: 3.0 -> 0.0, 5.0+ -> 1.0
        norm_k = (k - 3.0) / 3.0
        return float(np.clip(norm_k, 0.0, 1.0))

    def compute_bas_sqi(self, raw_signal: np.ndarray) -> float:
        """
        Baseline Wander Signal Quality Index (basSQI):
        Proportion of signal power residing in the ultra-low frequency drift band (< 1.0 Hz).
        Low basSQI is desirable; we invert it so 1.0 = pristine stable baseline, 0.0 = severe drift.
        """
        if len(raw_signal) < int(self.fs * 0.5):
            return 0.0

        freqs, psd = signal.welch(raw_signal, fs=self.fs, nperseg=min(len(raw_signal), self.fs * 2))
        idx_base = freqs < 1.0
        power_base = np.sum(psd[idx_base])
        power_total = np.sum(psd)

        if power_total <= 1e-12:
            return 0.0

        wander_ratio = power_base / power_total
        # Invert so 1.0 is best
        stability_score = 1.0 - np.clip(wander_ratio / self.config.BASSQI_MAX_BASELINE, 0.0, 1.0)
        return float(stability_score)

    def detect_lead_off(self, raw_signal: np.ndarray) -> bool:
        """Flags disconnected patch / open-circuit flatline conditions."""
        amp_range = np.ptp(raw_signal)
        std_val = np.std(raw_signal)
        return bool(amp_range < self.config.LEAD_OFF_AMPLITUDE_THRESHOLD_MV or std_val < 0.008)

    def evaluate_gatekeeper(
        self,
        raw_signal: np.ndarray,
        filtered_signal: np.ndarray,
        peaks_pantomptkins: np.ndarray,
        peaks_secondary: np.ndarray
    ) -> Dict[str, Any]:
        """
        Full Hierarchical SQI Gatekeeper Evaluation.
        Decides whether incoming window is acceptable for diagnostic classification
        or must be rejected to prevent clinical false alarms.
        """
        # 1. Lead-off check
        is_lead_off = self.detect_lead_off(raw_signal)
        if is_lead_off:
            return {
                "decision": "REJECT_LEAD_OFF",
                "passed_gate": False,
                "overall_sqi": 0.0,
                "bsqi": 0.0,
                "ssqi": 0.0,
                "ksqi": 0.0,
                "bassqi": 0.0,
                "reason": "Lead-off / Sensor Disconnected: Amplitude flatline detected."
            }

        # 2. Compute individual indices
        bsqi = self.compute_b_sqi(peaks_pantomptkins, peaks_secondary)
        ssqi = self.compute_s_sqi(filtered_signal)
        ksqi = self.compute_k_sqi(filtered_signal)
        bassqi = self.compute_bas_sqi(raw_signal)

        # 3. Weighted composite quality index
        # bSQI carries the highest weight as beat morphology concordance is the gold standard
        composite_sqi = 0.40 * bsqi + 0.25 * ssqi + 0.20 * ksqi + 0.15 * bassqi
        composite_sqi = float(np.clip(composite_sqi, 0.0, 1.0))

        # 4. Gatekeeper thresholding
        passed = (composite_sqi >= self.config.OVERALL_SQI_ACCEPT_THRESHOLD) and (bsqi >= 0.50)

        reason = "Clean diagnostic signal: Quality gate passed." if passed else (
            f"Corrupted ambulatory artifact (bSQI={bsqi:.2f}, sSQI={ssqi:.2f}). "
            "Abstaining from arrhythmia inference to prevent clinician false alarms."
        )

        return {
            "decision": "ACCEPT" if passed else "REJECT_MOTION_CORRUPTED",
            "passed_gate": passed,
            "overall_sqi": round(composite_sqi, 3),
            "bsqi": round(bsqi, 3),
            "ssqi": round(ssqi, 3),
            "ksqi": round(ksqi, 3),
            "bassqi": round(bassqi, 3),
            "reason": reason
        }

