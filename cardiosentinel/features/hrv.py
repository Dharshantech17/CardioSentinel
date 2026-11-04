"""
CardioSentinel HRV & Poincaré Feature Extraction Engine
Calculates time-domain Heart Rate Variability (HRV), statistical dynamics,
and non-linear Poincaré dispersion geometry for arrhythmia screening.
"""

import numpy as np
from typing import Dict, Any, Tuple


class HRVExtractor:
    """
    Extracts clinical HRV metrics from localized R-peak timestamps compliant
    with the European Society of Cardiology & North American Society of Pacing and Electrophysiology (ESC/NASPE).
    """

    def __init__(self, fs: int = 250):
        self.fs = fs

    def compute_rr_intervals(self, r_peaks: np.ndarray) -> np.ndarray:
        """Computes beat-to-beat RR intervals in milliseconds."""
        if len(r_peaks) < 2:
            return np.array([], dtype=float)
        rr_samples = np.diff(r_peaks)
        rr_ms = (rr_samples / self.fs) * 1000.0
        return rr_ms

    def compute_time_domain_hrv(self, rr_ms: np.ndarray) -> Dict[str, float]:
        """Calculates standard time-domain HRV indices."""
        if len(rr_ms) < 2:
            return {
                "mean_hr_bpm": 0.0,
                "mean_rr_ms": 0.0,
                "sdnn_ms": 0.0,
                "rmssd_ms": 0.0,
                "pnn50_pct": 0.0,
                "min_rr_ms": 0.0,
                "max_rr_ms": 0.0
            }

        mean_rr = float(np.mean(rr_ms))
        mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0
        sdnn = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 2 else 0.0

        # Successive differences
        diff_rr = np.diff(rr_ms)
        rmssd = float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) > 0 else 0.0
        nn50 = np.sum(np.abs(diff_rr) > 50.0)
        pnn50 = float((nn50 / len(diff_rr)) * 100.0) if len(diff_rr) > 0 else 0.0

        return {
            "mean_hr_bpm": round(mean_hr, 1),
            "mean_rr_ms": round(mean_rr, 1),
            "sdnn_ms": round(sdnn, 2),
            "rmssd_ms": round(rmssd, 2),
            "pnn50_pct": round(pnn50, 2),
            "min_rr_ms": round(float(np.min(rr_ms)), 1),
            "max_rr_ms": round(float(np.max(rr_ms)), 1)
        }

    def compute_poincare_dispersion(self, rr_ms: np.ndarray) -> Dict[str, float]:
        """
        Calculates non-linear Poincaré geometry ($RR_{n}$ vs $RR_{n+1}$):
        - SD1: Minor axis dispersion (instantaneous beat-to-beat variability)
        - SD2: Major axis dispersion (long-term continuous variability)
        - SD1/SD2 ratio
        """
        if len(rr_ms) < 3:
            return {"sd1_ms": 0.0, "sd2_ms": 0.0, "sd_ratio": 0.0}

        x = rr_ms[:-1]
        y = rr_ms[1:]

        diff_rr = x - y
        sd1 = float(np.sqrt(0.5 * np.var(diff_rr, ddof=1)))

        var_x = np.var(x, ddof=1)
        sd2_sq = 2.0 * var_x - 0.5 * np.var(diff_rr, ddof=1)
        sd2 = float(np.sqrt(max(1e-4, sd2_sq)))

        ratio = sd1 / sd2 if sd2 > 1e-4 else 0.0

        return {
            "sd1_ms": round(sd1, 2),
            "sd2_ms": round(sd2, 2),
            "sd_ratio": round(ratio, 3)
        }

    def analyze(self, r_peaks: np.ndarray) -> Dict[str, Any]:
        """Runs full HRV and non-linear dynamics analysis on detected R-peaks."""
        rr_ms = self.compute_rr_intervals(r_peaks)
        time_metrics = self.compute_time_domain_hrv(rr_ms)
        poincare_metrics = self.compute_poincare_dispersion(rr_ms)

        return {
            "num_detected_beats": len(r_peaks),
            "rr_intervals_ms": rr_ms.tolist(),
            **time_metrics,
            **poincare_metrics
        }

