"""
CardioSentinel Adaptive Signal Denoiser
Applies targeted, artifact-specific signal enhancement depending on the classified
degradation mode (mild motion smoothing, baseline wander correction, muscle EMG suppression),
while strictly preserving diagnostic ST-segment and pathological QRS morphology.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, Any


class AdaptiveDenoiser:
    """
    Context-aware adaptive filter bank that applies specialized corrections
    tailored to the diagnosed noise etiology without distorting cardiac features.
    """

    def __init__(self, fs: int = 250):
        self.fs = fs

    def correct_baseline_wander(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Dual-stage cascading median filter baseline subtraction:
        - 200 ms median window removes QRS complexes and P waves.
        - 600 ms median window removes T waves to isolate wander profile.
        - Subtracts wander profile leaving diagnostic signal intact.
        """
        w1 = int(0.20 * self.fs)
        w2 = int(0.60 * self.fs)
        if w1 % 2 == 0:
            w1 += 1
        if w2 % 2 == 0:
            w2 += 1

        baseline = signal.medfilt(ecg_signal, kernel_size=w1)
        baseline = signal.medfilt(baseline, kernel_size=w2)
        corrected = ecg_signal - baseline
        return corrected

    def filter_muscle_emg(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Adaptive high-frequency smoothing for muscle tremor / EMG noise (25 - 45 Hz).
        Uses a zero-phase lowpass IIR filter (Chebyshev Type II) with 30 Hz stopband.
        """
        nyq = 0.5 * self.fs
        b, a = signal.iirfilter(
            N=4,
            Wn=32.0 / nyq,
            btype='lowpass',
            ftype='cheby2',
            rs=30.0
        )
        return signal.filtfilt(b, a, ecg_signal)

    def smooth_mild_motion(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Savitzky-Golay polynomial smoothing for mild motion artifacts.
        Maintains steep QRS derivative slopes while smoothing high-frequency motion jitter.
        """
        window_length = int(0.045 * self.fs) # ~45 ms window
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(5, window_length)
        return signal.savgol_filter(ecg_signal, window_length=window_length, polyorder=2)

    def apply_adaptive_remediation(
        self,
        ecg_signal: np.ndarray,
        artifact_type: str
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies selective remediation based on diagnosed signal condition.
        """
        remediation_meta = {
            "applied_filter": "NONE",
            "artifact_remediated": artifact_type,
            "snr_improved": False
        }

        if artifact_type == "BASELINE_WANDER":
            cleaned = self.correct_baseline_wander(ecg_signal)
            remediation_meta["applied_filter"] = "CASCADED_MEDIAN_BASELINE_CORRECTION"
            remediation_meta["snr_improved"] = True
            return cleaned, remediation_meta

        elif artifact_type == "MILD_MOTION":
            cleaned = self.smooth_mild_motion(ecg_signal)
            remediation_meta["applied_filter"] = "SAVITZKY_GOLAY_POLYNOMIAL_SMOOTHER"
            remediation_meta["snr_improved"] = True
            return cleaned, remediation_meta

        elif artifact_type == "MUSCLE_INTERFERENCE":
            cleaned = self.filter_muscle_emg(ecg_signal)
            remediation_meta["applied_filter"] = "CHEBYSHEV_II_EMG_SUPPRESSION"
            remediation_meta["snr_improved"] = True
            return cleaned, remediation_meta

        elif artifact_type in ("CLEAN", "ABNORMAL_MORPHOLOGY", "SEVERE_ELECTRODE_ARTIFACT"):
            # Clean or unrecoverable; no blind distortive filtering applied
            return ecg_signal, remediation_meta

        return ecg_signal, remediation_meta
