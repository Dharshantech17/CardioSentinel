"""
CardioSentinel Signal Filtering & Denoising Module
Implements zero-phase forward-backward Butterworth bandpass, IIR notch,
and robust baseline wander suppression compliant with IEC 60601-2-47.
"""

import numpy as np
from scipy import signal
from typing import Tuple


class ECGPreprocessor:
    """
    Standard ambulatory ECG filter chain optimized for leadless wearable hardware.
    Preserves diagnostic ST-segment and QRS peak morphology without phase shifts.
    """

    def __init__(
        self,
        fs: int = 250,
        lowcut: float = 0.5,
        highcut: float = 45.0,
        notch_freq: float = 50.0,
        notch_q: float = 30.0
    ):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.notch_q = notch_q

        # Pre-calculate filter coefficients
        nyq = 0.5 * fs
        self.b_band, self.a_band = signal.butter(
            N=3,
            Wn=[lowcut / nyq, highcut / nyq],
            btype='bandpass'
        )

        if notch_freq < nyq:
            self.b_notch, self.a_notch = signal.iirnotch(
                w0=notch_freq / nyq,
                Q=notch_q
            )
        else:
            self.b_notch, self.a_notch = None, None

    def apply_notch(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Removes 50Hz/60Hz mains interference."""
        if self.b_notch is not None and len(ecg_signal) > 15:
            return signal.filtfilt(self.b_notch, self.a_notch, ecg_signal)
        return ecg_signal

    def apply_bandpass(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Zero-phase Butterworth filtering (0.5 - 45 Hz)."""
        if len(ecg_signal) > 20:
            return signal.filtfilt(self.b_band, self.a_band, ecg_signal)
        return ecg_signal

    def remove_baseline_wander(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Suppresses low-frequency respiration drift using cascading median filters:
        - 200ms median filter (removes QRS and P waves)
        - 600ms median filter (removes T waves)
        - Subtract baseline from raw signal
        """
        w1 = int(0.20 * self.fs)
        w2 = int(0.60 * self.fs)
        if w1 % 2 == 0:
            w1 += 1
        if w2 % 2 == 0:
            w2 += 1

        baseline_est = signal.medfilt(ecg_signal, kernel_size=w1)
        baseline_est = signal.medfilt(baseline_est, kernel_size=w2)
        return ecg_signal - baseline_est

    def normalize(self, ecg_signal: np.ndarray, method: str = "robust") -> np.ndarray:
        """
        Normalizes ECG amplitude for neural network input.
        Method 'robust' uses median and interquartile range (resilient to large motion spikes).
        """
        if method == "robust":
            med = np.median(ecg_signal)
            iqr = np.percentile(ecg_signal, 75) - np.percentile(ecg_signal, 25)
            scale = iqr if iqr > 1e-4 else np.std(ecg_signal) + 1e-6
            return (ecg_signal - med) / scale
        else:
            mean = np.mean(ecg_signal)
            std = np.std(ecg_signal) + 1e-6
            return (ecg_signal - mean) / std

    def process(self, raw_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes full preprocessing pipeline.
        Returns:
            filtered_signal: Bandpassed and notch-filtered signal (physiologic units mV)
            normalized_signal: Scaled signal ready for neural inference
        """
        # 1. Notch filter
        step1 = self.apply_notch(raw_signal)
        # 2. Bandpass filter
        filtered = self.apply_bandpass(step1)
        # 3. Robust normalize
        normalized = self.normalize(filtered, method="robust")
        return filtered, normalized

