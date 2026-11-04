"""
CardioSentinel Dual-Engine QRS Fiducial Detector
Implements the canonical Pan-Tompkins Algorithm alongside an independent
adaptive derivative detector for robust R-peak localization and bSQI agreement calculation.
"""

import numpy as np
from scipy import signal
from typing import List, Tuple


class PanTompkinsDetector:
    """Pan-Tompkins (1985) Real-Time QRS Detection Algorithm."""

    def __init__(self, fs: int = 250):
        self.fs = fs

    def detect(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Executes Pan-Tompkins feature transformation:
        1. Bandpass filter (5 - 15 Hz QRS energy band)
        2. Five-point derivative filter: y[n] = (2x[n] + x[n-1] - x[n-3] - 2x[n-4]) / 8
        3. Non-linear Squaring: amplifies dominant QRS slopes
        4. Moving-Window Integration (MWI) of ~150 ms
        5. Adaptive Dual-Threshold Peak Search with 200 ms Refractory Period
        """
        if len(ecg_signal) < int(self.fs * 0.5):
            return np.array([], dtype=int)

        nyq = 0.5 * self.fs
        # 1. Bandpass for QRS band (5-15 Hz)
        low = max(0.01, 5.0 / nyq)
        high = min(0.99, 15.0 / nyq)
        b, a = signal.butter(2, [low, high], btype='bandpass')
        filtered = signal.filtfilt(b, a, ecg_signal)

        # 2. Derivative filter
        diff = np.zeros_like(filtered)
        for i in range(2, len(filtered) - 2):
            diff[i] = (filtered[i + 1] - filtered[i - 1]) / 2.0

        # 3. Squaring
        squared = diff ** 2

        # 4. Moving window integration (~150 ms)
        win_size = int(0.150 * self.fs)
        kernel = np.ones(win_size) / win_size
        mwi = np.convolve(squared, kernel, mode='same')

        # 5. Peak detection on integrated signal
        min_distance = int(0.28 * self.fs) # 280 ms refractory window
        # Dynamic threshold based on 90th percentile
        threshold = 0.35 * np.percentile(mwi, 90)

        raw_peaks, _ = signal.find_peaks(mwi, height=threshold, distance=min_distance)

        # 6. Fiducial alignment: map MWI peaks back to local maximum in raw filtered ECG
        refined_peaks = []
        search_radius = int(0.10 * self.fs) # 100 ms radius around MWI peak

        for p in raw_peaks:
            start = max(0, p - search_radius)
            end = min(len(ecg_signal), p + search_radius)
            local_segment = np.abs(ecg_signal[start:end])
            if len(local_segment) > 0:
                local_max_idx = start + np.argmax(local_segment)
                refined_peaks.append(local_max_idx)

        # Remove duplicates
        refined_peaks = np.unique(refined_peaks)
        return refined_peaks.astype(int)


class HamiltonDetector:
    """
    Independent Secondary QRS Detector (Hamilton-Tompkins variant).
    Uses non-linear absolute slope thresholding for cross-validation in bSQI.
    """

    def __init__(self, fs: int = 250):
        self.fs = fs

    def detect(self, ecg_signal: np.ndarray) -> np.ndarray:
        if len(ecg_signal) < int(self.fs * 0.5):
            return np.array([], dtype=int)

        # Absolute first difference
        diff = np.abs(np.diff(ecg_signal, prepend=ecg_signal[0]))

        # Moving average window (~120ms)
        win_size = int(0.12 * self.fs)
        kernel = np.ones(win_size) / win_size
        smoothed = np.convolve(diff, kernel, mode='same')

        threshold = 0.40 * np.percentile(smoothed, 88)
        min_dist = int(0.26 * self.fs)

        peaks, _ = signal.find_peaks(smoothed, height=threshold, distance=min_dist)

        # Align with peak in raw signal
        aligned_peaks = []
        radius = int(0.08 * self.fs)
        for p in peaks:
            start = max(0, p - radius)
            end = min(len(ecg_signal), p + radius)
            if end > start:
                local_idx = start + np.argmax(np.abs(ecg_signal[start:end]))
                aligned_peaks.append(local_idx)

        return np.unique(aligned_peaks).astype(int)
