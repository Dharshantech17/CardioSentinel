"""
CardioSentinel PhysioNet & Standard Recording Loader
Parses standard ambulatory ECG data formats (CSV, WFDB records, NumPy)
for benchmark evaluation on MIT-BIH, CinC Challenge 2017/2021, and PTB-XL.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class PhysioNetLoader:
    """Loads and standardizes clinical ECG recordings from disk."""

    def __init__(self, target_fs: int = 250):
        self.target_fs = target_fs

    def load_from_csv(
        self,
        filepath: str,
        signal_column: int = 0,
        original_fs: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Loads an ECG lead column from a CSV file and resamples to target sampling rate.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"ECG file not found: {filepath}")

        df = pd.read_csv(filepath)
        raw_signal = df.iloc[:, signal_column].values.astype(np.float64)

        fs = original_fs if original_fs else self.target_fs
        if fs != self.target_fs:
            raw_signal = self.resample_signal(raw_signal, fs, self.target_fs)

        meta = {
            "source_file": str(path.name),
            "original_fs": fs,
            "target_fs": self.target_fs,
            "num_samples": len(raw_signal),
            "duration_sec": len(raw_signal) / self.target_fs
        }
        return raw_signal, meta

    @staticmethod
    def resample_signal(signal: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
        """Resamples ECG signal to match standard algorithmic sampling rate."""
        if orig_fs == target_fs:
            return signal
        from scipy.signal import resample
        target_len = int(len(signal) * target_fs / orig_fs)
        return resample(signal, target_len)
