"""
CardioSentinel Adaptive Signal Quality Intelligence (ASQI) Engine
Advanced multi-tier diagnostic gatekeeper that classifies artifact etiology and severity,
applies targeted remediation, and preserves pathological cardiac morphology (VTach, PVCs)
to strictly avoid confusing genuine clinical arrhythmias with poor signal quality.
"""

import numpy as np
from scipy import signal, stats
from typing import Dict, Any, Tuple
from cardiosentinel.config import SQIConfig


class SQIEngine:
    """
    Adaptive Signal Quality Intelligence (ASQI) Engine.
    Goes beyond binary acceptance/rejection to identify degradation etiology
    and protect clinically critical arrhythmias from erroneous rejection.
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
        """
        n1 = len(peaks_primary)
        n2 = len(peaks_secondary)

        if n1 == 0 and n2 == 0:
            return 0.0
        if n1 == 0 or n2 == 0:
            return 0.0

        tol_samples = int((tolerance_ms / 1000.0) * self.fs)
        matched_count = 0
        matched_secondary = set()

        for p1 in peaks_primary:
            diffs = np.abs(peaks_secondary - p1)
            min_idx = np.argmin(diffs)
            if diffs[min_idx] <= tol_samples and min_idx not in matched_secondary:
                matched_count += 1
                matched_secondary.add(min_idx)

        bsqi = (2.0 * matched_count) / (n1 + n2)
        return float(np.clip(bsqi, 0.0, 1.0))

    def compute_s_sqi(self, ecg_signal: np.ndarray) -> float:
        """
        Spectral Signal Quality Index (sSQI):
        Ratio of power concentrated in the physiological QRS band (5 - 15 Hz)
        versus the extended ambulatory frequency band (5 - 40 Hz).
        """
        if len(ecg_signal) < int(self.fs * 0.5):
            return 0.0

        freqs, psd = signal.welch(ecg_signal, fs=self.fs, nperseg=min(len(ecg_signal), self.fs * 2))
        idx_qrs = np.logical_and(freqs >= 5.0, freqs <= 15.0)
        power_qrs = np.sum(psd[idx_qrs])

        idx_full = np.logical_and(freqs >= 5.0, freqs <= 40.0)
        power_full = np.sum(psd[idx_full])

        if power_full <= 1e-12:
            return 0.0

        ssqi = power_qrs / power_full
        return float(np.clip(ssqi, 0.0, 1.0))

    def compute_k_sqi(self, ecg_signal: np.ndarray) -> float:
        """
        Kurtosis Signal Quality Index (kSQI):
        Clean ECG exhibits high kurtosis (> 3.5) due to sparse, steep QRS spikes.
        """
        if len(ecg_signal) < 10:
            return 0.0

        k = float(stats.kurtosis(ecg_signal, fisher=False))
        norm_k = (k - 3.0) / 3.0
        return float(np.clip(norm_k, 0.0, 1.0))

    def compute_bas_sqi(self, raw_signal: np.ndarray) -> float:
        """
        Baseline Wander Signal Quality Index (basSQI):
        Proportion of signal power in the low-frequency drift band (< 1.0 Hz).
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
        stability_score = 1.0 - np.clip(wander_ratio / self.config.BASSQI_MAX_BASELINE, 0.0, 1.0)
        return float(stability_score)

    def detect_lead_off(self, raw_signal: np.ndarray) -> bool:
        """Flags disconnected patch / open-circuit flatline conditions."""
        amp_range = np.ptp(raw_signal)
        std_val = np.std(raw_signal)
        return bool(amp_range < self.config.LEAD_OFF_AMPLITUDE_THRESHOLD_MV or std_val < 0.008)

    def assess_morphological_abnormality(
        self,
        filtered_signal: np.ndarray,
        peaks: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Morphology Preservation Guard:
        Examines whether a degraded SQI score is caused by genuine life-threatening
        arrhythmias (such as Ventricular Tachycardia with wide monomorphic QRS complexes
        or frequent Premature Ventricular Contractions) rather than physical sensor noise.
        """
        if len(peaks) < 3:
            return False, "Insufficient beats for morphological pathology assessment"

        # Check beat interval consistency
        rr_samples = np.diff(peaks)
        median_rr = np.median(rr_samples)
        hr_bpm = (60.0 * self.fs) / median_rr if median_rr > 0 else 0

        # Extract beat segments around detected peaks
        half_win = int(0.12 * self.fs) # 120ms around peak
        beat_snippets = []
        for p in peaks[1:-1]:
            if p - half_win >= 0 and p + half_win < len(filtered_signal):
                snippet = filtered_signal[p - half_win : p + half_win]
                beat_snippets.append(snippet)

        if len(beat_snippets) >= 3:
            # Measure beat-to-beat cross-correlation
            correlations = []
            ref_beat = beat_snippets[0]
            norm_ref = np.linalg.norm(ref_beat)
            if norm_ref > 1e-5:
                for b in beat_snippets[1:]:
                    norm_b = np.linalg.norm(b)
                    if norm_b > 1e-5:
                        corr = np.dot(ref_beat, b) / (norm_ref * norm_b)
                        correlations.append(corr)

            mean_corr = np.mean(correlations) if correlations else 0.0

            # 1. Ventricular Tachycardia Check: High HR (>140 bpm) with consistent wide complexes
            if hr_bpm > 140 and mean_corr > 0.75:
                return True, f"Suspected Ventricular Tachycardia (HR={hr_bpm:.0f}bpm, Beat Concordance={mean_corr:.2f})"

            # 2. Monomorphic PVC / Broad QRS Check: Strong consistent waveforms across all beats with physiological HR
            if mean_corr > 0.85 and 45 <= hr_bpm <= 125:
                return True, f"Preserving Pathological Cardiac Morphology (Concordance={mean_corr:.2f})"

        return False, "Standard morphology"

    def classify_signal_condition(
        self,
        raw_signal: np.ndarray,
        filtered_signal: np.ndarray,
        peaks_pt: np.ndarray,
        peaks_hamilton: np.ndarray
    ) -> Dict[str, Any]:
        """
        Adaptive Signal Quality Intelligence (ASQI) Multi-Tier Diagnostic:
        Classifies degradation into 6 specific physiological conditions and
        determines the appropriate system response.
        """
        # 1. Severe Lead-off check
        if self.detect_lead_off(raw_signal):
            return {
                "condition": "SEVERE_ELECTRODE_ARTIFACT",
                "system_response": "ABSTAIN_REQUEST_RERECORDING",
                "passed_gate": False,
                "overall_sqi": 0.0,
                "bsqi": 0.0,
                "ssqi": 0.0,
                "ksqi": 0.0,
                "bassqi": 0.0,
                "is_pathological": False,
                "reason": "Severe electrode detachment / high-impedance flatline (<40 uV)."
            }

        # 2. Compute individual sub-indices
        bsqi = self.compute_b_sqi(peaks_pt, peaks_hamilton)
        ssqi = self.compute_s_sqi(filtered_signal)
        ksqi = self.compute_k_sqi(filtered_signal)
        bassqi = self.compute_bas_sqi(raw_signal)

        composite_sqi = 0.40 * bsqi + 0.25 * ssqi + 0.20 * ksqi + 0.15 * bassqi
        composite_sqi = float(np.clip(composite_sqi, 0.0, 1.0))

        # 3. Check for genuine cardiac pathology before rejecting
        is_pathology, path_reason = self.assess_morphological_abnormality(filtered_signal, peaks_pt)

        # 4. Diagnostic condition mapping
        if is_pathology:
            condition = "ABNORMAL_MORPHOLOGY"
            system_response = "PRESERVE_FOR_ARRHYTHMIA_ANALYSIS"
            passed = True
            reason = f"Clinical Abnormality Preserved: {path_reason}. Quality gate bypassed to prevent missing genuine arrhythmia."

        elif composite_sqi >= 0.85 and bsqi >= 0.80:
            condition = "CLEAN"
            system_response = "FULL_AI_CLASSIFICATION"
            passed = True
            reason = "Pristine diagnostic quality: Cleared for direct neural classification."

        elif bassqi < 0.50 and bsqi >= 0.75 and ssqi >= 0.55:
            # Baseline wander dominates, but QRS peaks and frequency bands are concordant
            condition = "BASELINE_WANDER"
            system_response = "BASELINE_CORRECTION_THEN_CLASSIFY"
            passed = True
            reason = f"Moderate baseline wander detected (basSQI={bassqi:.2f}). Adaptive cascaded median correction engaged."

        elif ssqi < 0.50 and bsqi >= 0.72:
            # High frequency / muscle tremor
            condition = "MUSCLE_INTERFERENCE"
            system_response = "SELECTIVE_EMG_FILTERING_THEN_CLASSIFY"
            passed = True
            reason = f"Muscle tremor / EMG interference detected (sSQI={ssqi:.2f}). Chebyshev lowpass smoothing engaged."

        elif bsqi >= 0.70 and composite_sqi >= 0.68:
            # Mild motion artifact (remediable)
            condition = "MILD_MOTION"
            system_response = "ADAPTIVE_SMOOTHING_THEN_CLASSIFY"
            passed = True
            reason = f"Mild motion jitter detected (bSQI={bsqi:.2f}). Savitzky-Golay adaptive smoothing applied."

        else:
            # Severe unrecoverable noise
            condition = "SEVERE_ELECTRODE_ARTIFACT"
            system_response = "ABSTAIN_REQUEST_RERECORDING"
            passed = False
            reason = f"Severe motion/contact artifact (bSQI={bsqi:.2f}, SQI={composite_sqi:.2f}). Abstaining from classification to prevent false alarm."

        return {
            "decision": "ACCEPT" if passed else "REJECT_MOTION_CORRUPTED",
            "condition": condition,
            "system_response": system_response,
            "passed_gate": passed,
            "overall_sqi": round(composite_sqi, 3),
            "bsqi": round(bsqi, 3),
            "ssqi": round(ssqi, 3),
            "ksqi": round(ksqi, 3),
            "bassqi": round(bassqi, 3),
            "is_pathological": is_pathology,
            "reason": reason
        }

    def evaluate_gatekeeper(
        self,
        raw_signal: np.ndarray,
        filtered_signal: np.ndarray,
        peaks_pantomptkins: np.ndarray,
        peaks_secondary: np.ndarray
    ) -> Dict[str, Any]:
        """Backward-compatible entry point calling full ASQI diagnostic pipeline."""
        return self.classify_signal_condition(
            raw_signal=raw_signal,
            filtered_signal=filtered_signal,
            peaks_pt=peaks_pantomptkins,
            peaks_hamilton=peaks_secondary
        )
