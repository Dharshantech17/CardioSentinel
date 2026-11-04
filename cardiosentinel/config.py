"""
CardioSentinel Configuration Module
Defines signal processing constants, SQI threshold matrices, model hyper-parameters,
and clinical risk mappings aligned with ANSI/AAMI EC57 and IEC 60601-2-47.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SignalConfig:
    """ECG Acquisition and Processing Specifications."""
    SAMPLING_RATE_HZ: int = 250  # Leadless wearable standard (250 Hz)
    WINDOW_DURATION_SEC: float = 10.0  # Standard clinical rhythm strip window (seconds)
    WINDOW_SAMPLES: int = int(SAMPLING_RATE_HZ * WINDOW_DURATION_SEC)  # 2500 samples
    
    # Filter specifications
    BANDPASS_LOWCUT_HZ: float = 0.5   # Suppresses respiration & baseline wander
    BANDPASS_HIGHCUT_HZ: float = 45.0  # Suppresses electromyographic (EMG) noise
    NOTCH_FREQ_HZ: float = 50.0       # European/Asian mains; 60.0 configurable
    NOTCH_Q: float = 30.0             # Quality factor for notch filter
    
    # QRS Detection Parameters
    MIN_RR_SEC: float = 0.28   # Max physiological HR ~ 214 bpm
    MAX_RR_SEC: float = 2.00   # Min physiological HR ~ 30 bpm
    REFRACTORY_MS: int = 200   # Refractory period in ms


@dataclass(frozen=True)
class SQIConfig:
    """Multi-Tier Signal Quality Indexing (SQI) Thresholds."""
    BSQI_MIN_AGREEMENT: float = 0.70  # Dual-detector peak matching threshold
    SSQI_MIN_SPECTRAL: float = 0.45   # QRS power ratio (5-15 Hz vs overall band)
    KSQI_MIN_KURTOSIS: float = 3.50   # Gaussianity check (ECG QRS spikes > 3.5)
    BASSQI_MAX_BASELINE: float = 0.35 # Max fraction of baseline wander energy (< 1 Hz)
    
    # Overall Gatekeeper Threshold
    OVERALL_SQI_ACCEPT_THRESHOLD: float = 0.65  # Below this: trigger ABSTAIN / ARTIFACT
    LEAD_OFF_AMPLITUDE_THRESHOLD_MV: float = 0.04 # Flatline / disconnected detection (< 40 uV)


@dataclass(frozen=True)
class EvidentialConfig:
    """Evidential Deep Learning Uncertainty Calibration Parameters."""
    EVIDENTIAL_UNCERTAINTY_THRESHOLD: float = 0.38  # Predictions with uncertainty > 0.38 prompt clinician review
    DIRICHLET_LAMBDA_REG: float = 0.005             # KL-divergence regularization weight
    NUM_CLASSES: int = 6


# Diagnostic Classification Lexicon
ARRHYTHMIA_CLASSES: Dict[int, str] = {
    0: "Normal Sinus Rhythm (NSR)",
    1: "Atrial Fibrillation (AFib)",
    2: "Premature Ventricular Contraction (PVC)",
    3: "Premature Atrial Contraction (PAC)",
    4: "Ventricular Tachycardia (VT)",
    5: "Sinus Bradycardia / Tachycardia"
}

ARRHYTHMIA_SEVERITY: Dict[int, str] = {
    0: "NORMAL",
    1: "URGENT",       # Paroxysmal AFib requires anticoagulant / rate-control consideration
    2: "INTERMEDIATE", # PVC frequency correlates with cardiomyopathy risks
    3: "MILD",         # PAC is typically benign unless triggering SVT
    4: "CRITICAL",     # Sustained VT is life-threatening emergency
    5: "MODERATE"      # Brady/Tachy requires etiology review
}

# Standard LOINC Codes for HL7 FHIR Interoperability
LOINC_CODES = {
    "HEART_RATE": "8867-4",          # Heart rate (beats/min)
    "ECG_IMPRESSION": "131383",      # Electrocardiogram impression
    "R_R_INTERVAL": "8627-2",        # R-R interval
    "SIGNAL_QUALITY": "93041-2"      # ECG signal quality score
}
