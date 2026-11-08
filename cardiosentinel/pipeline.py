"""
CardioSentinel End-to-End Orchestrator Pipeline
Unifies preprocessing, dual QRS detection, Adaptive Signal Quality Intelligence (ASQI),
context-aware adaptive remediation, HRV extraction, evidential neural inference,
and Three-Zone clinical triage export (Accept / Review / Abstain).
"""

from pathlib import Path
import torch
import numpy as np
from typing import Dict, Any, Optional

from cardiosentinel.config import SignalConfig, SQIConfig, EvidentialConfig
from cardiosentinel.preprocessing.filters import ECGPreprocessor
from cardiosentinel.preprocessing.qrs_detector import PanTompkinsDetector, HamiltonDetector
from cardiosentinel.preprocessing.adaptive_denoiser import AdaptiveDenoiser
from cardiosentinel.sqi.sqi_engine import SQIEngine
from cardiosentinel.features.hrv import HRVExtractor
from cardiosentinel.models.hybrid_classifier import CardioSentinelClassifier
from cardiosentinel.models.evidential import ThreeZoneDecisionEngine
from cardiosentinel.export.fhir_export import FHIRObservationBuilder
from cardiosentinel.export.pdf_report import ClinicalReportGenerator


class CardioSentinelPipeline:
    """
    Production-grade end-to-end ambulatory arrhythmia screening engine.
    Ensures safe AI deployment with adaptive signal remediation and Three-Zone Dirichlet uncertainty bounds.
    """

    def __init__(
        self,
        fs: int = 250,
        model: Optional[CardioSentinelClassifier] = None,
        sqi_threshold: float = 0.65,
        uncertainty_threshold: float = 0.380
    ):
        self.fs = fs
        self.sqi_threshold = sqi_threshold
        self.uncertainty_threshold = uncertainty_threshold

        # Initialize subcomponents
        self.preprocessor = ECGPreprocessor(fs=fs)
        self.pan_tompkins = PanTompkinsDetector(fs=fs)
        self.hamilton = HamiltonDetector(fs=fs)
        self.adaptive_denoiser = AdaptiveDenoiser(fs=fs)
        self.sqi_engine = SQIEngine(fs=fs)
        self.hrv_extractor = HRVExtractor(fs=fs)
        self.decision_engine = ThreeZoneDecisionEngine(
            uncertainty_threshold=uncertainty_threshold,
            sqi_accept_threshold=sqi_threshold
        )

        # Classifier initialization
        if model is not None:
            self.model = model
        else:
            self.model = CardioSentinelClassifier(in_channels=1, num_classes=6)
            weights_path = Path(__file__).resolve().parent / "models" / "pretrained_weights.pt"
            if weights_path.exists():
                try:
                    state_dict = torch.load(weights_path, map_location="cpu")
                    self.model.load_state_dict(state_dict)
                except Exception:
                    pass
            self.model.eval()

        self.fhir_builder = FHIRObservationBuilder()
        self.report_generator = ClinicalReportGenerator()

    def process_window(
        self,
        raw_signal: np.ndarray,
        patient_id: str = "PATIENT-MRN-90214",
        patient_name: str = "Anonymous Patient"
    ) -> Dict[str, Any]:
        """
        Executes full CardioSentinel screening loop on a single ECG window (typically 10s).
        """
        raw_signal = np.asarray(raw_signal, dtype=np.float32)
        n_samples = len(raw_signal)

        # 1. Base Preprocessing (Notch + Bandpass)
        filtered_signal, normalized_signal = self.preprocessor.process(raw_signal)

        # 2. Dual QRS Detection
        peaks_pt = self.pan_tompkins.detect(filtered_signal)
        peaks_hamilton = self.hamilton.detect(filtered_signal)

        # 3. Adaptive Signal Quality Intelligence (ASQI) Evaluation
        sqi_result = self.sqi_engine.classify_signal_condition(
            raw_signal=raw_signal,
            filtered_signal=filtered_signal,
            peaks_pt=peaks_pt,
            peaks_hamilton=peaks_hamilton
        )

        # Backward compatibility for 'decision' key
        sqi_result["decision"] = "ACCEPT" if sqi_result["passed_gate"] else "REJECT_MOTION_CORRUPTED"

        condition = sqi_result.get("condition", "CLEAN")
        is_pathology = sqi_result.get("is_pathological", False)

        # 4. Adaptive Remediation (if mild artifact detected)
        remediation_meta = {"applied_filter": "NONE"}
        if condition in ("BASELINE_WANDER", "MILD_MOTION", "MUSCLE_INTERFERENCE"):
            filtered_signal, remediation_meta = self.adaptive_denoiser.apply_adaptive_remediation(
                filtered_signal, condition
            )
            # Re-normalize remediated signal
            normalized_signal = self.preprocessor.normalize(filtered_signal, method="robust")
            # Update detected peaks on cleaned signal
            peaks_pt = self.pan_tompkins.detect(filtered_signal)

        # 5. Gatekeeper & Decision Branching
        if not sqi_result["passed_gate"]:
            # ABSTAIN on severe noise to suppress false alarms
            decision = self.decision_engine.evaluate_decision(
                overall_sqi=sqi_result["overall_sqi"],
                epistemic_uncertainty=1.0,
                passed_sqi_gate=False,
                is_pathological_morphology=is_pathology
            )
            hrv_metrics = {
                "mean_hr_bpm": 0.0,
                "mean_rr_ms": 0.0,
                "sdnn_ms": 0.0,
                "rmssd_ms": 0.0,
                "pnn50_pct": 0.0,
                "num_detected_beats": len(peaks_pt)
            }
            inference_result = {
                "predicted_class_id": -1,
                "rhythm_diagnosis": "Artifact / Corrupted Recording",
                "confidence_score": 0.0,
                "epistemic_uncertainty": 1.0,
                "severity": "REJECTED_NOISE",
                "triage_status": "ABSTAIN_CORRUPTED_SIGNAL",
                "is_ambiguous": True,
                "three_zone_decision": decision,
                "reason": sqi_result["reason"]
            }
        else:
            # Signal cleared or pathology preserved: extract HRV & execute Evidential Inference
            hrv_metrics = self.hrv_extractor.analyze(peaks_pt)

            target_len = int(10.0 * self.fs)
            if len(normalized_signal) < target_len:
                padded_input = np.pad(normalized_signal, (0, target_len - len(normalized_signal)), mode='edge')
            else:
                padded_input = normalized_signal[:target_len]

            inference_result = self.model.predict_window(
                padded_input,
                uncertainty_threshold=self.uncertainty_threshold
            )

            # Evaluate 3-Zone Clinical Decision
            decision = self.decision_engine.evaluate_decision(
                overall_sqi=sqi_result["overall_sqi"],
                epistemic_uncertainty=inference_result["epistemic_uncertainty"],
                passed_sqi_gate=True,
                is_pathological_morphology=is_pathology
            )
            inference_result["three_zone_decision"] = decision

        payload = {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "sampling_rate": self.fs,
            "duration_sec": round(n_samples / self.fs, 2),
            "raw_signal": raw_signal.tolist(),
            "filtered_signal": filtered_signal.tolist(),
            "detected_r_peaks": peaks_pt.tolist(),
            "secondary_r_peaks": peaks_hamilton.tolist(),
            "sqi": sqi_result,
            "remediation": remediation_meta,
            "hrv": hrv_metrics,
            "inference": inference_result
        }
        return payload

    def export_fhir(self, analysis_result: Dict[str, Any], filepath: str) -> str:
        """Exports analysis results to an HL7 FHIR JSON bundle."""
        fhir_obj = self.fhir_builder.build_observation(
            analysis_result,
            patient_id=analysis_result.get("patient_id", "MRN-001"),
            patient_name=analysis_result.get("patient_name", "Anonymous")
        )
        return self.fhir_builder.export_to_json(fhir_obj, filepath)

    def export_pdf(self, analysis_result: Dict[str, Any], filepath: str) -> str:
        """Exports analysis results to a clinical PDF diagnostic summary."""
        return self.report_generator.generate(
            analysis_result,
            output_filepath=filepath,
            patient_name=analysis_result.get("patient_name", "Anonymous Patient"),
            patient_id=analysis_result.get("patient_id", "PATIENT-MRN-90214")
        )
