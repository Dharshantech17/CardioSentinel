"""
CardioSentinel End-to-End Clinical Demonstration Script
Simulates diverse ambulatory scenarios: Normal Sinus Rhythm, Atrial Fibrillation,
PVC, VTach, Motion Artifacts (False Alarm Suppression), and Lead-off Disconnect.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from cardiosentinel.pipeline import CardioSentinelPipeline
from cardiosentinel.data.synthetic_ecg import ECGSynthesizer


def print_banner():
    banner = """
========================================================================================
   CARDIOSENTINEL: AI-POWERED LEADLESS AMBULATORY CARDIAC SCREENING ENGINE
   Multi-Tier SQI Gatekeeper + Dirichlet Evidential Uncertainty Deep Learning
========================================================================================
"""
    print(banner)


def run_all_scenarios():
    print_banner()

    synthesizer = ECGSynthesizer(sampling_rate=250)
    pipeline = CardioSentinelPipeline(fs=250)

    demo_dir = PROJECT_ROOT / "outputs" / "demo_reports"
    demo_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        {
            "id": 1,
            "name": "Normal Sinus Rhythm (Clean Ambulatory Patch)",
            "rhythm": "NSR",
            "noise": {"baseline_wander_level": 0.05, "emg_tremor_level": 0.02, "motion_artifact_level": 0.0},
            "expected_gate": "ACCEPT"
        },
        {
            "id": 2,
            "name": "Atrial Fibrillation (Paroxysmal Ambulatory Episode)",
            "rhythm": "AFib",
            "noise": {"baseline_wander_level": 0.08, "emg_tremor_level": 0.04, "motion_artifact_level": 0.0},
            "expected_gate": "ACCEPT"
        },
        {
            "id": 3,
            "name": "Frequent Premature Ventricular Contractions (PVC)",
            "rhythm": "PVC",
            "noise": {"baseline_wander_level": 0.06, "emg_tremor_level": 0.03, "motion_artifact_level": 0.0},
            "expected_gate": "ACCEPT"
        },
        {
            "id": 4,
            "name": "Severe Motion Corrupted Window (False Alarm Suppression Test)",
            "rhythm": "NSR",
            "noise": {"baseline_wander_level": 0.40, "emg_tremor_level": 0.20, "motion_artifact_level": 1.5},
            "expected_gate": "REJECT_MOTION_CORRUPTED"
        },
        {
            "id": 5,
            "name": "Sensor Detached / Lead-Off Condition",
            "rhythm": "NSR",
            "noise": {"lead_off": True},
            "expected_gate": "REJECT_LEAD_OFF"
        }
    ]

    for sc in scenarios:
        print(f"\n--- Scenario {sc['id']}: {sc['name']} ---")

        # 1. Synthesize cardiac rhythm
        clean_sig, true_peaks, meta = synthesizer.generate_rhythm(
            rhythm_type=sc["rhythm"],
            duration_sec=10.0,
            random_seed=42 + sc["id"]
        )

        # 2. Inject noise
        noisy_sig, _ = synthesizer.inject_ambulatory_noise(clean_sig, **sc["noise"])

        # 3. Process through CardioSentinel pipeline
        result = pipeline.process_window(
            noisy_sig,
            patient_id=f"PT-DEMO-{sc['id']:03d}",
            patient_name=f"Patient Cohort #{sc['id']}"
        )

        sqi = result["sqi"]
        hrv = result["hrv"]
        inf = result["inference"]

        print(f"  [SQI Gatekeeper] Decision: {sqi['decision']} | Overall SQI: {sqi['overall_sqi']:.3f} | bSQI: {sqi['bsqi']:.3f} | sSQI: {sqi['ssqi']:.3f}")
        print(f"  [Gatekeeper Reasoning]: {sqi['reason']}")

        if sqi["passed_gate"]:
            print(f"  [HRV Telemetry] Mean HR: {hrv['mean_hr_bpm']} bpm | RMSSD: {hrv['rmssd_ms']} ms | SDNN: {hrv['sdnn_ms']} ms | Poincaré SD1/SD2: {hrv['sd_ratio']:.3f}")
            print(f"  [Evidential AI] Diagnosis: {inf['rhythm_diagnosis']}")
            print(f"                  Confidence: {inf['confidence_score']*100:.1f}% | Dirichlet Epistemic Uncertainty: {inf['epistemic_uncertainty']:.4f}")
            print(f"                  Triage Status: {inf['triage_status']} (Severity: {inf['severity']})")
        else:
            print(f"  [SAFETY ACTION] Arrhythmia inference ABSTAINED. Clinical false alarm successfully suppressed!")

        # Export sample outputs for scenario 1 and 2
        if sc["id"] in (1, 2):
            fhir_path = demo_dir / f"scenario_{sc['id']}_fhir_observation.json"
            pdf_path = demo_dir / f"scenario_{sc['id']}_clinical_summary.pdf"

            pipeline.export_fhir(result, str(fhir_path))
            pipeline.export_pdf(result, str(pdf_path))
            print(f"  [Exports Generated] FHIR: {fhir_path.name} | PDF: {pdf_path.name}")

    print("\n" + "="*88)
    print(f"SUCCESS: CardioSentinel demonstration completed. Clinical reports saved to {demo_dir}")
    print("="*88 + "\n")


if __name__ == "__main__":
    run_all_scenarios()
