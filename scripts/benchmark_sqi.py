"""
CardioSentinel SQI False-Alarm Suppression Benchmark
Quantitatively evaluates the gatekeeper performance over 200 simulated ambulatory windows:
100 clean diagnostic windows vs 100 noise/motion-corrupted windows.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from cardiosentinel.pipeline import CardioSentinelPipeline
from cardiosentinel.data.synthetic_ecg import ECGSynthesizer


def run_benchmark(n_trials: int = 100):
    print("=" * 80)
    print(f"CARDIOSENTINEL BENCHMARK: Testing False-Alarm Suppression across {n_trials * 2} windows")
    print("=" * 80)

    synthesizer = ECGSynthesizer(sampling_rate=250)
    pipeline = CardioSentinelPipeline(fs=250)

    # 1. Clean Cohort
    clean_accepted = 0
    clean_sqi_list = []

    print("\n[Phase 1] Evaluating Clean Diagnostic Windows...")
    for i in range(n_trials):
        rhythm = np.random.choice(["NSR", "AFib", "PVC", "PAC", "VT"])
        sig, _, _ = synthesizer.generate_rhythm(rhythm_type=rhythm, duration_sec=10.0, random_seed=1000 + i)
        noisy, _ = synthesizer.inject_ambulatory_noise(
            sig,
            baseline_wander_level=np.random.uniform(0.01, 0.08),
            emg_tremor_level=np.random.uniform(0.005, 0.03),
            motion_artifact_level=0.0
        )
        res = pipeline.process_window(noisy)
        clean_sqi_list.append(res["sqi"]["overall_sqi"])
        if res["sqi"]["passed_gate"]:
            clean_accepted += 1

    clean_pass_rate = (clean_accepted / n_trials) * 100.0
    mean_clean_sqi = np.mean(clean_sqi_list)

    # 2. Corrupted Noise Cohort (Walking / Motion / Loose electrode)
    noise_rejected = 0
    noise_sqi_list = []

    print("[Phase 2] Evaluating Motion-Corrupted / Lead-Off Windows...")
    for i in range(n_trials):
        sig, _, _ = synthesizer.generate_rhythm(rhythm_type="NSR", duration_sec=10.0, random_seed=5000 + i)
        is_lead_off = (i % 5 == 0)
        noisy, _ = synthesizer.inject_ambulatory_noise(
            sig,
            baseline_wander_level=np.random.uniform(0.25, 0.60),
            emg_tremor_level=np.random.uniform(0.12, 0.35),
            motion_artifact_level=np.random.uniform(0.8, 2.5),
            lead_off=is_lead_off
        )
        res = pipeline.process_window(noisy)
        noise_sqi_list.append(res["sqi"]["overall_sqi"])
        if not res["sqi"]["passed_gate"]:
            noise_rejected += 1

    noise_rejection_rate = (noise_rejected / n_trials) * 100.0
    mean_noise_sqi = np.mean(noise_sqi_list)

    print("\n" + "=" * 80)
    print("                    BENCHMARK RESULTS & METRICS SUMMARY")
    print("=" * 80)
    print(f"Clean Diagnostic Windows Passed Gate : {clean_accepted}/{n_trials} ({clean_pass_rate:.1f}%)")
    print(f"Mean SQI (Clean Diagnostic Cohort)    : {mean_clean_sqi:.3f}")
    print(f"Corrupted Noise Windows Rejected     : {noise_rejected}/{n_trials} ({noise_rejection_rate:.1f}%)")
    print(f"Mean SQI (Corrupted Noise Cohort)     : {mean_noise_sqi:.3f}")
    print(f"Ambulatory False-Alarm Suppression   : {noise_rejection_rate:.1f}% Reduction in Spurious Triggers")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_benchmark(n_trials=50)
