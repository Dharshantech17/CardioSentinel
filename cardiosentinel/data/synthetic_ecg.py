"""
CardioSentinel Synthetic ECG & Ambulatory Noise Generator
Implements physiologically grounded ECG waveforms parameterized by beat morphology
and realistic ambulatory wearable artifacts (EMG tremor, motion pops, baseline wander, lead-off).
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


class ECGSynthesizer:
    """
    Physiological ECG signal synthesizer capable of generating clean cardiac rhythms
    and realistic leadless wearable ambulatory noise for bench testing and validation.
    """

    def __init__(self, sampling_rate: int = 250):
        self.fs = sampling_rate

    def _generate_single_beat(
        self,
        duration: float,
        p_amp: float = 0.15,
        p_dur: float = 0.08,
        q_amp: float = -0.15,
        q_dur: float = 0.04,
        r_amp: float = 1.0,
        r_dur: float = 0.05,
        s_amp: float = -0.25,
        s_dur: float = 0.04,
        t_amp: float = 0.25,
        t_dur: float = 0.16,
        pr_interval: float = 0.16,
        st_interval: float = 0.10,
    ) -> np.ndarray:
        """Constructs an individual cardiac P-Q-R-S-T beat using Gaussian bell mixtures."""
        t = np.linspace(-0.25, duration - 0.25, int(self.fs * duration), endpoint=False)
        beat = np.zeros_like(t)

        def gaussian(x, center, amp, width):
            return amp * np.exp(-((x - center) ** 2) / (2 * (width / 2.355) ** 2))

        # P-wave
        if p_amp != 0:
            p_center = -pr_interval
            beat += gaussian(t, p_center, p_amp, p_dur)

        # Q-wave
        q_center = -0.025
        beat += gaussian(t, q_center, q_amp, q_dur)

        # R-peak (fiducial center at t=0)
        beat += gaussian(t, 0.0, r_amp, r_dur)

        # S-wave
        s_center = 0.035
        beat += gaussian(t, s_center, s_amp, s_dur)

        # T-wave
        t_center = st_interval + t_dur / 2.0
        beat += gaussian(t, t_center, t_amp, t_dur)

        return beat

    def generate_rhythm(
        self,
        rhythm_type: str = "NSR",
        duration_sec: float = 10.0,
        target_hr: Optional[float] = None,
        random_seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Synthesizes a 10-second (or custom length) leadless single-lead ECG recording.
        Returns:
            signal: np.ndarray [N] (mV)
            r_peaks: np.ndarray [M] (indices of true R-peaks)
            meta: Dict describing rhythm parameters
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        n_samples = int(self.fs * duration_sec)
        time_axis = np.linspace(0, duration_sec, n_samples, endpoint=False)
        signal = np.zeros(n_samples)
        r_peaks = []

        if rhythm_type == "NSR":
            # Normal Sinus Rhythm: HR 60-85, subtle respiratory sinus arrhythmia (RSA)
            hr = target_hr if target_hr else np.random.uniform(68.0, 78.0)
            base_rr = 60.0 / hr
            current_time = 0.4

            while current_time < duration_sec - 0.3:
                # Add slight RSA breathing modulation
                rsa_mod = 0.03 * np.sin(2 * np.pi * 0.25 * current_time)
                rr = base_rr + rsa_mod + np.random.normal(0, 0.012)
                r_idx = int(current_time * self.fs)
                r_peaks.append(r_idx)

                beat = self._generate_single_beat(duration=rr)
                beat_len = len(beat)
                start_idx = r_idx - int(0.25 * self.fs)
                end_idx = start_idx + beat_len

                valid_start = max(0, start_idx)
                valid_end = min(n_samples, end_idx)
                b_start = valid_start - start_idx
                b_end = b_start + (valid_end - valid_start)

                if valid_end > valid_start and b_end <= beat_len:
                    signal[valid_start:valid_end] += beat[b_start:b_end]

                current_time += rr

        elif rhythm_type == "AFib":
            # Atrial Fibrillation: Irregularly irregular RR, no P-wave, fibrillatory f-waves (4-8 Hz)
            hr = target_hr if target_hr else np.random.uniform(95.0, 130.0)
            current_time = 0.3

            # Baseline fibrillatory waves
            f_wave_freq = np.random.uniform(5.5, 7.5)
            f_waves = 0.08 * np.sin(2 * np.pi * f_wave_freq * time_axis + np.random.uniform(0, 2*np.pi))
            f_waves += 0.04 * np.sin(2 * np.pi * (f_wave_freq * 1.4) * time_axis)
            signal += f_waves

            while current_time < duration_sec - 0.25:
                # Gamma/Poisson distributed RR intervals typical of AFib AV-conduction
                rr = np.random.exponential(scale=60.0 / hr)
                rr = np.clip(rr, 0.35, 1.1)
                r_idx = int(current_time * self.fs)
                r_peaks.append(r_idx)

                # Beat with NO P-WAVE
                beat = self._generate_single_beat(
                    duration=rr,
                    p_amp=0.0,  # Absent P-wave
                    r_amp=np.random.uniform(0.85, 1.15),
                    t_amp=0.20
                )
                beat_len = len(beat)
                start_idx = r_idx - int(0.15 * self.fs)
                end_idx = start_idx + beat_len

                valid_start = max(0, start_idx)
                valid_end = min(n_samples, end_idx)
                b_start = valid_start - start_idx
                b_end = b_start + (valid_end - valid_start)

                if valid_end > valid_start and b_end <= beat_len:
                    signal[valid_start:valid_end] += beat[b_start:b_end]

                current_time += rr

        elif rhythm_type == "PVC":
            # Premature Ventricular Contraction: normal beats interrupted by wide, bizarre QRS with compensatory pause
            hr = target_hr if target_hr else 72.0
            base_rr = 60.0 / hr
            current_time = 0.4
            beat_counter = 0

            while current_time < duration_sec - 0.4:
                beat_counter += 1
                is_pvc = (beat_counter % 4 == 0)  # Quadrigeminy PVC pattern

                if is_pvc:
                    rr = base_rr * 0.65  # Premature coupling interval
                    r_idx = int(current_time * self.fs)
                    r_peaks.append(r_idx)
                    # Wide, bizarre QRS: inverted/discordant T-wave, no P-wave
                    pvc_beat = self._generate_single_beat(
                        duration=base_rr * 1.35, # Compensatory pause follows
                        p_amp=0.0,
                        q_amp=0.0,
                        r_amp=-1.3,      # Deep broad monomorphic deflection
                        r_dur=0.13,      # QRS duration > 120 ms
                        s_amp=0.4,
                        t_amp=0.55,      # Discordant tall T
                        t_dur=0.22,
                        st_interval=0.12
                    )
                    start_idx = r_idx - int(0.20 * self.fs)
                    end_idx = start_idx + len(pvc_beat)
                    valid_start = max(0, start_idx)
                    valid_end = min(n_samples, end_idx)
                    b_start = valid_start - start_idx
                    b_end = b_start + (valid_end - valid_start)
                    if valid_end > valid_start and b_end <= len(pvc_beat):
                        signal[valid_start:valid_end] += pvc_beat[b_start:b_end]

                    current_time += base_rr * 1.35  # Compensatory pause
                else:
                    rr = base_rr + np.random.normal(0, 0.015)
                    r_idx = int(current_time * self.fs)
                    r_peaks.append(r_idx)
                    beat = self._generate_single_beat(duration=rr)
                    start_idx = r_idx - int(0.25 * self.fs)
                    end_idx = start_idx + len(beat)
                    valid_start = max(0, start_idx)
                    valid_end = min(n_samples, end_idx)
                    b_start = valid_start - start_idx
                    b_end = b_start + (valid_end - valid_start)
                    if valid_end > valid_start and b_end <= len(beat):
                        signal[valid_start:valid_end] += beat[b_start:b_end]
                    current_time += rr

        elif rhythm_type == "PAC":
            # Premature Atrial Contraction: early abnormal P-wave, narrow QRS, non-compensatory pause
            hr = target_hr if target_hr else 75.0
            base_rr = 60.0 / hr
            current_time = 0.4
            beat_counter = 0

            while current_time < duration_sec - 0.35:
                beat_counter += 1
                is_pac = (beat_counter % 5 == 0)

                if is_pac:
                    rr = base_rr * 0.70  # Early beat
                    r_idx = int(current_time * self.fs)
                    r_peaks.append(r_idx)
                    pac_beat = self._generate_single_beat(
                        duration=base_rr * 1.15,
                        p_amp=0.28,      # Peaked ectopic P-wave
                        p_dur=0.06,
                        pr_interval=0.12,
                        r_amp=1.05
                    )
                    start_idx = r_idx - int(0.20 * self.fs)
                    end_idx = start_idx + len(pac_beat)
                    valid_start = max(0, start_idx)
                    valid_end = min(n_samples, end_idx)
                    b_start = valid_start - start_idx
                    b_end = b_start + (valid_end - valid_start)
                    if valid_end > valid_start and b_end <= len(pac_beat):
                        signal[valid_start:valid_end] += pac_beat[b_start:b_end]
                    current_time += base_rr * 1.15
                else:
                    rr = base_rr + np.random.normal(0, 0.012)
                    r_idx = int(current_time * self.fs)
                    r_peaks.append(r_idx)
                    beat = self._generate_single_beat(duration=rr)
                    start_idx = r_idx - int(0.25 * self.fs)
                    end_idx = start_idx + len(beat)
                    valid_start = max(0, start_idx)
                    valid_end = min(n_samples, end_idx)
                    b_start = valid_start - start_idx
                    b_end = b_start + (valid_end - valid_start)
                    if valid_end > valid_start and b_end <= len(beat):
                        signal[valid_start:valid_end] += beat[b_start:b_end]
                    current_time += rr

        elif rhythm_type == "VT":
            # Ventricular Tachycardia: Rapid regular broad ventricular complexes (160-190 bpm)
            hr = target_hr if target_hr else 175.0
            rr = 60.0 / hr
            current_time = 0.25

            while current_time < duration_sec - 0.2:
                r_idx = int(current_time * self.fs)
                r_peaks.append(r_idx)
                vt_beat = self._generate_single_beat(
                    duration=rr,
                    p_amp=0.0,
                    q_amp=-0.2,
                    r_amp=1.45,
                    r_dur=0.12,  # Broad QRS
                    s_amp=-0.65,
                    s_dur=0.08,
                    t_amp=-0.4,
                    t_dur=0.14
                )
                start_idx = r_idx - int(0.12 * self.fs)
                end_idx = start_idx + len(vt_beat)
                valid_start = max(0, start_idx)
                valid_end = min(n_samples, end_idx)
                b_start = valid_start - start_idx
                b_end = b_start + (valid_end - valid_start)
                if valid_end > valid_start and b_end <= len(vt_beat):
                    signal[valid_start:valid_end] += vt_beat[b_start:b_end]
                current_time += rr

        elif rhythm_type in ("Bradycardia", "Tachycardia"):
            hr = target_hr if target_hr else (44.0 if rhythm_type == "Bradycardia" else 125.0)
            rr = 60.0 / hr
            current_time = 0.4

            while current_time < duration_sec - 0.3:
                r_idx = int(current_time * self.fs)
                r_peaks.append(r_idx)
                beat = self._generate_single_beat(duration=rr)
                start_idx = r_idx - int(0.25 * self.fs)
                end_idx = start_idx + len(beat)
                valid_start = max(0, start_idx)
                valid_end = min(n_samples, end_idx)
                b_start = valid_start - start_idx
                b_end = b_start + (valid_end - valid_start)
                if valid_end > valid_start and b_end <= len(beat):
                    signal[valid_start:valid_end] += beat[b_start:b_end]
                current_time += rr
        else:
            raise ValueError(f"Unknown rhythm type: {rhythm_type}")

        meta = {
            "rhythm_type": rhythm_type,
            "sample_rate": self.fs,
            "duration_sec": duration_sec,
            "total_samples": n_samples,
            "num_beats": len(r_peaks),
            "estimated_hr": (len(r_peaks) / duration_sec) * 60.0
        }
        return signal, np.array(r_peaks, dtype=int), meta

    def inject_ambulatory_noise(
        self,
        clean_signal: np.ndarray,
        baseline_wander_level: float = 0.15,
        emg_tremor_level: float = 0.05,
        motion_artifact_level: float = 0.0,
        mains_50hz_level: float = 0.02,
        lead_off: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Simulates realistic leadless wearable noise sources encountered during ambulatory monitoring.
        """
        n = len(clean_signal)
        t = np.linspace(0, n / self.fs, n, endpoint=False)
        noisy_signal = clean_signal.copy()
        noise_info = {}

        if lead_off:
            # Complete patch detachment / high impedance: flatline with 50Hz mains hum
            flatline = 0.02 * np.sin(2 * np.pi * 50.0 * t) + np.random.normal(0, 0.01, n)
            noise_info["lead_off"] = True
            return flatline, noise_info

        # 1. Respiration & Baseline Wander (< 0.5 Hz)
        if baseline_wander_level > 0:
            bw_freq1 = np.random.uniform(0.18, 0.30) # Respiration
            bw_freq2 = np.random.uniform(0.05, 0.12) # Postural slow drift
            bw = baseline_wander_level * (
                np.sin(2 * np.pi * bw_freq1 * t) + 0.6 * np.cos(2 * np.pi * bw_freq2 * t)
            )
            noisy_signal += bw

        # 2. Electromyographic (EMG) Muscle Tremor (High frequency Gaussian noise filtered)
        if emg_tremor_level > 0:
            emg = np.random.normal(0, emg_tremor_level, n)
            noisy_signal += emg

        # 3. Motion Artifacts (Sudden baseline shifts / electrode contact pops)
        if motion_artifact_level > 0:
            # 2 to 3 sudden capacitive step pops
            num_pops = np.random.randint(1, 4)
            for _ in range(num_pops):
                pop_idx = np.random.randint(int(0.15 * n), int(0.85 * n))
                pop_dur = int(np.random.uniform(0.2, 0.6) * self.fs)
                pop_profile = np.hanning(min(pop_dur, n - pop_idx)) * motion_artifact_level * np.random.choice([-1.5, 1.8])
                noisy_signal[pop_idx : pop_idx + len(pop_profile)] += pop_profile

        # 4. Powerline Interference (50 Hz / 60 Hz hum)
        if mains_50hz_level > 0:
            mains = mains_50hz_level * np.sin(2 * np.pi * 50.0 * t)
            noisy_signal += mains

        noise_info.update({
            "baseline_wander_level": baseline_wander_level,
            "emg_tremor_level": emg_tremor_level,
            "motion_artifact_level": motion_artifact_level,
            "mains_50hz_level": mains_50hz_level,
            "lead_off": False
        })
        return noisy_signal, noise_info
