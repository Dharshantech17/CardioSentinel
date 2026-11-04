"""
CardioSentinel Interactive Clinical Telemetry Cockpit
A physician-facing web application for real-time leadless ECG rhythm visualization,
SQI gatekeeper analysis, evidential Dirichlet uncertainty quantification, and HL7 FHIR export.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import tempfile

from cardiosentinel.pipeline import CardioSentinelPipeline
from cardiosentinel.data.synthetic_ecg import ECGSynthesizer
from cardiosentinel.config import ARRHYTHMIA_CLASSES, ARRHYTHMIA_SEVERITY


st.set_page_config(
    page_title="CardioSentinel™ | Clinical Telemetry Cockpit",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical styling
st.markdown("""
<style>
    .reportview-container {
        background: #0f172a;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #334155;
        color: white;
    }
    .status-badge-pass {
        background-color: #166534;
        color: #4ade80;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .status-badge-fail {
        background-color: #991b1b;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .status-badge-warning {
        background-color: #854d0e;
        color: #facc15;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    return CardioSentinelPipeline(fs=250)


@st.cache_resource
def get_synthesizer():
    return ECGSynthesizer(sampling_rate=250)


pipeline = get_pipeline()
synthesizer = get_synthesizer()

# --- SIDEBAR CONTROLS ---
st.sidebar.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=64)
st.sidebar.title("CardioSentinel™")
st.sidebar.caption("AI Leadless Rhythm Monitor & SQI Engine")
st.sidebar.markdown("---")

st.sidebar.subheader("1. Clinical Scenario")
rhythm_option = st.sidebar.selectbox(
    "Select Cardiac Rhythm:",
    ["Normal Sinus Rhythm (NSR)", "Atrial Fibrillation (AFib)", "Premature Ventricular Contraction (PVC)",
     "Premature Atrial Contraction (PAC)", "Ventricular Tachycardia (VT)", "Sinus Bradycardia"],
    index=0
)

rhythm_key_map = {
    "Normal Sinus Rhythm (NSR)": "NSR",
    "Atrial Fibrillation (AFib)": "AFib",
    "Premature Ventricular Contraction (PVC)": "PVC",
    "Premature Atrial Contraction (PAC)": "PAC",
    "Ventricular Tachycardia (VT)": "VT",
    "Sinus Bradycardia": "Bradycardia"
}
selected_rhythm = rhythm_key_map[rhythm_option]

st.sidebar.markdown("---")
st.sidebar.subheader("2. Wearable Ambulatory Noise")
inject_lead_off = st.sidebar.checkbox("Trigger Lead-Off / Sensor Detached", value=False)
baseline_wander = st.sidebar.slider("Respiration / Baseline Drift (mV)", 0.0, 0.8, 0.05, 0.05)
emg_tremor = st.sidebar.slider("EMG Muscle Tremor (mV)", 0.0, 0.4, 0.02, 0.01)
motion_artifact = st.sidebar.slider("Motion Electrode Pops (mV)", 0.0, 2.0, 0.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Gatekeeper Thresholds")
sqi_threshold = st.sidebar.slider("Min SQI Acceptance Score", 0.40, 0.90, 0.65, 0.05)
uncertainty_limit = st.sidebar.slider("Max Evidential Uncertainty Bound", 0.20, 0.60, 0.38, 0.02)

st.sidebar.markdown("---")
patient_id = st.sidebar.text_input("Patient MRN:", value="PATIENT-MRN-90214")
patient_name = st.sidebar.text_input("Patient Name:", value="Eleanor Vance")


# --- GENERATE SIGNAL ---
clean_signal, true_peaks, meta = synthesizer.generate_rhythm(
    rhythm_type=selected_rhythm,
    duration_sec=10.0,
    random_seed=42
)

noisy_signal, _ = synthesizer.inject_ambulatory_noise(
    clean_signal,
    baseline_wander_level=baseline_wander,
    emg_tremor_level=emg_tremor,
    motion_artifact_level=motion_artifact,
    lead_off=inject_lead_off
)

# Process through pipeline
pipeline.sqi_threshold = sqi_threshold
pipeline.uncertainty_threshold = uncertainty_limit
results = pipeline.process_window(noisy_signal, patient_id=patient_id, patient_name=patient_name)

sqi = results["sqi"]
hrv = results["hrv"]
inference = results["inference"]


# --- TOP HEADER & CLINICAL BANNER ---
st.title("🫀 CardioSentinel™ Clinical Telemetry Cockpit")
st.markdown("**Edge-Aware Leadless Rhythm Monitor with Dynamic Signal Quality Assessment and Uncertainty Calibration**")

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    hr_val = hrv.get("mean_hr_bpm", 0.0)
    st.metric("Heart Rate", f"{hr_val} bpm", delta=f"{hrv.get('num_detected_beats', 0)} beats detected")

with col2:
    gate_passed = sqi.get("passed_gate", False)
    sqi_val = sqi.get("overall_sqi", 0.0)
    badge = "PASSED (Clean)" if gate_passed else "ABSTAINED (Noisy)"
    st.metric("Signal Quality (SQI)", f"{sqi_val:.3f}", delta=badge, delta_color="normal" if gate_passed else "inverse")

with col3:
    diag = inference.get("rhythm_diagnosis", "N/A")
    sev = inference.get("severity", "N/A")
    st.metric("Rhythm Diagnosis", diag, delta=f"Severity: {sev}")

with col4:
    uncert = inference.get("epistemic_uncertainty", 0.0)
    is_amb = inference.get("is_ambiguous", False)
    triage = inference.get("triage_status", "N/A")
    st.metric("Evidential Uncertainty", f"{uncert:.4f}", delta="Review Required" if is_amb else "Calibrated Safe", delta_color="inverse" if is_amb else "normal")


# --- RHYTHM STRIP DISPLAY ---
st.markdown("### 📈 Leadless ECG Rhythm Strip (10 Seconds @ 250 Hz)")

time_axis = np.linspace(0, 10.0, len(noisy_signal))
fig_ecg = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Raw Leadless Electrode Stream (mV)", "Filtered & QRS Denoised Rhythm Strip (mV)"))

# Raw
fig_ecg.add_trace(go.Scatter(x=time_axis, y=results["raw_signal"], name="Raw Leadless ECG",
                             line=dict(color="#94a3b8", width=1.2)), row=1, col=1)

# Filtered
fig_ecg.add_trace(go.Scatter(x=time_axis, y=results["filtered_signal"], name="Filtered Diagnostic Lead",
                             line=dict(color="#38bdf8", width=1.5)), row=2, col=1)

# Annotate R-Peaks if gatekeeper passed
if len(results["detected_r_peaks"]) > 0:
    r_times = [time_axis[idx] for idx in results["detected_r_peaks"] if idx < len(time_axis)]
    r_amps = [results["filtered_signal"][idx] for idx in results["detected_r_peaks"] if idx < len(results["filtered_signal"])]
    fig_ecg.add_trace(go.Scatter(
        x=r_times, y=r_amps,
        mode='markers',
        name='Pan-Tompkins R-Peaks',
        marker=dict(color='#ef4444', size=8, symbol='diamond')
    ), row=2, col=1)

fig_ecg.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=40, b=20),
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_ecg, use_container_width=True)


# --- TABBED CLINICAL DETAILS ---
tab_sqi, tab_evidential, tab_hrv, tab_export = st.tabs([
    "🛡️ SQI Gatekeeper", "🧠 Evidential AI & Uncertainty", "📊 HRV & Poincaré Dynamics", "📑 HL7 FHIR & PDF Export"
])

# Tab 1: SQI Gatekeeper
with tab_sqi:
    st.subheader("Multi-Tier Signal Quality Indexing (SQI)")
    st.write(f"**Gatekeeper Decision:** `{sqi.get('decision')}` — {sqi.get('reason')}")

    col_sqi1, col_sqi2 = st.columns([1, 1])

    with col_sqi1:
        metrics = ["Beat Concordance (bSQI)", "Spectral Ratio (sSQI)", "Kurtosis (kSQI)", "Baseline Stability (basSQI)"]
        values = [sqi.get("bsqi", 0.0), sqi.get("ssqi", 0.0), sqi.get("ksqi", 0.0), sqi.get("bassqi", 0.0)]
        thresholds = [0.70, 0.45, 0.20, 0.65]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=metrics, y=values, name="Measured Value",
            marker_color=['#10b981' if v >= t else '#f43f5e' for v, t in zip(values, thresholds)]
        ))
        fig_bar.add_trace(go.Scatter(
            x=metrics, y=thresholds, name="Clinical Min Threshold",
            mode='markers', marker=dict(color='yellow', size=12, symbol='line-ew-open')
        ))
        fig_bar.update_layout(
            title="Individual Quality Indices vs Physiological Thresholds",
            yaxis=dict(range=[0, 1.1]),
            template="plotly_dark",
            height=340
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_sqi2:
        st.markdown("""
        #### How CardioSentinel Prevents False Alarms:
        - **Beat Concordance ($bSQI$):** Cross-checks detections between the Pan-Tompkins filter bank and an independent derivative detector. If motion artifacts produce false peaks, $bSQI$ drops sharply.
        - **Spectral Energy ($sSQI$):** Verifies that signal energy is concentrated in the 5–15 Hz QRS depolarization band rather than high-frequency muscle tremor.
        - **Kurtosis ($kSQI$):** Assesses sharpness of physiological peaks over the isoelectric baseline.
        - **Baseline Drift ($basSQI$):** Flags low-frequency respiration distortion (< 1 Hz).
        """)
        if not gate_passed:
            st.warning("⚠️ **Gatekeeper Rejection Active:** Arrhythmia classification has abstained. Patient prompt: 'Please adjust patch sensor and avoid vigorous motion.'")
        else:
            st.success("✅ **Gatekeeper Cleared:** Signal is diagnostically pristine. Neural arrhythmia evaluation is authorized.")


# Tab 2: Evidential AI & Uncertainty
with tab_evidential:
    st.subheader("Dirichlet Evidential Deep Learning Analysis")

    if not gate_passed:
        st.info("Neural arrhythmia inference was bypassed by the SQI gatekeeper because the incoming window is corrupted by noise.")
    else:
        probs = inference.get("probabilities", {})
        classes = list(probs.keys())
        p_vals = [probs[c] for c in classes]

        col_ev1, col_ev2 = st.columns([1, 1])
        with col_ev1:
            fig_prob = go.Figure(go.Bar(
                x=p_vals, y=classes, orientation='h',
                marker=dict(color=['#38bdf8' if c != inference['rhythm_diagnosis'] else '#f59e0b' for c in classes])
            ))
            fig_prob.update_layout(
                title="Expected Class Probabilities E[p_k] = α_k / S",
                xaxis=dict(range=[0, 1.0]),
                template="plotly_dark",
                height=340
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        with col_ev2:
            st.markdown(f"""
            #### Uncertainty Calibration Diagnostics:
            - **Predicted Rhythm:** `{inference.get('rhythm_diagnosis')}`
            - **Model Confidence:** `{inference.get('confidence_score') * 100:.1f} %`
            - **Dirichlet Epistemic Uncertainty ($u = K / S$):** `{inference.get('epistemic_uncertainty'):.4f}`
            - **Safety Abstention Threshold:** `< {pipeline.uncertainty_threshold}`
            - **Triage Action:** `{inference.get('triage_status')}`
            """)
            if inference.get("is_ambiguous"):
                st.error("🚨 **High Epistemic Ambiguity:** Prediction confidence is low or contradictory. Flagged for priority cardiologist manual review.")
            else:
                st.success("🎯 **Calibrated Certainty:** Dirichlet strength is sufficient for automated ambulatory screening.")


# Tab 3: HRV & Poincaré Dynamics
with tab_hrv:
    st.subheader("Heart Rate Variability & Autonomic Tone")

    rr_intervals = hrv.get("rr_intervals_ms", [])
    col_hrv1, col_hrv2 = st.columns([1, 1])

    with col_hrv1:
        if len(rr_intervals) >= 3:
            rr_n = rr_intervals[:-1]
            rr_next = rr_intervals[1:]

            fig_poincare = go.Figure()
            fig_poincare.add_trace(go.Scatter(
                x=rr_n, y=rr_next, mode='markers',
                marker=dict(color='#38bdf8', size=9, opacity=0.8),
                name='RR[n] vs RR[n+1]'
            ))
            # Identity line
            min_rr = min(rr_intervals) * 0.9
            max_rr = max(rr_intervals) * 1.1
            fig_poincare.add_trace(go.Scatter(
                x=[min_rr, max_rr], y=[min_rr, max_rr],
                mode='lines', line=dict(color='#64748b', dash='dash'),
                name='Identity Line (Line of Identity)'
            ))
            fig_poincare.update_layout(
                title=f"Poincaré Plot (SD1={hrv.get('sd1_ms')}ms, SD2={hrv.get('sd2_ms')}ms)",
                xaxis_title="RR[n] (ms)",
                yaxis_title="RR[n+1] (ms)",
                template="plotly_dark",
                height=350
            )
            st.plotly_chart(fig_poincare, use_container_width=True)
        else:
            st.info("Insufficient R-peaks detected to construct Poincaré scatter plot.")

    with col_hrv2:
        st.markdown(f"""
        #### Time-Domain Autonomic Metrics:
        - **Mean Heart Rate:** `{hrv.get('mean_hr_bpm', 0.0)} bpm`
        - **Mean RR Interval:** `{hrv.get('mean_rr_ms', 0.0)} ms`
        - **SDNN (Total Autonomic Spread):** `{hrv.get('sdnn_ms', 0.0)} ms`
        - **RMSSD (Vagal Parasympathetic Tone):** `{hrv.get('rmssd_ms', 0.0)} ms`
        - **pNN50 (Erratic Rhythm Index):** `{hrv.get('pnn50_pct', 0.0)} %`
        - **Poincaré Dispersion Ratio (SD1/SD2):** `{hrv.get('sd_ratio', 0.0):.3f}`
        """)


# Tab 4: HL7 FHIR & Clinical PDF Export
with tab_export:
    st.subheader("Interoperability & Diagnostic Report Generation")

    col_exp1, col_exp2 = st.columns([1, 1])

    with col_exp1:
        st.markdown("#### 📄 Clinical Telemetry PDF Report")
        st.write("Generates an ANSI/AAMI EC57 compliant diagnostic summary with quality gate verification, telemetry tables, and cardiologist sign-off block.")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            tmp_pdf_path = tmp_pdf.name

        pipeline.export_pdf(results, tmp_pdf_path)
        with open(tmp_pdf_path, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="⬇️ Download Clinical Telemetry PDF Report",
            data=pdf_bytes,
            file_name=f"CardioSentinel_Report_{patient_id}.pdf",
            mime="application/pdf"
        )

    with col_exp2:
        st.markdown("#### 🏥 HL7 FHIR R4 Observation Bundle")
        st.write("Formats telemetry findings into standardized FHIR Observation resources with LOINC 8867-4, 131383, and 93041-2 codes for electronic health records.")

        fhir_obs = pipeline.fhir_builder.build_observation(results, patient_id=patient_id, patient_name=patient_name)
        fhir_json_str = json.dumps(fhir_obs, indent=2)

        st.download_button(
            label="⬇️ Download HL7 FHIR Observation (JSON)",
            data=fhir_json_str,
            file_name=f"fhir_observation_{patient_id}.json",
            mime="application/json"
        )

    st.markdown("---")
    st.markdown("#### FHIR Observation Resource Preview:")
    st.code(fhir_json_str[:1200] + "\n  ...\n}", language="json")

