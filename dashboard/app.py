"""
CardioSentinel™ Clinical Telemetry Cockpit
A physician-facing web application structured into five core clinical sections:
  1. Live ECG Visualization & RR Interval Dynamics
  2. Adaptive Signal Quality Monitoring (ASQI & Remediation)
  3. Arrhythmia Screening & Morphological Analysis
  4. Evidential Uncertainty & Three-Zone Decision Monitoring
  5. Clinician Review, Governance & Export (HL7 FHIR & PDF Report)
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

# Custom clinical styling
st.markdown("""
<style>
    .reportview-container { background: #0b1120; }
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 14px;
        border: 1px solid #334155;
        color: white;
    }
    .disclaimer-banner {
        background-color: #172554;
        border-left: 5px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 15px;
        color: #bfdbfe;
        font-size: 13px;
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

# --- SIDEBAR TELEMETRY CONTROLS ---
st.sidebar.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=60)
st.sidebar.title("CardioSentinel™")
st.sidebar.caption("Adaptive Quality-Aware Wearable Screening")
st.sidebar.markdown("---")

st.sidebar.subheader("1. Clinical Episode Selector")
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
st.sidebar.subheader("2. Ambulatory Noise Stress Simulator")
inject_lead_off = st.sidebar.checkbox("Electrode Detachment / Lead-Off", value=False)
baseline_wander = st.sidebar.slider("Respiration Drift (<1 Hz)", 0.0, 0.8, 0.04, 0.02)
emg_tremor = st.sidebar.slider("EMG Muscle Tremor (25-45 Hz)", 0.0, 0.4, 0.02, 0.01)
motion_artifact = st.sidebar.slider("Motion Electrode Pops (mV)", 0.0, 2.0, 0.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Patient Demographics")
patient_id = st.sidebar.text_input("Patient MRN:", value="PATIENT-MRN-90214")
patient_name = st.sidebar.text_input("Patient Name:", value="Eleanor Vance")

# --- GENERATE & PROCESS WINDOW ---
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

results = pipeline.process_window(noisy_signal, patient_id=patient_id, patient_name=patient_name)
sqi = results["sqi"]
hrv = results["hrv"]
inference = results["inference"]
remediation = results.get("remediation", {})
decision = inference.get("three_zone_decision", {})

# --- CLINICAL GOVERNANCE BANNER ---
st.markdown("""
<div class="disclaimer-banner">
    <b>ℹ️ CLINICAL SCREENING NOTICE:</b> CardioSentinel provides algorithmic screening decision-support adhering to ANSI/AAMI EC57 and IEC 60601-2-47. 
    This system distinguishes automated screening predictions from confirmed clinical diagnoses. Any episode tagged for <b>REVIEW</b> or classified with high evidential uncertainty requires confirmatory 12-lead ECG review by an attending cardiologist.
</div>
""", unsafe_allow_html=True)

st.title("🫀 CardioSentinel™ Clinical Cockpit")
st.caption("Adaptive Signal Quality Intelligence • Multi-Scale Evidential Inference • Dirichlet Uncertainty Calibration")

# Top KPI Summary Cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    hr_val = hrv.get("mean_hr_bpm", 0.0)
    st.metric("Mean Heart Rate", f"{hr_val} bpm", delta=f"{hrv.get('num_detected_beats', 0)} beats")

with kpi2:
    cond = sqi.get("condition", "N/A")
    passed = sqi.get("passed_gate", False)
    st.metric("Signal Quality (ASQI)", f"{sqi.get('overall_sqi', 0.0):.3f}", delta=cond, delta_color="normal" if passed else "inverse")

with kpi3:
    diag = inference.get("rhythm_diagnosis", "N/A")
    st.metric("Screening Impression", diag, delta=f"Sev: {inference.get('severity', 'N/A')}")

with kpi4:
    uncert = inference.get("epistemic_uncertainty", 0.0)
    st.metric("Evidential Uncertainty", f"{uncert:.4f}", delta="Safety < 0.380", delta_color="normal" if uncert <= 0.380 else "inverse")

with kpi5:
    zone = decision.get("decision_zone", "N/A")
    st.metric("Clinical Triage Zone", zone, delta="Action Authorized" if zone == "ACCEPT" else "Requires Review")


# --- THE FIVE CORE CLINICAL SECTIONS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. 📈 Live ECG Visualization",
    "2. 🛡️ Signal Quality & Remediation",
    "3. 🫀 Arrhythmia Analysis",
    "4. ⚖️ Uncertainty Monitoring",
    "5. 👨‍⚕️ Clinician Review & Export"
])


# -------------------------------------------------------------
# 1. LIVE ECG VISUALIZATION
# -------------------------------------------------------------
with tab1:
    st.subheader("1. Leadless Single-Lead Rhythm Strip (10 Seconds @ 250 Hz)")

    time_axis = np.linspace(0, 10.0, len(noisy_signal))
    fig_ecg = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Raw Leadless Electrode Stream (mV)", "Filtered & Adaptively Cleaned Rhythm Strip (mV)")
    )

    # Raw waveform
    fig_ecg.add_trace(go.Scatter(
        x=time_axis, y=results["raw_signal"], name="Raw Leadless Stream",
        line=dict(color="#94a3b8", width=1.2)
    ), row=1, col=1)

    # Filtered / remediated waveform
    fig_ecg.add_trace(go.Scatter(
        x=time_axis, y=results["filtered_signal"], name="Filtered Diagnostic Lead",
        line=dict(color="#38bdf8", width=1.6)
    ), row=2, col=1)

    # Annotate R-Peaks
    if len(results["detected_r_peaks"]) > 0:
        r_times = [time_axis[idx] for idx in results["detected_r_peaks"] if idx < len(time_axis)]
        r_amps = [results["filtered_signal"][idx] for idx in results["detected_r_peaks"] if idx < len(results["filtered_signal"])]
        fig_ecg.add_trace(go.Scatter(
            x=r_times, y=r_amps, mode='markers', name='QRS Fiducial Peaks',
            marker=dict(color='#ef4444', size=8, symbol='diamond')
        ), row=2, col=1)

    fig_ecg.update_layout(
        height=380, margin=dict(l=10, r=10, t=35, b=10), template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_ecg, use_container_width=True)

    # Sub-panel: RR interval dynamics
    rr_intervals = hrv.get("rr_intervals_ms", [])
    if len(rr_intervals) > 0:
        col_rr1, col_rr2, col_rr3, col_rr4 = st.columns(4)
        col_rr1.metric("Mean R-R Interval", f"{hrv.get('mean_rr_ms', 0.0)} ms")
        col_rr2.metric("Min R-R Interval", f"{hrv.get('min_rr_ms', 0.0)} ms")
        col_rr3.metric("Max R-R Interval", f"{hrv.get('max_rr_ms', 0.0)} ms")
        col_rr4.metric("Autonomic RMSSD", f"{hrv.get('rmssd_ms', 0.0)} ms")


# -------------------------------------------------------------
# 2. SIGNAL QUALITY MONITORING & ADAPTIVE REMEDIATION
# -------------------------------------------------------------
with tab2:
    st.subheader("2. Adaptive Signal Quality Intelligence (ASQI) & Remediation")

    col_sqi1, col_sqi2 = st.columns([1, 1])

    with col_sqi1:
        metrics = ["Concordance (bSQI)", "Spectral Band (sSQI)", "Kurtosis (kSQI)", "Baseline Stability (basSQI)"]
        values = [sqi.get("bsqi", 0.0), sqi.get("ssqi", 0.0), sqi.get("ksqi", 0.0), sqi.get("bassqi", 0.0)]
        thresholds = [0.70, 0.45, 0.20, 0.65]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=metrics, y=values, name="Measured Value",
            marker_color=['#10b981' if v >= t else '#f43f5e' for v, t in zip(values, thresholds)]
        ))
        fig_bar.add_trace(go.Scatter(
            x=metrics, y=thresholds, name="Minimum Quality Threshold",
            mode='markers', marker=dict(color='yellow', size=12, symbol='line-ew-open')
        ))
        fig_bar.update_layout(
            title="Individual Quality Indices vs Physiological Thresholds",
            yaxis=dict(range=[0, 1.1]), template="plotly_dark", height=320
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_sqi2:
        st.markdown(f"""
        #### ASQI Diagnostic Assessment:
        - **Classified Condition:** `{sqi.get('condition')}`
        - **System Action:** `{sqi.get('system_response')}`
        - **Remediation Filter Applied:** `{remediation.get('applied_filter', 'NONE')}`
        - **Clinical Abnormality Preserved:** `{'YES' if sqi.get('is_pathological') else 'NO'}`
        - **Diagnostic Reasoning:** {sqi.get('reason')}
        """)

        if sqi.get("is_pathological"):
            st.success("🛡️ **Morphology Pathology Guard:** Pathological QRS detected (VTach/PVC). System preserved episode for classification rather than rejecting as noise!")
        elif not sqi.get("passed_gate"):
            st.error("🚨 **Gatekeeper Abstention Active:** Severe motion/contact artifact. Arrhythmia inference bypassed to prevent clinician false alarms.")
        else:
            st.info("✅ **Adaptive Quality Cleared:** Signal is diagnostically reliable for automated screening.")


# -------------------------------------------------------------
# 3. ARRHYTHMIA ANALYSIS
# -------------------------------------------------------------
with tab3:
    st.subheader("3. Arrhythmia Screening & Probability Distribution")

    col_arr1, col_arr2 = st.columns([1, 1])

    with col_arr1:
        probs = inference.get("probabilities", {})
        if probs:
            classes = list(probs.keys())
            p_vals = [probs[c] for c in classes]

            fig_prob = go.Figure(go.Bar(
                x=p_vals, y=classes, orientation='h',
                marker=dict(color=['#38bdf8' if c != inference['rhythm_diagnosis'] else '#f59e0b' for c in classes])
            ))
            fig_prob.update_layout(
                title="Model Posterior Probabilities E[p_k] = α_k / S",
                xaxis=dict(range=[0, 1.0]), template="plotly_dark", height=320
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.warning("Prediction suppressed by ASQI gatekeeper due to unrecoverable noise.")

    with col_arr2:
        st.markdown(f"""
        #### Primary Screening Impression:
        - **Diagnostic Category:** `{inference.get('rhythm_diagnosis')}`
        - **Model Confidence:** `{inference.get('confidence_score', 0.0) * 100:.1f} %`
        - **Clinical Severity:** `{inference.get('severity')}`
        - **Screening Triage Status:** `{inference.get('triage_status')}`
        
        #### Autonomic & HRV Features:
        - **SDNN (Autonomic Spread):** `{hrv.get('sdnn_ms', 0.0)} ms`
        - **RMSSD (Parasympathetic Vagal Tone):** `{hrv.get('rmssd_ms', 0.0)} ms`
        - **pNN50 (Erratic RR Index):** `{hrv.get('pnn50_pct', 0.0)} %`
        - **Poincaré Ratio (SD1/SD2):** `{hrv.get('sd_ratio', 0.0):.3f}`
        """)


# -------------------------------------------------------------
# 4. UNCERTAINTY MONITORING & THREE-ZONE DECISION
# -------------------------------------------------------------
with tab4:
    st.subheader("4. Dirichlet Evidential Uncertainty & Three-Zone Decision Engine")

    col_unc1, col_unc2 = st.columns([1, 1])

    with col_unc1:
        # Gauge chart for epistemic uncertainty
        u_score = inference.get("epistemic_uncertainty", 0.0)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=u_score,
            title={'text': "Dirichlet Epistemic Uncertainty (u = K / S)"},
            gauge={
                'axis': {'range': [0, 1.0]},
                'bar': {'color': "#38bdf8"},
                'steps': [
                    {'range': [0, 0.380], 'color': "rgba(34, 197, 94, 0.25)"},
                    {'range': [0.380, 0.650], 'color': "rgba(234, 179, 8, 0.25)"},
                    {'range': [0.650, 1.0], 'color': "rgba(239, 68, 68, 0.25)"}
                ],
                'threshold': {
                    'line': {'color': "yellow", 'width': 3},
                    'thickness': 0.8,
                    'value': 0.380
                }
            }
        ))
        fig_gauge.update_layout(height=300, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_unc2:
        st.markdown(f"""
        #### Three-Zone Decision Matrix:
        """)
        if decision.get("decision_zone") == "ACCEPT":
            st.success(f"🟢 **ZONE 1: ACCEPT (Automated Screening)**\n\n{decision.get('clinical_guidance')}")
        elif decision.get("decision_zone") == "REVIEW":
            st.warning(f"🟡 **ZONE 2: REVIEW (Priority Clinician Overread)**\n\n{decision.get('clinical_guidance')}")
        else:
            st.error(f"🔴 **ZONE 3: ABSTAIN (Request Re-recording)**\n\n{decision.get('clinical_guidance')}")

        st.markdown("""
        *How it works:* CardioSentinel bridges signal physics with neural evidential logic. If noise corrupts the trace, it abstains (Zone 3). If the trace is clean but morphology is ambiguous or out-of-distribution, it flags for cardiologist overread (Zone 2). Only validated, certain traces are cleared automatically (Zone 1).
        """)


# -------------------------------------------------------------
# 5. CLINICIAN REVIEW, GOVERNANCE & EXPORT
# -------------------------------------------------------------
with tab5:
    st.subheader("5. Clinician Review Cockpit & Interoperable HL7 FHIR Export")

    col_rev1, col_rev2 = st.columns([1, 1])

    with col_rev1:
        st.markdown("#### 📄 Export Official Physician Diagnostic Summary (PDF)")
        st.write("Generates an ANSI/AAMI EC57 and IEC 60601-2-47 compliant report including SQI verification, HRV metrics, calibrated uncertainty, and doctor sign-off block.")

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

    with col_rev2:
        st.markdown("#### 🏥 Export HL7 FHIR R4 Observation Resource (JSON)")
        st.write("Formats findings into HL7 FHIR Observation JSON using standard LOINC codes (8867-4 for Heart rate, 131383 for ECG Impression, 93041-2 for Signal Quality).")

        fhir_obs = pipeline.fhir_builder.build_observation(results, patient_id=patient_id, patient_name=patient_name)
        fhir_json_str = json.dumps(fhir_obs, indent=2)

        st.download_button(
            label="⬇️ Download HL7 FHIR Observation JSON",
            data=fhir_json_str,
            file_name=f"fhir_observation_{patient_id}.json",
            mime="application/json"
        )

    st.markdown("---")
    st.markdown("#### HL7 FHIR R4 Observation Resource Preview:")
    st.code(fhir_json_str[:1200] + "\n  ...\n}", language="json")
