# Clinical Standards & Regulatory Alignment

CardioSentinel is built in strict adherence to international biomedical engineering and health informatics standards governing wearable ambulatory electrocardiography.

---

## 1. Algorithmic Reporting Standards (ANSI/AAMI EC57:2012)
ANSI/AAMI EC57 establishes testing protocols and statistical reporting conventions for automated cardiac rhythm evaluation algorithms:
- **Evaluation Categories:**
  - Ventricular Fibrillation / Tachycardia (VF/VT) detection
  - Atrial Fibrillation (AFib) detection
  - Premature Ventricular Contractions (PVC) classification
- **Statistical Metrics Reported:**
  - Episode Sensitivity ($Se$) and Specificity ($Sp$)
  - Positive Predictive Value ($PPV$) and False Positive Rate ($FPR$)
  - Gross beat-by-beat concordance matrix across benchmark PhysioNet databases (MIT-BIH Arrhythmia Database, CinC 2017 Challenge).

---

## 2. Ambulatory Medical Device Safety (IEC 60601-2-47:2012)
IEC 60601-2-47 governs the safety and essential performance of ambulatory ECG systems (Holter monitors and wearable event recorders):
- **Frequency Response:** Preserves bandwidth from 0.67 Hz to 40 Hz for diagnostic ST-segment and QRS fidelity.
- **Electrode Contact & Lead-Off Detection:** Requires immediate notification when electrode-skin impedance exceeds diagnostic limits or leads detach. CardioSentinel fulfills this via the zero-voltage flatline & high-impedance detector.
- **Signal Quality Gatekeeper:** Requires differentiation between physiological arrhythmias and movement/EMG artifacts to avoid inappropriate clinical action.

---

## 3. Risk Management & Safe AI Abstention (ISO 14971:2019)
Under medical device risk management (ISO 14971), an AI algorithm forcing predictions on corrupted signals introduces an unacceptable clinical hazard (Hazard: Inappropriate patient anxiety or unnecessary hospitalization due to false positive arrhythmia call).

CardioSentinel eliminates this hazard through a **Two-Tier Safety Abstention Architecture**:
1. **Tier 1 (Physics-based SQI Gate):** If composite Signal Quality Index $< 0.65$, prediction is rejected at the sensor level.
2. **Tier 2 (Epistemic Uncertainty Abstention):** If Dirichlet epistemic uncertainty $u > 0.380$, the system abstains from automated clearance and routes the strip for priority cardiologist overread.

---

## 4. Health Data Interoperability (HL7 FHIR Release 4)
Telemetry results and diagnostic findings are serialized as standard HL7 FHIR `Observation` resources using standard Logical Observation Identifiers Names and Codes (LOINC):

| Clinical Metric | LOINC Code | Display Name | Unit |
| :--- | :--- | :--- | :--- |
| **Heart Rate** | `8867-4` | Heart rate | beats/minute (`/min`) |
| **R-R Interval** | `8627-2` | R-R interval | milliseconds (`ms`) |
| **ECG Impression** | `131383` | Electrocardiogram impression | text string |
| **Signal Quality Score** | `93041-2` | ECG signal quality score | unitless score (`0.0 - 1.0`) |
