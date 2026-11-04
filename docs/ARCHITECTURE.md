# CardioSentinel™ Architecture Deep Dive

CardioSentinel is an ambulatory cardiac screening architecture designed for non-invasive, leadless single-patch wearables. It addresses the fundamental flaw of consumer cardiac wearables—**excessive false alarms triggered by motion noise**—by embedding self-evaluating Signal Quality Indices (SQI) and Dirichlet evidential neural uncertainty calibration directly into the screening pipeline.

---

## 1. End-to-End System Pipeline

```
+-----------------------------------------------------------------------------------+
|                        1. DATA ACQUISITION & SENSING                              |
|   - Leadless Single-Lead / Dry-Contact Electrode Patch (250 Hz sampling)          |
|   - Real-time continuous streaming buffer (10-second analysis windows)            |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                     2. PREPROCESSING & ARTIFACT FILTERING                         |
|   - Zero-phase Butterworth Bandpass (0.5 Hz - 45.0 Hz, 3rd Order filtfilt)         |
|   - IIR Notch Filter (50 Hz / 60 Hz mains interference, Q = 30)                  |
|   - Robust Median/MAD Scaling for motion-resistant amplitude normalization        |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                   3. HIERARCHICAL SQI GATEKEEPER (THE DEFENSE)                    |
|   - bSQI: Pan-Tompkins vs Hamilton QRS detector concordance (tolerance = 50ms)    |
|   - sSQI: Power spectral ratio in QRS depolarization band (5-15 Hz vs 5-40 Hz)    |
|   - kSQI: Pearson kurtosis check for distinct physiological QRS spikes (> 3.5)    |
|   - basSQI: Baseline drift ratio (< 1.0 Hz respiration/movement)                  |
|   - Lead-Off: High-impedance flatline / disconnection detection (< 40 uV)         |
|                                                                                   |
|   --> IF SQI < 0.65 OR Lead-Off:                                                  |
|       * Tag as "Motion Corrupted / Sensor Detached"                               |
|       * ABSTAIN from neural prediction                                            |
|       * Prompt patient adjustment -> FALSE ALARM SUPPRESSED!                      |
+------------------------------------------+----------------------------------------+
                                           | (Passed Gatekeeper)
                                           v
+-----------------------------------------------------------------------------------+
|                 4. MULTI-SCALE 1D-CNN + TEMPORAL SEQUENCE MODEL                   |
|   - Branch 1 (Kernel 5):  Micro-scale notch & spike capture                       |
|   - Branch 2 (Kernel 15): Meso-scale QRS duration & P/T wave morphology           |
|   - Branch 3 (Kernel 31): Macro-scale ST-segment elevation & broad ectopics       |
|   - Bidirectional GRU: Sequential beat-to-beat interval dynamics across strip    |
|   - Global Attention Pooling -> Dense Latent Representation (64-dim)             |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|              5. EVIDENTIAL DIRICHLET UNCERTAINTY QUANTIFICATION                   |
|   - Softplus non-negative evidence: e_k = softplus(z_k) >= 0                      |
|   - Dirichlet parameters: alpha_k = e_k + 1                                       |
|   - Total Dirichlet strength: S = sum(alpha_k)                                    |
|   - Expected class probability: E[p_k] = alpha_k / S                              |
|   - Epistemic uncertainty: u = K / S in [0, 1]                                    |
|                                                                                   |
|   --> IF uncertainty u > 0.380: Flag as "Ambiguous - Priority Overread Required"   |
|   --> IF uncertainty u <= 0.380: Confirmed Automated Screening Result             |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                 6. CLINICAL TRIAGE, HL7 FHIR & TELEMETRY REPORT                   |
|   - Autonomic HRV & Poincaré analysis (RMSSD, SDNN, pNN50, SD1/SD2)               |
|   - HL7 FHIR R4 Observation JSON generation (LOINC 8867-4, 131383, 93041-2)       |
|   - Automated Physician PDF Diagnostic Summary with Doctor Sign-off Block         |
+-----------------------------------------------------------------------------------+
```

---

## 2. Mathematical Formulations

### 2.1 Beat Signal Quality Index ($bSQI$)
Calculates the harmonic agreement ($F_1$ score) between the primary Pan-Tompkins QRS detector and the secondary derivative-based Hamilton detector:

$$bSQI = \frac{2 \cdot N_{\text{matched}}}{N_{\text{Pan-Tompkins}} + N_{\text{Hamilton}}}$$

where a beat is considered matched if:

$$|t_{\text{PT}} - t_{\text{Hamilton}}| \le 50 \text{ ms}$$

### 2.2 Spectral Signal Quality Index ($sSQI$)
Calculates the fraction of spectral power residing within the diagnostic ventricular depolarization band (5–15 Hz) relative to the extended bandwidth (5–40 Hz):

$$sSQI = \frac{\int_{5}^{15} S_{xx}(f) \, df}{\int_{5}^{40} S_{xx}(f) \, df}$$

where $S_{xx}(f)$ is the Welch power spectral density.

### 2.3 Evidential Deep Learning (Subjective Logic)
In contrast to standard softmax neural networks that yield overconfident predictions on corrupted inputs, CardioSentinel treats network outputs as evidence $e_k \ge 0$ parameterized by a Dirichlet distribution $\text{Dir}(\boldsymbol{\alpha})$:

$$\alpha_k = e_k + 1 = \text{softplus}(z_k) + 1$$

Total Dirichlet strength:
$$S = \sum_{k=1}^K \alpha_k$$

Expected classification probability:
$$\hat{p}_k = \frac{\alpha_k}{S}$$

Total Epistemic Uncertainty:
$$u = \frac{K}{S} \in (0, 1]$$

During training, the evidential loss minimizes Bayes risk under the Dirichlet distribution with a KL divergence penalty to a uniform prior $\text{Dir}(\mathbf{1})$ for incorrect classes:

$$\mathcal{L}(\boldsymbol{\alpha}) = \sum_{k=1}^K y_k \left[ \psi(S) - \psi(\alpha_k) \right] + \lambda_t \, \text{KL}\left[ \text{Dir}(\boldsymbol{\tilde{\alpha}}) \parallel \text{Dir}(\mathbf{1}) \right]$$

---

## 3. Regulatory & Standard Compliance
- **ANSI/AAMI EC57:** Testing and reporting performance results of cardiac rhythm and ST-segment measurement algorithms.
- **IEC 60601-2-47:** Particular requirements for basic safety and essential performance of ambulatory electrocardiographic systems.
- **HL7 FHIR Release 4:** Interoperable clinical data encoding with official LOINC codes.

