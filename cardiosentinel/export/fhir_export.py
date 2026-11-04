"""
CardioSentinel HL7 FHIR (Fast Healthcare Interoperability Resources) Exporter
Constructs standardized FHIR R4 Observation resource bundles for seamless
electronic health record (EHR / Epic / Cerner) integration.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any
from cardiosentinel.config import LOINC_CODES


class FHIRObservationBuilder:
    """Creates HL7 FHIR R4 Observation objects compliant with standard MedTech interoperability."""

    def __init__(self, device_id: str = "CardioSentinel-Patch-v1"):
        self.device_id = device_id

    def build_observation(
        self,
        analysis_result: Dict[str, Any],
        patient_id: str = "PATIENT-MRN-90214",
        patient_name: str = "Anonymous Patient"
    ) -> Dict[str, Any]:
        """Encodes CardioSentinel telemetry, SQI, and evidential diagnosis into a FHIR Observation."""
        obs_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        sqi_info = analysis_result.get("sqi", {})
        hrv_info = analysis_result.get("hrv", {})
        inference = analysis_result.get("inference", {})

        # Components for individual clinical parameters
        components = []

        # 1. Heart Rate Component
        mean_hr = hrv_info.get("mean_hr_bpm", 0.0)
        components.append({
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": LOINC_CODES["HEART_RATE"],
                    "display": "Heart rate"
                }]
            },
            "valueQuantity": {
                "value": mean_hr,
                "unit": "beats/minute",
                "system": "http://unitsofmeasure.org",
                "code": "/min"
            }
        })

        # 2. Signal Quality Index (SQI) Component
        overall_sqi = sqi_info.get("overall_sqi", 0.0)
        components.append({
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": LOINC_CODES["SIGNAL_QUALITY"],
                    "display": "ECG signal quality score"
                }]
            },
            "valueQuantity": {
                "value": overall_sqi,
                "unit": "score",
                "system": "http://unitsofmeasure.org",
                "code": "1"
            }
        })

        # 3. Evidential Uncertainty Component
        uncertainty = inference.get("epistemic_uncertainty", 0.0) if inference else 0.0
        components.append({
            "code": {
                "coding": [{
                    "system": "https://cardiosentinel.ai/fhir/codes",
                    "code": "EVIDENTIAL_UNCERTAINTY",
                    "display": "Dirichlet Epistemic Uncertainty"
                }]
            },
            "valueQuantity": {
                "value": uncertainty,
                "unit": "uncertainty",
                "system": "http://unitsofmeasure.org",
                "code": "1"
            }
        })

        # Interpretation Code
        diagnosis_str = inference.get("rhythm_diagnosis", "Signal Corrupted / Inconclusive")
        severity_str = inference.get("severity", "UNKNOWN")

        fhir_resource = {
            "resourceType": "Observation",
            "id": obs_id,
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "vital-signs",
                            "display": "Vital Signs"
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": LOINC_CODES["ECG_IMPRESSION"],
                        "display": "Electrocardiogram impression"
                    }
                ],
                "text": "CardioSentinel Leadless Wearable Arrhythmia Analysis"
            },
            "subject": {
                "reference": f"Patient/{patient_id}",
                "display": patient_name
            },
            "effectiveDateTime": now_iso,
            "issued": now_iso,
            "device": {
                "display": f"Wearable Leadless Patch Sensor ({self.device_id})"
            },
            "valueString": diagnosis_str,
            "interpretation": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                            "code": severity_str,
                            "display": f"Clinical Severity: {severity_str}"
                        }
                    ]
                }
            ],
            "note": [
                {
                    "text": f"Gatekeeper Decision: {sqi_info.get('decision', 'N/A')}. Triage: {inference.get('triage_status', 'N/A')}."
                }
            ],
            "component": components
        }

        return fhir_resource

    def export_to_json(self, fhir_resource: Dict[str, Any], filepath: str) -> str:
        """Saves FHIR Observation to a formatted JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(fhir_resource, f, indent=2)
        return filepath
