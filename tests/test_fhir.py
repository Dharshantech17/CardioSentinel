"""Unit tests for CardioSentinel HL7 FHIR Exporter."""

import sys
import unittest
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cardiosentinel.export.fhir_export import FHIRObservationBuilder
from cardiosentinel.config import LOINC_CODES


class TestFHIRExporter(unittest.TestCase):
    def test_fhir_observation_structure(self):
        builder = FHIRObservationBuilder(device_id="CardioSentinel-Patch-99")

        mock_analysis = {
            "sqi": {"overall_sqi": 0.88, "decision": "ACCEPT"},
            "hrv": {"mean_hr_bpm": 72.5},
            "inference": {
                "rhythm_diagnosis": "Normal Sinus Rhythm (NSR)",
                "severity": "NORMAL",
                "epistemic_uncertainty": 0.18,
                "triage_status": "CONFIRMED_ALGORITHM_SCREENING"
            }
        }

        obs = builder.build_observation(mock_analysis, patient_id="PAT-12345", patient_name="Jane Doe")

        self.assertEqual(obs["resourceType"], "Observation")
        self.assertEqual(obs["status"], "final")
        self.assertEqual(obs["subject"]["reference"], "Patient/PAT-12345")
        self.assertEqual(obs["subject"]["display"], "Jane Doe")
        self.assertEqual(obs["valueString"], "Normal Sinus Rhythm (NSR)")

        # Check LOINC codes inside components
        found_loincs = set()
        for comp in obs["component"]:
            for c in comp["code"]["coding"]:
                if c.get("system") == "http://loinc.org":
                    found_loincs.add(c.get("code"))

        self.assertIn(LOINC_CODES["HEART_RATE"], found_loincs)
        self.assertIn(LOINC_CODES["SIGNAL_QUALITY"], found_loincs)


if __name__ == "__main__":
    unittest.main()
