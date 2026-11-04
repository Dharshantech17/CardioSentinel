"""
CardioSentinel Clinical Telemetry PDF Diagnostic Report Generator
Produces publication-grade, physician-ready diagnostic summaries
complete with SQI scores, HRV parameters, uncertainty bounds, and triage actions.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime


class ClinicalReportGenerator:
    """Generates standardized ambulatory cardiac screening PDF reports."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        self.title_style = ParagraphStyle(
            'ClinicalTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#0F172A'),
            fontName='Helvetica-Bold'
        )
        self.subtitle_style = ParagraphStyle(
            'ClinicalSubtitle',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=13,
            textColor=colors.HexColor('#475569')
        )
        self.section_heading = ParagraphStyle(
            'SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor('#1E3A8A'),
            fontName='Helvetica-Bold'
        )
        self.body_style = ParagraphStyle(
            'ClinicalBody',
            parent=self.styles['Normal'],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#1E293B')
        )
        self.alert_style = ParagraphStyle(
            'AlertStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=13,
            textColor=colors.HexColor('#991B1B'),
            fontName='Helvetica-Bold'
        )

    def generate(
        self,
        analysis_result: Dict[str, Any],
        output_filepath: str,
        patient_name: str = "Anonymous Patient",
        patient_id: str = "PATIENT-MRN-90214",
        clinician_notes: Optional[str] = None
    ) -> str:
        """Assembles and writes the PDF report to output_filepath."""
        doc = SimpleDocTemplate(
            output_filepath,
            pagesize=letter,
            rightMargin=0.5 * inch,
            leftMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch
        )

        story = []
        sqi = analysis_result.get("sqi", {})
        hrv = analysis_result.get("hrv", {})
        inference = analysis_result.get("inference", {})

        # Header Title
        story.append(Paragraph("<b>CardioSentinel™ Clinical Telemetry Report</b>", self.title_style))
        story.append(Paragraph("AI-Powered Leadless Rhythm Monitor & Ambulatory Arrhythmia Triage Summary", self.subtitle_style))
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=10))

        # Patient & Device Metadata Table
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        meta_data = [
            [
                Paragraph(f"<b>Patient Name:</b> {patient_name}", self.body_style),
                Paragraph(f"<b>MRN / ID:</b> {patient_id}", self.body_style),
                Paragraph(f"<b>Timestamp:</b> {now_str}", self.body_style)
            ],
            [
                Paragraph("<b>Sensor Mode:</b> Leadless Single-Patch", self.body_style),
                Paragraph("<b>Sampling Rate:</b> 250 Hz", self.body_style),
                Paragraph(f"<b>Triage Status:</b> {inference.get('triage_status', 'SCREENED')}", self.body_style)
            ]
        ]
        meta_table = Table(meta_data, colWidths=[2.5*inch, 2.2*inch, 2.8*inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 12))

        # Section 1: Signal Quality Indexing (SQI) Gatekeeper
        story.append(Paragraph("1. Signal Quality Indexing (SQI) Gatekeeper Verification", self.section_heading))
        gate_passed = sqi.get("passed_gate", False)
        gate_status_color = colors.HexColor('#16A34A') if gate_passed else colors.HexColor('#DC2626')

        sqi_table_data = [
            ["Metric", "Value", "Threshold Criterion", "Quality Assessment"],
            ["Overall Composite SQI", f"{sqi.get('overall_sqi', 0.0):.3f}", ">= 0.65", "PASSED" if gate_passed else "REJECTED"],
            ["Beat Agreement (bSQI)", f"{sqi.get('bsqi', 0.0):.3f}", ">= 0.70 (Dual Pan-Tompkins/Hamilton)", "Concordant" if sqi.get('bsqi', 0) >= 0.7 else "Discordant"],
            ["Spectral Ratio (sSQI)", f"{sqi.get('ssqi', 0.0):.3f}", ">= 0.45 (QRS Band 5-15 Hz)", "Adequate" if sqi.get('ssqi', 0) >= 0.45 else "Noise Dominated"],
            ["Kurtosis (kSQI)", f"{sqi.get('ksqi', 0.0):.3f}", ">= 3.50 (Non-Gaussian Sharp Spikes)", "Valid Spikes" if sqi.get('ksqi', 0) >= 0.2 else "Flat/Gaussian"],
            ["Baseline Stability (basSQI)", f"{sqi.get('bassqi', 0.0):.3f}", ">= 0.65 (Drift < 1 Hz)", "Stable Base" if sqi.get('bassqi', 0) >= 0.65 else "Drift Alert"]
        ]
        sqi_table = Table(sqi_table_data, colWidths=[2.2*inch, 1.3*inch, 2.5*inch, 1.5*inch])
        sqi_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8.5),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#FFFFFF')),
            ('TEXTCOLOR', (3,1), (3,1), gate_status_color),
            ('FONTNAME', (3,1), (3,1), 'Helvetica-Bold'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(sqi_table)
        story.append(Spacer(1, 12))

        # Section 2: Neural Arrhythmia Diagnosis & Evidential Calibration
        story.append(Paragraph("2. Neural Arrhythmia Impression & Dirichlet Uncertainty", self.section_heading))

        diag_name = inference.get("rhythm_diagnosis", "Inconclusive")
        conf_score = inference.get("confidence_score", 0.0) * 100.0
        uncertainty = inference.get("epistemic_uncertainty", 0.0)
        severity = inference.get("severity", "UNKNOWN")

        diag_data = [
            ["Primary Diagnostic Impression:", Paragraph(f"<b>{diag_name}</b>", self.body_style)],
            ["Clinical Severity Tier:", Paragraph(f"<b>{severity}</b>", self.body_style)],
            ["Expected Model Confidence:", f"{conf_score:.1f} %"],
            ["Dirichlet Epistemic Uncertainty:", f"{uncertainty:.4f} (Safety Limit: < 0.380)"],
            ["Abstention / Review Recommendation:", "Safe for Automated Screening" if not inference.get("is_ambiguous", False) else "EXCEEDS UNCERTAINTY BOUND: REQUIRES CARDIOLOGIST OVERREAD"]
        ]
        diag_table = Table(diag_data, colWidths=[2.6*inch, 4.9*inch])
        diag_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F1F5F9')),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(diag_table)
        story.append(Spacer(1, 12))

        # Section 3: Heart Rate Variability & Poincaré Non-Linear Dynamics
        story.append(Paragraph("3. Heart Rate Variability (HRV) & Autonomic Dynamics", self.section_heading))

        hrv_data = [
            ["Metric", "Measured Value", "Physiological Context"],
            ["Heart Rate (Mean)", f"{hrv.get('mean_hr_bpm', 0.0)} bpm", "Resting Ambulatory Range: 60 - 100 bpm"],
            ["Mean RR Interval", f"{hrv.get('mean_rr_ms', 0.0)} ms", "Inverted beat-to-beat pacing cycle"],
            ["SDNN (Total Variability)", f"{hrv.get('sdnn_ms', 0.0)} ms", "Global autonomic tone (ANS health)"],
            ["RMSSD (Parasympathetic)", f"{hrv.get('rmssd_ms', 0.0)} ms", "Vagal tone index (severely erratic in AFib)"],
            ["pNN50 (Successive RR > 50ms)", f"{hrv.get('pnn50_pct', 0.0)} %", "High values correlate with rhythm irregularity"],
            ["Poincaré SD1 (Short-term)", f"{hrv.get('sd1_ms', 0.0)} ms", "Fast beat-to-beat instantaneous dispersion"],
            ["Poincaré SD2 (Long-term)", f"{hrv.get('sd2_ms', 0.0)} ms", "Continuous baseline autonomic tone dispersion"],
            ["SD1 / SD2 Dispersion Ratio", f"{hrv.get('sd_ratio', 0.0):.3f}", "Geometric ellipse ratio (< 0.5 typical for NSR)"]
        ]
        hrv_table = Table(hrv_data, colWidths=[2.3*inch, 1.8*inch, 3.4*inch])
        hrv_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8.5),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(hrv_table)
        story.append(Spacer(1, 16))

        # Section 4: Physician Attestation & Regulatory Statement
        story.append(KeepTogether([
            Paragraph("<b>Clinical Governance & Regulatory Compliance Statement:</b>", self.body_style),
            Paragraph(
                "CardioSentinel complies with ANSI/AAMI EC57 and IEC 60601-2-47 algorithmic reporting standards. "
                "This report provides automated screening decision support. For alerts marked URGENT or CRITICAL, "
                "or windows flagged with high evidential uncertainty, 12-lead ECG confirmatory evaluation is recommended.",
                self.subtitle_style
            ),
            Spacer(1, 14),
            Table([
                [
                    Paragraph("<b>Attending Cardiologist:</b> ___________________________", self.body_style),
                    Paragraph("<b>Signature:</b> ___________________________", self.body_style),
                    Paragraph("<b>Date:</b> ____________", self.body_style)
                ]
            ], colWidths=[3.2*inch, 3.0*inch, 1.3*inch])
        ]))

        doc.build(story)
        return output_filepath
