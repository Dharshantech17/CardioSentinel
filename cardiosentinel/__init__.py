"""
CardioSentinel: Edge-Aware Leadless Rhythm Monitor with Dynamic Signal Quality Assessment
and Uncertainty-Calibrated Arrhythmia Screening.
"""

__version__ = "1.0.0"
__author__ = "CardioSentinel Team - TECHgium MedTech Innovation"

from cardiosentinel.config import SignalConfig, SQIConfig, EvidentialConfig
from cardiosentinel.pipeline import CardioSentinelPipeline

__all__ = [
    "SignalConfig",
    "SQIConfig",
    "EvidentialConfig",
    "CardioSentinelPipeline"
]

