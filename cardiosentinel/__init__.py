"""
CardioSentinel: An Adaptive, Quality-Aware and Uncertainty-Calibrated Wearable Cardiac Screening System.
TECHgium Round 2 Innovation.
"""

__version__ = "2.0.0"
__author__ = "CardioSentinel Team - TECHgium MedTech Innovation"

from cardiosentinel.config import SignalConfig, SQIConfig, EvidentialConfig
from cardiosentinel.pipeline import CardioSentinelPipeline

__all__ = [
    "SignalConfig",
    "SQIConfig",
    "EvidentialConfig",
    "CardioSentinelPipeline"
]
