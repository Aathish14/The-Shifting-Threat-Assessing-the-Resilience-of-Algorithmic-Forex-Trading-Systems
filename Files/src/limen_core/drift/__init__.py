"""Distribution-shift detector package.

Core detectors: KS, CUSUM, parametric scale, quantile, MAD.
All expose the common causal interface: update(feature_value, bar_id, timestamp)
-> DetectorResult(score, status, reason, ...).
"""

from .detectors import (
    DETECT_OK,
    DETECT_ALARM,
    DETECT_UNKNOWN,
    DetectorResult,
    BaseDetector,
    KSShiftDetector,
    CUSUMDetector,
    ParametricScaleDetector,
    QuantileMonitor,
    MADMonitor,
)

__all__ = [
    "DETECT_OK",
    "DETECT_ALARM",
    "DETECT_UNKNOWN",
    "DetectorResult",
    "BaseDetector",
    "KSShiftDetector",
    "CUSUMDetector",
    "ParametricScaleDetector",
    "QuantileMonitor",
    "MADMonitor",
]
