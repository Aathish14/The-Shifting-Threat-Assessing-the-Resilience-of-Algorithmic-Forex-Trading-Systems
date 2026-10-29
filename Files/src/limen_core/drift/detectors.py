"""Distribution-shift monitors behind a common causal interface.

Each detector consumes only a declared causal feature stream and returns
(score, status, reason, input_end_bar_id, version). Status is OK | ALARM |
UNKNOWN. A missing score is UNKNOWN, never an implicit zero.

Core methods (implemented here):
  - Two-sample KS statistic on clipped absolute log returns
  - CUSUM on training-standardized scalar
  - Parametric scale (variance ratio)
  - Quantile monitor
  - Robust MAD monitor

Phase II methods (ADWIN, MMD/kernel, classifier, multivariate) are deferred
until core acceptance and explicit protocol approval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


DETECT_OK = "OK"
DETECT_ALARM = "ALARM"
DETECT_UNKNOWN = "UNKNOWN"


@dataclass
class DetectorResult:
    """Output of one shift detector evaluation at one bar."""

    detector_name: str
    version: str
    score: float | None
    status: str  # OK | ALARM | UNKNOWN
    reason: str
    input_end_bar_id: int
    input_end_timestamp: str
    threshold: float | None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # True only when this call actually produced a score. Declared as a field so
    # callers can rely on it; v1.0.x attached it dynamically, so any result that
    # skipped the assignment raised AttributeError instead of being usable.
    checked: bool = False

    def is_alarm(self) -> bool:
        return self.status == DETECT_ALARM

    def is_unknown(self) -> bool:
        return self.status == DETECT_UNKNOWN


class BaseDetector:
    """Common interface for all shift detectors."""

    name: str = "base"
    version: str = "1.0.0"

    def reset(self) -> None:
        """Reset detector state (e.g., at session boundary)."""
        raise NotImplementedError

    def update(self, feature_value: float | None, bar_id: int, timestamp: str) -> DetectorResult:
        """Process one causal feature value and return a DetectorResult."""
        raise NotImplementedError


class KSShiftDetector(BaseDetector):
    """Two-sample Kolmogorov-Smirnov statistic on clipped absolute log returns.

    Compares a reference window (N bars) with a recent window (M bars).
    Scores every `check_cadence` bars. Alarms after `consecutive_alerts`
    consecutive scores strictly above threshold.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.reference_window = config.reference_window
        self.recent_window = config.recent_window
        self.check_cadence = config.check_cadence
        self.consecutive_alerts = config.consecutive_alerts
        self.quantile = config.quantile
        self.clip = config.clip_abs_return
        self.name = "ks_shift"
        self.version = "1.0.0"

        self._values: list[float] = []
        self._bar_ids: list[int] = []
        self._timestamps: list[str] = []
        self._checked_since_alarm = 0
        self._threshold: float | None = None
        self._calibrated = False
        self._bars_since_last_check = 0

    def set_threshold(self, threshold: float) -> None:
        """Set threshold from validation data only; frozen before test.

        ``None`` is accepted and means "no score is comparable", i.e. every
        checked result is UNKNOWN rather than an implicit OK.
        """
        self._threshold = threshold
        self._calibrated = True

    def clear_alarm_streak(self) -> None:
        """Reset the consecutive-above-threshold counter without dropping history.

        Used at the validation/test boundary: the reference window legitimately
        carries validation history, but a streak that accumulated before the
        threshold was frozen must not contribute to the first test alarm.
        """
        self._checked_since_alarm = 0

    def reset(self) -> None:
        self._values = []
        self._bar_ids = []
        self._timestamps = []
        self._checked_since_alarm = 0
        self._bars_since_last_check = 0

    def _ks_statistic(self, ref: list[float], recent: list[float]) -> float:
        if not ref or not recent:
            return float("nan")
        # Empirical CDF difference at all unique points
        combined = sorted(set(ref + recent))
        max_diff = 0.0
        n_ref = len(ref)
        n_rec = len(recent)
        ref_sorted = sorted(ref)
        rec_sorted = sorted(recent)
        i_ref = 0
        i_rec = 0
        for x in combined:
            while i_ref < n_ref and ref_sorted[i_ref] <= x:
                i_ref += 1
            while i_rec < n_rec and rec_sorted[i_rec] <= x:
                i_rec += 1
            diff = abs(i_ref / n_ref - i_rec / n_rec)
            if diff > max_diff:
                max_diff = diff
        return max_diff

    def update(self, feature_value: float | None, bar_id: int, timestamp: str) -> DetectorResult:
        # Feature value is clipped absolute log return
        if feature_value is None:
            self._bars_since_last_check += 1
            return DetectorResult(
                self.name, self.version, None, DETECT_UNKNOWN,
                "missing_feature", bar_id, timestamp, self._threshold,
                {"reason": "feature_unavailable"},
            )

        v = feature_value
        if self.clip is not None:
            v = min(v, self.clip)

        self._values.append(v)
        self._bar_ids.append(bar_id)
        self._timestamps.append(timestamp)

        # Keep only the needed window
        if len(self._values) > self.reference_window + self.recent_window:
            drop = len(self._values) - (self.reference_window + self.recent_window)
            self._values = self._values[drop:]
            self._bar_ids = self._bar_ids[drop:]
            self._timestamps = self._timestamps[drop:]

        self._bars_since_last_check += 1

        # Default result: no check performed this bar
        result = DetectorResult(
            self.name, self.version, None, DETECT_OK,
            "no_check", bar_id, timestamp, self._threshold,
            {"checked": False},
        )

        if self._bars_since_last_check >= self.check_cadence:
            self._bars_since_last_check = 0
            n_total = len(self._values)
            if n_total >= self.reference_window + self.recent_window:
                ref = self._values[-(self.reference_window + self.recent_window):-self.recent_window]
                recent = self._values[-self.recent_window:]
                d = self._ks_statistic(ref, recent)
                result.score = d
                result.checked = True
                result.diagnostics = {
                    "reference_window": self.reference_window,
                    "recent_window": self.recent_window,
                    "ks_distance": d,
                }
                if self._calibrated and self._threshold is not None:
                    if d > self._threshold:
                        self._checked_since_alarm += 1
                        if self._checked_since_alarm >= self.consecutive_alerts:
                            result.status = DETECT_ALARM
                            result.reason = "ks_alarm"
                        else:
                            result.status = DETECT_OK
                            result.reason = f"ks_consecutive_{self._checked_since_alarm}"
                    else:
                        self._checked_since_alarm = 0
                        result.status = DETECT_OK
                        result.reason = "ks_below_threshold"
                else:
                    result.status = DETECT_UNKNOWN
                    result.reason = "threshold_not_calibrated"
            else:
                result.status = DETECT_UNKNOWN
                result.reason = "insufficient_history"
                result.diagnostics = {
                    "have": n_total,
                    "need": self.reference_window + self.recent_window,
                }

        return result


class CUSUMDetector(BaseDetector):
    """Two-sided CUSUM on a training-standardized scalar feature.

    G_t^+ = max(0, G_{t-1}^+ + z_t - k), G_t^- = max(0, G_{t-1}^- - z_t - k).
    Signal when either accumulator exceeds h.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.allowance = config.allowance
        self.decision_limit = config.decision_limit
        self.reset_on_gap = config.reset_on_gap
        self.reset_on_missing = config.reset_on_missing
        self.name = "cusum_shift"
        self.version = "1.0.0"
        self._g_plus = 0.0
        self._g_minus = 0.0
        self._mean: float | None = None
        self._std: float | None = None
        self._calibrated = False

    def set_training_reference(self, mean: float, std: float) -> None:
        self._mean = mean
        self._std = std
        self._calibrated = True

    def reset(self) -> None:
        self._g_plus = 0.0
        self._g_minus = 0.0

    def update(self, feature_value: float | None, bar_id: int, timestamp: str) -> DetectorResult:
        if feature_value is None or not self._calibrated:
            if self.reset_on_missing and feature_value is None:
                self.reset()
            return DetectorResult(
                self.name, self.version, None, DETECT_UNKNOWN,
                "missing_feature_or_uncalibrated", bar_id, timestamp, self.decision_limit,
                {"calibrated": self._calibrated},
            )

        mean = self._mean
        std = self._std
        if mean is None or std is None or std == 0:
            return DetectorResult(
                self.name, self.version, None, DETECT_UNKNOWN,
                "invalid_reference", bar_id, timestamp, self.decision_limit,
            )

        z = (feature_value - mean) / std
        self._g_plus = max(0.0, self._g_plus + z - self.allowance)
        self._g_minus = max(0.0, self._g_minus - z - self.allowance)

        score = max(self._g_plus, self._g_minus)
        status = DETECT_ALARM if score > self.decision_limit else DETECT_OK
        reason = "cusum_alarm" if status == DETECT_ALARM else "cusum_below_limit"

        return DetectorResult(
            self.name, self.version, score, status, reason,
            bar_id, timestamp, self.decision_limit,
            {"g_plus": self._g_plus, "g_minus": self._g_minus, "z": z},
        )


class ParametricScaleDetector(BaseDetector):
    """Compare rolling return scale/variance to training reference via ratio."""

    def __init__(self, config) -> None:
        self.config = config
        self.name = "parametric_scale"
        self.version = "1.0.0"
        self._values: list[float] = []
        self._train_variance: float | None = None
        self._calibrated = False
        self.threshold: float | None = None

    def set_training_reference(self, variance: float, threshold: float) -> None:
        self._train_variance = variance
        self.threshold = threshold
        self._calibrated = True

    def reset(self) -> None:
        self._values = []

    def update(self, feature_value: float | None, bar_id: int, timestamp: str) -> DetectorResult:
        if feature_value is None or not self._calibrated:
            return DetectorResult(
                self.name, self.version, None, DETECT_UNKNOWN,
                "missing_feature_or_uncalibrated", bar_id, timestamp, self.threshold,
            )
        self._values.append(feature_value)
        if len(self._values) < 10:
            return DetectorResult(
                self.name, self.version, None, DETECT_UNKNOWN,
                "insufficient_history", bar_id, timestamp, self.threshold,
                {"have": len(self._values)},
            )
        recent_var = _variance(self._values[-30:])
        if self._train_variance is None or self._train_variance == 0:
            return DetectorResult(
                self.name, self.version, None, DETECT_UNKNOWN,
                "invalid_reference", bar_id, timestamp, self.threshold,
            )
        ratio = recent_var / self._train_variance
        status = DETECT_ALARM if self.threshold is not None and ratio > self.threshold else DETECT_OK
        reason = "scale_alarm" if status == DETECT_ALARM else "scale_normal"
        return DetectorResult(
            self.name, self.version, ratio, status, reason,
            bar_id, timestamp, self.threshold,
            {"recent_variance": recent_var, "train_variance": self._train_variance},
        )


class QuantileMonitor(BaseDetector):
    """Compare recent return quantile to training reference quantile."""

    def __init__(self, config) -> None:
        self.config = config
        self.name = "quantile_monitor"
        self.version = "1.0.0"
        self._values: list[float] = []
        self._train_quantile: float | None = None
        self._calibrated = False
        self.threshold: float | None = None

    def set_training_reference(self, quantile_value: float, threshold: float) -> None:
        self._train_quantile = quantile_value
        self.threshold = threshold
        self._calibrated = True

    def reset(self) -> None:
        self._values = []

    def update(self, feature_value: float | None, bar_id: int, timestamp: str) -> DetectorResult:
        if feature_value is None or not self._calibrated:
            return DetectorResult(
                self.name, self.version, None, DETECT_UNKNOWN,
                "missing_feature_or_uncalibrated", bar_id, timestamp, self.threshold,
            )
        self._values.append(feature_value)
        if len(self._values) < self.config.recent_window:
            return DetectorResult(
                self.name, self.version, None, DETECT_UNKNOWN,
                "insufficient_history", bar_id, timestamp, self.threshold,
                {"have": len(self._values)},
            )
        recent = sorted(self._values[-self.config.recent_window:])
        q = self.config.quantile_level
        idx = max(0, min(len(recent) - 1, int(math.ceil(q * len(recent))) - 1))
        recent_q = recent[idx]
        score = recent_q
        status = DETECT_ALARM if self._train_quantile is not None and recent_q > self._train_quantile * (1 + self.threshold) else DETECT_OK
        reason = "quantile_alarm" if status == DETECT_ALARM else "quantile_normal"
        return DetectorResult(
            self.name, self.version, score, status, reason,
            bar_id, timestamp, self.threshold,
            {"recent_quantile": recent_q, "train_quantile": self._train_quantile},
        )


class MADMonitor(BaseDetector):
    """Robust MAD monitor: standardized deviation using training median/MAD."""

    def __init__(self, config) -> None:
        self.config = config
        self.name = "mad_monitor"
        self.version = "1.0.0"
        self._values: list[float] = []
        self._train_median: float | None = None
        self._train_mad: float | None = None
        self._calibrated = False
        self.threshold: float | None = None

    def set_training_reference(self, median: float, mad: float, threshold: float) -> None:
        self._train_median = median
        self._train_mad = mad
        self.threshold = threshold
        self._calibrated = True

    def reset(self) -> None:
        self._values = []

    def update(self, feature_value: float | None, bar_id: int, timestamp: str) -> DetectorResult:
        if feature_value is None or not self._calibrated:
            return DetectorResult(
                self.name, self.version, None, DETECT_UNKNOWN,
                "missing_feature_or_uncalibrated", bar_id, timestamp, self.threshold,
            )
        self._values.append(feature_value)
        if self._train_mad is None or self._train_mad == 0:
            fallback = self.config.zero_mad_fallback
            if fallback == "unknown":
                return DetectorResult(
                    self.name, self.version, None, DETECT_UNKNOWN,
                    "zero_mad", bar_id, timestamp, self.threshold,
                )
            elif fallback == "inf":
                score = float("inf")
            else:
                score = 0.0
        else:
            score = abs(feature_value - (self._train_median or 0.0)) / self._train_mad
        status = DETECT_ALARM if self.threshold is not None and score > self.threshold else DETECT_OK
        reason = "mad_alarm" if status == DETECT_ALARM else "mad_normal"
        return DetectorResult(
            self.name, self.version, score, status, reason,
            bar_id, timestamp, self.threshold,
            {"median": self._train_median, "mad": self._train_mad},
        )


def _variance(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((x - mean) ** 2 for x in values) / (n - 1)


__all__ = [
    "DETECT_OK", "DETECT_ALARM", "DETECT_UNKNOWN",
    "DetectorResult", "BaseDetector",
    "KSShiftDetector", "CUSUMDetector", "ParametricScaleDetector",
    "QuantileMonitor", "MADMonitor",
]
