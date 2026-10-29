"""Quote-integrity monitor: hard failures and soft evidence checks.

Every check returns a reason code with severity INVALID / SUSPECT / UNKNOWN /
OK. Hard failures (INVALID) block entry permission in gated arms; soft evidence
(SUSPECT) contributes to policy decisions per the frozen precedence.

Reporting contract (proved in ``tests/test_integrity.py`` and
``tests/test_stale.py``):
  * ``IntegrityAssessment.hard_reasons`` holds INVALID codes only.
  * ``IntegrityAssessment.soft_reasons`` holds SUSPECT codes only.
  * A check that cannot be evaluated is ``UNKNOWN:...`` and never appears in
    either list: a missing score is unknown, never implicit OK or zero.

Checks do not consume evaluator labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


SEV_OK = "OK"
SEV_UNKNOWN = "UNKNOWN"
SEV_SUSPECT = "SUSPECT"
SEV_INVALID = "INVALID"

# Check names (also used as ablation group targets).
CHECK_MISSING = "missing_quote"
CHECK_NONPOSITIVE_BID = "nonpositive_bid"
CHECK_NONPOSITIVE_ASK = "nonpositive_ask"
CHECK_CROSSED = "crossed_quote"
CHECK_SPREAD = "spread_outlier"
CHECK_INNOVATION = "price_innovation"
CHECK_REPEATED = "repeated_quote"
CHECK_GAP = "inter_arrival_gap"
CHECK_STALE = "stale_quote"
CHECK_TIMESTAMP = "timestamp_order"
CHECK_TICK_REASON = "tick_reason"


@dataclass
class MonitorResult:
    """Result of a single integrity check on one bar."""

    check_name: str
    severity: str  # OK | SUSPECT | INVALID | UNKNOWN
    reason: str
    score: float | None
    bar_id: int
    timestamp: str
    details: dict[str, Any] = field(default_factory=dict)

    def is_hard_invalid(self) -> bool:
        return self.severity == SEV_INVALID

    def is_soft_suspect(self) -> bool:
        return self.severity == SEV_SUSPECT

    def is_unknown(self) -> bool:
        return self.severity == SEV_UNKNOWN


@dataclass
class IntegrityAssessment:
    """Aggregate integrity assessment for one bar across all checks."""

    bar_id: int
    timestamp: str
    hard_invalid: bool
    hard_reasons: list[str] = field(default_factory=list)
    soft_reasons: list[str] = field(default_factory=list)
    check_results: list[MonitorResult] = field(default_factory=list)
    eligible_for_entry: bool = True

    @property
    def has_soft_reason(self) -> bool:
        return len(self.soft_reasons) > 0

    @property
    def has_hard_reason(self) -> bool:
        return len(self.hard_reasons) > 0


class IntegrityMonitor:
    """Collection of explicit quote-integrity checks."""

    def __init__(self, config) -> None:
        self.config = config
        # Training-fitted limits (set during calibration, frozen before test)
        self.spread_threshold: float | None = None
        self.innovation_threshold: float | None = None
        self._prior_midpoint: float | None = None
        self._repeat_count = 0
        self._last_bid: float | None = None
        self._last_ask: float | None = None
        self._last_timestamp: str | None = None
        self._calibrated = False

    def set_training_limits(
        self,
        spread_threshold: float | None,
        innovation_threshold: float | None,
    ) -> None:
        """Fit limits from training data only. Must precede test evaluation."""
        self.spread_threshold = spread_threshold
        self.innovation_threshold = innovation_threshold
        self._calibrated = True

    def assess(self, bar) -> IntegrityAssessment:
        """Run all integrity checks against one observed bar."""
        results: list[MonitorResult] = []
        hard_reasons: list[str] = []
        soft_reasons: list[str] = []

        def hard(name: str, reason: str, score: float | None, details: dict[str, Any]) -> None:
            results.append(MonitorResult(name, SEV_INVALID, reason, score, bar.bar_id, ts, details))
            hard_reasons.append(reason)

        def soft(name: str, reason: str, score: float | None, details: dict[str, Any]) -> None:
            results.append(MonitorResult(name, SEV_SUSPECT, reason, score, bar.bar_id, ts, details))
            soft_reasons.append(reason)

        def unknown(name: str, detail: str, score: float | None = None) -> None:
            results.append(MonitorResult(
                name, SEV_UNKNOWN, f"UNKNOWN:{detail.upper()}", score,
                bar.bar_id, ts, {"reason": detail},
            ))

        ts = bar.last_event_timestamp
        bid = bar.bid_close
        ask = bar.ask_close
        midpoint = bar.midpoint_close
        spread = bar.spread_close

        # --- Hard checks: evidence carried from the tick layer -------------
        # A bar with *any* INVALID tick is hard-invalid even when its OHLC came
        # from other valid ticks (documented "any tick" rule).
        for reason in getattr(bar, "hard_invalid_reasons", []) or []:
            if reason and reason not in hard_reasons:
                hard(CHECK_TICK_REASON, reason, None, {"source": "tick_layer"})

        # --- Hard checks on the reconstructed bar --------------------------
        if self.config.enable_missing:
            if bid is None or ask is None:
                hard(CHECK_MISSING, "INVALID:MISSING_QUOTE", None,
                     {"bid": bid, "ask": ask})

        if self.config.enable_nonpositive and bid is not None and ask is not None:
            if bid <= 0:
                hard(CHECK_NONPOSITIVE_BID, "INVALID:NONPOSITIVE_BID", bid, {"value": bid})
            if ask <= 0:
                hard(CHECK_NONPOSITIVE_ASK, "INVALID:NONPOSITIVE_ASK", ask, {"value": ask})

        if self.config.enable_crossed and bid is not None and ask is not None:
            if bid > ask:
                hard(CHECK_CROSSED, "INVALID:CROSSED_QUOTE", spread, {"bid": bid, "ask": ask})

        # --- Runtime timestamp ordering (Item 4) ---------------------------
        if getattr(self.config, "enable_timestamp_order", True) and ts:
            if self._last_timestamp is not None and ts < self._last_timestamp:
                hard(CHECK_TIMESTAMP, "INVALID:TIMESTAMP_ORDER", None,
                     {"bar_timestamp": ts, "prior_timestamp": self._last_timestamp})

        # --- Soft checks ----------------------------------------------------
        if self.config.enable_spread_outlier:
            if spread is None:
                unknown(CHECK_SPREAD, "spread_missing")
            elif self.spread_threshold is None:
                unknown(CHECK_SPREAD, "threshold_not_calibrated", spread)
            elif spread > self.spread_threshold:
                soft(CHECK_SPREAD, "SUSPECT:SPREAD_OUTLIER", spread,
                     {"spread": spread, "threshold": self.spread_threshold})

        if self.config.enable_price_innovation and midpoint is not None:
            innovation = None
            if self._prior_midpoint is not None and self._prior_midpoint > 0 and midpoint > 0:
                innovation = abs(math.log(midpoint / self._prior_midpoint))
            if self.innovation_threshold is None:
                unknown(CHECK_INNOVATION, "threshold_not_calibrated", innovation)
            elif innovation is not None and innovation > self.innovation_threshold:
                soft(CHECK_INNOVATION, "SUSPECT:PRICE_INNOVATION", innovation,
                     {"innovation": innovation, "threshold": self.innovation_threshold})

        # Repeated quote (operates on observed bars, not elapsed time)
        if self.config.enable_repeated_quote and bid is not None and ask is not None:
            if self._last_bid == bid and self._last_ask == ask:
                self._repeat_count += 1
            else:
                self._repeat_count = 1  # current bar counts as one
            if self._repeat_count >= self.config.repeat_count_threshold:
                soft(CHECK_REPEATED, "SUSPECT:REPEATED_QUOTE",
                     float(self._repeat_count),
                     {"repeat_count": self._repeat_count,
                      "threshold": self.config.repeat_count_threshold})

        # Inter-arrival gap
        if self.config.enable_gap_check and bar.elapsed_seconds_since_prior is not None:
            if bar.elapsed_seconds_since_prior > self.config.min_gap_seconds:
                soft(CHECK_GAP, "SUSPECT:INTER_ARRIVAL_GAP",
                     bar.elapsed_seconds_since_prior,
                     {"elapsed_seconds": bar.elapsed_seconds_since_prior,
                      "threshold": self.config.min_gap_seconds})

        # --- Stale quote (Item 1): soft unless configured hard -------------
        if getattr(self.config, "enable_stale_quote", True):
            bid_age = getattr(bar, "bid_age_seconds_close", None)
            ask_age = getattr(bar, "ask_age_seconds_close", None)
            if bid_age is None and ask_age is None:
                unknown(CHECK_STALE, "stale_age_missing")
            else:
                threshold = self.config.stale_quote_seconds
                stale = []
                if bid_age is not None and bid_age > threshold:
                    stale.append(("bid", bid_age))
                if ask_age is not None and ask_age > threshold:
                    stale.append(("ask", ask_age))
                if stale:
                    details = {
                        "bid_age_seconds": bid_age,
                        "ask_age_seconds": ask_age,
                        "threshold": threshold,
                        "stale_sides": [side for side, _ in stale],
                    }
                    worst = max(a for _, a in stale)
                    if getattr(self.config, "stale_is_hard", True):
                        # Per-side hard codes (INVALID:STALE_BID /
                        # INVALID:STALE_ASK) are emitted when only one side is
                        # stale, so a downstream reader can tell which book was
                        # frozen; both stale keeps the combined code. The tick
                        # layer may already have carried the same code, so only
                        # record it here when it is new.
                        sides = {side for side, _ in stale}
                        if sides == {"bid"}:
                            code = "INVALID:STALE_BID"
                        elif sides == {"ask"}:
                            code = "INVALID:STALE_ASK"
                        else:
                            code = "INVALID:STALE_QUOTE"
                        if code not in hard_reasons:
                            hard(CHECK_STALE, code, worst, details)
                    else:
                        soft(CHECK_STALE, "SUSPECT:STALE_QUOTE", worst, details)

        # --- Update state for the next bar ---------------------------------
        if midpoint is not None:
            self._prior_midpoint = midpoint
        self._last_bid = bid
        self._last_ask = ask
        # Never move the ordering high-water mark backward.
        if ts and (self._last_timestamp is None or ts >= self._last_timestamp):
            self._last_timestamp = ts

        hard_invalid = bool(hard_reasons)
        return IntegrityAssessment(
            bar_id=bar.bar_id,
            timestamp=ts,
            hard_invalid=hard_invalid,
            hard_reasons=hard_reasons,
            soft_reasons=soft_reasons,
            check_results=results,
            eligible_for_entry=not hard_invalid,
        )

    def reset(self) -> None:
        """Reset monitor state (e.g., at a session boundary)."""
        self._prior_midpoint = None
        self._repeat_count = 0
        self._last_bid = None
        self._last_ask = None
        self._last_timestamp = None


__all__ = [
    "CHECK_CROSSED",
    "CHECK_GAP",
    "CHECK_INNOVATION",
    "CHECK_MISSING",
    "CHECK_NONPOSITIVE_ASK",
    "CHECK_NONPOSITIVE_BID",
    "CHECK_REPEATED",
    "CHECK_SPREAD",
    "CHECK_STALE",
    "CHECK_TICK_REASON",
    "CHECK_TIMESTAMP",
    "IntegrityAssessment",
    "IntegrityMonitor",
    "MonitorResult",
    "SEV_INVALID",
    "SEV_OK",
    "SEV_SUSPECT",
    "SEV_UNKNOWN",
]
