"""Causal EMA(10, 30) reference decision rule.

Computes a discrete entry intent (ENTER_LONG, ENTER_SHORT, NO_ENTRY) from
midpoint close prices using exponential moving averages with explicit
warmup, missing-value, gap, equality, and no-lookahead semantics.

The intent at bar t is computed from prices at bars <= t only. It is
actionable at bar t's close (or next observation, per config). The same
intent stream is shared across all policy arms for a given scenario input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator

from .config import StrategyConfig


INTENT_ENTER_LONG = "ENTER_LONG"
INTENT_ENTER_SHORT = "ENTER_SHORT"
INTENT_NO_ENTRY = "NO_ENTRY"


@dataclass
class Intent:
    """A single reference-strategy intent with causal provenance."""

    bar_id: int
    timestamp: str
    midpoint_close: float | None
    ema_fast: float | None
    ema_slow: float | None
    intent: str  # ENTER_LONG | ENTER_SHORT | NO_ENTRY
    warmup: bool  # True while either EMA is still warming up
    eligible: bool  # True when both EMAs are computed and intent is defined
    crossover: bool  # True when a crossover occurred at this bar
    reason: str  # human-readable reason


@dataclass
class IntentStream:
    """Ordered sequence of intents."""

    intents: list[Intent] = field(default_factory=list)
    config: StrategyConfig = field(default_factory=StrategyConfig)

    def __len__(self) -> int:
        return len(self.intents)

    def __iter__(self) -> Iterator[Intent]:
        return iter(self.intents)

    def first_eligible_index(self) -> int | None:
        for i, intent in enumerate(self.intents):
            if intent.eligible:
                return i
        return None

    def last_intent(self) -> Intent | None:
        return self.intents[-1] if self.intents else None


class EMACalculator:
    """Incremental causal EMA calculator (adjust=False semantics).

    ema_t = alpha * price_t + (1 - alpha) * ema_{t-1}, alpha = 2 / (span + 1).
    Warmup: min_periods observations required before the EMA is considered
    computed. Before that, the EMA is None and the strategy is not eligible.
    """

    def __init__(self, span: int, min_periods: int, adjust: bool = False) -> None:
        if span < 1:
            raise ValueError("span must be >= 1")
        if min_periods < 1 or min_periods > span:
            raise ValueError("min_periods must be in [1, span]")
        self.span = span
        self.min_periods = min_periods
        self.adjust = adjust
        self.alpha = 2.0 / (span + 1.0)
        self._ema: float | None = None
        self._count = 0

    def update(self, price: float | None) -> float | None:
        """Feed one price (causal, current-or-past only) and return the EMA.

        If price is None, the EMA carries forward unchanged (no invention).
        The count of valid observations still advances for warmup tracking.
        """
        if price is None:
            # Missing observation: carry forward the last EMA; count does not
            # advance because no new price was observed.
            return self._ema
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"price must be finite and positive, got {price!r}")
        if self._ema is None:
            self._ema = price
        else:
            self._ema = self.alpha * price + (1.0 - self.alpha) * self._ema
        self._count += 1
        return self._ema

    @property
    def ready(self) -> bool:
        return self._count >= self.min_periods


class ReferenceStrategy:
    """Causal EMA(10, 30) crossover reference strategy.

    Produces one Intent per observed bar. A crossover is detected when the
    fast EMA crosses the slow EMA relative to the previous bar's relationship.
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()
        self._fast = EMACalculator(
            self.config.fast_span, self.config.min_periods_fast, self.config.adjust
        )
        self._slow = EMACalculator(
            self.config.slow_span, self.config.min_periods_slow, self.config.adjust
        )
        self.stream = IntentStream(config=self.config)
        self._prev_fast: float | None = None
        self._prev_slow: float | None = None
        self._prev_relationship: str | None = None  # "above" | "below" | "equal" | None

    def update(self, bar) -> Intent:
        """Process one observed bar and produce its causal intent.

        Uses the bar's midpoint_close. If midpoint_close is None, both EMAs
        carry forward and the intent is NO_ENTRY (not eligible).
        """
        price = bar.midpoint_close

        fast_ema = self._fast.update(price)
        slow_ema = self._slow.update(price)

        warmup = not (self._fast.ready and self._slow.ready)
        eligible = not warmup and fast_ema is not None and slow_ema is not None

        intent = INTENT_NO_ENTRY
        crossover = False
        reason = "warmup" if warmup else "no_cross"

        if eligible and fast_ema is not None and slow_ema is not None:
            if fast_ema > slow_ema:
                relationship = "above"
            elif fast_ema < slow_ema:
                relationship = "below"
            else:
                relationship = "equal"

            if self._prev_relationship is not None:
                if (
                    relationship == "above"
                    and self._prev_relationship in ("below", "equal")
                ):
                    intent = INTENT_ENTER_LONG
                    crossover = True
                    reason = "fast_crossed_above_slow"
                elif (
                    relationship == "below"
                    and self._prev_relationship in ("above", "equal")
                ):
                    intent = INTENT_ENTER_SHORT
                    crossover = True
                    reason = "fast_crossed_below_slow"
                else:
                    reason = "no_cross"
            else:
                reason = "first_eligible_no_cross"

            self._prev_relationship = relationship

        self._prev_fast = fast_ema
        self._prev_slow = slow_ema

        intent_obj = Intent(
            bar_id=bar.bar_id,
            timestamp=bar.last_event_timestamp,
            midpoint_close=price,
            ema_fast=fast_ema,
            ema_slow=slow_ema,
            intent=intent,
            warmup=warmup,
            eligible=eligible,
            crossover=crossover,
            reason=reason,
        )
        self.stream.intents.append(intent_obj)
        return intent_obj

    def reset(self) -> None:
        """Reset strategy state (e.g., at a session/gap boundary if configured)."""
        self._fast = EMACalculator(
            self.config.fast_span, self.config.min_periods_fast, self.config.adjust
        )
        self._slow = EMACalculator(
            self.config.slow_span, self.config.min_periods_slow, self.config.adjust
        )
        self.stream = IntentStream(config=self.config)
        self._prev_fast = None
        self._prev_slow = None
        self._prev_relationship = None


__all__ = [
    "INTENT_ENTER_LONG",
    "INTENT_ENTER_SHORT",
    "INTENT_NO_ENTRY",
    "Intent",
    "IntentStream",
    "EMACalculator",
    "ReferenceStrategy",
]
