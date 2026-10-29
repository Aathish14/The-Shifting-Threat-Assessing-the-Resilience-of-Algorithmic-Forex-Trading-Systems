"""Observed-minute bar construction from reconstructed quote states.

Groups ticks only into minutes that have at least one *valid* observed event.
Does NOT create empty minutes, does NOT fill gaps, and does NOT treat
irregular observed bars as equally spaced elapsed time.

Bar quality rules (each has a proving test in ``tests/test_bars.py``):

* ``quality_flags`` is the ordered, de-duplicated union of every tick's reason
  codes in the bar -- not just the first tick's.
* A bar's OHLC is seeded only by a *valid* tick. An invalid leading tick can no
  longer smuggle a crossed/None price into ``bid_close``.
* ``hard_invalid_reasons`` / ``invalid_tick_count`` expose invalid evidence so
  the integrity monitor can mark the whole bar hard-invalid.
* Only valid, in-order ticks advance ``last_event_timestamp`` or open a new
  minute bucket; a backward or invalid tick cannot create a bar.
* ``bid_age_seconds_close`` / ``ask_age_seconds_close`` are the per-side ages at
  the bar's last valid tick; ``bid_update_count`` / ``ask_update_count`` count
  the ticks in the bar that changed each side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterator

from .timestamp import seconds_between


@dataclass
class Bar:
    """A single observed-minute bar with full lineage and quality flags."""

    bar_id: int
    source_row_start: int
    source_row_stop: int
    minute_start: str  # "YYYY.MM.DD HH:MM" bucket of first event
    last_event_timestamp: str  # timestamp of last valid event in the bucket
    bid_open: float | None
    bid_high: float | None
    bid_low: float | None
    bid_close: float | None
    ask_open: float | None
    ask_high: float | None
    ask_low: float | None
    ask_close: float | None
    midpoint_open: float | None
    midpoint_high: float | None
    midpoint_low: float | None
    midpoint_close: float | None
    spread_open: float | None
    spread_high: float | None
    spread_low: float | None
    spread_close: float | None
    tick_count: int
    elapsed_seconds_since_prior: float | None
    gap_indicator: bool  # True if elapsed > configured gap or session break
    log_return: float | None  # log(midpoint_close / prior midpoint_close)
    quality_flags: list[str] = field(default_factory=list)
    hard_invalid_reasons: list[str] = field(default_factory=list)
    invalid_tick_count: int = 0
    bid_age_seconds_close: float | None = None
    ask_age_seconds_close: float | None = None
    bid_update_count: int = 0
    ask_update_count: int = 0

    def is_hard_invalid(self) -> bool:
        return bool(self.hard_invalid_reasons)


@dataclass
class BarSeries:
    """Ordered sequence of observed bars."""

    bars: list[Bar] = field(default_factory=list)
    # Invalid ticks whose minute never produced a valid observed bar. They are
    # counted for auditability but cannot form a bar (no empty minutes).
    invalid_ticks_without_bar: int = 0

    def __len__(self) -> int:
        return len(self.bars)

    def __iter__(self) -> Iterator[Bar]:
        return iter(self.bars)

    def last(self) -> Bar | None:
        return self.bars[-1] if self.bars else None

    def __getitem__(self, index: int) -> Bar:
        return self.bars[index]


def _minute_bucket(timestamp: str) -> str:
    """Truncate 'YYYY.MM.DD HH:MM:SS.mmm' to 'YYYY.MM.DD HH:MM'."""
    date_part, time_part = timestamp.split(" ", 1)
    time_fields = time_part.split(":")
    hh_mm = f"{time_fields[0]}:{time_fields[1]}"
    return f"{date_part} {hh_mm}"


def _reason_of(state: Any) -> str:
    reason = getattr(state, "reason", None)
    if reason is None:
        return "OK" if getattr(state, "valid", True) else "INVALID:UNSPECIFIED"
    return reason


def _soft_reasons_of(state: Any) -> tuple[str, ...]:
    soft = getattr(state, "soft_reasons", ())
    return tuple(soft) if soft else ()


@dataclass
class _BarAcc:
    """Mutable accumulator for one in-progress bar."""

    row_start: int
    minute_start: str
    row_stop: int = 0
    last_ts: str = ""
    bid_o: float | None = None
    bid_h: float | None = None
    bid_l: float | None = None
    bid_c: float | None = None
    ask_o: float | None = None
    ask_h: float | None = None
    ask_l: float | None = None
    ask_c: float | None = None
    mid_o: float | None = None
    mid_h: float | None = None
    mid_l: float | None = None
    mid_c: float | None = None
    spr_o: float | None = None
    spr_h: float | None = None
    spr_l: float | None = None
    spr_c: float | None = None
    tick_count: int = 0
    invalid_tick_count: int = 0
    bid_age: float | None = None
    ask_age: float | None = None
    bid_updates: int = 0
    ask_updates: int = 0
    reasons: list[str] = field(default_factory=list)
    hard_reasons: list[str] = field(default_factory=list)

    def note(self, reason: str) -> None:
        if reason and reason != "OK" and reason not in self.reasons:
            self.reasons.append(reason)
        if reason.startswith("INVALID:") and reason not in self.hard_reasons:
            self.hard_reasons.append(reason)


def _apply_tick(acc: _BarAcc, state: Any) -> None:
    """Fold one tick into the bar accumulator (never a bucket decision)."""
    acc.row_stop = getattr(state, "source_row_id", acc.row_stop)
    acc.tick_count += 1
    acc.note(_reason_of(state))
    for soft in _soft_reasons_of(state):
        acc.note(soft)

    if not getattr(state, "valid", True):
        acc.invalid_tick_count += 1
        return

    acc.last_ts = getattr(state, "timestamp", acc.last_ts)
    bid = getattr(state, "bid", None)
    ask = getattr(state, "ask", None)
    mid = getattr(state, "midpoint", None)
    spr = getattr(state, "spread", None)

    if bid is not None:
        if acc.bid_o is None:
            acc.bid_o = bid
        acc.bid_c = bid
        acc.bid_h = bid if acc.bid_h is None else max(acc.bid_h, bid)
        acc.bid_l = bid if acc.bid_l is None else min(acc.bid_l, bid)
    if ask is not None:
        if acc.ask_o is None:
            acc.ask_o = ask
        acc.ask_c = ask
        acc.ask_h = ask if acc.ask_h is None else max(acc.ask_h, ask)
        acc.ask_l = ask if acc.ask_l is None else min(acc.ask_l, ask)
    if mid is not None:
        if acc.mid_o is None:
            acc.mid_o = mid
        acc.mid_c = mid
        acc.mid_h = mid if acc.mid_h is None else max(acc.mid_h, mid)
        acc.mid_l = mid if acc.mid_l is None else min(acc.mid_l, mid)
    if spr is not None:
        if acc.spr_o is None:
            acc.spr_o = spr
        acc.spr_c = spr
        acc.spr_h = spr if acc.spr_h is None else max(acc.spr_h, spr)
        acc.spr_l = spr if acc.spr_l is None else min(acc.spr_l, spr)

    acc.bid_age = getattr(state, "bid_age_seconds", None)
    acc.ask_age = getattr(state, "ask_age_seconds", None)
    if getattr(state, "bid_updated", False):
        acc.bid_updates += 1
    if getattr(state, "ask_updated", False):
        acc.ask_updates += 1


def _finalize_bar(
    acc: _BarAcc,
    bar_id: int,
    prior_last_ts: str | None,
    prior_mid_close: float | None,
    gap_seconds: float,
) -> Bar:
    elapsed = None
    if prior_last_ts is not None:
        elapsed = seconds_between(prior_last_ts, acc.last_ts)
    gap = elapsed is not None and elapsed > gap_seconds

    log_return = None
    if (
        prior_mid_close is not None
        and acc.mid_c is not None
        and prior_mid_close > 0.0
        and acc.mid_c > 0.0
    ):
        log_return = math.log(acc.mid_c / prior_mid_close)

    return Bar(
        bar_id=bar_id,
        source_row_start=acc.row_start,
        source_row_stop=acc.row_stop,
        minute_start=acc.minute_start,
        last_event_timestamp=acc.last_ts,
        bid_open=acc.bid_o, bid_high=acc.bid_h, bid_low=acc.bid_l, bid_close=acc.bid_c,
        ask_open=acc.ask_o, ask_high=acc.ask_h, ask_low=acc.ask_l, ask_close=acc.ask_c,
        midpoint_open=acc.mid_o, midpoint_high=acc.mid_h,
        midpoint_low=acc.mid_l, midpoint_close=acc.mid_c,
        spread_open=acc.spr_o, spread_high=acc.spr_h,
        spread_low=acc.spr_l, spread_close=acc.spr_c,
        tick_count=acc.tick_count,
        elapsed_seconds_since_prior=elapsed,
        gap_indicator=gap,
        log_return=log_return,
        quality_flags=list(acc.reasons),
        hard_invalid_reasons=list(acc.hard_reasons),
        invalid_tick_count=acc.invalid_tick_count,
        bid_age_seconds_close=acc.bid_age,
        ask_age_seconds_close=acc.ask_age,
        bid_update_count=acc.bid_updates,
        ask_update_count=acc.ask_updates,
    )


def build_bars(quote_states, gap_seconds: float = 60.0) -> BarSeries:
    """Aggregate valid quote states into observed-minute bars.

    ``quote_states`` is any iterable of tick-like objects (``_TickInput`` from
    ``ingest.iter_ticks`` or a sequence of ``QuoteState``). Consumed once, so a
    generator keeps memory bounded.

    ``gap_seconds`` is the inter-bar elapsed time above which ``gap_indicator``
    is set; the value comes from ``BarConfig.gap_seconds``.
    """
    series = BarSeries()
    bar_id = 0
    current_bucket: str | None = None
    acc: _BarAcc | None = None
    pending_invalid: list[Any] = []
    prior_mid_close: float | None = None
    prior_last_ts: str | None = None

    for state in quote_states:
        valid = getattr(state, "valid", True)
        if not valid:
            if current_bucket is not None:
                try:
                    bucket = _minute_bucket(getattr(state, "timestamp"))
                except (ValueError, AttributeError):
                    bucket = ""
                if bucket == current_bucket:
                    _apply_tick(acc, state)  # type: ignore[arg-type]
                    continue
            pending_invalid.append(state)
            continue

        bucket = _minute_bucket(getattr(state, "timestamp"))
        if current_bucket is None:
            first_row = getattr(state, "source_row_id", 0)
            for pending in pending_invalid:
                try:
                    pending_bucket = _minute_bucket(getattr(pending, "timestamp"))
                except (ValueError, AttributeError):
                    pending_bucket = ""
                if pending_bucket == bucket:
                    first_row = min(first_row, getattr(pending, "source_row_id", first_row))
            acc = _BarAcc(row_start=first_row, minute_start=bucket)
            current_bucket = bucket
            for pending in pending_invalid:
                try:
                    pending_bucket = _minute_bucket(getattr(pending, "timestamp"))
                except (ValueError, AttributeError):
                    pending_bucket = ""
                if pending_bucket == bucket:
                    _apply_tick(acc, pending)
                else:
                    series.invalid_ticks_without_bar += 1
            pending_invalid.clear()
            _apply_tick(acc, state)
            continue

        if bucket == current_bucket:
            assert acc is not None
            for pending in pending_invalid:
                try:
                    pending_bucket = _minute_bucket(getattr(pending, "timestamp"))
                except (ValueError, AttributeError):
                    pending_bucket = ""
                if pending_bucket == bucket:
                    _apply_tick(acc, pending)
                else:
                    series.invalid_ticks_without_bar += 1
            pending_invalid.clear()
            _apply_tick(acc, state)
            continue

        # A valid tick in a new bucket closes the current bar and opens a new
        # one. Pending invalid ticks belong to minutes that never materialised.
        series.invalid_ticks_without_bar += len(pending_invalid)
        pending_invalid.clear()
        assert acc is not None
        series.bars.append(
            _finalize_bar(acc, bar_id, prior_last_ts, prior_mid_close, gap_seconds)
        )
        prior_mid_close = acc.mid_c
        prior_last_ts = acc.last_ts
        bar_id += 1

        acc = _BarAcc(
            row_start=getattr(state, "source_row_id", 0),
            minute_start=bucket,
        )
        current_bucket = bucket
        _apply_tick(acc, state)

    if acc is not None:
        series.bars.append(
            _finalize_bar(acc, bar_id, prior_last_ts, prior_mid_close, gap_seconds)
        )
    series.invalid_ticks_without_bar += len(pending_invalid)
    return series


__all__ = ["Bar", "BarSeries", "build_bars"]
