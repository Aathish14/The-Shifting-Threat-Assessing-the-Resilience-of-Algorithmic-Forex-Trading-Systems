"""Stateful bid/ask quote-state reconstruction from parsed source rows.

One state machine is shared by every consumer (``QuoteReconstructor`` for the
public object API and ``ingest.iter_ticks`` for the streaming path), so the
two cannot drift apart. The parity test in ``tests/test_quotes.py`` asserts it.

Contract
--------
* The FLAGS bit for a side says that side changed on this row; when it did not
  change, the prior value is carried forward.
* A changed side without a finite, strictly positive value is rejected (the
  prior value is kept) and the row is ``INVALID:MISSING_OR_NONPOSITIVE_BID`` /
  ``..._ASK``.
* A row before both sides exist persists the side it did receive and reports
  ``INVALID:BOTH_SIDES_NOT_INITIALIZED`` until both sides exist.
* A row whose timestamp is earlier than the high-water mark is
  ``INVALID:BACKWARD_TIMESTAMP``: no bid/ask state and no mark are updated.
  Equal timestamps are valid.
* A reconstructed state with bid > ask is ``INVALID:CROSSED_QUOTE``.
* Staleness is *soft* evidence only: the tick remains valid and stays in OHLC.
  ``SUSPECT:STALE_BID`` / ``SUSPECT:STALE_ASK`` are recorded when a side's age
  (epoch seconds since that side's last update) is strictly greater than the
  configured threshold. Side ages are reset across a session gap larger than
  ``gap_reset_seconds`` so a weekend close does not look like a stale feed.

Precedence when several conditions apply is deterministic:
``PARSE_FAILURE`` > ``BACKWARD_TIMESTAMP`` > ``MISSING_OR_NONPOSITIVE_*`` >
``BOTH_SIDES_NOT_INITIALIZED`` > ``CROSSED_QUOTE``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, NamedTuple

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from .ingest import ParsedRow


class QuoteReconstructionError(ValueError):
    """Raised when a source row violates the quote-state contract."""


# Hard-invalid reason codes (one definition, used by monitors and tests).
REASON_OK = "OK"
REASON_PARSE_FAILURE = "INVALID:PARSE_FAILURE"
REASON_BACKWARD = "INVALID:BACKWARD_TIMESTAMP"
REASON_MISSING_BID = "INVALID:MISSING_OR_NONPOSITIVE_BID"
REASON_MISSING_ASK = "INVALID:MISSING_OR_NONPOSITIVE_ASK"
REASON_BOTH_UNINIT = "INVALID:BOTH_SIDES_NOT_INITIALIZED"
REASON_CROSSED = "INVALID:CROSSED_QUOTE"
REASON_STALE_BID = "INVALID:STALE_BID"
REASON_STALE_ASK = "INVALID:STALE_ASK"
REASON_STALE = "INVALID:STALE_QUOTE"

# Soft reason codes: evidence that never blocks a tick by itself. Only used
# when ``stale_is_hard`` is explicitly disabled.
SOFT_STALE_BID = "SUSPECT:STALE_BID"
SOFT_STALE_ASK = "SUSPECT:STALE_ASK"

HARD_INVALID_REASONS: frozenset[str] = frozenset({
    REASON_PARSE_FAILURE,
    REASON_BACKWARD,
    REASON_MISSING_BID,
    REASON_MISSING_ASK,
    REASON_BOTH_UNINIT,
    REASON_CROSSED,
    REASON_STALE_BID,
    REASON_STALE_ASK,
    REASON_STALE,
})

FLAGS_BID = 2
FLAGS_ASK = 4
KNOWN_FLAG_BITS = FLAGS_BID | FLAGS_ASK

# Timestamp used to keep a malformed leading row's timestamp usable.
EPOCH_TS = "1970.01.01 00:00:00.000"


def is_hard_invalid(reason: str) -> bool:
    """Return True when *reason* is a hard-invalid reason code."""
    return bool(reason) and reason.startswith("INVALID:")


class TickResult(NamedTuple):
    """Result of applying one source row to the quote state machine."""

    valid: bool
    reason: str
    soft_reasons: tuple[str, ...]
    bid: float | None
    ask: float | None
    midpoint: float | None
    spread: float | None
    bid_age_seconds: float | None
    ask_age_seconds: float | None
    bid_updated: bool
    ask_updated: bool


@dataclass
class QuoteState:
    """Reconstructed quote state for a single observation."""

    row_id: int
    raw_date: str
    raw_time: str
    timestamp: str  # "DATE TIME" as exported
    bid: float | None
    ask: float | None
    midpoint: float | None
    spread: float | None
    flags_int: int
    bid_updated: bool
    ask_updated: bool
    valid: bool
    reason: str
    quality_flags: list[str] = field(default_factory=list)
    soft_reasons: tuple[str, ...] = ()
    bid_age_seconds: float | None = None
    ask_age_seconds: float | None = None
    error: str | None = None


def _as_positive(value: Any) -> float | None:
    """Parse *value* to a finite, strictly positive float, else ``None``."""
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


class QuoteStateMachine:
    """Carry-forward reconstruction with explicit, tested reason codes."""

    def __init__(
        self,
        stale_threshold_seconds: float | None = None,
        gap_reset_seconds: float | None = None,
        enable_timestamp_order: bool = True,
        stale_is_hard: bool = True,
    ) -> None:
        self.bid: float | None = None
        self.ask: float | None = None
        self.bid_timestamp: str | None = None
        self.ask_timestamp: str | None = None
        self._bid_epoch: float | None = None
        self._ask_epoch: float | None = None
        self._last_epoch: float | None = None
        self._last_timestamp: str | None = None
        self._any_tick = False
        self.stale_threshold_seconds = stale_threshold_seconds
        self.gap_reset_seconds = gap_reset_seconds
        self.enable_timestamp_order = enable_timestamp_order
        self.stale_is_hard = stale_is_hard
        # Reconstructed-state audit counters (single definition for both paths).
        self.crossed_quotes = 0
        self.bid_nonpositive = 0
        self.ask_nonpositive = 0
        self.parse_failures = 0
        self.backward_count = 0

    # -- helpers ---------------------------------------------------------
    @property
    def both_initialized(self) -> bool:
        return self.bid is not None and self.ask is not None

    def _mid_spread(self) -> tuple[float | None, float | None]:
        if self.bid is None or self.ask is None:
            return None, None
        return (self.bid + self.ask) / 2.0, self.ask - self.bid

    def _result(
        self,
        valid: bool,
        reason: str,
        soft_reasons: tuple[str, ...],
        bid_updated: bool,
        ask_updated: bool,
        age_epoch: float | None,
    ) -> TickResult:
        midpoint, spread = self._mid_spread()
        bid_age = None
        ask_age = None
        if age_epoch is not None:
            if self._bid_epoch is not None:
                bid_age = age_epoch - self._bid_epoch
            if self._ask_epoch is not None:
                ask_age = age_epoch - self._ask_epoch
        return TickResult(
            valid=valid,
            reason=reason,
            soft_reasons=soft_reasons,
            bid=self.bid,
            ask=self.ask,
            midpoint=midpoint,
            spread=spread,
            bid_age_seconds=bid_age,
            ask_age_seconds=ask_age,
            bid_updated=bid_updated,
            ask_updated=ask_updated,
        )

    # -- public API ------------------------------------------------------
    def apply(
        self,
        *,
        ts: str,
        epoch: float | None,
        flags_int: int = 0,
        bid_updated: bool = False,
        ask_updated: bool = False,
        raw_bid: Any = None,
        raw_ask: Any = None,
        parse_ok: bool = True,
    ) -> TickResult:
        """Apply one source row and return the reconstructed state.

        ``epoch`` is the row timestamp in epoch seconds (``None`` when the
        timestamp is malformed), computed by the caller with ``ts_to_epoch``.
        """
        # 1) Parse failure: never valid, even with an initialised state.
        if not parse_ok:
            self.parse_failures += 1
            return self._result(False, REASON_PARSE_FAILURE, (), False, False, None)
        if epoch is None:
            self.parse_failures += 1
            return self._result(False, REASON_PARSE_FAILURE, (), False, False, None)

        # 2) Backward timestamp: reject the whole row, corrupt nothing.
        if (
            self.enable_timestamp_order
            and self._last_epoch is not None
            and epoch < self._last_epoch
        ):
            self.backward_count += 1
            return self._result(False, REASON_BACKWARD, (), False, False, None)

        # The last usable timestamp is needed to keep a malformed row's
        # timestamp non-empty without inventing ordering information.
        self._last_timestamp = ts

        # Session-gap reset: a long market close must not look like a stale feed.
        gap_reset = (
            self.gap_reset_seconds is not None
            and self._last_epoch is not None
            and (epoch - self._last_epoch) > self.gap_reset_seconds
        )
        if gap_reset:
            self._bid_epoch = epoch
            self._ask_epoch = epoch

        # 3) Apply changed sides; a changed side with a bad value is rejected.
        invalid_reason = ""
        if bid_updated:
            parsed = _as_positive(raw_bid)
            if parsed is None:
                self.bid_nonpositive += 1
                invalid_reason = REASON_MISSING_BID
            else:
                self.bid = parsed
                self._bid_epoch = epoch
        if ask_updated:
            parsed = _as_positive(raw_ask)
            if parsed is None:
                self.ask_nonpositive += 1
                invalid_reason = invalid_reason or REASON_MISSING_ASK
            else:
                self.ask = parsed
                self._ask_epoch = epoch

        if invalid_reason:
            self._advance_mark(epoch)
            return self._result(False, invalid_reason, (), bid_updated, ask_updated, epoch)

        # 4) Both sides must exist before the state is usable.
        if not self.both_initialized:
            self._advance_mark(epoch)
            return self._result(
                False, REASON_BOTH_UNINIT, (), bid_updated, ask_updated, epoch
            )

        # 5) Crossed quote is hard-invalid (state is still updated: the latest
        #    observed quotes are what subsequent carry-forward must use).
        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            self.crossed_quotes += 1
            self._advance_mark(epoch)
            return self._result(False, REASON_CROSSED, (), bid_updated, ask_updated, epoch)

        # 6) Staleness. A side that has not been updated within the threshold
        #    makes the observation INVALID:STALE_BID / INVALID:STALE_ASK
        #    (Item 1). Soft mode is retained for experiments that want the
        #    evidence without withdrawing the observation.
        stale_sides: list[str] = []
        if self.stale_threshold_seconds is not None:
            if (
                self._bid_epoch is not None
                and (epoch - self._bid_epoch) > self.stale_threshold_seconds
            ):
                stale_sides.append("bid")
            if (
                self._ask_epoch is not None
                and (epoch - self._ask_epoch) > self.stale_threshold_seconds
            ):
                stale_sides.append("ask")

        if stale_sides and self.stale_is_hard:
            if len(stale_sides) == 2:
                reason = REASON_STALE
            elif stale_sides[0] == "bid":
                reason = REASON_STALE_BID
            else:
                reason = REASON_STALE_ASK
            self._advance_mark(epoch)
            return self._result(False, reason, (), bid_updated, ask_updated, epoch)

        soft = tuple(
            SOFT_STALE_BID if side == "bid" else SOFT_STALE_ASK
            for side in stale_sides
        )
        self._advance_mark(epoch)
        return self._result(True, REASON_OK, soft, bid_updated, ask_updated, epoch)

    def _advance_mark(self, epoch: float) -> None:
        if self._last_epoch is None or epoch >= self._last_epoch:
            self._last_epoch = epoch


class _QuoteStream:
    """Accumulates QuoteState objects for the object-oriented API."""

    def __init__(self) -> None:
        self.states: list[QuoteState] = []

    def append(self, state: QuoteState) -> None:
        self.states.append(state)


class QuoteReconstructor:
    """Object-oriented wrapper around :class:`QuoteStateMachine`.

    This is the reconstruction object the ingest path drives: ``TickStream``
    instantiates it with ``retain_states=False`` so the 22 M-row replay never
    materialises a per-row list, while tests use the default ``True`` to keep
    ``self.stream.states`` for inspection.
    """

    def __init__(
        self,
        fail_on_unknown_flags: bool = False,
        stale_threshold_seconds: float | None = None,
        gap_reset_seconds: float | None = None,
        enable_timestamp_order: bool = True,
        stale_is_hard: bool = True,
        retain_states: bool = True,
    ) -> None:
        self.fail_on_unknown_flags = fail_on_unknown_flags
        self.retain_states = retain_states
        self.machine = QuoteStateMachine(
            stale_threshold_seconds=stale_threshold_seconds,
            gap_reset_seconds=gap_reset_seconds,
            enable_timestamp_order=enable_timestamp_order,
            stale_is_hard=stale_is_hard,
        )
        self.stream = _QuoteStream()

    def add_row(self, row: "ParsedRow") -> QuoteState:
        """Apply a parsed row and return the reconstructed state."""
        return self.add_fast_row(
            row_id=row.row_id,
            raw_date=row.raw_date,
            raw_time=row.raw_time,
            raw_bid=row.raw_bid,
            raw_ask=row.raw_ask,
            raw_flags=row.flags_int,
            parse_ok=row.parse_ok,
            error=row.error,
        )

    def apply_tick(
        self,
        *,
        ts: str,
        epoch: float | None,
        flags_int: int = 0,
        bid_updated: bool = False,
        ask_updated: bool = False,
        raw_bid: Any = None,
        raw_ask: Any = None,
        parse_ok: bool = True,
        row_id: int | None = None,
    ) -> TickResult:
        """Apply one already-timestamped row without materialising a QuoteState.

        This is the streaming entry point. ``TickStream`` calls it once per
        source row, so ``QuoteReconstructor`` (not a bare machine) is the object
        the production ingest path drives, and the 22 M-row replay stays
        bounded because no ``QuoteState`` is retained.
        """
        unknown_bits = flags_int & ~KNOWN_FLAG_BITS
        if unknown_bits and self.fail_on_unknown_flags:
            where = f"row {row_id}: " if row_id is not None else ""
            raise QuoteReconstructionError(
                f"{where}unknown FLAGS bits {unknown_bits} in {flags_int}"
            )
        return self.machine.apply(
            ts=ts,
            epoch=epoch,
            flags_int=flags_int,
            bid_updated=bid_updated,
            ask_updated=ask_updated,
            raw_bid=raw_bid,
            raw_ask=raw_ask,
            parse_ok=parse_ok,
        )

    def add_fast_row(
        self,
        row_id: int,
        raw_date: str,
        raw_time: str,
        raw_bid: str,
        raw_ask: str,
        raw_flags: int,
        parse_ok: bool = True,
        error: str | None = None,
    ) -> QuoteState:
        """Apply already-parsed primitives and return the reconstructed state."""
        from .timestamp import ts_to_epoch

        bid_updated = bool(raw_flags & FLAGS_BID)
        ask_updated = bool(raw_flags & FLAGS_ASK)

        ts = f"{raw_date} {raw_time}" if (raw_date and raw_time) else (
            self.machine._last_timestamp or EPOCH_TS
        )
        epoch: float | None = None
        if parse_ok:
            try:
                epoch = ts_to_epoch(ts)
            except (ValueError, TypeError):
                epoch = None

        result = self.apply_tick(
            ts=ts,
            epoch=epoch,
            flags_int=raw_flags,
            bid_updated=bid_updated,
            ask_updated=ask_updated,
            raw_bid=raw_bid,
            raw_ask=raw_ask,
            parse_ok=parse_ok,
            row_id=row_id,
        )
        state = QuoteState(
            row_id=row_id,
            raw_date=raw_date,
            raw_time=raw_time,
            timestamp=ts,
            bid=result.bid,
            ask=result.ask,
            midpoint=result.midpoint,
            spread=result.spread,
            flags_int=raw_flags,
            bid_updated=result.bid_updated,
            ask_updated=result.ask_updated,
            valid=result.valid,
            reason=result.reason,
            quality_flags=list(result.soft_reasons),
            soft_reasons=result.soft_reasons,
            bid_age_seconds=result.bid_age_seconds,
            ask_age_seconds=result.ask_age_seconds,
            error=error,
        )
        if self.retain_states:
            self.stream.append(state)
        return state

    def iter_states(self) -> Iterator[QuoteState]:
        return iter(self.stream.states)


__all__ = [
    "EPOCH_TS",
    "FLAGS_ASK",
    "FLAGS_BID",
    "HARD_INVALID_REASONS",
    "KNOWN_FLAG_BITS",
    "QuoteReconstructionError",
    "QuoteReconstructor",
    "QuoteState",
    "QuoteStateMachine",
    "REASON_BACKWARD",
    "REASON_BOTH_UNINIT",
    "REASON_CROSSED",
    "REASON_MISSING_ASK",
    "REASON_MISSING_BID",
    "REASON_OK",
    "REASON_PARSE_FAILURE",
    "REASON_STALE",
    "REASON_STALE_ASK",
    "REASON_STALE_BID",
    "SOFT_STALE_ASK",
    "SOFT_STALE_BID",
    "TickResult",
    "is_hard_invalid",
]
