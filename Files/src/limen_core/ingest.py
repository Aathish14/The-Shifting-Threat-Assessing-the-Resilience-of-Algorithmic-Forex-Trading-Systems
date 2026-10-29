"""Streaming ingestion and source audit for the immutable EUR/USD CSV.

The input is treated as unverified until its schema, delimiter, timestamp
basis, ordering, and quote/flag semantics are established. Exactly one pass
parses the file; the SHA-256 digest is a separate streaming read (it needs the
raw bytes, not the parsed rows).

Source schema observed from the local file:
    <DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>
    e.g. 2026.03.05\t00:00:10.657\t1.16299\t1.16348\t\t\t6

FLAGS bit semantics (current adapter contract, MT5-like):
    bit value 2  -> bid side updated this row
    bit value 4  -> ask side updated this row
    FLAGS=6      -> both bid and ask updated
    FLAGS=2      -> bid only (carry forward prior ask)
    FLAGS=4      -> ask only (carry forward prior bid)
    FLAGS=0      -> neither side updated (carry forward both)

Memory
------
``iter_ticks`` is a generator: ticks are produced one at a time as immutable
``_TickInput`` named tuples and consumed incrementally by ``build_bars``. Audit
statistics accumulate in a small fixed-size object, so a 22M-row file costs
O(1) memory rather than O(rows).

Audit counter definition
------------------------
``crossed_quotes``, ``bid_nonpositive`` and ``ask_nonpositive`` are counted on
the *reconstructed* (carry-forward) quote state by the same state machine the
runtime replay uses, so the audit and the runtime cannot disagree. This is
recorded in ``SourceAudit.notes``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, NamedTuple

from .quotes import (
    EPOCH_TS,
    FLAGS_ASK,
    FLAGS_BID,
    QuoteReconstructor,
    QuoteStateMachine,
    TickResult,
)
from .timestamp import ts_to_epoch


class _TickInput(NamedTuple):
    """Lightweight tick input for bar construction.

    Immutable and allocation-cheap (no per-tick list). ``reason`` carries the
    hard-invalid code for this tick and ``soft_reasons`` any soft evidence
    (staleness). ``bid_age_seconds`` / ``ask_age_seconds`` are epoch seconds
    since each side last changed at this tick.
    """

    row_id: int
    timestamp: str
    source_row_id: int
    bid: float | None
    ask: float | None
    midpoint: float | None
    spread: float | None
    valid: bool
    reason: str
    soft_reasons: tuple[str, ...]
    bid_age_seconds: float | None
    ask_age_seconds: float | None
    bid_updated: bool
    ask_updated: bool
    flags_int: int

    @property
    def quality_flags(self) -> list[str]:
        """Ordered union of this tick's reason codes (hard first)."""
        out: list[str] = []
        if not self.valid:
            out.append(self.reason)
        out.extend(self.soft_reasons)
        return out


@dataclass
class SourceAudit:
    """Record of the source-file audit results."""

    path: str
    byte_size: int
    sha256: str
    delimiter: str
    encoding: str
    header: list[str]
    column_count: int
    total_rows: int
    parse_failures: int
    first_timestamp: str | None
    last_timestamp: str | None
    timestamp_basis: str  # "raw_export" until verified
    ordering: str  # "nondecreasing" | "mixed" | "unknown"
    equal_adjacent_count: int
    backward_count: int
    duplicate_rows: int
    missing_values_by_column: dict[str, int]
    flag_value_counts: dict[int, int]
    bid_nonpositive: int
    ask_nonpositive: int
    crossed_quotes: int
    active_dates: list[str]
    timestamp: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "delimiter": self.delimiter,
            "encoding": self.encoding,
            "header": self.header,
            "column_count": self.column_count,
            "total_rows": self.total_rows,
            "parse_failures": self.parse_failures,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "timestamp_basis": self.timestamp_basis,
            "ordering": self.ordering,
            "equal_adjacent_count": self.equal_adjacent_count,
            "backward_count": self.backward_count,
            "duplicate_rows": self.duplicate_rows,
            "missing_values_by_column": self.missing_values_by_column,
            "flag_value_counts": self.flag_value_counts,
            "bid_nonpositive": self.bid_nonpositive,
            "ask_nonpositive": self.ask_nonpositive,
            "crossed_quotes": self.crossed_quotes,
            "active_dates": self.active_dates,
            "timestamp": self.timestamp,
            "notes": list(self.notes),
        }


@dataclass
class ParsedRow:
    """A single parsed source row, preserving original fields."""

    row_id: int  # 1-based source row index (excluding header)
    raw_date: str
    raw_time: str
    raw_bid: str
    raw_ask: str
    raw_last: str
    raw_volume: str
    raw_flags: str
    flags_int: int
    bid_updated: bool
    ask_updated: bool
    parse_ok: bool
    error: str | None = None


class StreamingReader:
    """Chunked reader for the immutable CSV. Yields ParsedRow objects."""

    def __init__(
        self,
        path: str | Path,
        delimiter: str = "\t",
        max_rows: int | None = None,
        chunk_rows: int = 100_000,
    ) -> None:
        self.path = Path(path)
        self.delimiter = delimiter
        self.max_rows = max_rows
        self.chunk_rows = chunk_rows
        self._header: list[str] | None = None
        self._columns: int | None = None

    @property
    def header(self) -> list[str]:
        if self._header is None:
            self._detect_header()
        return self._header

    @property
    def column_count(self) -> int:
        if self._columns is None:
            self._detect_header()
        return self._columns  # type: ignore[return-value]

    def _detect_header(self) -> None:
        with self.path.open("r", encoding="utf-8", errors="replace") as fh:
            first_line = fh.readline()
        self._header = first_line.rstrip("\n").rstrip("\r").split(self.delimiter)
        self._columns = len(self._header)

    def iter_rows(self) -> Iterator[ParsedRow]:
        """Yield parsed rows in strict source order, bounded by max_rows."""
        row_id = 0
        emitted = 0
        with self.path.open("r", encoding="utf-8", errors="replace") as fh:
            header_line = fh.readline()  # skip header
            if not header_line:
                return
            for line in fh:
                row_id += 1
                if self.max_rows is not None and emitted >= self.max_rows:
                    break
                row = self._parse_line(row_id, line)
                if row.parse_ok:
                    emitted += 1
                yield row

    def _parse_line(self, row_id: int, line: str) -> ParsedRow:
        line = line.rstrip("\n").rstrip("\r")
        parts = line.split(self.delimiter)
        if len(parts) != self.column_count:
            return ParsedRow(
                row_id=row_id,
                raw_date="", raw_time="", raw_bid="", raw_ask="",
                raw_last="", raw_volume="", raw_flags="",
                flags_int=0, bid_updated=False, ask_updated=False,
                parse_ok=False,
                error=f"column_count {len(parts)} != expected {self.column_count}",
            )
        (
            raw_date, raw_time, raw_bid, raw_ask,
            raw_last, raw_volume, raw_flags,
        ) = parts
        flags_int = 0
        bid_updated = False
        ask_updated = False
        error: str | None = None
        try:
            flags_int = int(raw_flags)
            bid_updated = bool(flags_int & FLAGS_BID)
            ask_updated = bool(flags_int & FLAGS_ASK)
        except ValueError:
            error = f"non-integer FLAGS value: {raw_flags!r}"
            flags_int = 0
        return ParsedRow(
            row_id=row_id,
            raw_date=raw_date, raw_time=raw_time, raw_bid=raw_bid,
            raw_ask=raw_ask, raw_last=raw_last, raw_volume=raw_volume,
            raw_flags=raw_flags, flags_int=flags_int,
            bid_updated=bid_updated, ask_updated=ask_updated,
            parse_ok=error is None, error=error,
        )


def compute_sha256(path: str | Path) -> str:
    """Stream-compute the SHA-256 digest of a file without loading it into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1_048_576), b""):
            h.update(block)
    return h.hexdigest()


class _AuditAccumulator:
    """Fixed-size audit statistics, finalised after the tick stream is drained."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.byte_size = path.stat().st_size
        self.header: list[str] = []
        self.column_count = 0
        self.total_rows = 0
        self.first_timestamp: str | None = None
        self.last_timestamp: str | None = None
        self.equal_adjacent = 0
        self.raw_backward = 0
        self.duplicates = 0
        self.missing_by_col: dict[str, int] = {}
        self.flag_counts: dict[int, int] = {}
        self.active_dates: set[str] = set()

    def finalize(self, machine: QuoteStateMachine) -> SourceAudit:
        if self.raw_backward > 0:
            ordering = "mixed"
        elif self.column_count == 0 and self.total_rows == 0:
            ordering = "unknown"
        else:
            ordering = "nondecreasing"

        notes: list[str] = []
        notes.append("Timestamps preserved exactly as exported; no conversion applied.")
        notes.append("FLAGS bits: 2=bid updated, 4=ask updated (current adapter contract).")
        notes.append(
            "crossed_quotes / bid_nonpositive / ask_nonpositive are counted on the "
            "reconstructed carry-forward state, identical to the runtime path."
        )
        if self.raw_backward:
            notes.append(
                f"{self.raw_backward} raw row(s) out of timestamp order; "
                f"{machine.backward_count} flagged INVALID:BACKWARD_TIMESTAMP at runtime."
            )
        if machine.parse_failures:
            notes.append(
                f"{machine.parse_failures} row(s) failed to parse and were excluded."
            )
        if machine.crossed_quotes:
            notes.append(
                f"{machine.crossed_quotes} crossed reconstructed quote row(s)."
            )

        return SourceAudit(
            path=str(self.path),
            byte_size=self.byte_size,
            sha256=compute_sha256(self.path),
            delimiter="\\t",
            encoding="utf-8",
            header=list(self.header),
            column_count=self.column_count,
            total_rows=self.total_rows,
            parse_failures=machine.parse_failures,
            first_timestamp=self.first_timestamp,
            last_timestamp=self.last_timestamp,
            timestamp_basis="raw_export_unverified",
            ordering=ordering,
            equal_adjacent_count=self.equal_adjacent,
            backward_count=machine.backward_count,
            duplicate_rows=self.duplicates,
            missing_values_by_column=dict(self.missing_by_col),
            flag_value_counts=dict(self.flag_counts),
            bid_nonpositive=machine.bid_nonpositive,
            ask_nonpositive=machine.ask_nonpositive,
            crossed_quotes=machine.crossed_quotes,
            active_dates=sorted(self.active_dates),
            notes=notes,
        )


class TickStream:
    """Iterator of ``_TickInput`` plus the accumulating source audit.

    Usage::

        stream = iter_ticks(path, ...)
        series = build_bars(stream)      # consumes incrementally
        audit = stream.finalize()        # after exhaustion
    """

    def __init__(
        self,
        path: str | Path,
        max_rows: int | None = None,
        track_duplicates: bool = False,
        stale_threshold_seconds: float | None = None,
        enable_timestamp_order: bool = True,
        gap_reset_seconds: float | None = None,
        stale_is_hard: bool = True,
    ) -> None:
        self.path = Path(path)
        self.max_rows = max_rows
        self.track_duplicates = track_duplicates
        # Item 1: the production ingest path drives ``QuoteReconstructor``.
        # ``retain_states=False`` keeps memory bounded over 22 M rows.
        self._reconstructor = QuoteReconstructor(
            stale_threshold_seconds=stale_threshold_seconds,
            gap_reset_seconds=gap_reset_seconds,
            enable_timestamp_order=enable_timestamp_order,
            stale_is_hard=stale_is_hard,
            retain_states=False,
        )
        self._machine: QuoteStateMachine = self._reconstructor.machine
        self._acc = _AuditAccumulator(self.path)
        self._seen_rows: set[bytes] | None = set() if track_duplicates else None
        self._finished = False
        self._gen = self._generate()

    @property
    def reconstructor(self) -> QuoteReconstructor:
        """The reconstruction object driving this stream."""
        return self._reconstructor

    def __iter__(self) -> "TickStream":
        return self

    def __next__(self) -> _TickInput:
        return next(self._gen)

    @property
    def machine(self) -> QuoteStateMachine:
        """The reconstruction state machine (inspection/test access)."""
        return self._machine

    def finalize(self) -> SourceAudit:
        """Drain any remaining ticks and return the audit record."""
        if not self._finished:
            for _ in self._gen:
                pass
        return self._acc.finalize(self._machine)

    # -- internals -------------------------------------------------------
    def _make_tick(self, row_id: int, ts: str, flags_int: int, res: TickResult) -> _TickInput:
        return _TickInput(
            row_id=row_id,
            timestamp=ts,
            source_row_id=row_id,
            bid=res.bid,
            ask=res.ask,
            midpoint=res.midpoint,
            spread=res.spread,
            valid=res.valid,
            reason=res.reason,
            soft_reasons=res.soft_reasons,
            bid_age_seconds=res.bid_age_seconds,
            ask_age_seconds=res.ask_age_seconds,
            bid_updated=res.bid_updated,
            ask_updated=res.ask_updated,
            flags_int=flags_int,
        )

    def _generate(self) -> Iterator[_TickInput]:
        acc = self._acc
        prev_date: bytes | None = None
        prev_time: bytes | None = None
        try:
            with self.path.open("rb") as fh:
                header_line = fh.readline()
                if not header_line:
                    self._finished = True
                    return
                header = [
                    h.decode("utf-8")
                    for h in header_line.rstrip(b"\n").rstrip(b"\r").split(b"\t")
                ]
                acc.header = header
                acc.column_count = len(header)
                acc.missing_by_col = {c: 0 for c in header}

                for line in fh:
                    if self.max_rows is not None and acc.total_rows >= self.max_rows:
                        break
                    acc.total_rows += 1
                    row_id = acc.total_rows
                    line = line.rstrip(b"\n").rstrip(b"\r")
                    parts = line.split(b"\t")

                    if len(parts) != len(header):
                        ts = self._machine._last_timestamp or EPOCH_TS
                        res = self._reconstructor.apply_tick(
                            ts=ts, epoch=None, parse_ok=False, row_id=row_id
                        )
                        yield self._make_tick(row_id, ts, 0, res)
                        continue

                    (
                        raw_date, raw_time, raw_bid, raw_ask,
                        raw_last, raw_volume, raw_flags,
                    ) = parts
                    ts = (
                        raw_date.decode("utf-8") + " " + raw_time.decode("utf-8")
                    )

                    if acc.first_timestamp is None:
                        acc.first_timestamp = ts
                    acc.last_timestamp = ts
                    acc.active_dates.add(raw_date.decode("utf-8"))

                    try:
                        flags_int = int(raw_flags)
                        parse_ok = True
                    except ValueError:
                        flags_int = 0
                        parse_ok = False
                    acc.flag_counts[flags_int] = (
                        acc.flag_counts.get(flags_int, 0) + 1
                    )

                    if not raw_bid:
                        acc.missing_by_col["<BID>"] = (
                            acc.missing_by_col.get("<BID>", 0) + 1
                        )
                    if not raw_ask:
                        acc.missing_by_col["<ASK>"] = (
                            acc.missing_by_col.get("<ASK>", 0) + 1
                        )
                    if not raw_last:
                        acc.missing_by_col["<LAST>"] = (
                            acc.missing_by_col.get("<LAST>", 0) + 1
                        )
                    if not raw_volume:
                        acc.missing_by_col["<VOLUME>"] = (
                            acc.missing_by_col.get("<VOLUME>", 0) + 1
                        )
                    if not raw_flags:
                        acc.missing_by_col["<FLAGS>"] = (
                            acc.missing_by_col.get("<FLAGS>", 0) + 1
                        )

                    if self._seen_rows is not None:
                        row_key = b"\t".join(parts)
                        if row_key in self._seen_rows:
                            acc.duplicates += 1
                        self._seen_rows.add(row_key)

                    # Raw source ordering (audit provenance, not the runtime flag).
                    if prev_date is not None:
                        if raw_date < prev_date or (
                            raw_date == prev_date and raw_time < prev_time  # type: ignore[operator]
                        ):
                            acc.raw_backward += 1
                        elif raw_date == prev_date and raw_time == prev_time:
                            acc.equal_adjacent += 1
                    prev_date, prev_time = raw_date, raw_time

                    epoch: float | None = None
                    if parse_ok:
                        try:
                            epoch = ts_to_epoch(ts)
                        except (ValueError, TypeError):
                            epoch = None

                    res = self._reconstructor.apply_tick(
                        ts=ts,
                        epoch=epoch,
                        flags_int=flags_int,
                        bid_updated=bool(flags_int & FLAGS_BID),
                        ask_updated=bool(flags_int & FLAGS_ASK),
                        raw_bid=raw_bid,
                        raw_ask=raw_ask,
                        parse_ok=parse_ok,
                        row_id=row_id,
                    )
                    yield self._make_tick(row_id, ts, flags_int, res)
        finally:
            self._finished = True


def iter_ticks(
    path: str | Path,
    max_rows: int | None = None,
    track_duplicates: bool = False,
    stale_threshold_seconds: float | None = None,
    enable_timestamp_order: bool = True,
    gap_reset_seconds: float | None = None,
    stale_is_hard: bool = True,
) -> TickStream:
    """Return a streaming tick iterator with an accumulating audit.

    The returned object is both an iterator of ``_TickInput`` and the holder of
    the audit statistics; call ``.finalize()`` after exhaustion.
    """
    return TickStream(
        path,
        max_rows=max_rows,
        track_duplicates=track_duplicates,
        stale_threshold_seconds=stale_threshold_seconds,
        enable_timestamp_order=enable_timestamp_order,
        gap_reset_seconds=gap_reset_seconds,
        stale_is_hard=stale_is_hard,
    )


def audit_source(path: str | Path, max_rows: int | None = None) -> SourceAudit:
    """Produce a full bounded-memory audit of the source CSV."""
    stream = iter_ticks(path, max_rows=max_rows)
    for _ in stream:
        pass
    return stream.finalize()


def audit_and_reconstruct(
    path: str | Path,
    max_rows: int | None = None,
    track_duplicates: bool = False,
    stale_threshold_seconds: float | None = None,
) -> tuple[SourceAudit, list]:
    """Compatibility wrapper: audit plus a materialised list of ticks.

    Prefer :func:`iter_ticks` for large files; this helper exists for callers
    (and tests) that need the whole tick list in memory at once.
    """
    stream = iter_ticks(
        path,
        max_rows=max_rows,
        track_duplicates=track_duplicates,
        stale_threshold_seconds=stale_threshold_seconds,
    )
    ticks = list(stream)
    return stream.finalize(), ticks


def save_audit(audit: SourceAudit, path: str | Path) -> None:
    """Atomically write the SourceAudit as JSON."""
    path = Path(path)
    payload = json.dumps(audit.to_dict(), indent=2, sort_keys=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


__all__ = [
    "ParsedRow",
    "SourceAudit",
    "StreamingReader",
    "TickStream",
    "_TickInput",
    "audit_and_reconstruct",
    "audit_source",
    "compute_sha256",
    "iter_ticks",
    "save_audit",
]
