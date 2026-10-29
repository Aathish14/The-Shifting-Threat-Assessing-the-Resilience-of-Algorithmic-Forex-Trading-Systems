"""Canonical UTC timestamp utilities for Limen Core.

Every generated record, log line, CSV row, and report produced by this
project must carry an explicit timestamp. This module is the single source
of truth for timestamp generation and validation.

Timestamp format (ISO 8601, UTC, milliseconds):
    datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


# Format used by the source export: "YYYY.MM.DD HH:MM:SS.mmm"
SOURCE_TS_FORMAT = "%Y.%m.%d %H:%M:%S.%f"

# Cache of per-date midnight epoch seconds. The number of distinct dates in a
# multi-year tick file is small, so this stays bounded in practice.
_DATE_MIDNIGHT_CACHE: dict[str, float] = {}


def _date_midnight_epoch(date_part: str) -> float:
    cached = _DATE_MIDNIGHT_CACHE.get(date_part)
    if cached is None:
        dt = datetime.strptime(date_part, "%Y.%m.%d").replace(tzinfo=timezone.utc)
        cached = dt.timestamp()
        _DATE_MIDNIGHT_CACHE[date_part] = cached
    return cached


def ts_to_epoch(ts: str) -> float:
    """Convert a source timestamp to epoch seconds (UTC) without strptime.

    Input format is ``YYYY.MM.DD HH:MM:SS.mmm`` (the ``.mmm`` fraction is
    optional). The date portion is parsed once per distinct date and cached;
    the time portion is converted arithmetically. This is the single shared
    replacement for the per-row ``datetime.strptime`` calls that previously
    dominated ingestion cost.

    Raises ``ValueError`` for malformed input so callers can preserve the
    "invalid timestamp is a failed contract" behaviour.
    """
    date_part, sep, time_part = ts.partition(" ")
    if not sep:
        raise ValueError(f"malformed timestamp (missing time): {ts!r}")
    base = _date_midnight_epoch(date_part)
    hh, sep1, rest = time_part.partition(":")
    mm, sep2, rest2 = rest.partition(":")
    if not (sep1 and sep2):
        raise ValueError(f"malformed time portion: {ts!r}")
    ss, _, frac = rest2.partition(".")
    seconds = int(hh) * 3600 + int(mm) * 60 + int(ss)
    if frac:
        if not frac.isdigit():
            raise ValueError(f"malformed fractional seconds: {ts!r}")
        seconds += int(frac) / (10.0 ** len(frac))
    return base + seconds


def seconds_between(ts_a: str, ts_b: str) -> float | None:
    """Return ``ts_b - ts_a`` in seconds, or ``None`` if either is malformed."""
    try:
        return ts_to_epoch(ts_b) - ts_to_epoch(ts_a)
    except (ValueError, TypeError):
        return None


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string with milliseconds and a Z suffix.

    Example: 2026-10-27T10:01:47.725Z
    """
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def utc_now_dt() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def is_utc_aware(dt: datetime) -> bool:
    """Return True if *dt* is timezone-aware and uses UTC."""
    if dt.tzinfo is None:
        return False
    return dt.utcoffset() == timezone.utc.utcoffset(None)


def ensure_utc_iso(dt: datetime | str) -> str:
    """Normalize *dt* to the canonical UTC ISO 8601 milliseconds string.

    Accepts a timezone-aware datetime or an ISO 8601 string ending in ``Z``.
    Raises ValueError for naive datetimes or non-UTC offsets.
    """
    if isinstance(dt, str):
        s = dt.strip()
        if not s.endswith("Z"):
            raise ValueError(f"timestamp string must end with Z: {s!r}")
        # Validate parse round-trip
        parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError(f"timestamp string is not timezone-aware: {s!r}")
        return s
    if dt.tzinfo is None:
        raise ValueError("naive datetime is not permitted; use UTC-aware datetimes")
    if dt.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError(f"non-UTC offset is not permitted: {dt.utcoffset()}")
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def timestamped(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *record* with a ``timestamp`` field set to now (UTC).

    Does not mutate the input. If the record already has a ``timestamp`` key,
    it is overwritten with the generation time.
    """
    out = dict(record)
    out["timestamp"] = utc_now_iso()
    return out


def iso_to_dt(iso_str: str) -> datetime:
    """Parse a canonical UTC ISO 8601 string into a timezone-aware datetime."""
    s = iso_str.strip()
    if not s.endswith("Z"):
        raise ValueError(f"expected Z-suffixed UTC timestamp: {iso_str!r}")
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


__all__ = [
    "SOURCE_TS_FORMAT",
    "ts_to_epoch",
    "seconds_between",
    "utc_now_iso",
    "utc_now_dt",
    "is_utc_aware",
    "ensure_utc_iso",
    "timestamped",
    "iso_to_dt",
]
