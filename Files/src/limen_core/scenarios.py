"""Deterministic synthetic fault / control scenario generator.

Applies one specified mutation per *episode* to a copy of canonical observations
and returns a hidden evaluator-only label per episode. Runtime components
(strategy, monitors, policy, drift) never see labels or masks.

Scenario classes:
  - identity                   (no mutation, negative control)
  - missing_quote              (bid or ask becomes unavailable)
  - stale_replay               (prior valid quote replayed beyond expected cadence)
  - crossed_quote              (bid > ask, explicitly constructed)
  - symmetric_shift            (bounded additive shift to both sides)
  - return_scale_shift         (scale a return stream over a declared interval)
  - combined_shift_and_quote   (independent shift + quote mutation)
  - benign_volatility          (widened spread under a predeclared rule, no fault)

LABEL ISOLATION (enforced, not merely intended)
-----------------------------------------------
A mutation never writes a scenario name into any runtime-visible field. The old
generator appended ``"missing_bid"`` / ``"stale_replay"`` to ``quality_flags``,
which leaked the answer key into the monitor's own evidence. ``apply_plan`` now
asserts that every bar's ``quality_flags`` is byte-identical before and after
mutation (when ``ScenarioConfig.label_isolation`` is set, which is the default).

ONE LABEL PER EPISODE
---------------------
An episode spanning N bars produces exactly one ``ScenarioLabel`` with
``onset_bar_id`` / ``offset_bar_id`` (inclusive) and ``duration_bars == N``.
``duration_bars == 1`` per label used to make a 5-bar episode look like five
independent events, inflating ``eligible_fault_episodes``.

DERIVED FIELDS AFTER MUTATION
-----------------------------
Every mutated bar keeps ``midpoint = (bid + ask) / 2`` and
``spread = ask - bid`` (except missing / crossed episodes, where those fields
are legitimately undefined), and ``log_return`` is recomputed for the episode
plus the bar after it, because the first post-episode return depends on the last
mutated midpoint.

``return_scale_shift`` reads returns from the *original* series (never from the
already-mutated copy), so scaling is applied to the true return stream exactly
once. The final cumulative ratio is carried forward to later bars so no spurious
jump is introduced at the boundary and returns after the episode are preserved
(EMA crossover detection is scale-invariant, so this keeps the apparent signal
continuous).
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .timestamp import seconds_between


# Class names are labels only: they must never appear in runtime fields.
SCENARIO_CLASSES: tuple[str, ...] = (
    "identity",
    "missing_quote",
    "stale_replay",
    "crossed_quote",
    "symmetric_shift",
    "return_scale_shift",
    "combined_shift_and_quote",
    "benign_volatility",
)

# Classes placed by the planner (identity is the control and is not injected).
FAULT_CLASSES: tuple[str, ...] = (
    "missing_quote",
    "stale_replay",
    "crossed_quote",
    "symmetric_shift",
    "return_scale_shift",
    "combined_shift_and_quote",
    "benign_volatility",
)

_BID_FIELDS = ("bid_open", "bid_high", "bid_low", "bid_close")
_ASK_FIELDS = ("ask_open", "ask_high", "ask_low", "ask_close")
_MID_FIELDS = ("midpoint_open", "midpoint_high", "midpoint_low", "midpoint_close")
_SPREAD_FIELDS = ("spread_open", "spread_high", "spread_low", "spread_close")


@dataclass
class ScenarioLabel:
    """Hidden evaluator-only label for one scenario episode. Never consumed at runtime."""

    episode_id: str
    scenario_class: str
    onset_bar_id: int
    offset_bar_id: int | None  # inclusive end of the episode
    duration_bars: int
    severity: float
    affected_fields: list[str]
    seed: int
    recovery_behavior: str
    expected_invariants: list[str] = field(default_factory=list)
    label_digest: str | None = None
    onset_index: int | None = None  # index into the mutated list (evaluator only)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "scenario_class": self.scenario_class,
            "onset_bar_id": self.onset_bar_id,
            "offset_bar_id": self.offset_bar_id,
            "duration_bars": self.duration_bars,
            "severity": self.severity,
            "affected_fields": list(self.affected_fields),
            "seed": self.seed,
            "recovery_behavior": self.recovery_behavior,
            "expected_invariants": list(self.expected_invariants),
            "label_digest": self.label_digest,
            "onset_index": self.onset_index,
        }


@dataclass
class ScenarioManifest:
    """Collection of scenario labels for one run."""

    episodes: list[ScenarioLabel] = field(default_factory=list)
    run_id: str = ""
    seed: int = 0
    version: str = "1.1.0"

    def to_csv_rows(self) -> list[dict[str, Any]]:
        rows = []
        for ep in self.episodes:
            rows.append({
                "episode_id": ep.episode_id,
                "scenario_class": ep.scenario_class,
                "onset_bar_id": ep.onset_bar_id,
                "offset_bar_id": ep.offset_bar_id if ep.offset_bar_id is not None else "",
                "duration_bars": ep.duration_bars,
                "severity": ep.severity,
                "affected_fields": ";".join(ep.affected_fields),
                "seed": ep.seed,
                "recovery_behavior": ep.recovery_behavior,
                "label_digest": ep.label_digest or "",
            })
        return rows

    def by_class(self) -> dict[str, list[ScenarioLabel]]:
        grouped: dict[str, list[ScenarioLabel]] = {}
        for ep in self.episodes:
            grouped.setdefault(ep.scenario_class, []).append(ep)
        return grouped


@dataclass(frozen=True)
class EpisodePlan:
    """A planned, not-yet-applied episode.

    ``onset_index`` is relative to the slice handed to :func:`apply_plan`, not
    to the full bar series and not a ``bar_id``.
    """

    episode_id: str
    scenario_class: str
    onset_index: int
    duration_bars: int
    severity: float
    side: str = "bid"


@dataclass(frozen=True)
class ScenarioPlan:
    """Deterministic, non-overlapping episode placement for one evaluation run."""

    seed: int
    placement: str
    episodes: tuple[EpisodePlan, ...] = ()

    def by_class(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ep in self.episodes:
            counts[ep.scenario_class] = counts.get(ep.scenario_class, 0) + 1
        return counts


def _digest(
    episode_id: str,
    onset: int,
    offset: int | None,
    severity: float,
    fields: Sequence[str],
) -> str:
    h = hashlib.sha256()
    h.update(
        f"{episode_id}|{onset}|{offset}|{severity}|{','.join(fields)}".encode("utf-8")
    )
    return h.hexdigest()


def _class_duration(config, scenario_class: str) -> int:
    return {
        "missing_quote": config.missing_quote_duration_bars,
        "stale_replay": config.stale_replay_duration_bars,
        "crossed_quote": config.crossed_quote_duration_bars,
        "symmetric_shift": config.symmetric_shift_duration_bars,
        "return_scale_shift": config.return_scale_duration_bars,
        "combined_shift_and_quote": config.combined_duration_bars,
        "benign_volatility": config.benign_volatility_duration_bars,
        "identity": 1,
    }[scenario_class]


def _class_severity(config, scenario_class: str) -> float:
    return {
        "missing_quote": 1.0,
        "stale_replay": 1.0,
        "crossed_quote": 1.0,
        "symmetric_shift": config.symmetric_shift_magnitude,
        "return_scale_shift": config.return_scale_factor,
        "combined_shift_and_quote": config.symmetric_shift_magnitude,
        "benign_volatility": config.benign_volatility_multiplier,
        "identity": 0.0,
    }[scenario_class]


def plan_episodes(
    config,
    start_index: int,
    end_index: int,
    entry_indices: Iterable[int] = (),
) -> ScenarioPlan:
    """Plan non-overlapping episodes inside ``[start_index, end_index)``.

    ``start_index`` / ``end_index`` and ``entry_indices`` are indices into the
    full series. The returned ``EpisodePlan.onset_index`` is **relative to
    ``start_index``**, because :func:`apply_plan` is applied to the slice
    ``bars[start_index:end_index]`` and must not deep-copy the whole series.

    ``placement="uniform"`` draws from every eligible index in the range.
    ``placement="entry_anchored"`` draws only from indices that are entry
    opportunities (so ``min_faulted_entries`` can actually be met), which is
    the placement needed for a per-class pass-through comparison.

    Every accepted episode reserves ``recovery_window_bars`` on both sides, so
    two episodes can never overlap and a post-episode recovery window is never
    itself faulted.
    """
    n = end_index - start_index
    if n <= 0:
        return ScenarioPlan(seed=config.seed, placement=config.placement, episodes=())

    rng = random.Random(config.seed)
    recovery = max(0, config.recovery_window_bars)
    accepted: list[tuple[int, int]] = []  # [start, end) reserved intervals
    episodes: list[EpisodePlan] = []
    counter = 0

    def overlaps(start: int, duration: int) -> bool:
        lo = start - recovery
        hi = start + duration + recovery
        for a_lo, a_hi in accepted:
            if lo < a_hi and a_lo < hi:
                return True
        return False

    entry_set = sorted({int(i) for i in entry_indices if start_index <= int(i) < end_index})

    for scenario_class in FAULT_CLASSES:
        duration = _class_duration(config, scenario_class)
        severity = _class_severity(config, scenario_class)

        if config.placement == "entry_anchored":
            candidates = [
                i for i in entry_set
                if i >= start_index + recovery
                and i + duration <= end_index - recovery
            ]
        else:
            first = start_index + recovery
            last = end_index - duration - recovery
            candidates = list(range(first, last + 1)) if last >= first else []

        if not candidates:
            continue

        shuffled = list(candidates)
        rng.shuffle(shuffled)
        placed = 0
        for onset in shuffled:
            if placed >= config.episodes_per_fault:
                break
            if overlaps(onset, duration):
                continue
            accepted.append((onset - recovery, onset + duration + recovery))
            episodes.append(EpisodePlan(
                episode_id=f"{scenario_class}_{counter:03d}",
                scenario_class=scenario_class,
                onset_index=onset - start_index,  # relative to the applied slice
                duration_bars=duration,
                severity=severity,
                side="bid",
            ))
            counter += 1
            placed += 1

    episodes.sort(key=lambda ep: ep.onset_index)
    return ScenarioPlan(
        seed=config.seed,
        placement=config.placement,
        episodes=tuple(episodes),
    )


# --------------------------------------------------------------------------
# Mutation helpers (each operates on the already-copied bar list)
# --------------------------------------------------------------------------
def _set_side(bar, fields: Sequence[str], value: float | None) -> None:
    for name in fields:
        setattr(bar, name, value)


def _set_mid_spread(bar, midpoint: float | None, spread: float | None) -> None:
    _set_side(bar, _MID_FIELDS, midpoint)
    _set_side(bar, _SPREAD_FIELDS, spread)


def _sync_sides_from_mid(bar, midpoint: float, spread: float) -> None:
    """Recompute bid/ask from a midpoint and spread (keeps consistency)."""
    half = spread / 2.0
    _set_side(bar, _BID_FIELDS, midpoint - half)
    _set_side(bar, _ASK_FIELDS, midpoint + half)


def _scale_fields(bar, fields: Sequence[str], ratio: float) -> None:
    for name in fields:
        value = getattr(bar, name)
        if value is not None:
            setattr(bar, name, value * ratio)


def _shift_fields(bar, fields: Sequence[str], delta: float) -> None:
    for name in fields:
        value = getattr(bar, name)
        if value is not None:
            setattr(bar, name, value + delta)


def _apply_missing_quote(mutated, onset, end, side, config) -> list[str]:
    for idx in range(onset, end):
        bar = mutated[idx]
        _set_side(bar, _BID_FIELDS if side == "bid" else _ASK_FIELDS, None)
        _set_mid_spread(bar, None, None)
    return [side, "midpoint", "spread"]


def _apply_stale_replay(mutated, onset, end, config) -> list[str]:
    if onset == 0:
        return []
    prior = mutated[onset - 1]
    prior_bid = prior.bid_close
    prior_ask = prior.ask_close
    prior_mid = prior.midpoint_close
    prior_spread = prior.spread_close
    prior_bid_age = getattr(prior, "bid_age_seconds_close", None)
    prior_ask_age = getattr(prior, "ask_age_seconds_close", None)
    prev_ts = prior.last_event_timestamp

    bid_age = prior_bid_age
    ask_age = prior_ask_age
    for idx in range(onset, end):
        bar = mutated[idx]
        _set_side(bar, _BID_FIELDS, prior_bid)
        _set_side(bar, _ASK_FIELDS, prior_ask)
        _set_mid_spread(bar, prior_mid, prior_spread)
        # A replayed quote is by definition not a fresh update, and each replayed
        # bar ages the stale side further. Both fields are runtime-visible, so a
        # stale detector can see the episode (v1.0.x left them unchanged, which
        # made a replayed quote indistinguishable from a live one).
        bar.bid_update_count = 0
        bar.ask_update_count = 0
        elapsed = seconds_between(prev_ts, bar.last_event_timestamp)
        if bid_age is not None and elapsed is not None:
            bid_age = bid_age + elapsed
        if ask_age is not None and elapsed is not None:
            ask_age = ask_age + elapsed
        bar.bid_age_seconds_close = bid_age
        bar.ask_age_seconds_close = ask_age
        prev_ts = bar.last_event_timestamp
    return ["bid", "ask", "midpoint", "spread", "ages", "update_counts"]


def _apply_crossed_quote(mutated, onset, end, config) -> list[str]:
    for idx in range(onset, end):
        bar = mutated[idx]
        if bar.bid_close is None or bar.ask_close is None:
            continue
        # Explicit construction: bid is the old ask, ask is the old bid. Since a
        # sane quote has ask > bid, the result always has bid > ask.
        bid = bar.ask_close
        ask = bar.bid_close
        _set_side(bar, _BID_FIELDS, bid)
        _set_side(bar, _ASK_FIELDS, ask)
        _set_mid_spread(bar, None, None)
    return ["bid", "ask", "midpoint", "spread"]


def _apply_symmetric_shift(mutated, onset, end, magnitude, config) -> list[str]:
    for idx in range(onset, end):
        bar = mutated[idx]
        _shift_fields(bar, _BID_FIELDS, magnitude)
        _shift_fields(bar, _ASK_FIELDS, magnitude)
        _shift_fields(bar, _MID_FIELDS, magnitude)
        # spread is invariant under a common additive shift
    return ["bid", "ask", "midpoint"]


def _apply_return_scale(mutated, original, onset, end, scale_factor, config) -> list[str]:
    if onset == 0:
        return []
    anchor_bar = original[onset - 1]
    anchor = anchor_bar.midpoint_close if anchor_bar is not None else None
    if anchor is None or anchor <= 0.0:
        return []

    current = anchor
    prev_original = anchor
    last_ratio = 1.0
    for idx in range(onset, end):
        orig = original[idx]
        orig_mid = orig.midpoint_close
        if orig_mid is None or orig_mid <= 0.0 or prev_original <= 0.0:
            continue
        # Returns come from the ORIGINAL series, never from the mutated copy.
        original_return = math.log(orig_mid / prev_original)
        current = current * math.exp(original_return * scale_factor)
        if not math.isfinite(current) or current <= 0.0:
            break
        bar = mutated[idx]
        spread = orig.spread_close
        if spread is None and orig.bid_close is not None and orig.ask_close is not None:
            spread = orig.ask_close - orig.bid_close
        _set_mid_spread(bar, current, spread)
        if spread is not None:
            _sync_sides_from_mid(bar, current, spread)
        prev_original = orig_mid
        last_ratio = current / orig_mid

    # Carry the final ratio forward: later returns are preserved exactly, and no
    # jump appears at the episode boundary. Fields are scaled *individually* and
    # spread is left untouched: recomputing bid/ask from mid +/- spread/2 would
    # perturb exact-equality relations and change the repeated-quote check on
    # bars that were not faulted (the leak assertion in the replay calls that
    # out). Scaling preserves equality because equal inputs produce equal
    # products.
    if last_ratio != 1.0:
        for idx in range(end, len(mutated)):
            bar = mutated[idx]
            mid = bar.midpoint_close
            if mid is None or mid <= 0.0:
                continue
            _scale_fields(bar, _BID_FIELDS + _ASK_FIELDS + _MID_FIELDS, last_ratio)
    return ["midpoint", "bid", "ask"]


def _apply_benign_volatility(mutated, onset, end, multiplier, config) -> list[str]:
    for idx in range(onset, end):
        bar = mutated[idx]
        spread = bar.spread_close
        mid = bar.midpoint_close
        if spread is None or mid is None:
            continue
        widened = spread * multiplier
        _set_mid_spread(bar, mid, widened)
        _sync_sides_from_mid(bar, mid, widened)
    return ["spread", "bid", "ask"]


def _recompute_log_return(bars: list, indices: Iterable[int]) -> None:
    """Recompute ``log_return`` for *indices* (and their predecessor link)."""
    for idx in sorted(set(indices)):
        if idx < 0 or idx >= len(bars):
            continue
        bar = bars[idx]
        prev_mid = bars[idx - 1].midpoint_close if idx > 0 else None
        mid = bar.midpoint_close
        if (
            prev_mid is not None
            and mid is not None
            and prev_mid > 0.0
            and mid > 0.0
            and math.isfinite(prev_mid)
            and math.isfinite(mid)
        ):
            bar.log_return = math.log(mid / prev_mid)
        else:
            bar.log_return = None


def _assert_label_isolation(original: list, mutated: list) -> None:
    """Fail loudly if a mutation leaked a label into a runtime-visible field."""
    if len(original) != len(mutated):
        raise AssertionError("mutation changed the bar count")
    for before, after in zip(original, mutated):
        if list(getattr(before, "quality_flags", [])) != list(
            getattr(after, "quality_flags", [])
        ):
            raise AssertionError(
                f"label isolation violated at bar {before.bar_id}: quality_flags changed"
            )
        for flag in getattr(after, "quality_flags", []) or []:
            for class_name in SCENARIO_CLASSES:
                if class_name in flag:
                    raise AssertionError(
                        f"label isolation violated at bar {after.bar_id}: "
                        f"scenario name {class_name!r} in quality_flags"
                    )


def _apply_episode(mutated: list, original: list, ep: EpisodePlan, config) -> tuple[ScenarioLabel, set[int]]:
    """Apply one planned episode; return its label and dirty indices."""
    onset = ep.onset_index
    end = min(onset + ep.duration_bars, len(mutated))
    if onset < 0 or onset >= len(mutated) or end <= onset:
        raise ValueError(
            f"episode {ep.episode_id} out of range: onset={onset} end={end} "
            f"n={len(mutated)}"
        )

    cls = ep.scenario_class
    if cls == "missing_quote":
        affected = _apply_missing_quote(mutated, onset, end, ep.side, config)
        invariants = [f"{ep.side}_set_to_none", "other_side_unchanged"]
        recovery = "immediate_after_offset"
    elif cls == "stale_replay":
        affected = _apply_stale_replay(mutated, onset, end, config)
        invariants = ["values_equal_prior_bar", "no_side_updates", "ages_advance"]
        recovery = "immediate_after_offset"
    elif cls == "crossed_quote":
        affected = _apply_crossed_quote(mutated, onset, end, config)
        invariants = ["bid_greater_than_ask"]
        recovery = "immediate_after_offset"
    elif cls == "symmetric_shift":
        affected = _apply_symmetric_shift(mutated, onset, end, ep.severity, config)
        invariants = ["spread_preserved", "shift_applied_to_both_sides"]
        recovery = "immediate_after_offset"
    elif cls == "return_scale_shift":
        affected = _apply_return_scale(
            mutated, original, onset, end, ep.severity, config
        )
        invariants = ["consistent_price_path", "return_scaled", "post_episode_returns_preserved"]
        recovery = "ratio_carried_forward"
    elif cls == "benign_volatility":
        affected = _apply_benign_volatility(mutated, onset, end, ep.severity, config)
        invariants = ["no_fault_injected", "spread_widened"]
        recovery = "immediate_after_offset"
    elif cls == "combined_shift_and_quote":
        _apply_symmetric_shift(mutated, onset, end, ep.severity, config)
        affected = _apply_missing_quote(mutated, onset, end, ep.side, config)
        affected = sorted(set(affected) | {"bid", "ask", "midpoint"})
        invariants = [
            "spread_preserved_before_quote_removal",
            f"{ep.side}_set_to_none",
        ]
        recovery = "immediate_after_offset"
    elif cls == "identity":
        affected = []
        invariants = ["all_fields_unchanged"]
        recovery = "none"
    else:
        raise ValueError(f"unknown scenario class: {cls!r}")

    last_index = end - 1
    label = ScenarioLabel(
        episode_id=ep.episode_id,
        scenario_class=cls,
        onset_bar_id=mutated[onset].bar_id,
        offset_bar_id=mutated[last_index].bar_id,
        duration_bars=end - onset,
        severity=float(ep.severity),
        affected_fields=list(affected),
        seed=int(getattr(config, "seed", 0)),
        recovery_behavior=recovery,
        expected_invariants=invariants,
        onset_index=onset,
    )
    label.label_digest = _digest(
        label.episode_id,
        label.onset_bar_id,
        label.offset_bar_id,
        label.severity,
        label.affected_fields,
    )
    return label, set(range(onset, min(end + 1, len(mutated))))


def apply_plan(
    bars: Sequence,
    plan: ScenarioPlan,
    config=None,
) -> tuple[list, list[ScenarioLabel]]:
    """Return (mutated copy of *bars*, one label per applied episode).

    The input list is never mutated in place. Episodes are non-overlapping by
    construction, so application order cannot change the result.
    """
    original = list(bars)
    mutated = copy.deepcopy(original)
    labels: list[ScenarioLabel] = []
    dirty: set[int] = set()

    for ep in sorted(plan.episodes, key=lambda e: e.onset_index):
        label, touched = _apply_episode(mutated, original, ep, config)
        labels.append(label)
        dirty |= touched

    if dirty:
        _recompute_log_return(mutated, dirty)

    if config is not None and getattr(config, "label_isolation", True):
        _assert_label_isolation(original, mutated)

    return mutated, labels


class ScenarioGenerator:
    """Deterministically applies synthetic mutations to observed bars.

    The ``make_*`` helpers are single-episode conveniences over :func:`apply_plan`;
    they exist for focused tests and manual experiments. Batch evaluation uses
    :func:`plan_episodes` + :func:`apply_plan` so placement stays non-overlapping
    and seeded.
    """

    def __init__(self, config, seed: int = 12345) -> None:
        self.config = config
        self.seed = seed

    # -- internal --------------------------------------------------------
    def _one(self, bars, episode: EpisodePlan):
        plan = ScenarioPlan(
            seed=self.seed,
            placement=getattr(self.config, "placement", "uniform"),
            episodes=(episode,),
        )
        return apply_plan(bars, plan, self.config)

    def plan(self, start_index: int, end_index: int, entry_indices: Iterable[int] = ()) -> ScenarioPlan:
        return plan_episodes(self.config, start_index, end_index, entry_indices)

    # -- single-episode API ----------------------------------------------
    def make_identity(self, bars, start_bar: int = 0):
        """Negative control: an unchanged copy plus one control label."""
        n = len(bars)
        duration = max(1, n - start_bar)
        return self._one(
            bars,
            EpisodePlan("identity_000", "identity", start_bar, duration, 0.0),
        )

    def make_missing_quote(self, bars, start_bar: int, duration_bars: int, side: str = "bid"):
        return self._one(
            bars,
            EpisodePlan(
                "missing_quote_000", "missing_quote", start_bar,
                duration_bars, 1.0, side=side,
            ),
        )

    def make_stale_replay(self, bars, start_bar: int, duration_bars: int):
        return self._one(
            bars,
            EpisodePlan(
                "stale_replay_000", "stale_replay", start_bar,
                duration_bars, 1.0,
            ),
        )

    def make_crossed_quote(self, bars, start_bar: int, duration_bars: int):
        return self._one(
            bars,
            EpisodePlan(
                "crossed_quote_000", "crossed_quote", start_bar,
                duration_bars, 1.0,
            ),
        )

    def make_symmetric_shift(
        self, bars, start_bar: int, duration_bars: int, magnitude: float
    ):
        return self._one(
            bars,
            EpisodePlan(
                "symmetric_shift_000", "symmetric_shift", start_bar,
                duration_bars, magnitude,
            ),
        )

    def make_return_scale_shift(
        self, bars, start_bar: int, duration_bars: int, scale_factor: float
    ):
        return self._one(
            bars,
            EpisodePlan(
                "return_scale_shift_000", "return_scale_shift", start_bar,
                duration_bars, scale_factor,
            ),
        )

    def make_benign_volatility(
        self, bars, start_bar: int, duration_bars: int, multiplier: float
    ):
        return self._one(
            bars,
            EpisodePlan(
                "benign_volatility_000", "benign_volatility", start_bar,
                duration_bars, multiplier,
            ),
        )

    def make_combined(
        self, bars, start_bar: int, duration_bars: int, magnitude: float,
        side: str = "bid",
    ):
        return self._one(
            bars,
            EpisodePlan(
                "combined_shift_and_quote_000", "combined_shift_and_quote",
                start_bar, duration_bars, magnitude, side=side,
            ),
        )


__all__ = [
    "FAULT_CLASSES",
    "EpisodePlan",
    "SCENARIO_CLASSES",
    "ScenarioGenerator",
    "ScenarioLabel",
    "ScenarioManifest",
    "ScenarioPlan",
    "apply_plan",
    "plan_episodes",
]
