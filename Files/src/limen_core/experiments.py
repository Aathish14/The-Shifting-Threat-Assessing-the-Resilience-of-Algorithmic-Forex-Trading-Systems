"""Shared harness for the ablation and sensitivity experiments.

Both experiment families ask "how do the headline metrics move when exactly one
declared thing changes?". They therefore need the *replay* semantics -- the same
calibration rules, the same scenario plan, the same leak assertion, the same
policy arms -- applied to a variant config, without paying for a second ingest
of the 22M-row source file.

Two entry points:

``prepare(config)``
    Ingests, splits and builds the intent stream. Depends only on the runtime,
    bar and ingest-order parameters, so it is cached by
    :func:`ingest_signature`.

``evaluate(config, prepared)``
    Calibrates from train/validation only, plans and applies the scenario, runs
    the clean and mutated passes, and returns everything the metrics need.

Calibration discipline
----------------------
No function here fits a threshold on test data. Integrity limits are fitted on
``bars[:train_end]`` and the KS threshold on the causal validation scores,
exactly as in :meth:`ReplayEngine.run`. ``evaluate`` re-runs that calibration
for every variant, so a swept threshold is always re-fitted on train/validation
rather than inherited from the base run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import timestamp as ts_mod
from .audit import append_log
from .bars import build_bars
from .config import LimenConfig
from .drift import KSShiftDetector
from .ingest import SourceAudit, iter_ticks
from .integrity import IntegrityMonitor
from .policy import INTENT_ENTRY, PolicyArmResult, build_arm
from .replay import (
    PASS_THROUGH_CLASSES,
    EpisodeLayout,
    ReplayEngine,
    _PassResult,
    assert_mutation_contained,
    injected_bars,
    plan_matched_arm,
    plan_test_episodes,
    unmodified_positions,
)
from .strategy import ReferenceStrategy


@dataclass
class PreparedData:
    """Ingest-derived inputs shared by every variant of one experiment family."""

    bars: list
    intents: list
    entry_flags: list[bool]
    audit: SourceAudit
    train_end: int
    val_end: int
    test_end: int
    ingest_signature: tuple

    @property
    def n_bars(self) -> int:
        return len(self.bars)


@dataclass
class Evaluation:
    """One full evaluation of one config against a prepared data set."""

    clean: _PassResult
    mutated: _PassResult
    results: dict[str, PolicyArmResult]
    matched_rate: dict[str, Any]
    thresholds: dict[str, Any]
    calibrated: bool
    layout: EpisodeLayout
    injected: set[int]
    unmodified: set[int]
    entry_positions: set[int]
    fault_positions: set[int]

    def arm_rows(self, engine: ReplayEngine, timestamp: str) -> list[dict[str, Any]]:
        return engine._arm_rows(
            self.results,
            self.entry_positions,
            self.fault_positions,
            self.unmodified,
            self.mutated.results,
            self.layout.pos_by_bar_id,
            timestamp,
        )


def ingest_signature(config: LimenConfig) -> tuple:
    """Parameters that change the bar series and therefore require a re-ingest.

    Everything else (integrity thresholds, KS windows, scenario shape, policy)
    acts on an already built bar series and can be swept without re-ingesting.
    """
    return (
        config.runtime.input_path,
        config.runtime.max_rows,
        config.bar.gap_seconds,
        bool(config.integrity.enable_stale_quote),
        config.integrity.stale_quote_seconds,
        bool(config.integrity.enable_timestamp_order),
        config.integrity.min_gap_seconds,
    )


def prepare(config: LimenConfig, log_path=None) -> PreparedData:
    """Ingest, split, and build the reference intent stream."""
    cfg = config
    cfg.validate_for_run()
    stream = iter_ticks(
        cfg.runtime.input_path,
        max_rows=cfg.runtime.max_rows,
        track_duplicates=False,
        stale_threshold_seconds=(
            cfg.integrity.stale_quote_seconds
            if cfg.integrity.enable_stale_quote
            else None
        ),
        enable_timestamp_order=cfg.integrity.enable_timestamp_order,
        gap_reset_seconds=cfg.integrity.min_gap_seconds,
    )
    series = build_bars(stream, gap_seconds=cfg.bar.gap_seconds)
    audit = stream.finalize()
    audit.timestamp = ts_mod.utc_now_iso()
    bars = series.bars

    split = cfg.split
    train_end = min(split.train_end_index, len(bars))
    val_end = min(split.validation_end_index, len(bars))
    test_end = (
        len(bars) if split.test_end_index is None else min(split.test_end_index, len(bars))
    )
    if not (0 <= train_end <= val_end <= test_end <= len(bars)):
        raise ValueError(
            f"invalid split after clamping: train_end={train_end} "
            f"val_end={val_end} test_end={test_end} n_bars={len(bars)}"
        )

    strategy = ReferenceStrategy(cfg.strategy)
    for bar in bars:
        strategy.update(bar)
    intents = strategy.stream.intents
    entry_flags = [bool(i.eligible and i.intent in INTENT_ENTRY) for i in intents]

    if log_path is not None:
        append_log(
            log_path,
            f"event=prepare status=ok rows={audit.total_rows} bars={len(bars)} "
            f"train=[0:{train_end}] val=[{train_end}:{val_end}] "
            f"test=[{val_end}:{test_end}]",
        )
    return PreparedData(
        bars=bars,
        intents=intents,
        entry_flags=entry_flags,
        audit=audit,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        ingest_signature=ingest_signature(cfg),
    )


def evaluate(
    config: LimenConfig,
    prepared: PreparedData,
    log_path=None,
    collect_rows: bool = True,
) -> Evaluation:
    """Calibrate on train/validation, then run the clean and mutated passes."""
    cfg = config
    engine = ReplayEngine(cfg, "<experiment>")
    bars = prepared.bars
    test_start, test_end = prepared.val_end, prepared.test_end

    # --- calibration (train for limits, validation for the KS threshold) ---
    integrity = IntegrityMonitor(cfg.integrity)
    spread_threshold, innovation_threshold = engine._fit_integrity_limits(
        bars[: prepared.train_end]
    )
    integrity.set_training_limits(spread_threshold, innovation_threshold)
    ks = KSShiftDetector(cfg.ks)
    ks_threshold, ks_scores, prior_mid = engine._warm_and_calibrate_ks(
        bars, prepared.train_end, prepared.val_end, integrity, ks, calibrate=True
    )
    calibrated = (
        spread_threshold is not None
        and innovation_threshold is not None
        and ks_threshold is not None
    )

    # --- scenario plan over the test range (same planner as the replay) ---
    layout = plan_test_episodes(
        cfg, bars, test_start, test_end, prepared.entry_flags, "<experiment>"
    )

    # --- clean pass --------------------------------------------------------
    arms = {
        name: build_arm(name, cfg.policy)
        for name in cfg.policy.arms
        if name != "matched_rate_abstention"
    }
    clean = engine._signal_pass(
        bars, prepared.intents, test_start, test_end,
        integrity, ks, arms, prior_mid,
        collect_rows=collect_rows,
    )
    results, matched_rate = plan_matched_arm(
        cfg, clean.results, clean.rows, engine, log_path
    )

    # --- mutated pass (same monitors, frozen thresholds) -------------------
    scenario_integrity = IntegrityMonitor(cfg.integrity)
    scenario_integrity.set_training_limits(spread_threshold, innovation_threshold)
    scenario_ks = KSShiftDetector(cfg.ks)
    scenario_ks.set_threshold(ks_threshold)
    scenario_ks.clear_alarm_streak()
    scenario_arms = {name: build_arm(name, cfg.policy) for name in cfg.policy.arms}
    if "matched_rate_abstention" in scenario_arms:
        scenario_arms["matched_rate_abstention"].set_plan(
            [d.bar_id for d in (results.get("fused").decisions
                                if results.get("fused") else [])],
            matched_rate.get("target_rate", 0.0),
        )
    scenario_prior = engine._warm_and_calibrate_ks(
        layout.mutated_all, prepared.train_end, test_start,
        scenario_integrity, scenario_ks, calibrate=False,
    )[2]
    mutated = engine._signal_pass(
        layout.mutated_all, prepared.intents, test_start, test_end,
        scenario_integrity, scenario_ks, scenario_arms, scenario_prior,
        collect_rows=False,
    )

    # --- containment and unmodified set -----------------------------------
    assert_mutation_contained(
        layout.clean_test, layout.mutated_test, layout.episode_positions
    )
    injected = injected_bars(layout.episode_positions)
    unmodified = unmodified_positions(clean, mutated, injected)

    entry_positions = {
        i for i in range(len(layout.clean_test)) if prepared.entry_flags[test_start + i]
    }
    fault_positions: set[int] = set()
    for label in layout.labels:
        if label.scenario_class not in PASS_THROUGH_CLASSES:
            continue
        onset = layout.pos_by_bar_id.get(label.onset_bar_id)
        offset = layout.pos_by_bar_id.get(label.offset_bar_id)
        if onset is None or offset is None:
            continue
        fault_positions.update(range(onset, offset + 1))

    return Evaluation(
        clean=clean,
        mutated=mutated,
        results=results,
        matched_rate=matched_rate,
        thresholds={
            "spread": spread_threshold,
            "innovation": innovation_threshold,
            "ks": ks_threshold,
            "ks_scores_used": ks_scores,
        },
        calibrated=calibrated,
        layout=layout,
        injected=injected,
        unmodified=unmodified,
        entry_positions=entry_positions,
        fault_positions=fault_positions,
    )


__all__ = [
    "Evaluation",
    "PreparedData",
    "evaluate",
    "ingest_signature",
    "prepare",
]
