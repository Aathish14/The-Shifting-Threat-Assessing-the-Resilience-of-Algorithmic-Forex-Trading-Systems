"""Replay and experiment orchestration.

Ties ingestion, quote reconstruction, bar construction, causal strategy,
monitors, scenarios, policy arms, and metrics into one deterministic replay.
All policy arms receive the identical intent stream and aligned observation IDs.
Evaluator labels stay in a separate sidecar and never reach a runtime component.

What changed from v1.0.x (each item is proved by a test):
  * Bars are built directly from the streaming tick iterator, so the full tick
    list never materialises (22M rows for the frozen input).
  * Monitor limits come from the training range only. The KS threshold is the
    ``ks.quantile`` of the *causal KS statistics observed on the validation
    range* (same windows, cadence, and consecutive-alert rule), not the 99.9th
    percentile of raw absolute returns. The old value was ~2e-4 while every real
    KS statistic is >= 0.02, so 100% of checks alarmed.
  * ``test`` entry opportunities are ``eligible AND intent in {ENTER_LONG,
    ENTER_SHORT}``. v1.0.x counted every warm bar, inflating the denominator.
  * Scenario episodes are planned over the test range and applied to a copy of
    the test bars. The *clean* intent stream is used for the mutated pass so an
    injected quote fault cannot change which entries are opportunities; that is
    the controlled counterfactual the pass-through metric needs.
  * The matched-rate arm is planned after the fused arm's realized abstention
    rate is known (post hoc comparator, labelled as such in the summary).
  * ``config_digest`` has one definition (``audit.config_digest``).
  * The execution log is appended as events happen, so the final log line is
    written before ``artifact_manifest.json`` is generated.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import timestamp as ts_mod
from .audit import (
    append_log,
    atomic_write_json,
    build_artifact_manifest,
    build_frozen_config,
    build_run_manifest,
    config_digest,
)
from .bars import build_bars
from .config import LimenConfig
from .drift import KSShiftDetector
from .ingest import iter_ticks
from .integrity import IntegrityMonitor
from .metrics import MetricRecord, Metrics
from .policy import (
    BLOCKING_RESPONSES,
    INTENT_ENTRY,
    PolicyArmResult,
    RESP_ALLOW,
    build_arm,
)
from .scenarios import ScenarioManifest, apply_plan, plan_episodes
from .strategy import ReferenceStrategy


# Classes whose bars count as "faulted" for pass-through purposes. A benign
# volatility episode is a control, not a fault, so it is excluded.
PASS_THROUGH_CLASSES: frozenset[str] = frozenset({
    "missing_quote",
    "stale_replay",
    "crossed_quote",
    "symmetric_shift",
    "return_scale_shift",
    "combined_shift_and_quote",
})

_BAR_SIGNAL_FIELDS: tuple[str, ...] = (
    "bar_id",
    "timestamp",
    "minute_start",
    "intent",
    "intent_eligible",
    "is_entry",
    "ema_fast",
    "ema_slow",
    "midpoint_close",
    "spread_close",
    "integrity_hard_invalid",
    "integrity_hard_reasons",
    "integrity_soft_reasons",
    "integrity_eligible",
    "shift_alarm",
    "shift_checked",
    "shift_score",
    "log_return",
    "elapsed_seconds",
    "gap_indicator",
    "tick_count",
    "bid_age_seconds_close",
    "ask_age_seconds_close",
    "bid_update_count",
    "ask_update_count",
)


@dataclass
class _FlagShim:
    """Minimal integrity-surface used when replaying a stored decision."""

    hard_invalid: bool
    has_soft_reason: bool


@dataclass
class _PassResult:
    rows: list[dict[str, Any]]
    results: dict[str, PolicyArmResult]
    hard: list[bool]
    soft: list[bool]
    shift: list[bool]


def _abs_return(
    midpoint: float | None, prior: float | None
) -> tuple[float | None, float | None]:
    """Absolute log return against the last observed midpoint (None if unknown)."""
    ret: float | None = None
    if (
        midpoint is not None
        and prior is not None
        and midpoint > 0.0
        and prior > 0.0
    ):
        ret = abs(math.log(midpoint / prior))
    return ret, (midpoint if midpoint is not None else prior)


def _quantile(values: Sequence[float], q: float) -> float | None:
    """Empirical quantile using the same ceil-index rule as v1.0.x."""
    if not values:
        return None
    ordered = sorted(values)
    idx = int(math.ceil(q * len(ordered))) - 1
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


def _integrity_signature(bar, prev) -> tuple:
    """The exact input the integrity monitor consumes, made scale-invariant.

    Used by the leak assertion: outside its own episode, a mutation must leave
    every one of these values identical to the clean series. Prices appear only
    through the predicates the monitor evaluates (missing / nonpositive /
    crossed / repeated-for-the-first-time), so ``return_scale_shift``'s
    carried-forward level ratio -- which preserves every return and every
    equality -- is not counted as a leak, while an actual write outside the
    episode is.
    """
    bid = bar.bid_close
    ask = bar.ask_close
    prev_bid = prev.bid_close if prev is not None else None
    prev_ask = prev.ask_close if prev is not None else None
    both = bid is not None and ask is not None
    prev_both = prev_bid is not None and prev_ask is not None
    return (
        bid is None,
        ask is None,
        (bid <= 0.0) if bid is not None else None,
        (ask <= 0.0) if ask is not None else None,
        (bid > ask) if both else None,
        bar.spread_close,
        bar.log_return,
        getattr(bar, "bid_age_seconds_close", None),
        getattr(bar, "ask_age_seconds_close", None),
        getattr(bar, "bid_update_count", None),
        getattr(bar, "ask_update_count", None),
        (bid == prev_bid and ask == prev_ask) if (both and prev_both) else None,
        getattr(bar, "elapsed_seconds_since_prior", None),
        getattr(bar, "gap_indicator", None),
    )


def _manifest_digest(manifest: ScenarioManifest) -> str:
    import hashlib

    h = hashlib.sha256()
    for ep in sorted(manifest.episodes, key=lambda e: e.episode_id):
        h.update(
            (
                f"{ep.episode_id}|{ep.scenario_class}|{ep.onset_bar_id}|"
                f"{ep.offset_bar_id}|{ep.severity}|{ep.label_digest}"
            ).encode("utf-8")
        )
    return h.hexdigest()


@dataclass
class EpisodeLayout:
    """Result of planning and applying the scenario plan to the test range.

    Shared by the replay and by the ablation/sensitivity experiments so that
    episode placement, the containment assertion and the unmodified-set
    definition have exactly one implementation.
    """

    clean_test: list
    mutated_test: list
    mutated_all: list
    plan: Any
    labels: list
    manifest: ScenarioManifest
    pos_by_bar_id: dict[int, int]
    episode_positions: dict[int, tuple[int, int]]

    @property
    def injected(self) -> set[int]:
        return injected_bars(self.episode_positions)


def plan_test_episodes(
    cfg, bars: Sequence, test_start: int, test_end: int,
    entry_flags: Sequence[bool], run_id: str,
) -> EpisodeLayout:
    """Plan episodes over ``[test_start, test_end)`` and apply them to a copy."""
    clean_test = list(bars[test_start:test_end])
    absolute_entries = [i for i in range(test_start, test_end) if entry_flags[i]]
    plan = plan_episodes(
        cfg.scenario, test_start, test_end, entry_indices=absolute_entries
    )
    mutated_test, labels = apply_plan(clean_test, plan, cfg.scenario)
    if not labels:
        mutated_test = list(clean_test)
    mutated_all = list(bars[:test_start]) + mutated_test
    manifest = ScenarioManifest(
        episodes=labels, run_id=run_id, seed=cfg.scenario.seed
    )
    pos_by_bar_id = {bar.bar_id: i for i, bar in enumerate(clean_test)}
    episode_positions: dict[int, tuple[int, int]] = {}
    for label in labels:
        onset = pos_by_bar_id.get(label.onset_bar_id)
        offset = pos_by_bar_id.get(label.offset_bar_id)
        if onset is None or offset is None:
            continue
        episode_positions[onset] = (onset, offset)
    return EpisodeLayout(
        clean_test=clean_test,
        mutated_test=mutated_test,
        mutated_all=mutated_all,
        plan=plan,
        labels=labels,
        manifest=manifest,
        pos_by_bar_id=pos_by_bar_id,
        episode_positions=episode_positions,
    )


def injected_bars(episode_positions: dict[int, tuple[int, int]]) -> set[int]:
    """Every test position inside a declared episode interval."""
    out: set[int] = set()
    for start, end in episode_positions.values():
        out.update(range(start, end + 1))
    return out


def assert_mutation_contained(
    clean_test: Sequence, mutated_test: Sequence,
    episode_positions: dict[int, tuple[int, int]],
) -> None:
    """Fail if a mutation changed a bar outside its own episode.

    The bar immediately after an episode is exempt: its return is recomputed
    against the mutated final midpoint, which is the declared recovery
    behaviour. Every other position must present the identical monitor inputs.
    """
    injected = injected_bars(episode_positions)
    boundary = {end + 1 for _, end in episode_positions.values()}
    for i in range(len(clean_test)):
        if i in injected or i in boundary:
            continue
        prev_clean = clean_test[i - 1] if i > 0 else None
        prev_mutated = mutated_test[i - 1] if i > 0 else None
        if _integrity_signature(clean_test[i], prev_clean) != _integrity_signature(
            mutated_test[i], prev_mutated
        ):
            raise AssertionError(
                f"mutation leaked outside its episode at test bar {i}: "
                f"bar_id={clean_test[i].bar_id} "
                f"episodes={len(episode_positions)}"
            )


def unmodified_positions(clean: "_PassResult", mutated: "_PassResult",
                         injected: set[int]) -> set[int]:
    """Positions where the mutated pass saw exactly the clean monitor evidence.

    Equality of ``(hard, soft, shift_alarm)`` is what makes the policy decision
    necessarily identical, so the clean-intervention rate on this set is a
    like-for-like measurement.
    """
    return {
        i
        for i in range(len(clean.hard))
        if i not in injected
        and (clean.hard[i], clean.soft[i], clean.shift[i])
        == (mutated.hard[i], mutated.soft[i], mutated.shift[i])
    }


def plan_matched_arm(cfg, results: dict, rows: Sequence[dict[str, Any]],
                     engine: "ReplayEngine", log_path: Path | None = None):
    """Plan the post-hoc rate-matched comparator from the fused arm.

    Shared by the replay and the experiment runners so the comparator is
    defined once: the target rate is the fused arm's *realized* abstention rate
    on the same test opportunities, and abstentions are drawn from the same
    ordered opportunity list by cumulative rounding.
    """
    results = dict(results)
    info: dict[str, Any] = {}
    if "matched_rate_abstention" not in cfg.policy.arms:
        return results, info
    fused = results.get("fused")
    if fused is None or not fused.decisions:
        rate, blocked, opportunities = 0.0, 0, 0
    else:
        opportunities = len(fused.decisions)
        blocked = fused.blocked_entry_opportunities()
        rate = blocked / opportunities if opportunities else 0.0
    matched = build_arm("matched_rate_abstention", cfg.policy)
    matched.set_plan(
        [d.bar_id for d in (fused.decisions if fused else [])], rate
    )
    results["matched_rate_abstention"] = engine._matched_pass(rows, matched)
    info = {
        "source_arm": "fused",
        "target_rate": rate,
        "planned_abstentions": matched.planned_abstentions,
        "realized_rate": (
            results["matched_rate_abstention"].interventions()
            / len(results["matched_rate_abstention"].decisions)
            if results["matched_rate_abstention"].decisions
            else None
        ),
        "entry_opportunities": opportunities,
        "fused_blocked": blocked,
        "note": (
            "post hoc comparator: the rate is measured from the fused "
            "arm on the same test opportunities; abstentions are "
            "time-stratified and seeded"
        ),
    }
    if log_path is not None:
        append_log(
            log_path,
            f"event=matched_rate status=ok target={rate:.6f} "
            f"planned={matched.planned_abstentions} "
            f"opportunities={opportunities}",
        )
    return results, info


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class ReplayEngine:
    """Orchestrates one deterministic replay run."""

    def __init__(
        self,
        config: LimenConfig,
        run_id: str,
        config_path: str | None = None,
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.config_path = config_path or "Files/configs/<not-recorded>.json"

    # -- public API ------------------------------------------------------
    def run(self, output_dir: str | Path) -> dict[str, Any]:
        """Execute the full replay, write artifacts, return the run manifest."""
        output_dir = Path(output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty run directory: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "execution.log"

        cfg = self.config
        cfg.validate_for_run()
        append_log(log_path, f"event=init status=started run_id={self.run_id}")

        # --- 1. Ingest + reconstruct + build bars in one streaming pass ---
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
            stale_is_hard=cfg.integrity.stale_is_hard,
        )
        bar_series = build_bars(stream, gap_seconds=cfg.bar.gap_seconds)
        audit = stream.finalize()
        audit.timestamp = ts_mod.utc_now_iso()
        atomic_write_json(output_dir / "data_audit.json", audit.to_dict())
        append_log(
            log_path,
            f"event=ingest status=ok rows={audit.total_rows} "
            f"parse_failures={audit.parse_failures} bars={len(bar_series)}",
        )
        bars = bar_series.bars
        n_bars = len(bars)

        # --- 2. Split ------------------------------------------------------
        split = cfg.split
        train_end = min(split.train_end_index, n_bars)
        val_end = min(split.validation_end_index, n_bars)
        test_end = (
            n_bars if split.test_end_index is None else min(split.test_end_index, n_bars)
        )
        if not (0 <= train_end <= val_end <= test_end <= n_bars):
            raise ValueError(
                f"invalid split after clamping: train_end={train_end} "
                f"val_end={val_end} test_end={test_end} n_bars={n_bars}"
            )
        append_log(
            log_path,
            f"event=split status=ok train=[0:{train_end}] "
            f"val=[{train_end}:{val_end}] test=[{val_end}:{test_end}]",
        )

        # --- 3. Reference intent stream (clean, shared by every arm) ------
        strategy = ReferenceStrategy(cfg.strategy)
        for bar in bars:
            strategy.update(bar)
        intents = strategy.stream.intents
        entry_flags = [
            bool(i.eligible and i.intent in INTENT_ENTRY) for i in intents
        ]
        append_log(
            log_path,
            f"event=intents status=ok total={len(intents)} "
            f"entry_train={sum(entry_flags[:train_end])} "
            f"entry_val={sum(entry_flags[train_end:val_end])} "
            f"entry_test={sum(entry_flags[val_end:test_end])}",
        )

        # --- 4. Fit integrity limits on training bars only ----------------
        integrity = IntegrityMonitor(cfg.integrity)
        spread_threshold, innovation_threshold = self._fit_integrity_limits(
            bars[:train_end]
        )
        integrity.set_training_limits(spread_threshold, innovation_threshold)

        # --- 5. KS monitor: causal scores on validation set the threshold --
        ks = KSShiftDetector(cfg.ks)
        ks_threshold, ks_scores_used, prior_mid = self._warm_and_calibrate_ks(
            bars, train_end, val_end, integrity, ks, calibrate=True
        )
        calibrated = (
            spread_threshold is not None
            and innovation_threshold is not None
            and ks_threshold is not None
        )
        append_log(
            log_path,
            f"event=calibration status=ok calibrated={calibrated} "
            f"spread_q={spread_threshold} innovation_q={innovation_threshold} "
            f"ks_q={ks_threshold} ks_scores={ks_scores_used}",
        )

        # --- 6. Scenario plan over the test range --------------------------
        test_start = val_end
        layout = plan_test_episodes(
            cfg, bars, test_start, test_end, entry_flags, self.run_id
        )
        clean_test = layout.clean_test
        mutated_test = layout.mutated_test
        mutated_all = layout.mutated_all
        plan = layout.plan
        labels = layout.labels
        manifest = layout.manifest
        pos_by_bar_id = layout.pos_by_bar_id
        episode_positions = layout.episode_positions
        append_log(
            log_path,
            f"event=scenario status=ok episodes={len(labels)} "
            f"by_class={plan.by_class()}",
        )

        # --- 7. Clean pass: monitors + policy on unmodified test bars ------
        arm_names = list(cfg.policy.arms)
        post_hoc = "matched_rate_abstention"
        primary = [a for a in arm_names if a != post_hoc]
        arms = {name: build_arm(name, cfg.policy) for name in primary}
        clean = self._signal_pass(
            bars, intents, test_start, test_end,
            integrity, ks, arms, prior_mid,
            collect_rows=True,
        )
        append_log(
            log_path,
            f"event=clean_pass status=ok rows={len(clean.rows)} "
            f"entry_test={len(clean.results[primary[0]].decisions) if primary else 0}",
        )

        # --- 8. Matched-rate comparator, planned post hoc from the fused arm
        results, matched_rate_info = plan_matched_arm(
            cfg, clean.results, clean.rows, self, log_path
        )

        # --- 9. Scenario pass: same monitors on the mutated test bars ------
        scenario_integrity = IntegrityMonitor(cfg.integrity)
        scenario_integrity.set_training_limits(spread_threshold, innovation_threshold)
        scenario_ks = KSShiftDetector(cfg.ks)
        scenario_ks.set_threshold(ks_threshold)
        scenario_ks.clear_alarm_streak()
        scenario_arms = {name: build_arm(name, cfg.policy) for name in arm_names}
        if post_hoc in scenario_arms:
            scenario_arms[post_hoc].set_plan(
                [d.bar_id for d in (results.get("fused").decisions if results.get("fused") else [])],
                matched_rate_info.get("target_rate", 0.0),
            )
        scenario_prior = self._warm_and_calibrate_ks(
            mutated_all, train_end, test_start,
            scenario_integrity, scenario_ks, calibrate=False,
        )[2]
        mutated = self._signal_pass(
            mutated_all, intents, test_start, test_end,
            scenario_integrity, scenario_ks, scenario_arms, scenario_prior,
            collect_rows=False,
        )
        append_log(
            log_path,
            f"event=scenario_pass status=ok episodes={len(labels)} "
            f"episode_bars={sum(e - s + 1 for s, e in episode_positions.values())}",
        )

        # --- 10. Unmodified set, leak check, and metrics -------------------
        injected = injected_bars(episode_positions)
        assert_mutation_contained(clean_test, mutated_test, episode_positions)
        unmodified = unmodified_positions(clean, mutated, injected)

        metrics = Metrics()
        ts_now = ts_mod.utc_now_iso()
        fault_positions: set[int] = set()
        for label in labels:
            if label.scenario_class not in PASS_THROUGH_CLASSES:
                continue
            onset = pos_by_bar_id.get(label.onset_bar_id)
            offset = pos_by_bar_id.get(label.offset_bar_id)
            if onset is None or offset is None:
                continue
            fault_positions.update(range(onset, offset + 1))
        entry_positions = {
            i for i in range(len(clean_test)) if entry_flags[test_start + i]
        }
        self._arm_metrics(
            metrics, results, entry_positions, fault_positions,
            unmodified, len(clean_test), ts_now,
        )
        self._detection_metrics(
            metrics, labels, clean_test, pos_by_bar_id,
            mutated.hard, mutated.soft, mutated.shift,
            clean.hard, clean.soft, fault_positions,
            cfg.scenario.recovery_window_bars, ts_now,
        )

        # --- 11. Write artifacts ------------------------------------------
        arms_in_order = [a for a in arm_names]
        fields = list(_BAR_SIGNAL_FIELDS)
        for name in arms_in_order:
            fields.extend([f"arm_{name}_response", f"arm_{name}_multiplier"])
        _write_csv(output_dir / "bar_signals.csv", clean.rows, fields)

        arm_rows = self._arm_rows(
            results, entry_positions, fault_positions, unmodified,
            mutated.results, pos_by_bar_id, ts_now,
        )
        _write_csv(
            output_dir / "episode_metrics.csv",
            arm_rows,
            [
                "arm", "entry_opportunities", "allowed_entries",
                "blocked_eligible_entries", "cautioned_entries",
                "abstention_rate", "faulted_entry_opportunities",
                "faulted_allowed", "fault_pass_through",
                "unmodified_entry_opportunities", "unmodified_blocked",
                "unmodified_abstention_rate", "clean_intervention_rate",
                "timestamp",
            ],
        )

        scenario_rows = self._scenario_rows(
            metrics, labels, pos_by_bar_id, mutated.hard, mutated.soft,
            mutated.shift, cfg.scenario.recovery_window_bars, ts_now,
        )
        _write_csv(
            output_dir / "scenario_metrics.csv",
            scenario_rows,
            [
                "scenario_class", "episodes", "detected_integrity",
                "detected_shift", "detected_either", "episode_detection_recall",
                "shift_detection_recall", "fault_bars", "fault_bars_flagged",
                "fault_bar_flag_rate", "mean_detection_delay_bars",
                "missed_episodes", "timestamp",
            ],
        )

        scenario_csv: list[dict[str, Any]] = []
        for row in manifest.to_csv_rows():
            row = dict(row)
            row["timestamp"] = ts_now
            scenario_csv.append(row)
        _write_csv(
            output_dir / "scenario_manifest.csv",
            scenario_csv,
            [
                "episode_id", "scenario_class", "onset_bar_id", "offset_bar_id",
                "duration_bars", "severity", "affected_fields", "seed",
                "recovery_behavior", "label_digest", "timestamp",
            ],
        )

        summary = {
            "run_id": self.run_id,
            "timestamp": ts_now,
            "status": "completed",
            "config_version": cfg.version,
            "config_digest": config_digest(cfg),
            "calibrated": bool(calibrated),
            "input_sha256": audit.sha256,
            "input_byte_size": audit.byte_size,
            "total_source_rows": audit.total_rows,
            "parse_failures": audit.parse_failures,
            "total_observed_bars": n_bars,
            "train_bars": train_end,
            "validation_bars": val_end - train_end,
            "test_bars": test_end - val_end,
            "entry_opportunities": {
                "train": sum(entry_flags[:train_end]),
                "validation": sum(entry_flags[train_end:val_end]),
                "test": sum(entry_flags[val_end:test_end]),
            },
            "thresholds": {
                "spread": spread_threshold,
                "innovation": innovation_threshold,
                "ks": ks_threshold,
                "ks_scores_used": ks_scores_used,
                "ks_rule": (
                    "quantile of causal KS statistics observed on the validation "
                    "range, using the same reference/recent windows, check "
                    "cadence, and consecutive-alert rule as evaluation"
                ),
            },
            "arms": arms_in_order,
            "matched_rate": matched_rate_info,
            "scenario": {
                "seed": cfg.scenario.seed,
                "placement": cfg.scenario.placement,
                "episodes": len(labels),
                "by_class": plan.by_class(),
                "episode_bars": len(injected),
                "unmodified_test_bars": len(unmodified),
                "labels_digest": _manifest_digest(manifest),
                "note": (
                    "intents for the mutated pass come from the clean bar series; "
                    "this is the controlled counterfactual used by the "
                    "pass-through metric"
                ),
            },
            "metrics": metrics.all(),
            "source_audit": audit.to_dict(),
        }
        atomic_write_json(output_dir / "summary.json", summary)

        frozen = build_frozen_config(cfg, self.config_path)
        atomic_write_json(output_dir / "frozen_config.json", frozen)

        command = (
            f"python Files/run_limen.py run --config {self.config_path} "
            f"--run-id {self.run_id}"
        )
        manifest_dict = build_run_manifest(
            run_id=self.run_id,
            command=command,
            input_path=cfg.runtime.input_path,
            input_digest=audit.sha256,
            input_size=audit.byte_size,
            config_obj=cfg,
            status="completed",
            output_dir=output_dir,
            seed=cfg.scenario.seed,
        )
        atomic_write_json(output_dir / "run_manifest.json", manifest_dict)

        (output_dir / "README.md").write_text(
            self._readme(output_dir, audit, command), encoding="utf-8"
        )

        # Final log line BEFORE the artifact manifest so the manifest covers it.
        append_log(
            log_path,
            f"event=complete status=ok run_id={self.run_id} dir={output_dir}",
        )
        artifact_manifest = build_artifact_manifest(output_dir)
        atomic_write_json(output_dir / "artifact_manifest.json", artifact_manifest)
        return manifest_dict

    # -- calibration -----------------------------------------------------
    def _fit_integrity_limits(self, train_bars: Sequence) -> tuple[float | None, float | None]:
        spreads = [b.spread_close for b in train_bars if b.spread_close is not None]
        innovations: list[float] = []
        prior_mid: float | None = None
        for bar in train_bars:
            if (
                bar.midpoint_close is not None
                and prior_mid is not None
                and prior_mid > 0.0
                and bar.midpoint_close > 0.0
            ):
                innovations.append(abs(math.log(bar.midpoint_close / prior_mid)))
            if bar.midpoint_close is not None:
                prior_mid = bar.midpoint_close
        return (
            _quantile(spreads, self.config.integrity.spread_quantile),
            _quantile(innovations, self.config.integrity.innovation_quantile),
        )

    def _warm_and_calibrate_ks(
        self,
        bars: Sequence,
        train_end: int,
        val_end: int,
        integrity: IntegrityMonitor,
        detector: KSShiftDetector,
        calibrate: bool,
    ) -> tuple[float | None, int, float | None]:
        """Warm both monitors over the pre-test prefix.

        With ``calibrate=True`` the KS threshold is the configured quantile of
        the causal statistics observed inside the validation range; the same
        detector then continues into test with the threshold frozen, so the
        reference window at the boundary is legitimate causal history.
        """
        scores: list[float] = []
        prior_mid: float | None = None
        for i in range(val_end):
            bar = bars[i]
            ret, prior_mid = _abs_return(bar.midpoint_close, prior_mid)
            result = detector.update(ret, bar.bar_id, bar.last_event_timestamp)
            integrity.assess(bar)
            if calibrate and result.checked and i >= train_end and result.score is not None:
                scores.append(result.score)
        threshold: float | None = None
        if calibrate:
            threshold = _quantile(scores, detector.quantile)
            detector.set_threshold(threshold)  # type: ignore[arg-type]
            detector.clear_alarm_streak()
        return threshold, len(scores), prior_mid

    # -- passes ----------------------------------------------------------
    def _signal_pass(
        self,
        bars: Sequence,
        intents: Sequence,
        start: int,
        end: int,
        integrity: IntegrityMonitor,
        detector: KSShiftDetector,
        arms: dict[str, Any],
        prior_mid: float | None,
        collect_rows: bool,
    ) -> _PassResult:
        rows: list[dict[str, Any]] = []
        results = {name: PolicyArmResult(arm=name) for name in arms}
        hard: list[bool] = []
        soft: list[bool] = []
        shift: list[bool] = []

        for i in range(start, end):
            bar = bars[i]
            ret, prior_mid = _abs_return(bar.midpoint_close, prior_mid)
            shift_result = detector.update(ret, bar.bar_id, bar.last_event_timestamp)
            assessment = integrity.assess(bar)
            intent = intents[i]
            is_entry = bool(intent.eligible and intent.intent in INTENT_ENTRY)
            alarm = shift_result.is_alarm()

            hard.append(assessment.hard_invalid)
            soft.append(assessment.has_soft_reason)
            shift.append(alarm)

            row: dict[str, Any] = {}
            if collect_rows:
                row = {
                    "bar_id": bar.bar_id,
                    "timestamp": bar.last_event_timestamp,
                    "minute_start": bar.minute_start,
                    "intent": intent.intent,
                    "intent_eligible": intent.eligible,
                    "is_entry": is_entry,
                    "ema_fast": intent.ema_fast,
                    "ema_slow": intent.ema_slow,
                    "midpoint_close": bar.midpoint_close,
                    "spread_close": bar.spread_close,
                    "integrity_hard_invalid": assessment.hard_invalid,
                    "integrity_hard_reasons": ";".join(assessment.hard_reasons),
                    "integrity_soft_reasons": ";".join(assessment.soft_reasons),
                    "integrity_eligible": assessment.eligible_for_entry,
                    "shift_alarm": alarm,
                    "shift_checked": shift_result.checked,
                    "shift_score": shift_result.score,
                    "log_return": bar.log_return,
                    "elapsed_seconds": bar.elapsed_seconds_since_prior,
                    "gap_indicator": bar.gap_indicator,
                    "tick_count": bar.tick_count,
                    "bid_age_seconds_close": getattr(
                        bar, "bid_age_seconds_close", None
                    ),
                    "ask_age_seconds_close": getattr(
                        bar, "ask_age_seconds_close", None
                    ),
                    "bid_update_count": getattr(bar, "bid_update_count", 0),
                    "ask_update_count": getattr(bar, "ask_update_count", 0),
                }

            for name, arm in arms.items():
                decision = arm.decide(
                    intent.intent,
                    intent.eligible,
                    is_entry,
                    assessment,
                    alarm,
                    bar.bar_id,
                    bar.last_event_timestamp,
                )
                if collect_rows:
                    row[f"arm_{name}_response"] = decision.response
                    row[f"arm_{name}_multiplier"] = decision.multiplier
                if decision.is_entry:
                    results[name].decisions.append(decision)

            if collect_rows:
                rows.append(row)

        return _PassResult(rows=rows, results=results, hard=hard, soft=soft, shift=shift)

    def _matched_pass(
        self, rows: Sequence[dict[str, Any]], matched
    ) -> PolicyArmResult:
        result = PolicyArmResult(arm=matched.ARM)
        for row in rows:
            shim = _FlagShim(
                hard_invalid=bool(row["integrity_hard_invalid"]),
                has_soft_reason=bool(row["integrity_soft_reasons"]),
            )
            decision = matched.decide(
                row["intent"],
                bool(row["intent_eligible"]),
                bool(row["is_entry"]),
                shim,
                bool(row["shift_alarm"]),
                int(row["bar_id"]),
                str(row["timestamp"]),
            )
            row[f"arm_{matched.ARM}_response"] = decision.response
            row[f"arm_{matched.ARM}_multiplier"] = decision.multiplier
            if decision.is_entry:
                result.decisions.append(decision)
        return result

    # -- metrics ---------------------------------------------------------
    def _arm_metrics(
        self,
        metrics: Metrics,
        results: dict[str, PolicyArmResult],
        entry_positions: set[int],
        fault_positions: set[int],
        unmodified_positions: set[int],
        test_bars: int,
        timestamp: str,
    ) -> None:
        opportunities = len(entry_positions)
        for arm, result in results.items():
            allowed = result.allowed_entry_opportunities()
            blocked = result.blocked_entry_opportunities()
            cautioned = result.cautioned_entry_opportunities()
            if allowed + blocked + cautioned != opportunities:
                raise AssertionError(
                    f"arm {arm}: allowed({allowed}) + blocked({blocked}) + "
                    f"cautioned({cautioned}) != entry opportunities "
                    f"({opportunities}); responses did not partition"
                )
            if len(result.decisions) != opportunities:
                raise AssertionError(
                    f"arm {arm}: {len(result.decisions)} entry decisions for "
                    f"{opportunities} entry opportunities (intent streams diverged)"
                )
            for name, value, num, den in (
                (f"allowed_entries_{arm}", None, allowed, opportunities),
                (f"blocked_eligible_entries_{arm}", None, blocked, opportunities),
            ):
                metrics.add(MetricRecord(
                    name=name,
                    value=(num / den) if den else None,
                    numerator=num,
                    denominator=den,
                    unit="count",
                    time_basis="test_entry_opportunity",
                    aggregation="pooled_bars",
                    exclusions=["train_bars", "validation_bars"],
                    caveat="test-range entry opportunities only",
                    timestamp=timestamp,
                ))
            metrics.intervention_burden(result.interventions(), test_bars)

        fused = results.get("fused")
        no_gate = results.get("no_gate")
        if fused is not None and no_gate is not None:
            metrics.action_divergence(
                fused.action_divergence(no_gate), opportunities
            )

    def _detection_metrics(
        self,
        metrics: Metrics,
        labels: Sequence,
        clean_test: Sequence,
        pos_by_bar_id: dict[int, int],
        hard: Sequence[bool],
        soft: Sequence[bool],
        shift: Sequence[bool],
        clean_hard: Sequence[bool],
        clean_soft: Sequence[bool],
        fault_positions: set[int],
        recovery: int,
        timestamp: str,
    ) -> None:
        by_class: dict[str, list] = {}
        for label in labels:
            by_class.setdefault(label.scenario_class, []).append(label)

        total_detected = 0
        total_fault_episodes = 0
        for cls, episodes in sorted(by_class.items()):
            detected_int = 0
            detected_shift = 0
            detected_either = 0
            delays: list[int] = []
            fault_bars = 0
            fault_bars_flagged = 0
            for label in episodes:
                onset = pos_by_bar_id.get(label.onset_bar_id)
                offset = pos_by_bar_id.get(label.offset_bar_id)
                if onset is None or offset is None:
                    continue
                window_end = min(offset + recovery + 1, len(hard))
                window = range(onset, window_end)
                int_flags = [hard[p] or soft[p] for p in window]
                shift_flags = [shift[p] for p in window]
                if any(int_flags):
                    detected_int += 1
                    first = int_flags.index(True)
                    delays.append(first + 1)
                if any(shift_flags):
                    detected_shift += 1
                if any(int_flags) or any(shift_flags):
                    detected_either += 1
                fault_bars += offset - onset + 1
                fault_bars_flagged += sum(
                    1 for p in range(onset, offset + 1) if hard[p] or soft[p]
                )
            n = len(episodes)
            if cls in PASS_THROUGH_CLASSES:
                total_detected += detected_either
                total_fault_episodes += n
            for name, num, den, caveat in (
                (
                    f"episode_detection_recall_{cls}",
                    detected_int,
                    n,
                    "integrity monitor (hard or soft) within the episode window",
                ),
                (
                    f"shift_episode_detection_recall_{cls}",
                    detected_shift,
                    n,
                    "shift monitor alarm within the episode window",
                ),
                (
                    f"episode_detection_recall_either_{cls}",
                    detected_either,
                    n,
                    "integrity or shift evidence within the episode window",
                ),
            ):
                metrics.add(MetricRecord(
                    name=name,
                    value=(num / den) if den else None,
                    numerator=num,
                    denominator=den,
                    unit="ratio",
                    time_basis="episode",
                    aggregation="episode_macro",
                    exclusions=["censored_episodes"],
                    caveat=caveat,
                    timestamp=timestamp,
                ))
            metrics.add(MetricRecord(
                name=f"fault_bar_flag_rate_{cls}",
                value=(fault_bars_flagged / fault_bars) if fault_bars else None,
                numerator=fault_bars_flagged,
                denominator=fault_bars,
                unit="ratio",
                time_basis="pooled_bar",
                aggregation="pooled_bars",
                exclusions=[],
                caveat="bars within an episode are dependent",
                timestamp=timestamp,
            ))
            metrics.add(MetricRecord(
                name=f"mean_detection_delay_bars_{cls}",
                value=(sum(delays) / len(delays)) if delays else None,
                numerator=len(delays),
                denominator=n,
                unit="observed_bars",
                time_basis="episode",
                aggregation="episode_macro",
                exclusions=["missed_censored"],
                caveat="missed episodes are censored, not zero-delay",
                timestamp=timestamp,
            ))

        if total_fault_episodes:
            metrics.add(MetricRecord(
                name="episode_detection_recall_all_faults",
                value=total_detected / total_fault_episodes,
                numerator=total_detected,
                denominator=total_fault_episodes,
                unit="ratio",
                time_basis="episode",
                aggregation="episode_micro",
                exclusions=["benign_volatility", "identity"],
                caveat="benign volatility is a control, not a fault",
                timestamp=timestamp,
            ))

        # False-alert burden on clean test bars outside injected intervals.
        excluded = set(fault_positions)
        for label in labels:
            onset = pos_by_bar_id.get(label.onset_bar_id)
            offset = pos_by_bar_id.get(label.offset_bar_id)
            if onset is not None and offset is not None:
                excluded.update(range(onset, offset + 1))
        runs = 0
        in_run = False
        eligible = 0
        for i in range(len(clean_hard)):
            if i in excluded:
                in_run = False
                continue
            eligible += 1
            flagged = clean_hard[i] or clean_soft[i]
            if flagged and not in_run:
                runs += 1
                in_run = True
            elif not flagged:
                in_run = False
        metrics.false_alert_burden(runs, eligible, denominator_unit="eligible_clean_bar")
        metrics.add(MetricRecord(
            name="false_alert_rate_per_1000_clean_bars",
            value=(1000.0 * runs / eligible) if eligible else None,
            numerator=runs,
            denominator=eligible,
            unit="ratio_per_1000_bars",
            time_basis="pooled_bar",
            aggregation="pooled_bars",
            exclusions=["injected_fault_intervals"],
            caveat="alert episodes counted on clean bars; raw run count also reported",
            timestamp=timestamp,
        ))

    def _arm_rows(
        self,
        results: dict[str, PolicyArmResult],
        entry_positions: set[int],
        fault_positions: set[int],
        unmodified_positions: set[int],
        scenario_results: dict[str, PolicyArmResult],
        pos_by_bar_id: dict[int, int],
        timestamp: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for arm, result in results.items():
            blocked = result.blocked_entry_opportunities()
            cautioned = result.cautioned_entry_opportunities()
            allowed = result.allowed_entry_opportunities()
            opps = len(result.decisions)
            # Faulted pass-through uses the mutated pass (same arm, same plan).
            scenario = scenario_results.get(arm)
            faulted_total = 0
            faulted_allowed = 0
            if scenario is not None:
                for decision in scenario.decisions:
                    pos = pos_by_bar_id.get(decision.bar_id)
                    if pos is None or pos not in fault_positions:
                        continue
                    faulted_total += 1
                    if decision.response == RESP_ALLOW:
                        faulted_allowed += 1
            unmodified_blocked = 0
            unmodified_total = 0
            unmodified_interventions = 0
            for decision in result.decisions:
                pos = pos_by_bar_id.get(decision.bar_id)
                if pos is None or pos not in unmodified_positions:
                    continue
                unmodified_total += 1
                if decision.response in BLOCKING_RESPONSES:
                    unmodified_blocked += 1
                if decision.response not in (RESP_ALLOW, "ALLOW_OR_IDLE"):
                    unmodified_interventions += 1
            rows.append({
                "arm": arm,
                "entry_opportunities": opps,
                "allowed_entries": allowed,
                "blocked_eligible_entries": blocked,
                "cautioned_entries": cautioned,
                "abstention_rate": (blocked / opps) if opps else None,
                "faulted_entry_opportunities": faulted_total,
                "faulted_allowed": faulted_allowed,
                "fault_pass_through": (
                    faulted_allowed / faulted_total if faulted_total else None
                ),
                "unmodified_entry_opportunities": unmodified_total,
                "unmodified_blocked": unmodified_blocked,
                "unmodified_abstention_rate": (
                    unmodified_blocked / unmodified_total
                    if unmodified_total
                    else None
                ),
                "clean_intervention_rate": (
                    unmodified_interventions / unmodified_total
                    if unmodified_total
                    else None
                ),
                "timestamp": timestamp,
            })
        return rows

    def _scenario_rows(
        self,
        metrics: Metrics,
        labels: Sequence,
        pos_by_bar_id: dict[int, int],
        hard: Sequence[bool],
        soft: Sequence[bool],
        shift: Sequence[bool],
        recovery: int,
        timestamp: str,
    ) -> list[dict[str, Any]]:
        records = {r.name: r for r in metrics.records}
        rows: list[dict[str, Any]] = []
        by_class: dict[str, list] = {}
        for label in labels:
            by_class.setdefault(label.scenario_class, []).append(label)
        for cls, episodes in sorted(by_class.items()):
            delays: list[int] = []
            fault_bars = 0
            flagged_bars = 0
            for label in episodes:
                onset = pos_by_bar_id.get(label.onset_bar_id)
                offset = pos_by_bar_id.get(label.offset_bar_id)
                if onset is None or offset is None:
                    continue
                window = range(onset, min(offset + recovery + 1, len(hard)))
                flags = [hard[p] or soft[p] for p in window]
                if any(flags):
                    delays.append(flags.index(True) + 1)
                fault_bars += offset - onset + 1
                flagged_bars += sum(
                    1 for p in range(onset, offset + 1) if hard[p] or soft[p]
                )
            recall = records.get(f"episode_detection_recall_{cls}")
            shift_recall = records.get(f"shift_episode_detection_recall_{cls}")
            either = records.get(f"episode_detection_recall_either_{cls}")
            rows.append({
                "scenario_class": cls,
                "episodes": len(episodes),
                "detected_integrity": recall.numerator if recall else 0,
                "detected_shift": shift_recall.numerator if shift_recall else 0,
                "detected_either": either.numerator if either else 0,
                "episode_detection_recall": recall.value if recall else None,
                "shift_detection_recall": shift_recall.value if shift_recall else None,
                "fault_bars": fault_bars,
                "fault_bars_flagged": flagged_bars,
                "fault_bar_flag_rate": (
                    flagged_bars / fault_bars if fault_bars else None
                ),
                "mean_detection_delay_bars": (
                    sum(delays) / len(delays) if delays else None
                ),
                "missed_episodes": len(episodes) - len(delays),
                "timestamp": timestamp,
            })
        return rows

    def _readme(self, output_dir: Path, audit, command: str) -> str:
        return f"""# Limen Core Run Reproduction README

run_id: {self.run_id}
generated: {ts_mod.utc_now_iso()}
status: completed
config: {self.config_path}
config_version: {self.config.version}
config_digest: {config_digest(self.config)}

## Reproduction

From the project root:

    {command}

## Input

- Path: {self.config.runtime.input_path}
- SHA-256: {audit.sha256}
- Byte size: {audit.byte_size}
- Rows: {audit.total_rows}

## Outputs

- run_manifest.json: run identity, command, digests, status
- data_audit.json: source schema, range, reconstructed-state counters
- frozen_config.json: all frozen parameters and the canonical digest
- bar_signals.csv: per-bar intent, monitor scores, policy responses (test range)
- episode_metrics.csv: per-arm metrics on test entry opportunities
- scenario_metrics.csv: per-class detection and false-alert metrics
- scenario_manifest.csv: evaluator-only labels (isolated from runtime)
- summary.json: aggregate summary including thresholds and source audit
- artifact_manifest.json: digests of every output file above
- execution.log: timestamped event log

## Notes

- All timestamps are UTC ISO 8601 with milliseconds (Z suffix).
- The source CSV is immutable and was not modified.
- Metrics use test entry opportunities only; train and validation are calibration.
- The KS threshold is fitted on causal validation scores, never on test data.
- Evaluator labels never enter a runtime component: mutations never write a
  scenario name into a bar field, and the mutated pass uses the clean intent
  stream.
- bar_signals.csv covers the test range; the replay still passes the whole
  series through the monitors so their state is causal.
- This is an offline research prototype; no broker or execution path exists.
"""


__all__ = [
    "PASS_THROUGH_CLASSES",
    "ReplayEngine",
]
