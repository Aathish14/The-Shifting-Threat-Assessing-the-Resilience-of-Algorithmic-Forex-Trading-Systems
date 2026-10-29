"""Ablation study (Phase 6).

Answers "which evidence actually drives the decisions?" by removing exactly one
signal source -- or disabling exactly one fusion rule -- and re-running the
*whole* replay semantics on the same bars, the same scenario plan, the same
calibration rules, and the same frozen thresholds.

What is deliberately NOT ablated
--------------------------------
The reference intent stream never changes: every variant is evaluated on the
identical intent stream, so a difference in the arm columns can only come from
the removed signal. The runner asserts that the entry-opportunity count is
identical in every variant (a mismatch would mean the intent stream diverged).

Label isolation is untouched: variants only toggle runtime parameters; the
scenario labels are produced by the same planner and never reach a runtime
component.

Signal groups (``config.ABLATION_SIGNAL_GROUPS``)
    hard_checks        remove the hard quote checks (missing/nonpositive/crossed)
    spread_outlier     remove the spread outlier check
    price_innovation   remove the price innovation check
    repeated_quote     remove the repeated-quote check
    inter_arrival_gap  remove the inter-arrival gap check
    stale_quote        remove the stale-quote check
    timestamp_order    remove the backward-timestamp check (requires a re-ingest)
    shift_monitor      remove the shift alarm from the policy input

Fusion rules (``config.FUSION_RULES``) are ablated one at a time through
``PolicyConfig.disabled_rules``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import timestamp as ts_mod
from .audit import (
    append_log,
    atomic_write_json,
    build_artifact_manifest,
    build_run_manifest,
    build_frozen_config,
    config_digest,
)
from .config import LimenConfig
from .experiments import Evaluation, PreparedData, evaluate, ingest_signature, prepare
from .metrics import Metrics
from .replay import PASS_THROUGH_CLASSES, ReplayEngine, _write_csv


@dataclass(frozen=True)
class Variant:
    """One declared one-factor change relative to the frozen configuration."""

    name: str
    kind: str  # base | signal_group | fusion_rule
    detail: str
    config: LimenConfig


def _signal_group_variant(config: LimenConfig, group: str) -> LimenConfig:
    integ = config.integrity
    policy = config.policy
    if group == "hard_checks":
        integ = dataclasses.replace(
            integ, enable_missing=False, enable_nonpositive=False, enable_crossed=False
        )
    elif group == "spread_outlier":
        integ = dataclasses.replace(integ, enable_spread_outlier=False)
    elif group == "price_innovation":
        integ = dataclasses.replace(integ, enable_price_innovation=False)
    elif group == "repeated_quote":
        integ = dataclasses.replace(integ, enable_repeated_quote=False)
    elif group == "inter_arrival_gap":
        integ = dataclasses.replace(integ, enable_gap_check=False)
    elif group == "stale_quote":
        integ = dataclasses.replace(integ, enable_stale_quote=False)
    elif group == "timestamp_order":
        integ = dataclasses.replace(integ, enable_timestamp_order=False)
    elif group == "shift_monitor":
        policy = dataclasses.replace(policy, enable_shift=False)
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"unknown ablation signal group: {group!r}")
    return dataclasses.replace(config, integrity=integ, policy=policy)


def _fusion_rule_variant(config: LimenConfig, rule: str) -> LimenConfig:
    policy = dataclasses.replace(
        config.policy, disabled_rules=tuple(config.policy.disabled_rules) + (rule,)
    )
    return dataclasses.replace(config, policy=policy)


def _faulted_counts(
    fault_positions: set[int],
    metric_rows: list[dict[str, Any]],
) -> tuple[int, int]:
    """Return (faulted episode bars, faulted entry opportunities).

    ``len(fault_positions)`` is the number of *bars* touched by an injected
    episode. The pass-through ratio's denominator is the number of *entry
    opportunities* that fall inside those bars, which is smaller. Reporting one
    under the other's name makes the summary look self-contradictory, so the two
    are computed separately and named separately.
    """
    episode_bars = len(fault_positions)
    entry_opportunities = max(
        (
            int(row["faulted_entry_opportunities"])
            for row in metric_rows
            if row.get("variant") == "base"
        ),
        default=0,
    )
    return episode_bars, entry_opportunities


class AblationRunner:
    """Run the declared ablation matrix and write the comparison artifacts."""

    def __init__(
        self,
        config: LimenConfig,
        run_id: str,
        config_path: str | None = None,
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.config_path = config_path or "Files/configs/<not-recorded>.json"

    @staticmethod
    def _label_signature(ev: Evaluation) -> tuple:
        """Everything an ablation must hold fixed about the scenario plan.

        ``onset_bar_id`` / ``offset_bar_id`` are absolute bar IDs, so this
        catches a variant that moved an episode, changed a severity, or drew a
        different episode set even if it happened to keep the same count.
        """
        return tuple(
            (
                label.episode_id,
                label.scenario_class,
                label.onset_bar_id,
                label.offset_bar_id,
                label.duration_bars,
                label.severity,
                label.label_digest,
            )
            for label in ev.layout.labels
        )

    def variants(self) -> list[Variant]:
        cfg = self.config
        out = [Variant("base", "base", "frozen configuration", cfg)]
        groups = list(cfg.ablation.signal_groups)
        # ``timestamp_order`` is the only group that changes the bar series, so
        # it is evaluated last: the prepared data is rebuilt once and the
        # previous 240k-bar series can be released instead of being cached.
        groups.sort(key=lambda g: g == "timestamp_order")
        for group in groups:
            out.append(Variant(
                f"signal:{group}", "signal_group", group,
                _signal_group_variant(cfg, group),
            ))
        for rule in cfg.ablation.fusion_rules:
            out.append(Variant(
                f"rule:{rule}", "fusion_rule", rule,
                _fusion_rule_variant(cfg, rule),
            ))
        return out

    # -- public API ------------------------------------------------------
    def run(self, output_dir: str | Path) -> dict[str, Any]:
        output_dir = Path(output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty run directory: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "execution.log"
        cfg = self.config
        cfg.validate_for_run()
        append_log(log_path, f"event=ablation status=started run_id={self.run_id}")

        variants = self.variants()
        prepared: PreparedData | None = None
        prepared_sig: tuple | None = None
        evaluations: dict[str, Evaluation] = {}

        for variant in variants:
            sig = ingest_signature(variant.config)
            if prepared is None or sig != prepared_sig:
                append_log(
                    log_path,
                    f"event=prepare status=start variant={variant.name}",
                )
                prepared = prepare(variant.config, log_path)
                prepared_sig = sig
            evaluations[variant.name] = evaluate(
                variant.config, prepared, log_path
            )
            append_log(
                log_path,
                f"event=variant status=ok name={variant.name} "
                f"opportunities={len(evaluations[variant.name].entry_positions)}",
            )

        # --- invariants ----------------------------------------------------
        base = evaluations["base"]
        base_opps = len(base.entry_positions)
        base_labels = self._label_signature(base)
        for name, ev in evaluations.items():
            if len(ev.entry_positions) != base_opps:
                raise AssertionError(
                    f"variant {name}: {len(ev.entry_positions)} entry "
                    f"opportunities vs {base_opps} on the base config; the "
                    "intent stream must be identical across variants"
                )
            if self._label_signature(ev) != base_labels:
                raise AssertionError(
                    f"variant {name}: scenario labels differ from the base "
                    "config; an ablation must hold scenario IDs, onsets, and "
                    "severities fixed so the only difference is the removed "
                    "signal"
                )
            no_gate = ev.results.get("no_gate")
            if no_gate is not None and no_gate.blocked_entry_opportunities():
                raise AssertionError(
                    f"variant {name}: no_gate blocked "
                    f"{no_gate.blocked_entry_opportunities()} entries"
                )

        ts_now = ts_mod.utc_now_iso()
        engine = ReplayEngine(cfg, self.run_id)

        # --- per-variant metrics ------------------------------------------
        metric_rows: list[dict[str, Any]] = []
        detection_rows: list[dict[str, Any]] = []
        delta_rows: list[dict[str, Any]] = []
        for variant in variants:
            ev = evaluations[variant.name]
            rows = ev.arm_rows(engine, ts_now)
            for row in rows:
                row = dict(row)
                row["variant"] = variant.name
                row["kind"] = variant.kind
                row["detail"] = variant.detail
                metric_rows.append(row)

            detection = self._detection_records(
                cfg, ev, variant.config, ts_now
            )
            for row in detection:
                row["variant"] = variant.name
                detection_rows.append(row)

            delta_rows.append(self._delta_row(variant, ev, base, cfg, ts_now))

        _write_csv(
            output_dir / "ablation_metrics.csv",
            metric_rows,
            [
                "variant", "kind", "detail", "arm", "entry_opportunities",
                "allowed_entries", "blocked_eligible_entries", "cautioned_entries",
                "abstention_rate", "faulted_entry_opportunities", "faulted_allowed",
                "fault_pass_through", "unmodified_entry_opportunities",
                "unmodified_blocked", "unmodified_abstention_rate",
                "clean_intervention_rate", "timestamp",
            ],
        )
        _write_csv(
            output_dir / "ablation_detection.csv",
            detection_rows,
            [
                "variant", "scenario_class", "episodes", "detected_integrity",
                "detected_shift", "detected_either", "episode_detection_recall",
                "shift_detection_recall", "fault_bars", "fault_bars_flagged",
                "fault_bar_flag_rate", "mean_detection_delay_bars",
                "missed_episodes", "timestamp",
            ],
        )
        _write_csv(
            output_dir / "ablation_delta.csv",
            delta_rows,
            [
                "variant", "kind", "detail", "blocked_delta",
                "fault_pass_through_delta", "false_alert_rate_delta",
                "detection_recall_delta", "abstention_rate_delta", "timestamp",
            ],
        )

        faulted_episode_bars, faulted_entries = _faulted_counts(
            base.fault_positions, metric_rows
        )
        summary = {
            "run_id": self.run_id,
            "timestamp": ts_now,
            "status": "completed",
            "config_version": cfg.version,
            "config_digest": config_digest(cfg),
            "input_sha256": prepared.audit.sha256,
            "input_byte_size": prepared.audit.byte_size,
            "total_observed_bars": prepared.n_bars,
            "train_bars": prepared.train_end,
            "validation_bars": prepared.val_end - prepared.train_end,
            "test_bars": prepared.test_end - prepared.val_end,
            "entry_opportunities": base_opps,
            "faulted_episode_bars": faulted_episode_bars,
            "faulted_entry_opportunities": faulted_entries,
            "variant_count": len(variants),
            "variants": [
                {
                    "variant": v.name,
                    "kind": v.kind,
                    "detail": v.detail,
                    "ingest_signature_changed": ingest_signature(v.config)
                    != ingest_signature(cfg),
                    "thresholds": evaluations[v.name].thresholds,
                }
                for v in variants
            ],
            "episodes": len(base.layout.labels),
            "labels_digest": hashlib.sha256(
                json.dumps(base_labels, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "labels_identical_across_variants": True,
            "episodes_by_class": base.layout.plan.by_class(),
            "episode_bars": len(base.injected),
            "unmodified_test_bars": len(base.unmodified),
            "small_sample": {
                "min_faulted_entries": cfg.ablation.min_faulted_entries,
                "observed_faulted_entries": faulted_entries,
                "below_minimum": faulted_entries < cfg.ablation.min_faulted_entries,
                "caveat": (
                    "pass-through is a ratio over faulted entry opportunities; "
                    "below the configured minimum it is descriptive only"
                ),
            },
            "notes": [
                "every variant shares the reference intent stream; the entry "
                "opportunity count is asserted equal across variants",
                "ablation removes one signal source or one fusion rule at a time",
                "thresholds are re-fitted per variant on train/validation only",
                "the timestamp_order variant requires a re-ingest because the "
                "check runs during tick reconstruction",
            ],
        }
        atomic_write_json(output_dir / "summary.json", summary)
        atomic_write_json(
            output_dir / "frozen_config.json",
            build_frozen_config(cfg, self.config_path),
        )

        command = (
            f"python Files/run_limen.py ablation --config {self.config_path} "
            f"--run-id {self.run_id}"
        )
        manifest = build_run_manifest(
            run_id=self.run_id,
            command=command,
            input_path=cfg.runtime.input_path,
            input_digest=prepared.audit.sha256,
            input_size=prepared.audit.byte_size,
            config_obj=cfg,
            status="completed",
            output_dir=output_dir,
            seed=cfg.scenario.seed,
            extra={"variant_count": len(variants)},
        )
        atomic_write_json(output_dir / "run_manifest.json", manifest)

        append_log(
            log_path,
            f"event=complete status=ok run_id={self.run_id} variants={len(variants)}",
        )
        atomic_write_json(
            output_dir / "artifact_manifest.json",
            build_artifact_manifest(output_dir),
        )
        return {
            "variant_count": len(variants),
            "entry_opportunities": base_opps,
            "summary_path": str(output_dir / "summary.json"),
        }

    # -- helpers ---------------------------------------------------------
    def _detection_records(
        self, cfg: LimenConfig, ev: Evaluation, variant_cfg: LimenConfig, ts_now: str
    ) -> list[dict[str, Any]]:
        metrics = Metrics()
        engine = ReplayEngine(variant_cfg, self.run_id)
        engine._detection_metrics(
            metrics,
            ev.layout.labels,
            ev.layout.clean_test,
            ev.layout.pos_by_bar_id,
            ev.mutated.hard,
            ev.mutated.soft,
            ev.mutated.shift,
            ev.clean.hard,
            ev.clean.soft,
            ev.fault_positions,
            variant_cfg.scenario.recovery_window_bars,
            ts_now,
        )
        return engine._scenario_rows(
            metrics,
            ev.layout.labels,
            ev.layout.pos_by_bar_id,
            ev.mutated.hard,
            ev.mutated.soft,
            ev.mutated.shift,
            variant_cfg.scenario.recovery_window_bars,
            ts_now,
        )

    def _delta_row(
        self, variant: Variant, ev: Evaluation, base: Evaluation,
        cfg: LimenConfig, ts_now: str,
    ) -> dict[str, Any]:
        def headline(evaluation: Evaluation) -> tuple[float | None, float | None,
                                                      float | None, float | None]:
            rows = evaluation.arm_rows(ReplayEngine(cfg, self.run_id), ts_now)
            fused = next((r for r in rows if r["arm"] == "fused"), None)
            no_gate = next((r for r in rows if r["arm"] == "no_gate"), None)
            blocked = fused["blocked_eligible_entries"] if fused else None
            pass_through = fused["fault_pass_through"] if fused else None
            abstention = fused["abstention_rate"] if fused else None
            recall = self._overall_recall(ev)
            return blocked, pass_through, recall, abstention

        b_blocked, b_pass, b_recall, b_abst = headline(base)
        v_blocked, v_pass, v_recall, v_abst = headline(ev)
        b_false = self._false_alert_rate(base)
        v_false = self._false_alert_rate(ev)

        def delta(a, b):
            if a is None or b is None:
                return None
            return a - b

        return {
            "variant": variant.name,
            "kind": variant.kind,
            "detail": variant.detail,
            "blocked_delta": delta(v_blocked, b_blocked),
            "fault_pass_through_delta": delta(v_pass, b_pass),
            "false_alert_rate_delta": delta(v_false, b_false),
            "detection_recall_delta": delta(v_recall, b_recall),
            "abstention_rate_delta": delta(v_abst, b_abst),
            "timestamp": ts_now,
        }

    def _overall_recall(self, ev: Evaluation) -> float | None:
        """Fault episodes with any integrity/shift evidence, over all faults."""
        detected = 0
        total = 0
        recovery = self.config.scenario.recovery_window_bars
        by_class: dict[str, list] = {}
        for label in ev.layout.labels:
            by_class.setdefault(label.scenario_class, []).append(label)
        for cls, episodes in by_class.items():
            if cls not in PASS_THROUGH_CLASSES:
                continue
            for label in episodes:
                onset = ev.layout.pos_by_bar_id.get(label.onset_bar_id)
                offset = ev.layout.pos_by_bar_id.get(label.offset_bar_id)
                if onset is None or offset is None:
                    continue
                total += 1
                window = range(
                    onset,
                    min(offset + recovery + 1, len(ev.mutated.hard)),
                )
                if any(
                    ev.mutated.hard[p] or ev.mutated.soft[p] or ev.mutated.shift[p]
                    for p in window
                ):
                    detected += 1
        return (detected / total) if total else None

    def _false_alert_rate(self, ev: Evaluation) -> float | None:
        excluded = set(ev.injected)
        runs = 0
        in_run = False
        eligible = 0
        for i in range(len(ev.clean.hard)):
            if i in excluded:
                in_run = False
                continue
            eligible += 1
            flagged = ev.clean.hard[i] or ev.clean.soft[i]
            if flagged and not in_run:
                runs += 1
                in_run = True
            elif not flagged:
                in_run = False
        return (1000.0 * runs / eligible) if eligible else None


from .replay import PASS_THROUGH_CLASSES as _PASS_THROUGH  # noqa: E402

__all__ = ["AblationRunner", "Variant"]
