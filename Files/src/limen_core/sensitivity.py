"""One-at-a-time sensitivity sweeps (Phase 7).

For every declared sweep the runner changes **exactly one** frozen parameter,
re-fits every threshold on train/validation only, and re-runs the full replay
semantics. It never re-uses the base run's fitted thresholds: a swept
``spread_quantile`` or ``ks_quantile`` must produce its own calibration or the
experiment would not measure what it claims to measure.

Guards
------
* ``split`` is asserted unchanged for every variant. A sweep may not move the
  test boundary, so no experiment can accidentally tune on test data.
* Thresholds are always fitted inside :func:`experiments.evaluate`, which
  reads ``bars[:train_end]`` and the validation range only.
* ``experiments_per_parameter`` is recorded, and the frozen value is always
  included as an explicit reference row even when the sweep tuple omits it, so
  every delta is measured against a re-run of the base configuration rather
  than against a stored number.
* The ledger is append-only: re-running the sweep adds rows, never rewrites
  history.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from .experiments import Evaluation, evaluate, ingest_signature, prepare
from .replay import PASS_THROUGH_CLASSES, ReplayEngine, _write_csv

#: Whitelisted sweeps: name -> (section attribute, field name).
SWEEP_TARGETS: dict[str, tuple[str, str]] = {
    "spread_quantile": ("integrity", "spread_quantile"),
    "innovation_quantile": ("integrity", "innovation_quantile"),
    "repeat_count_threshold": ("integrity", "repeat_count_threshold"),
    "min_gap_seconds": ("integrity", "min_gap_seconds"),
    "stale_quote_seconds": ("integrity", "stale_quote_seconds"),
    "ks_reference_window": ("ks", "reference_window"),
    "ks_recent_window": ("ks", "recent_window"),
    "ks_check_cadence": ("ks", "check_cadence"),
    "ks_consecutive_alerts": ("ks", "consecutive_alerts"),
    "ks_quantile": ("ks", "quantile"),
    "scenario_missing_quote_duration_bars": ("scenario", "missing_quote_duration_bars"),
    "scenario_stale_replay_duration_bars": ("scenario", "stale_replay_duration_bars"),
    "scenario_symmetric_shift_magnitude": ("scenario", "symmetric_shift_magnitude"),
    "scenario_return_scale_factor": ("scenario", "return_scale_factor"),
}


@dataclass(frozen=True)
class Experiment:
    """One parameter value applied to the frozen configuration."""

    parameter: str
    value: Any
    is_reference: bool
    config: LimenConfig

    @property
    def name(self) -> str:
        return f"{self.parameter}={self.value}"


def apply_parameter(config: LimenConfig, parameter: str, value: Any) -> LimenConfig:
    """Return a copy of ``config`` with exactly one field replaced."""
    if parameter not in SWEEP_TARGETS:
        raise ValueError(f"unknown sensitivity parameter: {parameter!r}")
    section_name, field = SWEEP_TARGETS[parameter]
    section = getattr(config, section_name)
    if not hasattr(section, field):
        raise AttributeError(f"{section_name}.{field} does not exist")
    updated = dataclasses.replace(section, **{field: value})
    variant = dataclasses.replace(config, **{section_name: updated})
    if variant.split != config.split:
        raise AssertionError("a sensitivity variant must not change the split")
    # Prove the change is exactly one field.
    changed = [
        (name, f.name)
        for name, sec in (
            ("strategy", variant.strategy), ("split", variant.split),
            ("integrity", variant.integrity), ("ks", variant.ks),
            ("scenario", variant.scenario), ("policy", variant.policy),
            ("ablation", variant.ablation), ("sensitivity", variant.sensitivity),
            ("runtime", variant.runtime), ("bar", variant.bar),
        )
        for f in dataclasses.fields(sec)
        if getattr(sec, f.name) != getattr(getattr(config, name), f.name)
    ]
    expected = [(section_name, field)]
    frozen = getattr(getattr(config, section_name), field)
    # A reference row re-applies the frozen value: nothing may change. Any
    # other row must change exactly the targeted field and nothing else.
    allowed = [[], expected] if value == frozen else [expected]
    if changed not in allowed:
        raise AssertionError(
            f"parameter {parameter}={value!r} changed {changed}, expected {expected}"
        )
    return variant


def _value_for(config: LimenConfig, parameter: str) -> Any:
    section_name, field = SWEEP_TARGETS[parameter]
    return getattr(getattr(config, section_name), field)


class SensitivityRunner:
    """Run the declared one-at-a-time sweeps and write the comparison artifacts."""

    def __init__(
        self,
        config: LimenConfig,
        run_id: str,
        config_path: str | None = None,
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.config_path = config_path or "Files/configs/<not-recorded>.json"

    def experiments(self) -> list[Experiment]:
        cfg = self.config
        out: list[Experiment] = []
        for parameter in cfg.sensitivity.sweep_names():
            if parameter not in SWEEP_TARGETS:
                raise ValueError(
                    f"sensitivity.{parameter} has no sweep target registered in "
                    "sensitivity.SWEEP_TARGETS; refusing to guess what it means"
                )
            frozen = _value_for(cfg, parameter)
            values = list(getattr(cfg.sensitivity, parameter))
            is_reference = [v == frozen for v in values]
            if frozen not in values:
                values.append(frozen)
                is_reference.append(True)
            for value, reference in zip(values, is_reference):
                out.append(Experiment(
                    parameter=parameter,
                    value=value,
                    is_reference=reference,
                    config=apply_parameter(cfg, parameter, value),
                ))
        # Group by ingest signature so a re-ingest happens at most once per
        # distinct signature (only min_gap_seconds / stale_quote_seconds change
        # the bar series). Ties keep declaration order.
        out.sort(key=lambda e: ingest_signature(e.config))
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
        append_log(log_path, f"event=sensitivity status=started run_id={self.run_id}")

        experiments = self.experiments()
        # Deduplicate identical configs (the reference row repeats once per
        # parameter); evaluate each distinct config once.
        evaluations: dict[str, Evaluation] = {}
        prepared = None
        prepared_sig: tuple | None = None
        for experiment in experiments:
            digest = config_digest(experiment.config)
            if digest in evaluations:
                continue
            sig = ingest_signature(experiment.config)
            if prepared is None or sig != prepared_sig:
                append_log(log_path, f"event=prepare status=start name={experiment.name}")
                prepared = prepare(experiment.config, log_path)
                prepared_sig = sig
            evaluations[digest] = evaluate(experiment.config, prepared, log_path)
            append_log(
                log_path,
                f"event=experiment status=ok name={experiment.name} "
                f"opportunities={len(evaluations[digest].entry_positions)}",
            )

        ts_now = ts_mod.utc_now_iso()
        base_digest = config_digest(cfg)
        if base_digest not in evaluations:
            raise AssertionError("the frozen configuration was not evaluated")
        base = evaluations[base_digest]

        rows: list[dict[str, Any]] = []
        ledger_path = output_dir / "experiment_ledger.jsonl"
        for experiment in experiments:
            digest = config_digest(experiment.config)
            ev = evaluations[digest]
            row = self._row(experiment, ev, base, ts_now)
            row["config_digest"] = digest
            rows.append(row)
            self._append_ledger(ledger_path, row, ev, digest)

        # Evaluation order is grouped by ingest signature for speed; the CSV is
        # written in declaration order so a reader sees one parameter's sweep as
        # one contiguous block.
        order = {
            name: i for i, name in enumerate(cfg.sensitivity.sweep_names())
        }
        rows.sort(key=lambda r: (order[r["parameter"]], str(r["value"])))

        _write_csv(
            output_dir / "sensitivity_metrics.csv",
            rows,
            [
                "parameter", "value", "is_reference", "config_digest",
                "entry_opportunities",
                "blocked_entries", "abstention_rate", "cautioned_entries",
                "faulted_entry_opportunities", "fault_pass_through",
                "clean_intervention_rate", "false_alert_rate_per_1000_clean_bars",
                "detection_recall_all_faults", "spread_threshold",
                "innovation_threshold", "ks_threshold", "episodes",
                "blocked_delta", "fault_pass_through_delta",
                "false_alert_rate_delta", "detection_recall_delta", "timestamp",
            ],
        )

        by_parameter: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_parameter.setdefault(row["parameter"], []).append(row)
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
            "experiment_count": len(experiments),
            "distinct_configs": len(evaluations),
            "parameters": {
                name: {
                    "values": [r["value"] for r in rows_],
                    "experiments_per_parameter": len(rows_),
                    "has_reference": any(r["is_reference"] for r in rows_),
                    "max_abs_blocked_delta": max(
                        (abs(r["blocked_delta"]) for r in rows_
                         if r["blocked_delta"] is not None),
                        default=None,
                    ),
                    "max_abs_pass_through_delta": max(
                        (abs(r["fault_pass_through_delta"]) for r in rows_
                         if r["fault_pass_through_delta"] is not None),
                        default=None,
                    ),
                }
                for name, rows_ in sorted(by_parameter.items())
            },
            "guards": {
                "split_frozen_across_variants": True,
                "thresholds_fitted_on_train_and_validation_only": True,
                "reference_row_is_a_rerun_not_a_stored_number": True,
                "ledger_mode": "append_only",
            },
            "notes": [
                "one parameter changes per row; every other field is identical",
                "min_gap_seconds and stale_quote_seconds change the bar series and "
                "therefore trigger a re-ingest; all other sweeps reuse the bars",
                "the frozen value is included as an explicit reference row",
                "this is a descriptive sweep on a single EURUSD sample: it shows "
                "local stability, not statistical significance",
            ],
        }
        atomic_write_json(output_dir / "summary.json", summary)
        atomic_write_json(
            output_dir / "frozen_config.json",
            build_frozen_config(cfg, self.config_path),
        )

        command = (
            f"python Files/run_limen.py sensitivity --config {self.config_path} "
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
            extra={
                "experiment_count": len(experiments),
                "distinct_configs": len(evaluations),
            },
        )
        atomic_write_json(output_dir / "run_manifest.json", manifest)

        append_log(
            log_path,
            f"event=complete status=ok run_id={self.run_id} "
            f"experiments={len(experiments)} distinct={len(evaluations)}",
        )
        atomic_write_json(
            output_dir / "artifact_manifest.json",
            build_artifact_manifest(output_dir),
        )
        return {
            "experiment_count": len(experiments),
            "distinct_configs": len(evaluations),
            "summary_path": str(output_dir / "summary.json"),
        }

    # -- helpers ---------------------------------------------------------
    def _row(
        self, experiment: Experiment, ev: Evaluation, base: Evaluation, ts_now: str
    ) -> dict[str, Any]:
        engine = ReplayEngine(experiment.config, self.run_id)
        arm_rows = engine._arm_rows(
            ev.results,
            ev.entry_positions,
            ev.fault_positions,
            ev.unmodified,
            ev.mutated.results,
            ev.layout.pos_by_bar_id,
            ts_now,
        )
        fused = next((r for r in arm_rows if r["arm"] == "fused"), None)
        base_rows = engine._arm_rows(
            base.results,
            base.entry_positions,
            base.fault_positions,
            base.unmodified,
            base.mutated.results,
            base.layout.pos_by_bar_id,
            ts_now,
        )
        base_fused = next((r for r in base_rows if r["arm"] == "fused"), None)

        def delta(a, b):
            if a is None or b is None:
                return None
            return a - b

        blocked = fused["blocked_eligible_entries"] if fused else None
        base_blocked = base_fused["blocked_eligible_entries"] if base_fused else None
        pass_through = fused["fault_pass_through"] if fused else None
        base_pass = base_fused["fault_pass_through"] if base_fused else None
        recall = self._recall(ev)
        base_recall = self._recall(base)
        false_rate = self._false_alert_rate(ev)
        base_false = self._false_alert_rate(base)

        return {
            "parameter": experiment.parameter,
            "value": experiment.value,
            "is_reference": experiment.is_reference,
            "entry_opportunities": len(ev.entry_positions),
            "blocked_entries": blocked,
            "abstention_rate": fused["abstention_rate"] if fused else None,
            "cautioned_entries": fused["cautioned_entries"] if fused else None,
            "faulted_entry_opportunities": (
                fused["faulted_entry_opportunities"] if fused else None
            ),
            "fault_pass_through": pass_through,
            "clean_intervention_rate": (
                fused["clean_intervention_rate"] if fused else None
            ),
            "false_alert_rate_per_1000_clean_bars": false_rate,
            "detection_recall_all_faults": recall,
            "spread_threshold": ev.thresholds.get("spread"),
            "innovation_threshold": ev.thresholds.get("innovation"),
            "ks_threshold": ev.thresholds.get("ks"),
            "episodes": len(ev.layout.labels),
            "blocked_delta": delta(blocked, base_blocked),
            "fault_pass_through_delta": delta(pass_through, base_pass),
            "false_alert_rate_delta": delta(false_rate, base_false),
            "detection_recall_delta": delta(recall, base_recall),
            "timestamp": ts_now,
        }

    def _recall(self, ev: Evaluation) -> float | None:
        recovery = self.config.scenario.recovery_window_bars
        detected = 0
        total = 0
        for label in ev.layout.labels:
            if label.scenario_class not in PASS_THROUGH_CLASSES:
                continue
            onset = ev.layout.pos_by_bar_id.get(label.onset_bar_id)
            offset = ev.layout.pos_by_bar_id.get(label.offset_bar_id)
            if onset is None or offset is None:
                continue
            total += 1
            window = range(onset, min(offset + recovery + 1, len(ev.mutated.hard)))
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

    def _append_ledger(
        self,
        path: Path,
        row: dict[str, Any],
        ev: Evaluation,
        experiment_digest: str,
    ) -> None:
        record = {
            "timestamp": row["timestamp"],
            "run_id": self.run_id,
            "base_config_digest": config_digest(self.config),
            "experiment_config_digest": experiment_digest,
            "parameter": row["parameter"],
            "value": row["value"],
            "is_reference": row["is_reference"],
            "thresholds": ev.thresholds,
            "entry_opportunities": row["entry_opportunities"],
            "blocked_entries": row["blocked_entries"],
            "fault_pass_through": row["fault_pass_through"],
            "detection_recall_all_faults": row["detection_recall_all_faults"],
            "false_alert_rate_per_1000_clean_bars": (
                row["false_alert_rate_per_1000_clean_bars"]
            ),
            "test_bars_used_for_fitting": 0,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


__all__ = [
    "Experiment",
    "SWEEP_TARGETS",
    "SensitivityRunner",
    "apply_parameter",
]
