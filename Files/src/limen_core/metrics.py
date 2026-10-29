"""Evaluation-only metrics with explicit numerators, denominators, and censoring rules.

Metrics are computed after replay by joining evaluator labels with runtime
outputs. Labels never enter runtime components. Each metric names its unit,
numerator, denominator, time basis, aggregation rule, and missing/censored
handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricRecord:
    """A single computed metric with full provenance."""

    name: str
    value: float | None
    numerator: int | float | None
    denominator: int | float | None
    unit: str
    time_basis: str
    aggregation: str
    exclusions: list[str] = field(default_factory=list)
    caveat: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "unit": self.unit,
            "time_basis": self.time_basis,
            "aggregation": self.aggregation,
            "exclusions": list(self.exclusions),
            "caveat": self.caveat,
            "timestamp": self.timestamp,
        }


class Metrics:
    """Episode-aware and bar-aware metric computation."""

    def __init__(self) -> None:
        self.records: list[MetricRecord] = []

    def add(self, record: MetricRecord) -> None:
        self.records.append(record)

    def detection_recall(
        self,
        fault_episodes_detected: int,
        eligible_fault_episodes: int,
        time_basis: str = "episode",
    ) -> MetricRecord:
        value = (fault_episodes_detected / eligible_fault_episodes) if eligible_fault_episodes else None
        self.add(MetricRecord(
            name="episode_detection_recall",
            value=value,
            numerator=fault_episodes_detected,
            denominator=eligible_fault_episodes,
            unit="ratio",
            time_basis=time_basis,
            aggregation="episode_macro",
            exclusions=["censored_episodes"],
            caveat="episode-macro recall is not pooled-bar recall",
            timestamp="",
        ))
        return self.records[-1]

    def fault_bar_flag_rate(
        self,
        labeled_fault_bars_flagged: int,
        eligible_labeled_fault_bars: int,
    ) -> MetricRecord:
        value = (labeled_fault_bars_flagged / eligible_labeled_fault_bars) if eligible_labeled_fault_bars else None
        self.add(MetricRecord(
            name="fault_bar_flag_rate",
            value=value,
            numerator=labeled_fault_bars_flagged,
            denominator=eligible_labeled_fault_bars,
            unit="ratio",
            time_basis="pooled_bar",
            aggregation="pooled_bars",
            exclusions=[],
            caveat="bars within an episode are dependent; label as pooled bars",
            timestamp="",
        ))
        return self.records[-1]

    def fault_pass_through(
        self,
        faulted_eligible_entries_allowed: int,
        faulted_eligible_entry_opportunities: int,
        arm: str,
    ) -> MetricRecord:
        """Share of faulted entry opportunities that the arm allowed *unchanged*.

        Numerator counts responses that are exactly ``ALLOW``. ``CAUTION``
        (entry permitted but modified), ``DELAY``, ``PAUSE`` and ``REJECT`` are
        all non-pass-through, so this ratio is stricter than
        ``1 - abstention_rate``; the two are not complements. The comparison
        is made between arms on the same faulted set, in the mutated pass.

        ``caveat`` records why this is reported separately from blocking: it
        distinguishes *detecting* a signal from *acting* on it.
        """
        value = (faulted_eligible_entries_allowed / faulted_eligible_entry_opportunities) if faulted_eligible_entry_opportunities else None
        self.add(MetricRecord(
            name=f"fault_pass_through_{arm}",
            value=value,
            numerator=faulted_eligible_entries_allowed,
            denominator=faulted_eligible_entry_opportunities,
            unit="ratio",
            time_basis="pooled_bar",
            aggregation="pooled_bars",
            exclusions=[],
            caveat="distinguishes signal detection from policy blocking",
            timestamp="",
        ))
        return self.records[-1]

    def missed_event_rate(
        self,
        eligible_fault_episodes_no_detection: int,
        eligible_fault_episodes: int,
    ) -> MetricRecord:
        value = (eligible_fault_episodes_no_detection / eligible_fault_episodes) if eligible_fault_episodes else None
        self.add(MetricRecord(
            name="missed_event_rate",
            value=value,
            numerator=eligible_fault_episodes_no_detection,
            denominator=eligible_fault_episodes,
            unit="ratio",
            time_basis="episode",
            aggregation="episode_macro",
            exclusions=["censored_episodes"],
            caveat="episode-level complement of recall when eligibility is identical",
            timestamp="",
        ))
        return self.records[-1]

    def false_alert_burden(
        self,
        alert_episodes_outside_fault: int,
        eligible_unmodified_blocks: int,
        denominator_unit: str = "eligible_unmodified_block",
    ) -> MetricRecord:
        value = (alert_episodes_outside_fault / eligible_unmodified_blocks) if eligible_unmodified_blocks else None
        self.add(MetricRecord(
            name="false_alert_burden",
            value=value,
            numerator=alert_episodes_outside_fault,
            denominator=eligible_unmodified_blocks,
            unit="ratio",
            time_basis=denominator_unit,
            aggregation="episode",
            exclusions=["injected_fault_intervals"],
            caveat="unmodified does not mean verified fault-free; raw alert count and exposure reported separately",
            timestamp="",
        ))
        return self.records[-1]

    def detection_delay(
        self,
        delays_bars: list[int],
        missed_episodes: int,
    ) -> dict[str, Any]:
        """Return delay statistics; missed episodes reported as censored."""
        valid_delays = [d for d in delays_bars if d is not None]
        result = {
            "mean_delay_bars": (sum(valid_delays) / len(valid_delays)) if valid_delays else None,
            "min_delay_bars": min(valid_delays) if valid_delays else None,
            "max_delay_bars": max(valid_delays) if valid_delays else None,
            "count": len(valid_delays),
            "missed_censored": missed_episodes,
        }
        self.add(MetricRecord(
            name="detection_delay",
            value=result["mean_delay_bars"],
            numerator=len(valid_delays),
            denominator=len(valid_delays) + missed_episodes,
            unit="observed_bars",
            time_basis="episode",
            aggregation="episode_macro",
            exclusions=["missed_censored"],
            caveat="missed episodes reported as censored/no detection, not zero delay",
            timestamp="",
        ))
        return result

    def unmodified_opportunity_abstention(
        self,
        eligible_unmodified_blocked: int,
        eligible_unmodified_opportunities: int,
        arm: str,
    ) -> MetricRecord:
        value = (eligible_unmodified_blocked / eligible_unmodified_opportunities) if eligible_unmodified_opportunities else None
        self.add(MetricRecord(
            name=f"unmodified_opportunity_abstention_{arm}",
            value=value,
            numerator=eligible_unmodified_blocked,
            denominator=eligible_unmodified_opportunities,
            unit="ratio",
            time_basis="pooled_bar",
            aggregation="pooled_bars",
            exclusions=[],
            caveat="source is unmodified by harness, not known fault-free",
            timestamp="",
        ))
        return self.records[-1]

    def action_divergence(
        self,
        divergent_pairs: int,
        eligible_paired_opportunities: int,
    ) -> MetricRecord:
        value = (divergent_pairs / eligible_paired_opportunities) if eligible_paired_opportunities else None
        self.add(MetricRecord(
            name="action_divergence",
            value=value,
            numerator=divergent_pairs,
            denominator=eligible_paired_opportunities,
            unit="ratio",
            time_basis="paired_bar",
            aggregation="paired_bars",
            exclusions=[],
            caveat="intent divergence is not filled-trade loss",
            timestamp="",
        ))
        return self.records[-1]

    def intervention_burden(
        self,
        bars_with_response: int,
        total_bars: int,
    ) -> MetricRecord:
        value = (bars_with_response / total_bars) if total_bars else None
        self.add(MetricRecord(
            name="intervention_burden",
            value=value,
            numerator=bars_with_response,
            denominator=total_bars,
            unit="ratio",
            time_basis="pooled_bar",
            aggregation="pooled_bars",
            exclusions=[],
            caveat="kept separate from blocked-eligible-entry count",
            timestamp="",
        ))
        return self.records[-1]

    def clean_entry_intervention_rate(
        self,
        arm: str,
        clean_entry_interventions: int,
        clean_entry_opportunities: int,
    ) -> MetricRecord:
        """Interventions on entries whose bar was never mutated by the harness.

        Numerator: entry opportunities on clean (unmodified) bars whose response
        was anything other than ALLOW -- including CAUTION, which only modifies
        the entry rather than withdrawing it. Denominator: entry opportunities on
        clean bars. A CAUTION therefore counts as an intervention here even
        though ``blocked_entry_opportunities`` does not count it, so the two
        rates are not complements; read them together.

        Caveat: "clean" means unmodified by the harness, not verified fault-free.
        """
        value = (
            clean_entry_interventions / clean_entry_opportunities
            if clean_entry_opportunities
            else None
        )
        self.add(MetricRecord(
            name=f"clean_entry_intervention_rate_{arm}",
            value=value,
            numerator=clean_entry_interventions,
            denominator=clean_entry_opportunities,
            unit="ratio",
            time_basis="pooled_bar",
            aggregation="pooled_bars",
            exclusions=["injected_fault_intervals"],
            caveat=(
                "CAUTION counts as an intervention; clean means unmodified by the "
                "harness, not verified fault-free"
            ),
            timestamp="",
        ))
        return self.records[-1]

    def all(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.records]


__all__ = ["MetricRecord", "Metrics"]
