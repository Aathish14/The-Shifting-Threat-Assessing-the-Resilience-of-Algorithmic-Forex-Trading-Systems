"""Policy fusion and comparison arms.

Implements the frozen precedence policy and the comparison arms:
  - no_gate: intents pass through regardless of integrity signals
  - integrity_only: quote-integrity gate can abstain from entry opportunities
  - shift_only: shift detector controls hypothetical entry permission
  - fused: deterministic combination with hard-invalid precedence
  - matched_rate_abstention: seeded, time-stratified abstention matched to the
    fused arm's realized abstention rate (post hoc comparator)
  - shadow: computes/logs signals but never changes the reference action

All arms share one uniform ``decide`` signature::

    decide(intent, eligible, is_entry, integrity, shift_alarm, bar_id, timestamp)

``is_entry`` is an *entry opportunity*: ``eligible`` (warm-up complete) AND the
intent is ``ENTER_LONG`` or ``ENTER_SHORT``. ``eligible`` alone only means the
EMAs are warm, so gating on it inflated every intervention count in v1.0.x.

When ``PolicyConfig.entry_only_gate`` is True (the default) a bar that is not
an entry opportunity always returns ``ALLOW_OR_IDLE`` -- regardless of monitor
state -- while still recording hard/soft/shift evidence on the decision for
logging. The reference strategy still sees the same intent stream: a policy arm
may change only entry permission/response, never EMA history.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable


# Policy response labels
RESP_REJECT = "REJECT"
RESP_PAUSE = "PAUSE"
RESP_DELAY = "DELAY"
RESP_CAUTION = "CAUTION"
RESP_ALLOW = "ALLOW"
RESP_ALLOW_OR_IDLE = "ALLOW_OR_IDLE"
RESP_UNKNOWN = "UNKNOWN"

# Responses that withdraw (block) an entry outright.
BLOCKING_RESPONSES: frozenset[str] = frozenset({RESP_REJECT, RESP_PAUSE, RESP_DELAY})
# Responses that modify (but do not withdraw) an entry.
MODIFYING_RESPONSES: frozenset[str] = frozenset({RESP_CAUTION})
# Responses that do not intervene at all.
PASSIVE_RESPONSES: frozenset[str] = frozenset({RESP_ALLOW, RESP_ALLOW_OR_IDLE})

INTENT_ENTRY: frozenset[str] = frozenset({"ENTER_LONG", "ENTER_SHORT"})


@dataclass
class PolicyDecision:
    """Row-local policy decision for one bar."""

    bar_id: int
    timestamp: str
    intent: str  # ENTER_LONG | ENTER_SHORT | NO_ENTRY
    eligible: bool  # strategy warm-up complete
    response: str  # REJECT | PAUSE | DELAY | CAUTION | ALLOW | ALLOW_OR_IDLE | UNKNOWN
    multiplier: float  # proposed risk multiplier (research metadata only)
    reason: str
    arm: str
    is_entry: bool = False  # eligible AND entry intent
    hard_invalid: bool = False
    soft_reason: bool = False
    shift_alarm: bool = False
    hypothetical_response: str = ""  # shadow: what the fused policy would have done

    def is_intervention(self) -> bool:
        """True when the arm changed the reference action on an entry."""
        return self.is_entry and self.response not in PASSIVE_RESPONSES


@dataclass
class PolicyArmResult:
    """Results for one policy arm across all bars."""

    arm: str
    decisions: list[PolicyDecision] = field(default_factory=list)

    def allowed_entry_opportunities(self) -> int:
        return sum(
            1 for d in self.decisions
            if d.is_entry and d.response == RESP_ALLOW
        )

    def blocked_entry_opportunities(self) -> int:
        return sum(
            1 for d in self.decisions
            if d.is_entry and d.response in BLOCKING_RESPONSES
        )

    def cautioned_entry_opportunities(self) -> int:
        return sum(
            1 for d in self.decisions
            if d.is_entry and d.response in MODIFYING_RESPONSES
        )

    def entry_decisions(self) -> list[PolicyDecision]:
        return [d for d in self.decisions if d.is_entry]

    def interventions(self) -> int:
        """Entry opportunities whose response is not ALLOW (includes CAUTION)."""
        return sum(1 for d in self.decisions if d.is_intervention())

    def action_divergence(self, other: "PolicyArmResult") -> int:
        """Entry decisions whose response differs from *other* (aligned by bar_id)."""
        mine = {d.bar_id: d.response for d in self.decisions if d.is_entry}
        theirs = {d.bar_id: d.response for d in other.decisions if d.is_entry}
        return sum(1 for bar_id, resp in mine.items() if theirs.get(bar_id) != resp)


def _flags(integrity: Any) -> tuple[bool, bool]:
    if integrity is None:
        return False, False
    hard = bool(getattr(integrity, "hard_invalid", False))
    soft = bool(getattr(integrity, "has_soft_reason", False))
    return hard, soft


def _decision(
    *,
    arm: str,
    intent: str,
    eligible: bool,
    is_entry: bool,
    integrity: Any,
    shift_alarm: bool,
    bar_id: int,
    timestamp: str,
    response: str,
    multiplier: float,
    reason: str,
    hypothetical: str = "",
) -> PolicyDecision:
    hard, soft = _flags(integrity)
    return PolicyDecision(
        bar_id=bar_id,
        timestamp=timestamp,
        intent=intent,
        eligible=eligible,
        response=response,
        multiplier=multiplier,
        reason=reason,
        arm=arm,
        is_entry=is_entry,
        hard_invalid=hard,
        soft_reason=soft,
        shift_alarm=shift_alarm,
        hypothetical_response=hypothetical,
    )


def _idle_response(
    arm: str,
    intent: str,
    eligible: bool,
    integrity: Any,
    shift_alarm: bool,
    bar_id: int,
    timestamp: str,
    reason: str,
) -> PolicyDecision:
    """Non-entry opportunity: nothing to gate, evidence still recorded."""
    return _decision(
        arm=arm, intent=intent, eligible=eligible, is_entry=False,
        integrity=integrity, shift_alarm=shift_alarm, bar_id=bar_id,
        timestamp=timestamp, response=RESP_ALLOW_OR_IDLE, multiplier=1.0,
        reason=reason,
    )


class FusedPolicy:
    """Deterministic fused policy with the documented precedence:

    1. hard invalidity: REJECT, multiplier 0
    2. soft reason + shift alert: PAUSE, multiplier 0
    3. soft reason only: DELAY, multiplier 0
    4. shift alert only: CAUTION, configured multiplier
    5. monitor OK + entry: ALLOW, multiplier 1
    6. monitor OK + no entry: ALLOW_OR_IDLE, multiplier 1

    ``PolicyConfig.disabled_rules`` drops a rule and lets the next one apply
    (the ablation switches). ``PolicyConfig.enable_shift = False`` removes the
    shift term from rules 2 and 4.
    """

    ARM = "fused"
    RULE_ORDER = (
        "hard_reject",
        "soft_plus_shift_pause",
        "soft_only_delay",
        "shift_only_caution",
    )

    def __init__(self, config) -> None:
        self.config = config
        self.caution_multiplier = config.caution_multiplier
        self.entry_only_gate = config.entry_only_gate
        self.disabled_rules = frozenset(config.disabled_rules)
        self.enable_shift = config.enable_shift

    def decide(
        self,
        intent: str,
        eligible: bool,
        is_entry: bool,
        integrity: Any,
        shift_alarm: bool,
        bar_id: int = 0,
        timestamp: str = "",
    ) -> PolicyDecision:
        hard, soft = _flags(integrity)
        is_entry = bool(is_entry and eligible)
        shift = bool(shift_alarm) and self.enable_shift

        if self.entry_only_gate and not is_entry:
            return _idle_response(
                self.ARM, intent, eligible, integrity, shift, bar_id, timestamp,
                "allow_or_idle",
            )

        rule_hits = {
            "hard_reject": hard,
            "soft_plus_shift_pause": soft and shift,
            "soft_only_delay": soft,
            "shift_only_caution": shift,
        }
        for rule in self.RULE_ORDER:
            if rule in self.disabled_rules or not rule_hits[rule]:
                continue
            if rule == "hard_reject":
                return _decision(
                    arm=self.ARM, intent=intent, eligible=eligible, is_entry=is_entry,
                    integrity=integrity, shift_alarm=shift, bar_id=bar_id,
                    timestamp=timestamp, response=RESP_REJECT, multiplier=0.0,
                    reason="hard_invalid_reject",
                )
            if rule == "soft_plus_shift_pause":
                return _decision(
                    arm=self.ARM, intent=intent, eligible=eligible, is_entry=is_entry,
                    integrity=integrity, shift_alarm=shift, bar_id=bar_id,
                    timestamp=timestamp, response=RESP_PAUSE, multiplier=0.0,
                    reason="soft_reason_plus_shift_pause",
                )
            if rule == "soft_only_delay":
                return _decision(
                    arm=self.ARM, intent=intent, eligible=eligible, is_entry=is_entry,
                    integrity=integrity, shift_alarm=shift, bar_id=bar_id,
                    timestamp=timestamp, response=RESP_DELAY, multiplier=0.0,
                    reason="soft_reason_delay",
                )
            return _decision(
                arm=self.ARM, intent=intent, eligible=eligible, is_entry=is_entry,
                integrity=integrity, shift_alarm=shift, bar_id=bar_id,
                timestamp=timestamp, response=RESP_CAUTION,
                multiplier=self.caution_multiplier,
                reason="shift_alert_caution",
            )

        if is_entry:
            return _decision(
                arm=self.ARM, intent=intent, eligible=eligible, is_entry=True,
                integrity=integrity, shift_alarm=shift, bar_id=bar_id,
                timestamp=timestamp, response=RESP_ALLOW, multiplier=1.0,
                reason="allow_entry",
            )
        return _decision(
            arm=self.ARM, intent=intent, eligible=eligible, is_entry=False,
            integrity=integrity, shift_alarm=shift, bar_id=bar_id,
            timestamp=timestamp, response=RESP_ALLOW_OR_IDLE, multiplier=1.0,
            reason="allow_or_idle",
        )


class NoGateArm:
    """Intents pass through regardless of integrity/shift signals."""

    ARM = "no_gate"

    def decide(self, intent, eligible, is_entry, integrity=None, shift_alarm=False,
               bar_id=0, timestamp="") -> PolicyDecision:
        is_entry = bool(is_entry and eligible)
        return _decision(
            arm=self.ARM, intent=intent, eligible=eligible, is_entry=is_entry,
            integrity=integrity, shift_alarm=bool(shift_alarm), bar_id=bar_id,
            timestamp=timestamp,
            response=RESP_ALLOW if is_entry else RESP_ALLOW_OR_IDLE,
            multiplier=1.0,
            reason="no_gate_allow" if is_entry else "no_gate_idle",
        )


class IntegrityOnlyArm:
    """Quote-integrity gate can abstain from entry opportunities."""

    ARM = "integrity_only"

    def __init__(self, config) -> None:
        self.config = config
        self.entry_only_gate = config.entry_only_gate

    def decide(self, intent, eligible, is_entry, integrity=None, shift_alarm=False,
               bar_id=0, timestamp="") -> PolicyDecision:
        hard, _ = _flags(integrity)
        is_entry = bool(is_entry and eligible)
        if self.entry_only_gate and not is_entry:
            return _idle_response(
                self.ARM, intent, eligible, integrity, bool(shift_alarm),
                bar_id, timestamp, "integrity_idle",
            )
        if hard:
            return _decision(
                arm=self.ARM, intent=intent, eligible=eligible, is_entry=is_entry,
                integrity=integrity, shift_alarm=bool(shift_alarm), bar_id=bar_id,
                timestamp=timestamp, response=RESP_REJECT, multiplier=0.0,
                reason="integrity_hard_invalid_reject",
            )
        return _decision(
            arm=self.ARM, intent=intent, eligible=eligible, is_entry=is_entry,
            integrity=integrity, shift_alarm=bool(shift_alarm), bar_id=bar_id,
            timestamp=timestamp,
            response=RESP_ALLOW if is_entry else RESP_ALLOW_OR_IDLE,
            multiplier=1.0,
            reason="integrity_allow" if is_entry else "integrity_idle",
        )


class ShiftOnlyArm:
    """Shift detector controls hypothetical entry permission only."""

    ARM = "shift_only"

    def __init__(self, config) -> None:
        self.config = config
        self.caution_multiplier = config.caution_multiplier
        self.entry_only_gate = config.entry_only_gate
        self.enable_shift = config.enable_shift

    def decide(self, intent, eligible, is_entry, integrity=None, shift_alarm=False,
               bar_id=0, timestamp="") -> PolicyDecision:
        shift = bool(shift_alarm) and self.enable_shift
        is_entry = bool(is_entry and eligible)
        if self.entry_only_gate and not is_entry:
            return _idle_response(
                self.ARM, intent, eligible, integrity, shift,
                bar_id, timestamp, "shift_only_idle",
            )
        if shift:
            return _decision(
                arm=self.ARM, intent=intent, eligible=eligible, is_entry=is_entry,
                integrity=integrity, shift_alarm=True, bar_id=bar_id,
                timestamp=timestamp, response=RESP_CAUTION,
                multiplier=self.caution_multiplier,
                reason="shift_only_caution",
            )
        return _decision(
            arm=self.ARM, intent=intent, eligible=eligible, is_entry=is_entry,
            integrity=integrity, shift_alarm=False, bar_id=bar_id,
            timestamp=timestamp,
            response=RESP_ALLOW if is_entry else RESP_ALLOW_OR_IDLE,
            multiplier=1.0,
            reason="shift_only_allow" if is_entry else "shift_only_idle",
        )


class MatchedRateAbstentionArm:
    """Seeded, time-stratified abstention matched to the fused gate's rate.

    The rate is measured **after** the fused arm runs on the test range (post
    hoc comparator, labelled as such). ``set_plan`` converts the rate plus the
    ordered entry-opportunity bar IDs into a concrete abstention set: the
    opportunities are cut into consecutive strata of
    ``matched_rate_stratum_size`` and ``round(rate * n)`` in each stratum are
    selected without replacement from a single seeded RNG.
    """

    ARM = "matched_rate_abstention"

    def __init__(self, config, abstention_rate: float = 0.0) -> None:
        self.config = config
        self.entry_only_gate = config.entry_only_gate
        self.abstention_rate = float(abstention_rate)
        self._seed = config.matched_rate_seed
        self._stratum_size = config.matched_rate_stratum_size
        self.abstain_bar_ids: frozenset[int] = frozenset()
        self.planned_abstentions = 0
        self.planned_opportunities = 0

    def set_plan(self, entry_bar_ids: Iterable[int], abstention_rate: float) -> None:
        """Precompute the abstention set from ordered entry-opportunity bar IDs.

        Abstentions are allocated across consecutive strata of
        ``matched_rate_stratum_size`` by cumulative rounding: each stratum's
        expected count is accumulated and its integer part is taken. This keeps
        the comparator time-stratified **and** makes the realized count equal
        ``floor(rate * opportunities)`` instead of drifting to zero whenever the
        per-stratum expectation is below one (a 0.8% rate over strata of 50
        rounds to zero in every stratum, which silently disabled the
        comparator in the smoke run).
        """
        self.abstention_rate = float(abstention_rate)
        ordered = list(entry_bar_ids)
        self.planned_opportunities = len(ordered)
        rng = random.Random(self._seed)
        chosen: set[int] = set()
        carry = 0.0
        for start in range(0, len(ordered), self._stratum_size):
            stratum = ordered[start:start + self._stratum_size]
            carry += self.abstention_rate * len(stratum)
            n_take = int(math.floor(carry + 1e-9))
            carry -= n_take
            n_take = max(0, min(len(stratum), n_take))
            if n_take:
                chosen.update(rng.sample(stratum, n_take))
        self.abstain_bar_ids = frozenset(chosen)
        self.planned_abstentions = len(chosen)

    def decide(self, intent, eligible, is_entry, integrity=None, shift_alarm=False,
               bar_id=0, timestamp="") -> PolicyDecision:
        is_entry = bool(is_entry and eligible)
        if self.entry_only_gate and not is_entry:
            return _idle_response(
                self.ARM, intent, eligible, integrity, bool(shift_alarm),
                bar_id, timestamp, "matched_rate_idle",
            )
        if is_entry and bar_id in self.abstain_bar_ids:
            return _decision(
                arm=self.ARM, intent=intent, eligible=eligible, is_entry=True,
                integrity=integrity, shift_alarm=bool(shift_alarm), bar_id=bar_id,
                timestamp=timestamp, response=RESP_REJECT, multiplier=0.0,
                reason="matched_rate_abstain",
            )
        return _decision(
            arm=self.ARM, intent=intent, eligible=eligible, is_entry=is_entry,
            integrity=integrity, shift_alarm=bool(shift_alarm), bar_id=bar_id,
            timestamp=timestamp,
            response=RESP_ALLOW if is_entry else RESP_ALLOW_OR_IDLE,
            multiplier=1.0,
            reason="matched_rate_allow" if is_entry else "matched_rate_idle",
        )


class ShadowMode:
    """Computes and logs all signals but never changes the reference action."""

    ARM = "shadow"

    def __init__(self, config) -> None:
        self.config = config
        self.entry_only_gate = config.entry_only_gate
        self.fused = FusedPolicy(config)

    def decide(self, intent, eligible, is_entry, integrity=None, shift_alarm=False,
               bar_id=0, timestamp="") -> PolicyDecision:
        is_entry = bool(is_entry and eligible)
        hypothetical = self.fused.decide(
            intent, eligible, is_entry, integrity, shift_alarm, bar_id, timestamp
        )
        if self.entry_only_gate and not is_entry:
            return _decision(
                arm=self.ARM, intent=intent, eligible=eligible, is_entry=False,
                integrity=integrity, shift_alarm=bool(shift_alarm), bar_id=bar_id,
                timestamp=timestamp, response=RESP_ALLOW_OR_IDLE, multiplier=1.0,
                reason=f"shadow_hypothetical_{hypothetical.response}",
                hypothetical=hypothetical.response,
            )
        return _decision(
            arm=self.ARM, intent=intent, eligible=eligible, is_entry=is_entry,
            integrity=integrity, shift_alarm=bool(shift_alarm), bar_id=bar_id,
            timestamp=timestamp,
            response=RESP_ALLOW if is_entry else RESP_ALLOW_OR_IDLE,
            multiplier=1.0,
            reason=f"shadow_hypothetical_{hypothetical.response}",
            hypothetical=hypothetical.response,
        )


ARM_CLASSES = {
    "no_gate": NoGateArm,
    "integrity_only": IntegrityOnlyArm,
    "shift_only": ShiftOnlyArm,
    "fused": FusedPolicy,
    "matched_rate_abstention": MatchedRateAbstentionArm,
    "shadow": ShadowMode,
}


def build_arm(name: str, config, abstention_rate: float = 0.0):
    """Build exactly the requested policy arm (not all of them)."""
    cls = ARM_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"unknown policy arm: {name}")
    if name == "matched_rate_abstention":
        return MatchedRateAbstentionArm(config, abstention_rate=abstention_rate)
    if name == "no_gate":
        return NoGateArm()
    return cls(config)


__all__ = [
    "ARM_CLASSES",
    "BLOCKING_RESPONSES",
    "FusedPolicy",
    "INTENT_ENTRY",
    "IntegrityOnlyArm",
    "MODIFYING_RESPONSES",
    "MatchedRateAbstentionArm",
    "NoGateArm",
    "PASSIVE_RESPONSES",
    "PolicyArmResult",
    "PolicyDecision",
    "RESP_ALLOW",
    "RESP_ALLOW_OR_IDLE",
    "RESP_CAUTION",
    "RESP_DELAY",
    "RESP_PAUSE",
    "RESP_REJECT",
    "RESP_UNKNOWN",
    "ShadowMode",
    "ShiftOnlyArm",
    "build_arm",
]
