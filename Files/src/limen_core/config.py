"""Validated, frozen run configuration for Limen Core.

All scientific parameters (strategy, splits, monitor thresholds, scenarios,
policy arms) are declared here and frozen before any test-period evaluation.
The configuration is versioned and hashed for run reproducibility.

Rules enforced here:
  * every parameter has a single declared default (no magic numbers in logic);
  * every section validates its own invariants in ``__post_init__``;
  * ``from_dict`` is strict: an unknown key is a ``ConfigError`` rather than a
    silently ignored typo, so a newly added section can never be dropped on
    the floor by an older loader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when configuration validation fails."""


# Canonical names: policy arms and fusion rules are referenced by name in
# configuration, ablation groups and audit output. Declaring them here keeps a
# single source of truth and lets validation reject typos.
POLICY_ARMS: tuple[str, ...] = (
    "no_gate",
    "integrity_only",
    "shift_only",
    "fused",
    "matched_rate_abstention",
    "shadow",
)

FUSION_RULES: tuple[str, ...] = (
    "hard_reject",
    "soft_plus_shift_pause",
    "soft_only_delay",
    "shift_only_caution",
)

# Ablation groups that disable exactly one signal source (see ablation.py).
ABLATION_SIGNAL_GROUPS: tuple[str, ...] = (
    "hard_checks",
    "spread_outlier",
    "price_innovation",
    "repeated_quote",
    "inter_arrival_gap",
    "stale_quote",
    "timestamp_order",
    "shift_monitor",
)

# Top-level keys that are *annotations* written by the human owner of a frozen
# config (provenance), not runtime parameters. They are tolerated on load and
# never forwarded into runtime sections.
ANNOTATION_KEYS: frozenset[str] = frozenset({
    "split_metadata",
    "frozen_at",
    "frozen_by",
    "holdout_status",
    "holdout_selection",
    "holdout_unseen",
    "data_rights_confirmed",
    "date_range_confirmed",
})


@dataclass(frozen=True)
class StrategyConfig:
    """Frozen EMA(10, 30) reference strategy parameters."""

    fast_span: int = 10
    slow_span: int = 30
    price_field: str = "midpoint"  # canonical price input
    min_periods_fast: int = 10
    min_periods_slow: int = 30
    adjust: bool = False
    equality_tol: float = 0.0  # exact equality rule for crossover ties
    intent_actionable_at_close: bool = True  # intent at bar t close actionable at t

    def __post_init__(self) -> None:
        if self.fast_span <= 0 or self.slow_span <= 0:
            raise ConfigError("EMA spans must be positive integers")
        if self.fast_span >= self.slow_span:
            raise ConfigError("fast_span must be strictly less than slow_span")
        if self.min_periods_fast < 1 or self.min_periods_slow < 1:
            raise ConfigError("min_periods must be >= 1")
        if self.min_periods_fast > self.fast_span:
            raise ConfigError("min_periods_fast must not exceed fast_span")
        if self.min_periods_slow > self.slow_span:
            raise ConfigError("min_periods_slow must not exceed slow_span")
        if self.price_field not in ("midpoint", "bid", "ask"):
            raise ConfigError("price_field must be one of midpoint|bid|ask")
        if self.equality_tol < 0:
            raise ConfigError("equality_tol must be >= 0")


@dataclass(frozen=True)
class SplitConfig:
    """Chronological train/validation/test split by observed-bar index."""

    train_end_index: int = 0  # exclusive end index of training bars
    validation_end_index: int = 0  # exclusive end index of validation bars
    test_end_index: int | None = None  # None means to end of data
    purge_bars: int = 0  # bars excluded around boundary to avoid leakage
    allow_uncalibrated: bool = False  # explicit opt-in for split-free smoke runs

    def __post_init__(self) -> None:
        if self.train_end_index < 0:
            raise ConfigError("train_end_index must be >= 0")
        if self.validation_end_index < self.train_end_index:
            raise ConfigError("validation_end_index must be >= train_end_index")
        if self.test_end_index is not None and self.test_end_index < self.validation_end_index:
            raise ConfigError("test_end_index must be >= validation_end_index")
        if self.purge_bars < 0:
            raise ConfigError("purge_bars must be >= 0")


@dataclass(frozen=True)
class BarConfig:
    """Observed-minute bar construction parameters."""

    gap_seconds: float = 60.0  # inter-bar elapsed time that flags a session gap

    def __post_init__(self) -> None:
        if self.gap_seconds <= 0:
            raise ConfigError("bar gap_seconds must be > 0")


@dataclass(frozen=True)
class IntegrityConfig:
    """Quote-integrity monitor thresholds fitted on training data only."""

    spread_quantile: float = 0.999
    innovation_quantile: float = 0.999
    min_gap_seconds: float = 300.0
    repeat_count_threshold: int = 5  # consecutive repeated observed bars
    enable_spread_outlier: bool = True
    enable_price_innovation: bool = True
    enable_repeated_quote: bool = True
    enable_gap_check: bool = True
    enable_crossed: bool = True
    enable_nonpositive: bool = True
    enable_missing: bool = True
    stale_quote_seconds: float = 300.0  # seconds a quote side may be stale before it's flagged
    enable_stale_quote: bool = True  # whether to check for stale quotes
    stale_is_hard: bool = True  # stale evidence marks the observation INVALID
    enable_timestamp_order: bool = True  # hard-invalid on backward event timestamps

    def __post_init__(self) -> None:
        for name, q in (
            ("spread_quantile", self.spread_quantile),
            ("innovation_quantile", self.innovation_quantile),
        ):
            if not 0.0 < q < 1.0:
                raise ConfigError(f"{name} must be in (0, 1)")
        if self.min_gap_seconds <= 0:
            raise ConfigError("min_gap_seconds must be > 0")
        if self.repeat_count_threshold < 2:
            raise ConfigError("repeat_count_threshold must be >= 2")
        if self.stale_quote_seconds <= 0:
            raise ConfigError("stale_quote_seconds must be > 0")


@dataclass(frozen=True)
class KSConfig:
    """Two-sample KS shift monitor parameters."""

    reference_window: int = 120
    recent_window: int = 30
    check_cadence: int = 5  # score every N bars
    consecutive_alerts: int = 2
    quantile: float = 0.99  # validation 99th percentile threshold
    clip_abs_return: float | None = None  # clipping limit fitted on training

    def __post_init__(self) -> None:
        if self.reference_window < 2:
            raise ConfigError("ks.reference_window must be >= 2")
        if self.recent_window < 2:
            raise ConfigError("ks.recent_window must be >= 2")
        if self.check_cadence < 1:
            raise ConfigError("ks.check_cadence must be >= 1")
        if self.consecutive_alerts < 1:
            raise ConfigError("ks.consecutive_alerts must be >= 1")
        if not 0.0 < self.quantile < 1.0:
            raise ConfigError("ks.quantile must be in (0, 1)")
        if self.clip_abs_return is not None and self.clip_abs_return <= 0:
            raise ConfigError("ks.clip_abs_return must be > 0 when set")


@dataclass(frozen=True)
class CUSUMConfig:
    """CUSUM shift monitor parameters."""

    allowance: float = 0.5
    decision_limit: float = 5.0
    reset_on_gap: bool = True
    reset_on_missing: bool = True

    def __post_init__(self) -> None:
        if self.allowance < 0:
            raise ConfigError("cusum.allowance must be >= 0")
        if self.decision_limit <= 0:
            raise ConfigError("cusum.decision_limit must be > 0")


@dataclass(frozen=True)
class QuantileMonitorConfig:
    """Quantile shift monitor parameters."""

    quantile_level: float = 0.95
    reference_window: int = 120
    recent_window: int = 30

    def __post_init__(self) -> None:
        if not 0.0 < self.quantile_level < 1.0:
            raise ConfigError("quantile.quantile_level must be in (0, 1)")
        if self.reference_window < 2 or self.recent_window < 2:
            raise ConfigError("quantile windows must be >= 2")


@dataclass(frozen=True)
class MADConfig:
    """Robust MAD shift monitor parameters."""

    reference_window: int = 120
    recent_window: int = 30
    mad_scale: float = 3.0  # standardized MAD threshold
    zero_mad_fallback: str = "unknown"  # "unknown" | "inf" | "zero"

    def __post_init__(self) -> None:
        if self.reference_window < 2 or self.recent_window < 2:
            raise ConfigError("mad windows must be >= 2")
        if self.mad_scale <= 0:
            raise ConfigError("mad.mad_scale must be > 0")
        if self.zero_mad_fallback not in ("unknown", "inf", "zero"):
            raise ConfigError("mad.zero_mad_fallback must be unknown|inf|zero")


@dataclass(frozen=True)
class ScenarioConfig:
    """Deterministic synthetic fault / control scenario parameters."""

    seed: int = 12345
    missing_quote_duration_bars: int = 3
    stale_replay_duration_bars: int = 3
    crossed_quote_duration_bars: int = 1
    symmetric_shift_duration_bars: int = 3
    return_scale_duration_bars: int = 5
    benign_volatility_duration_bars: int = 5
    combined_duration_bars: int = 3
    symmetric_shift_magnitude: float = 0.0005  # additive price shift
    return_scale_factor: float = 2.0  # synthetic return-scale shift
    benign_volatility_multiplier: float = 2.0
    episodes_per_fault: int = 8
    recovery_window_bars: int = 30
    placement: str = "uniform"  # "uniform" | "entry_anchored"
    min_faulted_entries: int = 30  # below this a per-class delta is noise
    label_isolation: bool = True  # runtime must not see labels

    def __post_init__(self) -> None:
        for name in (
            "missing_quote_duration_bars",
            "stale_replay_duration_bars",
            "crossed_quote_duration_bars",
            "symmetric_shift_duration_bars",
            "return_scale_duration_bars",
            "benign_volatility_duration_bars",
            "combined_duration_bars",
        ):
            if getattr(self, name) < 1:
                raise ConfigError(f"scenario.{name} must be >= 1")
        if self.episodes_per_fault < 0:
            raise ConfigError("scenario.episodes_per_fault must be >= 0")
        if self.recovery_window_bars < 0:
            raise ConfigError("scenario.recovery_window_bars must be >= 0")
        if self.symmetric_shift_magnitude <= 0:
            raise ConfigError("scenario.symmetric_shift_magnitude must be > 0")
        if self.return_scale_factor <= 0:
            raise ConfigError("scenario.return_scale_factor must be > 0")
        if self.benign_volatility_multiplier < 1.0:
            raise ConfigError("scenario.benign_volatility_multiplier must be >= 1")
        if self.placement not in ("uniform", "entry_anchored"):
            raise ConfigError("scenario.placement must be uniform|entry_anchored")
        if self.min_faulted_entries < 0:
            raise ConfigError("scenario.min_faulted_entries must be >= 0")


@dataclass(frozen=True)
class PolicyConfig:
    """Policy arm enablement and fused-policy parameters."""

    arms: tuple[str, ...] = POLICY_ARMS
    caution_multiplier: float = 0.5  # shift-only multiplier
    matched_rate_seed: int = 999
    matched_rate_stratum_size: int = 50  # time-stratified matched-rate blocks
    entry_only_gate: bool = True  # count intervention only on entry opportunities
    require_aligned_intent: bool = True  # all arms see identical intent stream
    enable_shift: bool = True  # master switch for the shift monitor in fusion
    disabled_rules: tuple[str, ...] = ()  # fused rules skipped (ablation switches)

    def __post_init__(self) -> None:
        if not self.arms:
            raise ConfigError("policy.arms must not be empty")
        for arm in self.arms:
            if arm not in POLICY_ARMS:
                raise ConfigError(f"unknown policy arm: {arm!r}")
        if not 0.0 <= self.caution_multiplier <= 1.0:
            raise ConfigError("policy.caution_multiplier must be in [0, 1]")
        if self.matched_rate_stratum_size < 1:
            raise ConfigError("policy.matched_rate_stratum_size must be >= 1")
        for rule in self.disabled_rules:
            if rule not in FUSION_RULES:
                raise ConfigError(f"unknown fusion rule: {rule!r}")


@dataclass(frozen=True)
class AblationConfig:
    """Ablation study groups (Phase 6)."""

    signal_groups: tuple[str, ...] = ABLATION_SIGNAL_GROUPS
    fusion_rules: tuple[str, ...] = FUSION_RULES
    min_faulted_entries: int = 30

    def __post_init__(self) -> None:
        for group in self.signal_groups:
            if group not in ABLATION_SIGNAL_GROUPS:
                raise ConfigError(f"unknown ablation group: {group!r}")
        for rule in self.fusion_rules:
            if rule not in FUSION_RULES:
                raise ConfigError(f"unknown fusion rule: {rule!r}")
        if self.min_faulted_entries < 0:
            raise ConfigError("ablation.min_faulted_entries must be >= 0")


@dataclass(frozen=True)
class SensitivityConfig:
    """Sensitivity sweep values (Phase 7). Each tuple is swept one at a time."""

    spread_quantile: tuple[float, ...] = (0.99, 0.999, 0.9999)
    innovation_quantile: tuple[float, ...] = (0.99, 0.999, 0.9999)
    repeat_count_threshold: tuple[int, ...] = (3, 5, 8)
    min_gap_seconds: tuple[float, ...] = (120.0, 300.0, 600.0)
    stale_quote_seconds: tuple[float, ...] = (60.0, 300.0, 900.0)
    ks_reference_window: tuple[int, ...] = (90, 120, 180)
    ks_recent_window: tuple[int, ...] = (20, 30, 40)
    ks_check_cadence: tuple[int, ...] = (3, 5, 10)
    ks_consecutive_alerts: tuple[int, ...] = (1, 2, 3)
    ks_quantile: tuple[float, ...] = (0.95, 0.99, 0.999)
    scenario_missing_quote_duration_bars: tuple[int, ...] = (1, 3, 5)
    scenario_stale_replay_duration_bars: tuple[int, ...] = (3, 10, 30)
    scenario_symmetric_shift_magnitude: tuple[float, ...] = (0.0002, 0.0005, 0.001)
    scenario_return_scale_factor: tuple[float, ...] = (1.5, 2.0, 3.0)

    def __post_init__(self) -> None:
        for name in self.sweep_names():
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) < 2:
                raise ConfigError(f"sensitivity.{name} must be a tuple of >= 2 values")

    @staticmethod
    def sweep_names() -> tuple[str, ...]:
        return (
            "spread_quantile",
            "innovation_quantile",
            "repeat_count_threshold",
            "min_gap_seconds",
            "stale_quote_seconds",
            "ks_reference_window",
            "ks_recent_window",
            "ks_check_cadence",
            "ks_consecutive_alerts",
            "ks_quantile",
            "scenario_missing_quote_duration_bars",
            "scenario_stale_replay_duration_bars",
            "scenario_symmetric_shift_magnitude",
            "scenario_return_scale_factor",
        )


@dataclass(frozen=True)
class RuntimeConfig:
    """Run-level settings and paths.

    Paths are stored with forward slashes so a frozen config is portable across
    Windows and POSIX checkouts. Relative ``input_path`` values are resolved
    against the ``Files/`` directory by the CLI.
    """

    input_path: str = "../EURUSD_202603050000_202610262359.csv"
    max_rows: int | None = None  # None means process all
    streaming_chunk_rows: int = 100_000
    run_id: str | None = None
    output_root: str = "Files/artifacts/runs"
    shadow_mode: bool = False
    fail_on_label_access: bool = True

    def __post_init__(self) -> None:
        if self.max_rows is not None and self.max_rows <= 0:
            raise ConfigError("runtime.max_rows must be > 0 when set")
        if self.streaming_chunk_rows <= 0:
            raise ConfigError("runtime.streaming_chunk_rows must be > 0")
        if not self.input_path:
            raise ConfigError("runtime.input_path must not be empty")
        if not self.output_root:
            raise ConfigError("runtime.output_root must not be empty")


@dataclass(frozen=True)
class LimenConfig:
    """Top-level frozen configuration container."""

    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    bar: BarConfig = field(default_factory=BarConfig)
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)
    ks: KSConfig = field(default_factory=KSConfig)
    cusum: CUSUMConfig = field(default_factory=CUSUMConfig)
    quantile: QuantileMonitorConfig = field(default_factory=QuantileMonitorConfig)
    mad: MADConfig = field(default_factory=MADConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    version: str = "1.1.1"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (for JSON output and hashing)."""
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LimenConfig":
        """Build a LimenConfig from a dict, rejecting unknown keys.

        Section dicts are strict: a misspelled or newly introduced key raises
        ``ConfigError`` instead of being silently discarded. Top-level
        annotation keys (provenance only) are tolerated and dropped.
        """
        if not isinstance(data, dict):
            raise ConfigError(
                f"expected dict at top level, got {type(data).__name__}"
            )
        sections = {f.name for f in fields(cls) if f.name != "version"}
        unknown = set(data) - sections - {"version"} - ANNOTATION_KEYS
        if unknown:
            raise ConfigError(f"unknown top-level config keys: {sorted(unknown)}")
        return cls(
            strategy=_build(StrategyConfig, data.get("strategy", {})),
            split=_build(SplitConfig, data.get("split", {})),
            bar=_build(BarConfig, data.get("bar", {})),
            integrity=_build(IntegrityConfig, data.get("integrity", {})),
            ks=_build(KSConfig, data.get("ks", {})),
            cusum=_build(CUSUMConfig, data.get("cusum", {})),
            quantile=_build(QuantileMonitorConfig, data.get("quantile", {})),
            mad=_build(MADConfig, data.get("mad", {})),
            scenario=_build(ScenarioConfig, data.get("scenario", {})),
            policy=_build(PolicyConfig, data.get("policy", {})),
            ablation=_build(AblationConfig, data.get("ablation", {})),
            sensitivity=_build(SensitivityConfig, data.get("sensitivity", {})),
            runtime=_build(RuntimeConfig, data.get("runtime", {})),
            version=str(data.get("version", "1.1.1")),
        )

    def validate_for_run(self) -> None:
        """Reject configurations that cannot produce a calibrated evaluation."""
        split = self.split
        if split.purge_bars != 0:
            raise ConfigError(
                "split.purge_bars is not implemented; refusing to ignore it "
                f"(got {split.purge_bars})"
            )
        if split.allow_uncalibrated:
            return
        if split.train_end_index == 0:
            raise ConfigError(
                "split.train_end_index is 0 (no training range); set a real "
                "split or allow_uncalibrated=True"
            )
        if split.validation_end_index == split.train_end_index:
            raise ConfigError(
                "split.validation_end_index equals train_end_index (empty "
                "validation range); set a real split or allow_uncalibrated=True"
            )

    def save_json(self, path: str | Path) -> None:
        """Atomically write the frozen config as JSON with sorted keys."""
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)


def _build(dc_type: type, data_section: Any):
    """Instantiate ``dc_type`` from ``data_section``, rejecting unknown keys."""
    if not isinstance(data_section, dict):
        raise ConfigError(
            f"expected dict for {dc_type.__name__}, "
            f"got {type(data_section).__name__}"
        )
    declared = {f.name: f for f in fields(dc_type)}
    unknown = set(data_section) - set(declared)
    if unknown:
        raise ConfigError(
            f"unknown keys in section {dc_type.__name__}: {sorted(unknown)}"
        )
    init_kwargs: dict[str, Any] = {}
    for name, f in declared.items():
        if name in data_section:
            value = data_section[name]
            # JSON has no tuple type: convert list -> tuple for tuple defaults
            # (policy.arms, disabled_rules, sensitivity sweep tuples).
            if isinstance(f.default, tuple) and isinstance(value, list):
                value = tuple(value)
            init_kwargs[name] = value
    return dc_type(**init_kwargs)


def _dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass_instance(obj):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def is_dataclass_instance(obj: Any) -> bool:
    return hasattr(obj, "__dataclass_fields__")


def load_config(path: str | Path) -> LimenConfig:
    """Load and validate a LimenConfig from a JSON file."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return LimenConfig.from_dict(data)


def example_config() -> LimenConfig:
    """Return the default frozen example configuration."""
    return LimenConfig()


__all__ = [
    "ABLATION_SIGNAL_GROUPS",
    "ANNOTATION_KEYS",
    "AblationConfig",
    "BarConfig",
    "CUSUMConfig",
    "ConfigError",
    "FUSION_RULES",
    "IntegrityConfig",
    "KSConfig",
    "LimenConfig",
    "MADConfig",
    "POLICY_ARMS",
    "PolicyConfig",
    "QuantileMonitorConfig",
    "RuntimeConfig",
    "ScenarioConfig",
    "SensitivityConfig",
    "SplitConfig",
    "StrategyConfig",
    "example_config",
    "load_config",
]
