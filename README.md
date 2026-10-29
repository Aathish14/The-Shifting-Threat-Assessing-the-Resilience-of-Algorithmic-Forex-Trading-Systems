# Limen Core

*The Shifting Threat: Assessing the Resilience of Algorithmic Forex Trading Systems*

**Status:** Research prototype — offline replay, no broker, no live execution.
**Language/runtime:** Python 3.12, standard library only.
**CLI:** `Files/run_limen.py`

## Overview

Limen Core is an offline, deterministic research replay pipeline for studying how quote-integrity faults and synthetic distribution shifts affect a frozen algorithmic foreign-exchange decision rule. It reads an immutable EUR/USD tick export, reconstructs bid/ask quote state under source-specific FLAGS-bit semantics, builds observed-minute bars without gap filling, computes a causal EMA(10,30) reference intent, runs quote-integrity and distribution-shift monitors, applies synthetic fault scenarios with evaluator-only labels, and scores six policy arms against a common reference-intent stream.

### The problem it studies

Algorithmic forex systems make entry decisions from quote streams. If the stream is unreliable — stale, missing, crossed, non-positive, or shifted — the decision rule sees a distorted picture of the market. Limen Core investigates a narrow question: under controlled quote faults, how often does a frozen EMA(10,30) rule's entry intent change, and does a deterministic integrity gate reduce pass-through to invalid or shifted inputs at an acceptable cost on clean entry opportunities?

### Current maturity

Limen Core is a **research prototype**, not a production trading system. It does not connect to a broker, place orders, hold account credentials, simulate execution, or compute P&L. Results from synthetic fault-injection experiments on one EUR/USD feed are not evidence of profitable, resilient, or deployable trading.

## Key features

| Feature | Description |
|---------|-------------|
| Streaming ingestion | Tab-separated EUR/USD CSV parsed in one pass with bounded memory; source audit (schema, range, ordering, FLAGS counts, parse failures) |
| Quote reconstruction | FLAGS-bit-aware bid/ask carry-forward state machine with explicit reason codes and per-side staleness ages |
| Bar construction | Observed-minute bars only; no empty minutes, gap filling, resampling, or sorting |
| Reference strategy | Causal EMA(10,30) midpoint-crossover intent with explicit warmup, no-lookahead, and action-timing rules |
| Integrity monitor | Hard checks (missing/non-positive/crossed/stale quotes, backward timestamp) and soft checks (spread, innovation, repeat, gap); limits fitted on train data only |
| Drift monitor | Causal KS shift monitor; threshold from validation 99th percentile of causal KS statistics |
| Fault scenarios | Deterministic, seeded, non-overlapping synthetic episodes (missing quote, stale replay, crossed quote, symmetric shift, return-scale shift, combined shift-and-quote, benign volatility control) with evaluator-only labels in a sidecar |
| Policy evaluation | Six arms — no_gate, integrity_only, shift_only, fused, matched_rate_abstention, shadow — sharing one intent stream |
| Experiment tooling | Ablation (one signal/fusion rule at a time) and sensitivity (one parameter at a time) sweeps; run manifests, config digests, and artifact digests |

## Architecture

```mermaid
flowchart TD
    CSV[EUR/USD CSV tick export<br/>immutable, 22.07M rows] --> INGEST[ingest: streaming parse<br/>FLAGS bits, SourceAudit]
    INGEST --> QUOTES[quotes: carry-forward<br/>bid/ask state machine]
    QUOTES --> BARS[bars: observed-minute<br/>construction, no gap fill]
    BARS --> SPLIT[Split 70/10/20 by<br/>observed-bar index]
    SPLIT --> STRAT[strategy: causal EMA(10,30)<br/>reference intent stream]
    STRAT --> INTEGRITY[integrity: hard + soft<br/>limits fitted on train]
    BARS --> KS[drift: KS shift monitor<br/>threshold from validation]
    INTEGRITY --> REPLAY[replay: clean pass +<br/>mutated pass]
    KS --> REPLAY
    STRAT --> REPLAY
    SCEN[scenarios: synthetic episodes,<br/>evaluator-only labels] -.->|labels never reach runtime| REPLAY
    REPLAY --> POLICY[policy: six arms<br/>fused precedence]
    POLICY --> METRICS[metrics: recall, false<br/>alerts, pass-through]
    METRICS --> ARTIFACT[Persisted run artifacts<br/>manifests, CSVs, JSON digests]
```

## Repository structure

| Path | Purpose |
|------|---------|
| `Files/run_limen.py` | CLI entry point: `run`, `audit`, `ablation`, `sensitivity`, `config-example` |
| `Files/src/limen_core/` | Source package (stdlib only): config, ingest, quotes, bars, strategy, integrity, drift/detectors, scenarios, policy, replay, metrics, audit, ablation, experiments, sensitivity |
| `Files/configs/frozen_config_v1_1_1.json` | Frozen configuration, version 1.1.1 |
| `Files/artifacts/runs/confirmatory_v1_1_1_20261028/` | Confirmatory run bundle (manifests, metrics, config) |
| `Files/artifacts/runs/ablation_v1_1_1_20261028b/` | Post-fix ablation bundle |
| `Files/artifacts/runs/sensitivity_v1_1_1_20261028/` | Sensitivity sweep bundle |
| `.gitignore` | Excludes raw dataset, caches, venvs, generated outputs |

## Requirements and setup

- **Python:** 3.12.10 (the only version evidenced; no upper/lower bound declared in code)
- **Dependencies:** None. The code uses only the Python standard library. No `requirements.txt`, `setup.py`, `pyproject.toml`, or `environment.yml` exists in the repository.
- **Shell:** PowerShell, invoked from the project root
- **Virtual environment:** Not required and not used by this project

### Setup

1. Confirm Python 3.12 is available.
2. Place your EUR/USD CSV at the project root (the raw dataset is excluded from the repository — see *Data and reproducibility*).
3. Run the CLI from the project root:
   ```
   python Files/run_limen.py <command> [options]
   ```

## Usage

```
python Files/run_limen.py run --config Files/configs/frozen_config_v1_1_1.json --run-id <run_id>
python Files/run_limen.py audit --input <path_to_csv>
python Files/run_limen.py tests
python Files/run_limen.py ablation --config Files/configs/frozen_config_v1_1_1.json --run-id <run_id>
python Files/run_limen.py sensitivity --config Files/configs/frozen_config_v1_1_1.json --run-id <run_id>
python Files/run_limen.py config-example --out Files/configs/example_config.json
```

> The `tests` subcommand runs the unit test suite, which requires the test files from the full source tree. Test files are intentionally excluded from this curated GitHub copy, so `python Files/run_limen.py tests` will not run here.

### `--input` dataset override

The `run`, `ablation`, and `sensitivity` subcommands accept an optional `--input` argument that overrides `config.runtime.input_path` after the frozen configuration is loaded:

```
python Files/run_limen.py run --config Files/configs/frozen_config_v1_1_1.json --input <path_to_csv> --run-id <run_id>
```

- The path is resolved to an absolute path and validated to point to an existing file before execution begins.
- Invalid paths fail with a clear error and exit code 2; no experiment is started.
- Omitting `--input` preserves existing behavior (uses the path stored in the frozen config).
- Overriding the input path does not change the frozen configuration digest, because `runtime.input_path` is treated as environmental, not scientific.
- `audit --input` behavior is unchanged.

## Selected results

All figures below are offline experimental results from synthetic fault-injection on one EUR/USD feed (5 Mar–26 Oct 2026). They are point estimates, not proof of production readiness.

### Confirmatory run (`confirmatory_v1_1_1_20261028`)

| Metric | Value | Basis |
|--------|-------|-------|
| Input rows / size | 22,067,891 / 914,314,274 B | `run_manifest.json` |
| Observed-minute bars (total / test) | 239,708 / 47,942 | `frozen_config.json` (`split_metadata`), `summary.json` |
| Entry opportunities (test) | 1,749 (874 long / 875 short) | `summary.json` |
| Fault episodes / benign controls | 48 / 8 | `scenario_metrics.csv` |
| Faulted entry opportunities | 55 | `episode_metrics.csv` |
| Episode detection recall | 25/48 = 0.5208 | `scenario_metrics.csv` |
| False-alert rate | 56 / 47,758 = 1.173 per 1,000 clean bars | `summary.json` |

### Policy pass-through on faulted entries (test range)

| Arm | Fault pass-through | Clean blocks | Clean cautions |
|-----|-------------------|--------------|----------------|
| no_gate | 1.0000 (55/55) | 0 | 0 |
| integrity_only | 0.5091 (28/55) | 0 | 0 |
| shift_only | 1.0000 (55/55) | 0 | 5 |
| fused | 0.5091 (28/55) | 8 | 5 |
| shadow | 1.0000 (55/55) | 0 | 0 |
| matched_rate_abstention | 1.0000 (55/55) | 8 | 0 |

### Per-class episode detection recall

| Class | Episodes | Recall | Note |
|-------|----------|--------|------|
| missing_quote | 8 | 1.000 | Hard check fires |
| crossed_quote | 8 | 0.875 | Hard check fires on 7/8 |
| combined_shift_and_quote | 8 | 1.000 | Hard check fires |
| return_scale_shift | 8 | 0.125 | 0/40 fault bars flagged; detections fall in post-episode recovery window |
| stale_replay | 8 | 0.000 | Below 300 s stale threshold at frozen parameters |
| symmetric_shift | 8 | 0.000 | Below 0.0007294 innovation threshold at frozen parameters |
| benign_volatility (control) | 8 | 0.000 | Correctly no detection |

### Ablation and sensitivity

- **Ablation** (`ablation_v1_1_1_20261028b`): 13 variants sharing one intent stream and label set. Disabling `hard_reject` raises fault pass-through to 1.0000, showing hard checks drive all fault interception in this experiment.
- **Sensitivity** (`sensitivity_v1_1_1_20261028`): 42 rows across 14 parameters, 29 distinct config digests. Recall range 0.5000–0.7917; false-alert range 0.7329–7.0564 per 1,000. Every variant re-fits thresholds on train/validation only; test data contributes nothing to calibration.

## Data and reproducibility

### Raw dataset

The raw EUR/USD CSV (`EURUSD_202603050000_202610262359.csv`, 914 MB, 22.07M rows) is **excluded from the public repository** by `.gitignore`. You must supply your own copy at the project root to run full-data experiments. The SHA-256 recorded in the shipped artifacts is `dfcfb7ab72ebbfb54f9babde07bfd0ba655b8e176ff40f3dde1e9f845c455771` (recorded in `run_manifest.json` and `data_audit.json`; not independently re-derived in this copy).

### Configuration reproducibility

- Frozen config version: 1.1.1
- Config digest: `6225c2a28052f265727cbbf287cb374541d59fc86ac116cc37e38f45a9c9c463`
- `runtime.input_path` and `runtime.output_root` are excluded from the digest by design, so relocating the dataset or output directory does not invalidate the frozen scientific configuration.

### Reproducibility caveats

- The mutated-pass decision stream is **not persisted**; `fault_pass_through` cannot be independently recomputed from the shipped artifacts alone without a full rerun.
- The input SHA-256 is recorded but not independently re-derived in this copy.
- Code provenance in the source tree is modification-time based; no version-controlled history is included in the public clean copy.

## Limitations and future work

Reflecting findings from the independent audit:

- **Detection is dominated by hard quote-integrity checks.** Hard checks alone account for the fault interception (27/55). The KS drift monitor has zero measurable policy effect (shift_only passes 55/55 faulted entries).
- **Two fault classes are undetectable at frozen parameters.** `stale_replay` (3-bar replay ≈ 180 s, below the 300 s stale threshold) and `symmetric_shift` (0.0005 magnitude, below the 0.0007294 innovation threshold) both return 0/8 recall.
- **`return_scale_shift` detections occur outside the fault window.** No fault bar is flagged; the 2/8 detections fall in the 30-bar post-episode recovery window with a mean delay of 16 bars.
- **The fused policy adds no fault interception over integrity_only.** Both pass 28/55 faulted entries; fused adds 8 clean blocks and 5 clean cautions.
- **Shadow mode is inert** — byte-for-byte identical to no_gate on all bars; its counterfactual is not exported.
- **No independent evidence for headline pass-through.** `fault_pass_through` depends on the mutated pass that is not persisted.
- **No statistical uncertainty, economic evaluation, or live deployment.** Point estimates only; no confidence intervals, no spread/slippage/P&L model, no online feed or shadow deployment path.

Future work should address: real incident corpus, multi-instrument evaluation, statistical inference (bootstrap CIs, significance tests), economic cost model, persistent mutated-pass artifacts, observable shadow mode, and config portability across operating systems.

## License and citation

No license, publication, DOI, or institutional endorsement has been specified for this project. Citation details are not yet determined. If you reference this work, describe it as "Limen Core — a research prototype for algorithmic forex resilience evaluation" without implying peer review or institutional endorsement.
