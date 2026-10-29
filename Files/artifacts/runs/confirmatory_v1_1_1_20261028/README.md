# Limen Core Run Reproduction README

run_id: confirmatory_v1_1_1_20261028
generated: 2026-10-28T16:28:56.156Z
status: completed
config: Files/configs/frozen_config_v1_1_1.json
config_version: 1.1.1
config_digest: 6225c2a28052f265727cbbf287cb374541d59fc86ac116cc37e38f45a9c9c463

## Reproduction

From the project root:

    python Files/run_limen.py run --config Files/configs/frozen_config_v1_1_1.json --run-id confirmatory_v1_1_1_20261028

## Input

- Path: D:\Limen_Core\Files\..\EURUSD_202603050000_202610262359.csv
- SHA-256: dfcfb7ab72ebbfb54f9babde07bfd0ba655b8e176ff40f3dde1e9f845c455771
- Byte size: 914314274
- Rows: 22067891

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
