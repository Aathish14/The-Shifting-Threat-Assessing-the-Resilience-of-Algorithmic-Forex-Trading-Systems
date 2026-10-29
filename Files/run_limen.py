"""Limen Core command-line entry point.

Invoke from the project root (D:\\Limen_Core_GitHub):

    python Files/run_limen.py run --config Files/configs/frozen_config_v1_1_1.json --run-id <run_id>
    python Files/run_limen.py run --config Files/configs/frozen_config_v1_1_1.json --input <path_to_csv> --run-id <run_id>
    python Files/run_limen.py audit --input ../EURUSD_202603050000_202610262359.csv
    python Files/run_limen.py tests
    python Files/run_limen.py ablation --config <cfg> --run-id <run_id>
    python Files/run_limen.py ablation --config <cfg> --input <path_to_csv> --run-id <run_id>
    python Files/run_limen.py sensitivity --config <cfg> --run-id <run_id>
    python Files/run_limen.py sensitivity --config <cfg> --input <path_to_csv> --run-id <run_id>
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `Files.src.limen_core` imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILES_DIR = PROJECT_ROOT / "Files"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Files.src.limen_core.config import example_config, load_config
from Files.src.limen_core.audit import atomic_write_json
from Files.src.limen_core.ingest import audit_source
from Files.src.limen_core.timestamp import utc_now_iso
from Files.src.limen_core.replay import ReplayEngine


def _resolve_config(config_path: Path):
    """Load the config and resolve a config-relative input path.

    Uses ``dataclasses.replace`` so every section (bar, ablation, sensitivity,
    ...) survives; the v1.0.x reconstruction enumerated sections by hand and
    silently dropped any section added after it was written.
    """
    config = load_config(config_path)
    inp = Path(config.runtime.input_path)
    if not inp.is_absolute():
        resolved = FILES_DIR / inp
        if resolved.exists():
            runtime = dataclasses.replace(config.runtime, input_path=str(resolved))
            config = dataclasses.replace(config, runtime=runtime)
    return config


def _override_input_path(config, input_path_str: str):
    """Override ``config.runtime.input_path`` with a user-supplied path.

    Resolves the path to an absolute path and validates that it points to an
    existing file. Returns the updated config, or prints a clear error and
    returns ``None`` if the path is invalid.
    """
    inp = Path(input_path_str)
    if not inp.is_absolute():
        inp = Path.cwd() / inp
    if not inp.exists() or not inp.is_file():
        print(f"ERROR: input file not found: {inp}", file=sys.stderr)
        return None
    runtime = dataclasses.replace(config.runtime, input_path=str(inp))
    return dataclasses.replace(config, runtime=runtime)


def _default_run_id() -> str:
    return "run_" + utc_now_iso().replace(":", "").replace("-", "").replace(".", "")


def cmd_run(args) -> int:
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return 2
    config = _resolve_config(config_path)
    if args.input:
        config = _override_input_path(config, args.input)
        if config is None:
            return 2
    run_id = args.run_id or _default_run_id()
    output_dir = Path(config.runtime.output_root) / run_id
    engine = ReplayEngine(config, run_id, config_path=str(config_path.as_posix()))
    manifest = engine.run(output_dir)
    print(f"Run complete: {run_id}")
    print(f"Output directory: {output_dir}")
    print(f"Status: {manifest['status']}")
    return 0


def cmd_ablation(args) -> int:
    from Files.src.limen_core.ablation import AblationRunner

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return 2
    config = _resolve_config(config_path)
    if args.input:
        config = _override_input_path(config, args.input)
        if config is None:
            return 2
    run_id = args.run_id or _default_run_id()
    output_dir = Path(config.runtime.output_root) / run_id
    runner = AblationRunner(config, run_id, config_path=str(config_path.as_posix()))
    summary = runner.run(output_dir)
    print(f"Ablation complete: {run_id}")
    print(f"Output directory: {output_dir}")
    print(f"Variants: {summary['variant_count']}")
    return 0


def cmd_sensitivity(args) -> int:
    from Files.src.limen_core.sensitivity import SensitivityRunner

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return 2
    config = _resolve_config(config_path)
    if args.input:
        config = _override_input_path(config, args.input)
        if config is None:
            return 2
    run_id = args.run_id or _default_run_id()
    output_dir = Path(config.runtime.output_root) / run_id
    runner = SensitivityRunner(config, run_id, config_path=str(config_path.as_posix()))
    summary = runner.run(output_dir)
    print(f"Sensitivity complete: {run_id}")
    print(f"Output directory: {output_dir}")
    print(f"Experiments: {summary['experiment_count']}")
    return 0


def cmd_audit(args) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 2
    out_path = Path(args.out) if args.out else Path("Files") / "artifacts" / "audit_only.json"
    audit = audit_source(input_path, max_rows=args.max_rows)
    payload = {
        "timestamp": utc_now_iso(),
        "input_path": str(input_path),
        "audit": audit.to_dict(),
    }
    atomic_write_json(out_path, payload)
    print(f"Audit written to {out_path}")
    print(f"Rows: {audit.total_rows}, SHA-256: {audit.sha256}")
    print(f"First: {audit.first_timestamp}, Last: {audit.last_timestamp}")
    return 0


def cmd_tests(args) -> int:
    import subprocess
    tests_dir = PROJECT_ROOT / "Files" / "tests"
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", str(tests_dir), "-v"]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def cmd_config_example(args) -> int:
    out_path = Path(args.out) if args.out else Path("Files") / "configs" / "example_config.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = example_config()
    cfg.save_json(out_path)
    print(f"Example config written to {out_path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_limen.py",
        description="Limen Core offline research replay CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a deterministic replay")
    p_run.add_argument("--config", required=True, help="Path to frozen config JSON")
    p_run.add_argument("--input", help="Path to source CSV (overrides config.runtime.input_path)")
    p_run.add_argument("--run-id", help="Unique run ID (auto-generated if omitted)")
    p_run.set_defaults(func=cmd_run)

    p_audit = sub.add_parser("audit", help="Audit the source CSV")
    p_audit.add_argument("--input", required=True, help="Path to source CSV")
    p_audit.add_argument("--out", help="Output audit JSON path")
    p_audit.add_argument("--max-rows", type=int, default=None, help="Max rows to parse (bounded test)")
    p_audit.set_defaults(func=cmd_audit)

    p_tests = sub.add_parser("tests", help="Run the unit test suite")
    p_tests.set_defaults(func=cmd_tests)

    p_abl = sub.add_parser("ablation", help="Run the ablation matrix")
    p_abl.add_argument("--config", required=True, help="Path to frozen config JSON")
    p_abl.add_argument("--input", help="Path to source CSV (overrides config.runtime.input_path)")
    p_abl.add_argument("--run-id", help="Unique run ID (auto-generated if omitted)")
    p_abl.set_defaults(func=cmd_ablation)

    p_sens = sub.add_parser("sensitivity", help="Run one-at-a-time sensitivity sweeps")
    p_sens.add_argument("--config", required=True, help="Path to frozen config JSON")
    p_sens.add_argument("--input", help="Path to source CSV (overrides config.runtime.input_path)")
    p_sens.add_argument("--run-id", help="Unique run ID (auto-generated if omitted)")
    p_sens.set_defaults(func=cmd_sensitivity)

    p_cfg = sub.add_parser("config-example", help="Write an example frozen config")
    p_cfg.add_argument("--out", default=None, help="Output config path")
    p_cfg.set_defaults(func=cmd_config_example)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
