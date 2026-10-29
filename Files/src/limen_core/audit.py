"""Run manifests, data audit, hashes, and structured audit-trail writers.

Every generated artifact carries an explicit timestamp. Writes are atomic
(temp file + rename) and existing run directories are never overwritten.

Config digest
-------------
There is exactly one definition, :func:`config_digest`. It hashes the compact,
sorted JSON of the frozen configuration *excluding* ``runtime.input_path`` and
``runtime.output_root``: those are environmental, not scientific, so moving the
input file or writing to a different directory must not look like a different
experiment. The input's own SHA-256 is recorded separately in the manifest.

v1.0.x computed the digest three different ways (full-config JSON here, a
hand-built dict in ``replay``, another in the CLI), so two runs could claim the
same digest with different parameters. That is now impossible by construction.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from .timestamp import utc_now_iso


# Runtime fields excluded from the scientific config digest.
_DIGEST_EXCLUDED_RUNTIME_KEYS: tuple[str, ...] = ("input_path", "output_root")


def utc_iso() -> str:
    return utc_now_iso()


def compute_file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1_048_576), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Atomically write JSON to *path* via temp file + os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def append_log(log_path: str | Path, message: str, level: str = "INFO") -> None:
    """Append one timestamped log line to the execution log."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = utc_iso()
    line = f"{ts} [{level}] {message}\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def _config_dict(config_obj) -> dict[str, Any]:
    if hasattr(config_obj, "to_dict"):
        return copy.deepcopy(config_obj.to_dict())
    return copy.deepcopy(dict(config_obj))


def digestible_config(config_obj) -> dict[str, Any]:
    """Return the configuration payload that participates in the digest."""
    data = _config_dict(config_obj)
    runtime = data.get("runtime")
    if isinstance(runtime, dict):
        for key in _DIGEST_EXCLUDED_RUNTIME_KEYS:
            runtime.pop(key, None)
    return data


def config_digest(config_obj) -> str:
    """SHA-256 of the compact, sorted, environment-free frozen config."""
    payload = json.dumps(
        digestible_config(config_obj),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_run_manifest(
    *,
    run_id: str,
    command: str,
    input_path: str,
    input_digest: str,
    input_size: int,
    config_obj,
    status: str,
    output_dir: str | Path,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the run manifest dict (written by the caller).

    ``output_dir`` is the directory the caller is actually writing to, so the
    recorded path can never disagree with where the artifacts landed (v1.0.x
    re-derived it from ``output_root`` + ``run_id`` and silently drifted).
    """
    output_dir = Path(output_dir)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": utc_iso(),
        "command": command,
        "input_path": input_path,
        "input_byte_size": input_size,
        "input_sha256": input_digest,
        "config_digest": config_digest(config_obj),
        "config_version": getattr(config_obj, "version", ""),
        "python_version": sys.version,
        "platform": platform.platform(),
        "status": status,
        "output_dir": str(output_dir).replace("\\", "/"),
        "manifest_path": str(output_dir / "run_manifest.json").replace("\\", "/"),
    }
    if seed is not None:
        manifest["seed"] = seed
    if extra:
        manifest.update(extra)
    return manifest


def build_frozen_config(config_obj, config_path: str | Path) -> dict[str, Any]:
    """Serialize the frozen config and return its single canonical digest."""
    return {
        "config": _config_dict(config_obj),
        "config_digest": config_digest(config_obj),
        "config_version": getattr(config_obj, "version", ""),
        "config_path": str(config_path).replace("\\", "/"),
        "timestamp": utc_iso(),
    }


def build_artifact_manifest(run_dir: str | Path) -> dict[str, Any]:
    """List every output file under run_dir with byte size and SHA-256."""
    run_dir = Path(run_dir)
    entries = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(run_dir)
            entries.append({
                "path": str(rel).replace("\\", "/"),
                "byte_size": path.stat().st_size,
                "sha256": compute_file_sha256(path),
            })
    return {
        "run_dir": str(run_dir).replace("\\", "/"),
        "timestamp": utc_iso(),
        "artifact_count": len(entries),
        "artifacts": entries,
    }


def write_execution_log_entry(
    log_path: str | Path, event: str, status: str, message: str
) -> None:
    append_log(
        log_path,
        f"event={event} status={status} message={message}",
        level="INFO",
    )


__all__ = [
    "append_log",
    "atomic_write_json",
    "build_artifact_manifest",
    "build_frozen_config",
    "build_run_manifest",
    "compute_file_sha256",
    "config_digest",
    "digestible_config",
    "utc_iso",
    "write_execution_log_entry",
]
