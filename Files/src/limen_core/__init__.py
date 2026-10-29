"""Limen Core — offline EUR/USD resilience research prototype.

Package modules:
  config      — validated frozen run configuration
  timestamp   — canonical UTC timestamp utilities
  ingest      — streaming source audit and parser
  quotes      — stateful bid/ask quote-state reconstruction
  bars        — observed-minute bar construction
  strategy    — causal EMA(10, 30) reference decision rule
  integrity   — quote-integrity monitor (hard/soft checks)
  drift       — distribution-shift monitors (KS, CUSUM, scale, quantile, MAD)
  scenarios   — deterministic synthetic fault/control generator
  policy      — policy fusion and comparison arms
  replay      — experiment orchestration
  metrics     — evaluation-only scoring
  audit       — manifests, hashes, structured audit writers
"""

__version__ = "1.0.0"
