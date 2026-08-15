# Result Schema

The machine-readable contract for every InferCI run. Source of truth:
`inferci/schema.py`. This document is the human-readable form.

## Example

```json
{
  "run_id": "a1b2c3d4e5f6",
  "created_at": "2026-08-16T02:50:00+00:00",
  "spec": {
    "spec_version": "0.1.0",
    "id": "llama_cpp.Qwen2.5-0.5B-Instruct.Q4_K_M.pp512.tg128.b1",
    "model_id": "Qwen2.5-0.5B-Instruct",
    "model_file": "../models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "backend": "llama_cpp",
    "quantization": "Q4_K_M",
    "prompt_tokens": 512,
    "gen_tokens": 128,
    "repeats": 3,
    "warmup_repeats": 1,
    "batch": 1,
    "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "seed": 42},
    "extra": {"device": "cpu"}
  },
  "environment": {
    "host": "hostname",
    "os": "Darwin 24.0.0",
    "arch": "arm64",
    "cpu": "Apple M1 Pro",
    "ram_gb": 16.0,
    "accelerator": {"kind": "cpu"},
    "backend": "llama_cpp",
    "backend_version": "abc1234"
  },
  "metrics": {
    "pp_tps": 120.0, "pp_tps_std": 1.2,
    "tg_tps": 45.0, "tg_tps_std": 0.3,
    "ttft_ms": 4266.7,
    "itl": {"mean_ms": 22.2, "p50_ms": 22.2, "p95_ms": 0.0, "p99_ms": 0.0},
    "total_seconds": 7.1,
    "prompt_tokens": 512, "generated_tokens": 128,
    "model_size_mb": 468.6
  },
  "cost": null
}
```

## Field semantics

- `spec` — what was run. Comparable iff `spec.id` matches.
- `environment` — where it ran. Always captured automatically.
- `metrics` — the numbers. `pp_tps`/`tg_tps` are the canonical throughput
  signals; `ttft_ms`/`itl` are latency.
- `cost` — `null` for local hardware; populated when an instance price is given.
- `raw` — backend raw output (stored for forensics, never parsed for judgments).

## Ledger storage

Each `RunResult` is one row in the SQLite ledger (`inferci/store.py`), with
indexed columns (`backend`, `model_id`, `spec_id`, `tg_tps`, `pp_tps`,
`ttft_ms`) for fast queries plus full JSON for fidelity. History is append-only.
