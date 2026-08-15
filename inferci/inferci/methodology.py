"""Canonical benchmark methodology.

The neutrality claim rests on a fixed, published, reproducible methodology.
Every number that ships through InferCI MUST come from this file (or an
explicitly versioned override), otherwise comparisons are meaningless.
"""

from __future__ import annotations

# ---- Why these choices matter -------------------------------------------------
# LLM decode is memory-bound, so the two numbers that matter are:
#   * pp  (prompt processing / prefill) -> drives TTFT
#   * tg  (token generation / decode)   -> drives steady-state throughput & ITL
# We pin prompt/generation lengths, repeats, seed and sampling so runs are
# deterministic and comparable across time & machines.

# Deterministic synthetic prompt set (same seed -> same token stream).
DEFAULT_SEED = 42
DEFAULT_PROMPT_TOKENS = 512
DEFAULT_GEN_TOKENS = 128
DEFAULT_REPEATS = 3
DEFAULT_WARMUP_REPEATS = 1

# Relative thresholds used by the regression judge. Conservative enough to
# avoid crying wolf on CPU noise, tight enough to catch real regressions.
THRESHOLDS = {
    # throughput: regression if it DROPS by more than this fraction
    "tg_tps": -0.05,
    "pp_tps": -0.05,
    # latency: regression if it RISES by more than this fraction
    "ttft_ms": 0.10,
    "itl_p95_ms": 0.10,
}

# A metric change smaller than this is "noise" regardless of direction.
MIN_MEANINGFUL_FRACTION = 0.02


def canonical_spec_id(
    backend: str,
    model_id: str,
    quantization: str,
    prompt_tokens: int,
    gen_tokens: int,
    batch: int = 1,
) -> str:
    return ".".join(
        [
            backend,
            model_id,
            quantization,
            f"pp{prompt_tokens}",
            f"tg{gen_tokens}",
            f"b{batch}",
        ]
    )


# ---- Default benchmark matrix (CPU-first, GPU-free start) --------------------
# Small enough to run on a laptop, large enough to be meaningful.
DEFAULT_CPU_MODELS = [
    {
        "model_id": "Qwen2.5-0.5B-Instruct",
        "quantization": "Q4_K_M",
        "prompt_tokens": 512,
        "gen_tokens": 128,
    },
    {
        "model_id": "Qwen2.5-1.5B-Instruct",
        "quantization": "Q4_K_M",
        "prompt_tokens": 512,
        "gen_tokens": 128,
    },
]

# The serving (GPU) matrix is defined the same way but with batch>1 and a
# serving runner; it is the BYO-runner target, not the laptop default.
DEFAULT_SERVING_MATRIX = [
    {
        "model_id": "Qwen2.5-7B-Instruct",
        "quantization": "FP8",
        "prompt_tokens": 512,
        "gen_tokens": 128,
        "batch": 8,
    },
    {
        "model_id": "Llama-3.1-8B-Instruct",
        "quantization": "FP8",
        "prompt_tokens": 512,
        "gen_tokens": 128,
        "batch": 8,
    },
]
