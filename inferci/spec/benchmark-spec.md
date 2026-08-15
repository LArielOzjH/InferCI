# Benchmark Specification

The neutrality of InferCI rests on **reproducibility**. This document pins the
methodology. Any benchmark that does not follow it is not a valid InferCI run.

## 1. What we measure and why

Autoregressive decode is memory-bound, so the two numbers that matter are:

| metric | meaning | drives |
|---|---|---|
| `pp` (prompt processing / prefill) | tokens/sec when ingesting the prompt | TTFT |
| `tg` (token generation / decode) | tokens/sec when generating | ITL, steady-state throughput |

Latency (`ttft_ms`, `itl`) is measured directly by serving runners (batch≥1,
real scheduler). On the single-stream llama-bench runner it is **derived**
(`ttft ≈ prompt_tokens/pp_tps`, `itl ≈ 1/tg_tps`) and marked as such.

## 2. Determinism rules

- **Pinned lengths**: `prompt_tokens`, `gen_tokens` are fixed per spec.
- **Pinned seed / sampling**: `sampling` (temperature/top-p/top-k/seed) is
  applied by the *serving* runners (`llama_server`, `openai_serving`), which
  pass it to the engine. The `llama_cpp` (llama-bench) runner is greedy
  single-stream and takes no sampling parameters — it records the spec verbatim
  but never claims the engine used non-greedy sampling.
- **Warmup**: `warmup_repeats` runs are discarded before timing (serving
  runners); the `llama_cpp` runner honors warmup as on/off only.
- **Repeats**: `repeats` timed runs; mean and std are both recorded.
- **Environment captured automatically** (never hand-typed): backend version /
  commit, CPU, RAM, accelerator, OS.

### Latency: measured vs derived

| runner | TTFT / ITL | percentiles |
|---|---|---|
| `llama_server`, `openai_serving` | **measured** from a real HTTP stream | measured (mean/p50/p95/p99) |
| `llama_cpp` (llama-bench) | **derived** (`ttft ≈ prompt/pp_tps`, `itl.mean = 1/tg_tps`) | not measured (left 0) |

The `llama_cpp` runner is single-stream (`batch` must be 1); `batch>1` requires
a serving runner.

## 3. Default matrix (CPU-first)

| model | quantization | pp | tg | notes |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | Q4_K_M | 512 | 128 | laptop-runnable |
| Qwen2.5-1.5B-Instruct | Q4_K_M | 512 | 128 | laptop-runnable |

The serving (GPU) matrix extends this with `batch>1` and a real serving runner;
it is the BYO-runner target.

## 4. Spec fields (`BenchmarkSpec`)

See `inferci/schema.py` for the source of truth. Key fields:

- `id` — canonical comparable key (`backend.model.quant.pp<N>.tg<N>.b<batch>`).
  Two runs are comparable iff their `id` matches.
- `model_id`, `model_file`, `quantization`, `backend`
- `prompt_tokens`, `gen_tokens`, `repeats`, `warmup_repeats`, `batch`
- `sampling`, `prompt_set`, `extra` (device, threads, …)

## 5. Regression thresholds

Published in `inferci/methodology.py` (applied uniformly, never per-vendor):

| metric | regression when | threshold |
|---|---|---|
| `tg_tps` | drops by | > 5% |
| `pp_tps` | drops by | > 5% |
| `ttft_ms` | rises by | > 10% |
| `itl.p95_ms` | rises by | > 10% |

Changes under 2% are treated as noise regardless of direction.
