# InferCI

**Neutral, reproducible inference performance & cost regression CI.**

> The engine vendors will never benchmark their competitors fairly, never
> self-report a perf regression they shipped, and never publish their own
> cost-per-token honestly. That's a *structural* conflict of interest — not a
> matter of good faith. InferCI is the neutral layer that does it anyway.

InferCI continuously measures **throughput (tok/s), latency (TTFT/ITL), quality,
and $/token** across engines (llama.cpp, vLLM, SGLang, TRT-LLM, …), models,
quantizations and hardware — then **fails the build when a version ships a
regression**. It is designed around one principle:

> **The moat is not owning GPUs. It is the neutral spec, the orchestration, and
> the reproducible history.** Runners (CPU/GPU/NPU) are contributed resources.

---

## Why this exists

LLM inference engines merge PRs every day. Performance regressions are real,
frequent, and almost never caught before production:

| real regression (from the [evidence set](data/regression_evidence_summary.md)) | impact |
|---|---|
| vLLM #39004 — Qwen3.5-397B NVFP4 decode | **47,200 → 38,300 tok/s (−19%)** |
| llama.cpp #18107 — GPT-OSS 120B after image update | **~300 → ~30 t/s (≈10×)** |
| SGLang #7568 — 4090 v0.4.6→v0.4.8 | **27 → 20 tok/s** |

78 such reports (all with concrete numbers) are collected in
[`data/regression_evidence.jsonl`](data/regression_evidence.jsonl); 657 in the
full pool.

## How it works

```
  runner (llama-bench / vLLM / SGLang / ...)   ← BYO-runner: contributed resource
        │  measures pp/tg throughput, latency, memory
        ▼
  harness  (reproducible spec: pinned seeds/prompts/repeats)
        │
        ▼
  schema + sqlite ledger  (append-only, machine-readable history)
        │
        ▼
  regression judge  (per-metric threshold + noise band → PASS/REGRESSION)
        │
        ▼
  cost model  (throughput × instance price → $/1M tokens)
```

## Getting started (CPU-first, no GPU needed)

```bash
# 1. build llama.cpp (the runner we benchmark)
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cmake -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON
cmake --build llama.cpp/build -j8 --target llama-bench llama-cli llama-server

# 2. get a model (e.g. from ModelScope or HuggingFace)
mkdir -p models && cd models
curl -L -o qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/master/qwen2.5-0.5b-instruct-q4_k_m.gguf
cd ..

# 3. run a benchmark
cd inferci
python -m inferci.cli run --model-file ../models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --model-id Qwen2.5-0.5B-Instruct --quantization Q4_K_M --device cpu

# 4. list / diff / report
python -m inferci.cli list
python -m inferci.cli diff <base_run_id> <candidate_run_id>
python -m inferci.cli report
```

See [`inferci/README.md`](inferci/README.md) for the full CLI and
[`inferci/docs/ARCHITECTURE.md`](inferci/docs/ARCHITECTURE.md) for the design.

## Repository layout

```
inferci/      the project: package, tests, spec, docs
data/         real regression evidence (GitHub issues), with reproducible scripts
research/     the deep-dive that motivated this project (10 directions + red-team)
```

## Status

**v0.1 (CPU-first vertical).** Working: llama.cpp runner (CPU/Metal), schema,
SQLite ledger, regression judge, cost model, CLI, tests, CI. Next: serving
runners (vLLM/SGLang), long-context quality gate (quality-per-$), public
dashboard.

## Contributing

See [`CONTRIBUTING.md`](inferci/CONTRIBUTING.md). The fastest way to help is to
contribute a **runner** for a backend you care about, or a **benchmark spec**
for a workload that matters to you.

## License

Apache-2.0. See [`LICENSE`](inferci/LICENSE).
