# InferCI Architecture

## 1. The one idea

**The moat is neutrality + reproducibility, not hardware.** A benchmark whose
operator also owns the GPUs being compared is not neutral. So InferCI is a
**BYO-runner (bring-your-own-runner) neutral orchestrator**:

- *Runner* = an adapter around one inference backend (llama.cpp, vLLM, SGLang,
  TRT-LLM, …). Anyone can run it on their own CPU/GPU/NPU and contribute results.
- *Orchestrator* = the spec, the reproducible harness, the append-only ledger,
  and the regression judge. It is GPU-free and vendor-neutral.

This is also why InferCI can start on CPU: the first runner benchmarks
llama.cpp on CPU/Metal — a real user base of millions, where regressions happen
and are never caught.

## 2. Components

```
┌───────────── runners (BYO) ─────────────┐
│  llama_cpp (llama-bench)   [v0.1]       │
│  vllm / sglang / trt-llm   [planned]    │
└──────────────┬──────────────────────────┘
               │  RunResult (schema.py)
┌──────────────▼──────────────────────────┐
│  harness  (methodology.py: pinned spec) │
│  store    (sqlite ledger, append-only)  │
│  regression (judge: thresholds + noise) │
│  cost     (throughput × price → $/1M)   │
│  cli      (run / list / diff / report)  │
└─────────────────────────────────────────┘
```

### schema.py
Canonical, machine-readable data model (`BenchmarkSpec`, `Environment`,
`Metrics`, `CostResult`, `RunResult`). The schema *is* the contract: two runs
are comparable iff their spec `id` matches. Everything else speaks this schema.

### methodology.py
The pinned methodology: prompt/generation lengths, repeats, seed, sampling, and
the regression thresholds. Any number that ships through InferCI must come from
here (or an explicitly versioned override).

### store.py
Append-only SQLite ledger. The accumulated reproducible history is itself the
asset ("the ledger"), later feeding the dashboard and trend lines.

### regression.py
The judge. Per-metric relative change vs published thresholds, with a noise band
so it does not cry wolf on CPU jitter. Never picks a winner — only flags.

### cost.py
`$/1M tokens = (instance_hourly / 3600) × 1e6 / tokens_per_sec`. Local hardware
is priced as "local" (zero), cloud as catalog price (verify before quoting).

## 3. Trust model (why this is defensible)

| moat type | defensible? |
|---|---|
| code | ❌ copied |
| method | ❌ absorbed by engines |
| standard/protocol | ❌ taken over by a consortium |
| **neutrality + reproducible history** | ✅ engines *cannot* self-supply (conflict of interest); paywalled analysts *won't* self-prove (opaque) |

The only structurally uncopyable moat in this space is the accumulated,
long-term-neutral "fair arbiter" position — which is exactly what an
append-only, open, reproducible ledger builds.

## 4. Roadmap (first 90 days, CPU-first)

- **D1–14** ✅ core: schema/harness/store/judge/cost/CLI + llama.cpp runner
- **D15–30** ✅ real CPU PoC + unit tests + CI
- **D31–60** ➡️ serving runners (vLLM/SGLang), batch>1, measured (not derived) TTFT/ITL
- **D61–90** ➡️ long-context quality gate (quality-per-$), public dashboard, ROCm/AMD coverage

## 5. Design constraints

- **Zero runtime dependencies.** The harness must run unchanged anywhere.
- **Append-only, never rewrite history.** A "corrected" number is a new run.
- **Raw output is stored, never trusted.** Judgments parse the schema, not logs.
