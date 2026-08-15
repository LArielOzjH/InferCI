# Changelog

All notable changes to InferCI. Versions follow [SemVer](https://semver.org/).

## [0.2.0] — 2026-08-16

### Added
- **Concurrency (`batch>1`) serving benchmark**: `OpenAIServingRunner` fires N
  concurrent requests and reports aggregate system throughput (total tokens /
  wall span) plus per-request TTFT and pooled ITL percentiles. `LlamaServerRunner`
  sets `-np >= batch`.
- **vLLM / SGLang runner wrappers** (`runners/vllm.py`, `runners/sglang.py`):
  connect-to-existing-server or local-launch (GPU-gated), with automatic
  environment capture that degrades gracefully without the toolchain/GPU.
- **RecallGate** (`inferci/quality.py`): long-context quality-per-$ gate —
  `quality_per_dollar`, a deterministic needle-in-haystack probe, and a
  PASS/FAIL judge against a full-context baseline. CLI via
  `python -m inferci.quality`.

### Changed
- README / docs / roadmap updated for the v2 feature set.

## [0.1.0] — 2026-08-16

### Added
- Core harness: schema, methodology, `llama_cpp` (llama-bench) runner,
  `llama_server` runner (measured TTFT/ITL), generic `openai_serving` runner,
  append-only SQLite ledger, regression judge, cost model, CLI, static dashboard.
- Real regression-evidence dataset (`data/`): 78 curated / 657 full reports.
- 46 tests + GitHub Actions CI (green), Apache-2.0.
