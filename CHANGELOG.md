# Changelog

All notable changes to InferCI. Versions follow [SemVer](https://semver.org/).

## [0.2.0] — 2026-08-16

### Added
- **Concurrency (`batch>1`) serving benchmark**: aggregate system throughput +
  per-request TTFT + pooled ITL percentiles.
- **Runner wrappers**: `vllm`, `sglang`, `trt_llm`, `tgi` (GPU-gated
  connect-or-launch, graceful env capture). 8 runners total.
- **RecallGate**: long-context quality-per-$ gate — `quality_per_dollar`,
  deterministic needle-in-haystack probe, pluggable `Eval` protocol
  (`EVAL_REGISTRY`) with chat-completions mode, PASS/FAIL judge.
  CLI via `python -m inferci.quality`.
- **Mock runner** (`--backend mock`): deterministic, no model needed — exercises
  the whole pipeline and simulates regressions via `--set slowdown=0.9`.
- **CLI polish**: `python -m inferci` entry, generic `--set KEY=VALUE`,
  full-run-id display + prefix resolution in `diff`.

### Quality & tooling
- **ruff** (lint + format, 100-col) clean across the package and tests.
- **Coverage** gate in CI (CI floor 60; ~80% with the local integration env).
- **CI**: test matrix (py3.10/3.12/3.13) + lint + coverage jobs.
- Makefile, pre-commit config, issue/PR templates, CODE_OF_CONDUCT, README badges.
- Suite clean under `-W error::ResourceWarning` (no resource leaks).

## [0.1.0] — 2026-08-16

### Added
- Core harness: schema, methodology, `llama_cpp` (llama-bench) runner,
  `llama_server` runner (measured TTFT/ITL), generic `openai_serving` runner,
  append-only SQLite ledger, regression judge, cost model, CLI, static dashboard.
- Real regression-evidence dataset (`data/`): 78 curated / 657 full reports.
- 46 tests + GitHub Actions CI (green), Apache-2.0.
