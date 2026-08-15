# InferCI v0.1.0 — Code Review

> ## Resolution status (updated after review)
> All 5 `major` and 6 `minor` findings below have been fixed in the follow-up
> commit(s): comparability check + `--force`, failed-run rejection, unknown
> instance-type `ValueError`, sampling/threads/batch applied-or-documented,
> plain `INSERT` (append-only), full UUID ids, zero-baseline `NO_DATA`, honest
> ITL (percentiles only from serving runners), tolerant deserialization,
> positional `-o json` removal, and CLI/report polish. This file is kept as the
> review snapshot for transparency.

**Reviewer:** 资深 Python 评审（严格模式）
**Date:** 2026-08-16 (review timestamp)
**Scope:** `inferci/` 全部源码、`tests/`、`spec/`、`docs/`

## Test result

```
python3 -m unittest discover -s tests
----------------------------------------------------------------------
Ran 26 tests in 0.009s

OK (skipped=1)
```

26 tests pass, 1 skipped (the end-to-end llama-bench run, gated on
`INFERCI_LLAMA_BENCH` + `INFERCI_TEST_MODEL`). Unit tests are green but cover
only happy paths and the parser helpers; the correctness gaps below are all in
uncovered code paths.

## Summary verdict

The package is well-structured, zero-dependency, and honestly documented at the
"intent" level. However, several **core claims (append-only ledger, comparability
= spec id, pinned seed/sampling, honest failure recording) are not enforced by
the code** and would silently produce wrong or misleading benchmark data. No
shell-injection or SQL-injection was found (argv lists + parameterized queries
are used correctly everywhere). Priority: fix the five `major` items before
treating any number as trustworthy.

---

## Findings

### 1. major — Comparable-key omits device/accelerator and sampling → silent apples-to-oranges diff
- **File:** `inferci/schema.py` (`canonical_id`, L78-84), `inferci/methodology.py` (`canonical_spec_id`, L38-43), `inferci/cli.py` (`cmd_diff`, L87-91)
- **Description:** `canonical_id()` = `backend.model.quant.pp<N>.tg<N>.b<batch>`. It excludes `extra["device"]`, `accelerator.kind`, and `sampling` (temperature/top_p/top_k/seed). `cmd_diff` treats "same spec id" as "comparable" and only warns when the ids differ. So a CPU run and a CUDA run of the *same* model share an id, pass the comparability gate without warning, and the judge will flag an enormous "regression" (or "improvement") across hardware. The spec's "two runs are comparable iff their id matches" is therefore false as implemented.
- **Fix:** Either include `device`/accelerator in the canonical id, or (preferred) add an explicit comparability check in `cmd_diff`/`compare_runs` that compares `environment.accelerator.kind` (and `backend_version`, and `sampling`) and refuses/warns loudly when they differ — before computing relative changes.

### 2. major — Failed `llama-bench` runs are recorded as valid zero-metric rows
- **File:** `inferci/runners/llama_cpp.py` (`_run_bench` L139-159, `run` L182-221)
- **Description:** `subprocess.run(...)` is called without `check=True` and `proc.returncode`/`proc.stderr` are never inspected. If `llama-bench` crashes, times out, or the model file is missing, `_parse_json` returns `{}`, the fallback re-runs (also fails), and `run()` happily builds a `RunResult` with `pp_tps=0, tg_tps=0` and inserts it into the ledger. Downstream, `diff`/`report` interpret a 0-tps row as a real data point (a 100% "regression" or NaN-ish delta), polluting the neutral history with garbage.
- **Fix:** Check `proc.returncode != 0` and raise (or return an explicit `FAILED` result that is never inserted) with `stderr` included in the error message. Never persist a run whose `tg_tps==0.0 and pp_tps==0.0` with no explanation. Capture `proc.stderr` into `RunResult.raw` for forensics.

### 3. major — Unknown `instance_type` silently degrades to "local" $0 cost
- **File:** `inferci/cost.py` (`compute_cost` L27-50)
- **Description:** When `instance_type` is supplied but is not in `PRICE_CATALOG` and no `instance_hourly` is given, the function falls through to `CostResult(source="local (no cloud price; you own hardware)")` with zero price — while still storing the (unrecognized) `instance_type` string. A typo like `--instance gpu.t4.g4dn.xlargee` therefore produces a valid-looking **$0/1M-token** result instead of an error, which can silently understate cost in any published report.
- **Fix:** Raise `ValueError` (or return an error status) when a non-empty `instance_type` is not found in the catalog and no `instance_hourly` override is provided. Only treat the "local" case as local when *no* instance was requested.

### 4. major — Sampling / seed / threads / batch recorded but never applied to the runner
- **File:** `inferci/runners/llama_cpp.py` (`_run_bench` L139-159), `inferci/cli.py` (`_build_spec` L24-37), `spec/benchmark-spec.md` §2
- **Description:** `BenchmarkSpec.sampling` (temperature, top_p, top_k, seed) is captured and stored, and the spec claims "pinned seed / pinned sampling" and "same seed → same token stream". But `_run_bench` builds a command with only `-m -p -n -r -ngl -o json` — **no** `--seed`, `--temp`, `--top-p`, or `--top-k` is passed, and there is no seeded synthetic-prompt generator anywhere. `extra["threads"]` is likewise never passed, and `batch` changes the canonical id but never the single-stream execution. The result is misleading metadata: the ledger claims a run used `seed=42/temperature=1.0` that the engine never saw, and `--batch 8` produces a *different* spec id for an *identical* single-stream run.
- **Fix:** Either (a) pass the sampling/threads/batch values through to the backend where the backend supports them, or (b) stop recording them as if they were applied and document that the llama.cpp runner is greedy/single-stream by construction. At minimum, the canonical id must not include fields the runner ignores.

### 5. major — `INSERT OR REPLACE` breaks the append-only ledger contract
- **File:** `inferci/store.py` (`insert` L48-66; `_SCHEMA` L17-37)
- **Description:** `docs/ARCHITECTURE.md` and `spec/result-schema.md` both promise "append-only, never rewrite history; a corrected number is a new run". Yet `insert` uses `INSERT OR REPLACE`, so re-inserting the same `run_id` silently overwrites the prior row. Combined with finding 6 (truncated 48-bit ids), a collision quietly rewrites ledger history — directly contradicting the product's stated moat.
- **Fix:** Use plain `INSERT` (no `OR REPLACE`). If a duplicate `run_id` is attempted, raise an explicit integrity error so the operator must mint a new id for a new run.

### 6. minor — `run_id` truncated to 48 bits → realistic collision overwrites history
- **File:** `inferci/schema.py` (`new_run_id` L23-24)
- **Description:** `uuid.uuid4().hex[:12]` keeps only 48 bits of entropy. Birthday-collision probability reaches ~50% around ~16 million runs — plausible for a busy CI ledger — and, due to finding 5, a collision silently overwrites the earlier run rather than appending.
- **Fix:** Store the full 128-bit UUID (`str(uuid.uuid4())`) as the primary key; the shortening buys nothing and costs uniqueness.

### 7. minor — `compare_runs` mishandles zero baseline ("data appeared" hidden as NOISE)
- **File:** `inferci/regression.py` (`_rel` L60-63, `compare_runs` L66-116)
- **Description:** `_rel` returns `0.0` whenever `base == 0.0`, and only the `base==0 and cand==0` case is mapped to `NO_DATA`. If the baseline is 0 (unmeasured/failed) but the candidate is non-zero, `rel=0.0` falls through to `NOISE` — hiding the fact that data suddenly appeared, and never flagging a missing→present or present→missing transition. Also, when base>0 but `cand` is missing, this isn't distinguishable from a real zero.
- **Fix:** Treat any `base == 0.0 and cand != 0.0` (and vice versa) as `NO_DATA` (or a distinct "insufficient baseline") instead of computing a 0.0 relative change.

### 8. minor — Fabricated `itl.p50_ms`; p95/p99 never populated → latency regression is dead for llama.cpp
- **File:** `inferci/runners/llama_cpp.py` (`run` L199-204)
- **Description:** The runner sets `itl.mean_ms == itl.p50_ms == 1000/tg_tps` and leaves `p95_ms`/`p99_ms` at `0.0`. Median inter-token latency is **not** equal to the reciprocal of mean throughput, so `p50_ms` is a fabricated statistic. Because `p95_ms` is always 0, `compare_runs`' `itl_p95_ms` metric is permanently `NO_DATA` for this runner — making the published "itl.p95_ms rises > 10% → regression" threshold unreachable on the only implemented runner. The `raw` note says "no distribution", but the schema still ships a fake p50.
- **Fix:** Either populate real percentiles from llama-bench (if it can emit them) or leave `p50/p95/p99` at 0 and remove the false p50=mean assignment; do not advertise a latency regression path the runner cannot exercise.

### 9. minor — Fragile deserialization and no schema-version gate
- **File:** `inferci/schema.py` (`from_dict` L141-158)
- **Description:** `from_dict` does `BenchmarkSpec(**spec_d)` / `Environment(**env_d)` / `Metrics(**m_d)` with no filtering, so any extra/unknown key (e.g. from a future or foreign BYO-runner) raises `TypeError` and corrupts a whole `list`/`get` read. `SPEC_VERSION` ("0.1.0") is stored but never validated on load, so a version-mismatched ledger row is silently coerced. `run_id=d.get("run_id", new_run_id())` evaluates `new_run_id()` eagerly on every call (wasteful) and silently regenerates an id when one is missing.
- **Fix:** Filter dicts to known dataclass fields (or use a tolerant constructor), check `spec_version` on load and reject/handle unknown versions, and use a sentinel to avoid eager `new_run_id()`.

### 10. minor — `_run_bench` fallback: fragile arg filtering, double execution, uncaught timeout
- **File:** `inferci/runners/llama_cpp.py` (`_run_bench` L156-158)
- **Description:** The JSON→text fallback uses `[c for c in cmd if c not in ("-o", "json")]`, which would also delete a `model_file` literally named `"json"`. It re-runs the whole benchmark a second time on failure with no error surfaced, and `subprocess.run(timeout=1800)` raises `TimeoutExpired` uncaught, so a hang crashes the CLI with a raw traceback instead of a diagnostic.
- **Fix:** Remove `-o json` by position (index of `-o`), do not re-run when the first invocation returned non-zero, wrap `TimeoutExpired` with a clear message, and always attach `stderr` to `raw`.

### 11. minor — `warmup_repeats` only toggles `--no-warmup`; count never honored
- **File:** `inferci/runners/llama_cpp.py` (`_run_bench` L150-151), `inferci/methodology.py` (L21, comment L25)
- **Description:** `spec.warmup_repeats` is mapped only to a binary `--no-warmup` flag when `<= 0`. Any positive value (including the documented "1 warmup run discarded") is not passed to llama-bench (`-w` is never set), so the recorded warmup count has no effect on what is actually timed — another spec-vs-code gap.
- **Fix:** Pass llama-bench's `-w <n>` from `spec.warmup_repeats`, or document that warmup is controlled by llama-bench's own default and stop recording a per-spec count.

### 12. nit — CLI/report polish and mutable-catalog exposure
- **Files:** `inferci/cost.py` (`lookup_price` L23-24), `inferci/cli.py` (`cmd_diff` L95, `cmd_report` L98-115, `cmd_list` L74)
- **Description:**
  - `lookup_price` returns the live `PRICE_CATALOG` dict by reference; a caller can mutate the global price table.
  - `cmd_diff` re-implements `any_regression` inline (`f.verdict.value == "regression"`) instead of using the exported helper.
  - `cmd_report` reads `store.list(limit=1000)`, so with >1000 rows the "first" run is not the true first and the delta% is silently wrong.
  - `cmd_list` slices `created_at[:19]`, a hardcoded assumption about the ISO format that breaks if `now_utc()` ever changes precision.
  - `--model-id`/`--quantization` are optional (default `""`), so two different models can easily collide on an empty-containing spec id if the operator forgets them.
- **Fix:** Return a copy of the catalog entry; use `any_regression()`; add a "full history" path or warn on truncation in `report`; format timestamps via a helper; require `--model-id` (or validate non-empty) for `run`.

---

## Cross-cutting checks (clean)

- **Shell injection:** none — all `subprocess` calls use argv lists, no `shell=True`, no `os.system`.
- **SQL injection:** none — all `Store` queries use `?` placeholders; `list()` builds WHERE clauses from a fixed whitelist of columns.
- **Non-determinism:** metric determinism is delegated to llama-bench (greedy decode), but the *harness* over-claims determinism via seed/sampling it never applies (finding 4).

## Spec/doc vs code consistency

| Claim | Status |
|---|---|
| Append-only, never rewrite history | ❌ `INSERT OR REPLACE` (finding 5) |
| Comparable iff spec id matches | ❌ id omits device/accelerator/sampling (finding 1) |
| Pinned seed → same token stream | ❌ seed never passed / no synthetic generator (finding 4) |
| warmup_repeats runs discarded | ⚠️ only on/off toggle, count ignored (finding 11) |
| itl measured/derived honestly | ⚠️ p50 fabricated, p95 never filled (finding 8) |
| Raw output stored, never trusted | ✅ (but failed runs are not distinguished — finding 2) |
| Zero runtime deps | ✅ confirmed (stdlib only) |
