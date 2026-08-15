# Contributing to InferCI

InferCI's value is **neutrality + reproducibility**. Contributions must not
weaken either. Two rules follow from that:

1. **Every number ships from a pinned, documented methodology.** If you add a
   benchmark, put its methodology in `spec/` first.
2. **The judge never favors a vendor.** Regression thresholds are published in
   `inferci/methodology.py` and applied uniformly.

## Ways to contribute

### 1. Add a runner (highest impact)

A runner wraps one inference backend. Implement the two-method protocol in
`inferci/runners/base.py` and register it in `inferci/runners/__init__.py`:

```python
from inferci.runners.base import Runner
from inferci.schema import BenchmarkSpec, Environment, RunResult

class MyRunner(Runner):
    id = "my_backend"
    name = "My Backend"

    def capture_environment(self) -> Environment: ...
    def run(self, spec: BenchmarkSpec, env: Environment) -> RunResult: ...
```

Requirements:
- capture environment **automatically** (version, commit, device) — never typed
  by hand.
- fill `metrics.pp_tps` and `metrics.tg_tps` at minimum; latency if you can
  measure it per-token.
- put raw backend output in `result.raw`, never parse raw output for judgments.

### 2. Add a benchmark spec / workload

A spec is `BenchmarkSpec` (see `inferci/schema.py`). Contribute a workload
matrix that matters to real users (long context, code, RAG, agentic, batch).

### 3. Contribute regression evidence

The `data/` set is crowdsourced evidence that the problem is real. Scripts in
`data/collect.sh` + `data/process.py` reproduce it. Add more engines/issues.

### 4. Docs & bugs

Open an issue or PR. Keep the zero-dependency property: **no new runtime deps**
unless there is a very strong reason.

## Development

```bash
python -m unittest discover -s tests -v          # unit tests
INFERCI_TEST_MODEL=/path/to/model.gguf python -m unittest discover -s tests -v  # + integration
```

## Code of conduct

Be neutral about tools, kind to people.
