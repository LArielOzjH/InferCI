# InferCI — package

Zero-dependency Python harness for neutral inference performance & cost
regression CI. Standard-library only (dataclasses / sqlite3 / argparse /
subprocess) so it runs unchanged inside any CI/Docker/bare-metal.

## Install

```bash
pip install -e .          # or just use it in-place, no deps to install
```

## CLI

```bash
# run a benchmark (llama.cpp runner)
inferci run --model-file models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
            --model-id Qwen2.5-0.5B-Instruct --quantization Q4_K_M \
            --prompt-tokens 512 --gen-tokens 128 --repeats 3 --device cpu

# with cloud cost
inferci run ... --instance gpu.a10g.g5.xlarge

# history
inferci list
inferci report

# regression judge between two runs
inferci diff <base_run_id> <candidate_run_id>
```

`inferci diff` exits non-zero when a regression is detected — so it plugs
directly into CI as a gate.

## Metrics produced

- `pp_tps` — prompt-processing (prefill) throughput → drives TTFT
- `tg_tps` — token-generation (decode) throughput → drives ITL / steady state
- `ttft_ms`, `itl` — latency (measured by serving runners; derived on the
  llama-bench single-stream runner)
- `price_per_input_1m`, `price_per_output_1m` — $/1M tokens when an instance
  price is supplied

## Adding a runner

Implement `inferci.runners.base.Runner` (two methods: `capture_environment`,
`run`), then `register()` it in `inferci/runners/__init__.py`. See
`inferci/runners/llama_cpp.py` as the reference implementation.

## Tests

```bash
python -m unittest discover -s tests -v
```

The end-to-end test runs when both a llama-bench binary and a model are
available:

```bash
INFERCI_LLAMA_BENCH=../llama.cpp/build/bin/llama-bench \
INFERCI_TEST_MODEL=../models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
python -m unittest discover -s tests -v
```

## Methodology & schema

- [`spec/benchmark-spec.md`](spec/benchmark-spec.md) — what "reproducible" means
- [`spec/result-schema.md`](spec/result-schema.md) — the machine-readable result
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — BYO-runner neutral design
