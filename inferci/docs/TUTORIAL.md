# Tutorial

A 5-minute path from zero to a regression gate. InferCI has **zero runtime
dependencies** — everything below runs on stock Python 3.10+.

## 0. Install

```bash
cd inferci
pip install -e .          # or just use it in-place, nothing to install
inferci --version         # -> inferci 0.2.0
```

## 1. Try it with no model (mock runner)

The `mock` runner emits deterministic numbers, so you can exercise the entire
pipeline (ledger → diff → report → dashboard) on any machine:

```bash
# a baseline
inferci run --backend mock --model-id demo --quantization none --db demo.db

# a candidate that is 15% slower (simulates an upgrade regression)
inferci run --backend mock --model-id demo --quantization none \
            --set slowdown=0.85 --db demo.db

inferci list   --db demo.db
inferci report --db demo.db

# diff flags the regression (exit code 1)
inferci diff <base_run_id> <candidate_run_id> --db demo.db
```

`inferci diff` exits non-zero when a regression is detected, so it plugs
straight into CI as a gate.

## 2. Benchmark llama.cpp (CPU, real numbers)

```bash
# build llama.cpp once
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cmake -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON
cmake --build llama.cpp/build -j8 --target llama-bench llama-cli llama-server

# get a small model (HuggingFace or ModelScope mirror)
mkdir -p models && cd models
curl -L -o qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/master/qwen2.5-0.5b-instruct-q4_k_m.gguf
cd ..

# aggregate throughput (prefill/decode)
INFERCI_LLAMA_BENCH=llama.cpp/build/bin/llama-bench \
inferci run --backend llama_cpp --model-file models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
            --model-id Qwen2.5-0.5B-Instruct --quantization Q4_K_M \
            --prompt-tokens 512 --gen-tokens 128 --device cpu --db demo.db
```

## 3. Measure latency from a real HTTP stream

```bash
# starts a local llama-server, measures TTFT/ITL, tears it down
INFERCI_LLAMA_SERVER=llama.cpp/build/bin/llama-server \
inferci run --backend llama_server --model-file models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
            --model-id Qwen2.5-0.5B-Instruct --quantization Q4_K_M \
            --prompt-tokens 64 --gen-tokens 32 --device cpu --db demo.db

# concurrency: aggregate system throughput under N parallel requests
INFERCI_LLAMA_SERVER=llama.cpp/build/bin/llama-server \
inferci run --backend llama_server --model-file models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
            --model-id Qwen2.5-0.5B-Instruct --quantization Q4_K_M \
            --prompt-tokens 64 --gen-tokens 32 --batch 4 --device cpu --db demo.db
```

The `vllm` / `sglang` / `trt_llm` / `tgi` runners use the same `OpenAI-compatible`
measurement path — point them at any endpoint with `base_url` (see
`inferci/README.md`).

## 4. Long-context quality gate (RecallGate)

```bash
python -m inferci.quality --base-url http://127.0.0.1:8080 --model my-model \
    --budgets 512,1024,2048 --instance cpu.m7i.xlarge
```

Runs a deterministic needle-in-haystack probe at each context budget and emits a
PASS/FAIL verdict against the full-context baseline, with `quality_per_dollar`.

## 5. Visualize

```bash
inferci dashboard --db demo.db --out dashboard.html
open dashboard.html
```

## 6. Develop & verify

```bash
make test              # python -m unittest discover -s tests
make lint              # ruff check inferci tests
make format-check      # ruff format --check inferci tests
```

Full coverage (`~80%`) requires the local integration environment — set
`INFERCI_LLAMA_BENCH` / `INFERCI_LLAMA_SERVER` / `INFERCI_TEST_MODEL` and the
integration tests will actually run against real binaries and a real model.
