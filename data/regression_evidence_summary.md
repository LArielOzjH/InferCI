# Performance Regression Evidence Summary

- Curated regression reports: **78** (full raw pool: 657 unique issues)
- With concrete numeric claim: **78**

## Per-repo breakdown (curated)

- **ggml-org/llama.cpp**: 26 issues (26 with numeric claim)
- **sgl-project/sglang**: 26 issues (26 with numeric claim)
- **vllm-project/vllm**: 26 issues (26 with numeric claim)

## Representative 'slower after upgrade' cases

- **[Vulkan TG Performance Degradation at Large Context Sizes on RDNA4](https://github.com/ggml-org/llama.cpp/issues/24483)** (ggml-org/llama.cpp#24483, closed, created 2026-06-11)
  - claim: **Is this expected behavior for MoE models on Vulkan?** The 35.7% degradation from 2k→50k context seems high compared to dense models.
- **[[Bug]: FP8 speed regression in version 0.16.0rc2.dev87+g0b20469c6 (latest nightly)](https://github.com/vllm-project/vllm/issues/34377)** (vllm-project/vllm#34377, closed, created 2026-02-11)
  - claim: Qwen Coder Next FP8 in TP went from 100+ TPS to 38 TPS Stepfun3.5 FP8 using PP 66 TPS -> 18 TPS vllm/vllm-openai:nightly \ --model /models/step35flash \ --served-model-name step3p5-flash \ --host 0.0.0.0 \ --port 5000 \…
- **[Performance regression: 4090 GPUs slower on v0.4.8-cu126 (was 27→20 tokens/sec, A100 unaffected)](https://github.com/sgl-project/sglang/issues/7568)** (sgl-project/sglang#7568, closed, created 2025-06-26)
  - claim: After upgrading from lmsysorg/sglang:v0.4.6.post5-cu124 to lmsysorg/sglang:v0.4.8-cu126, I've noticed a **significant drop in generation speed** and GPU utilization on my setup with 2x4090 (48Gb each).
- **[[Performance]: Eagle3 speculative decoding latency regression in v0.19 vs v0.18](https://github.com/vllm-project/vllm/issues/39940)** (vllm-project/vllm#39940, open, created 2026-04-15)
  - claim: On v0.19, the same Eagle3 setup only reduces TPOT from 3.10ms to 2.84ms (8.4% speedup) — the regression cuts the spec decode benefit nearly in half.
- **[[Perf]: ~23% output throughput regression on Qwen3.5-397B NVFP4 decode (8×B200) over the last 10 days](https://github.com/vllm-project/vllm/issues/39004)** (vllm-project/vllm#39004, closed, created 2026-04-04)
  - claim: ## Summary Output token throughput for nvidia/Qwen3.5-397B-A17B-NVFP4 with -dp 8 -ep on 8×B200 has regressed **from ~47,200 tok/s to ~38,300 tok/s (−19%)** between March 28 and April 4.
- **[[Performance] TTFT regression from v0.5.4 to 0.6.2](https://github.com/vllm-project/vllm/issues/8918)** (vllm-project/vllm#8918, closed, created 2024-09-27)
  - claim: Your current environment Model Input Dumps 🐛 Describe the bug ## TLDR We are seeing TTFT regression when upgrading from v0.5.4 to v0.6.2, tldr, on a low QPS/batch size workload, in particularly 15% to 30% TTFT…
