# 推理 Infra 行业格局、趋势与专家判断

> 调研方式：多次联网检索（英文关键词为主），覆盖 arXiv 顶会论文、公司技术博客、GitHub 仓库、行业报道与专家 Newsletter。所有关键事实尽量附出处 URL。GitHub star 数据为调研当天通过 GitHub 页面抓取的近似值（量级），活跃度以 `pushed_at`/发布节奏判断。
> 调研时间基准：2025 年中后段——2026 年初。

---

## 1. 现状总览（为什么是推理 Infra，瓶颈在哪）

**为什么现在是"推理时代"而不是"训练时代"**：过去 3 年，大模型从"单点训练"进入"全民调用"阶段。SemiAnalysis 等机构持续指出，AI 的资本开支与收入重心正从训练集群向推理集群迁移（NVIDIA GTC 2025 的整个叙事围绕"reasoning-era inference"展开）：
- [SemiAnalysis — NVIDIA GTC 2025: Built For Reasoning, Dynamo Inference, Jensen Math](https://newsletter.semianalysis.com/p/nvidia-gtc-2025-built-for-reasoning-vera-rubin-kyber-cpo-dynamo-inference-jensen-math-feynman)

**本质瓶颈（三句话讲清）**：
1. **解码是访存密集（memory-bound）而非算力密集（compute-bound）**：自回归每生成一个 token 都要读一遍全部权重 + 读/写全部 KV cache，GPU 算力（FLOPS）大量空转，卡在 HBM 带宽与 KV cache 容量上。这解释了为什么几乎所有优化都围绕"省内存 / 省带宽 / 扩大有效 batch"。
2. **prefill 与 decode 特性相反**：prefill 是算力密集、可并行；decode 是访存密集、串行。二者抢同一张卡会互相拖累（TTFT vs TPOT 矛盾），于是催生**预填/解码分离（disaggregation）**。
3. **推理成本 = 硬件利用率 × 算法效率**：token 单价暴跌的背后是三层叠加——算法（MoE、MLA、量化、投机解码）、系统（连续批处理、PagedAttention、KV 复用）、硬件（Blackwell、TPU、专用 ASIC）。当前卡点已从"单卡 kernel"上移到**多节点/机架级编排、异构设备、长上下文 KV 管理、多模态与 Agent 工作流**。

**一句话格局**：底层推理引擎已收敛到 **vLLM / SGLang / TensorRT-LLM** 三强，上层编排层（NVIDIA Dynamo、Ray Serve、自建 Model Runner）和 KVCache/异构/多模态是新的主战场。

---

## 2. 关键技术（带出处）

| 技术 | 解决什么瓶颈 | 出处 |
|---|---|---|
| **PagedAttention**（vLLM） | KV cache 碎片化→按"页"管理，内存利用率/吞吐大幅提升 | [vLLM 项目](https://github.com/vllm-project/vllm)；SOSP 2023 论文（见论文清单） |
| **FlashAttention** | 注意力访存瓶颈→分块+SRAM 计算，降低显存读写 | 见论文清单（NeurIPS 2022） |
| **连续批处理 (Continuous Batching)** | 请求级 batch，消除 padding 浪费 | vLLM/TGI/SGLang 核心机制 |
| **预填/解码分离 (Disaggregation)** | prefill/decode 抢资源→拆到不同资源池 | [DistServe (OSDI 2024)](https://www.usenix.org/system/files/osdi24_proceedings_interior.pdf)；Mooncake（见下） |
| **MLA (Multi-head Latent Attention)** | 把 KV cache 压缩到低秩 latent，长上下文显存/带宽大降 | [DeepSeek-V3 Technical Report (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437) |
| **MoE (Mixture-of-Experts)** | 每次只激活部分参数，算力/带宽按需 | DeepSeek-V3、Qwen3、Llama 4 均 MoE 化 |
| **投机解码 / 稀疏自投机** | 用小模型/自身草稿并行猜 token，降低串行延迟 | [SparseSpec 仓库](https://github.com/sspec-project/SparseSpec) |
| **KVCache 中心化池化**（Mooncake） | 长上下文复用 KV，跨实例共享 | [Mooncake (arXiv 2407.00079)](https://arxiv.org/abs/2407.00079) |
| **TensorRT-LLM** | NVIDIA 专用：kernel 融合、量化、In-flight batching | [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |
| **NVIDIA Dynamo** | 多节点编排：KV 路由、disaggregated serving、SLO 自动扩缩 | [NVIDIA Dynamo 博客](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) |
| **Triton (GPU kernel 语言)** | 用 Python 写高性能 kernel，编译到多硬件 | [triton-lang/triton](https://github.com/triton-lang/triton) |
| **XLA/JetStream**（TPU） | TPU 上的吞吐/内存优化引擎 | [google/JetStream](https://github.com/google/JetStream) |
| **MLX**（Apple Silicon 统一内存） | 端侧/本机推理，共享内存模型 | [ml-explore/mlx](https://github.com/ml-explore/mlx) |

---

## 3. 论文清单（名称 + 年份 + venue + 一句话核心）

- **DeepSeek-V3 Technical Report** — 2025, arXiv（2412.19437）— 671B/37B 激活 MoE + MLA + FP8 + aux-loss-free 负载均衡，把训练成本压到 ~$5.6M。
  [arXiv](https://arxiv.org/abs/2412.19437)
- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL** — 2025, arXiv — 纯 RL 涌现推理能力，引爆 test-time compute 范式。
- **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving** — 2025, **FAST 2025 Best Paper**（Moonshot AI + 清华）— 以 KV cache 为中心解耦预填/解码，服务 Kimi。
  [arXiv](https://arxiv.org/abs/2407.00079) · [USENIX PDF](https://www.usenix.org/system/files/fast25-qin.pdf)
- **Efficient Memory Management for Large Language Model Serving with PagedAttention**（vLLM）— 2023, **SOSP 2023** — 分页 KV cache，奠定主流引擎内存管理。
- **FlashAttention / FlashAttention-2** — 2022/2023, NeurIPS 2022 — IO-aware 注意力，降低显存访存。
- **Splitwise: Efficient Generative LLM Inference Using Phase Splitting** — 2024, ISCA 2024（Microsoft）— 按 phase 把 prefill/decode 分到异构硬件（含 CPU）。
- **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving** — 2024, **OSDI 2024** — 面向 goodput 的预填/解码分离。
  [USENIX OSDI'24](https://www.usenix.org/system/files/osdi24_proceedings_interior.pdf)
- **HexGen-2: Disaggregated Generative Inference of LLMs in Heterogeneous Environment** — 2025, ICLR 2025 — 异构设备上的分布式生成推理。
- **Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations** — 2019, MAPL（OpenAI）— 奠定"Python 写 kernel"路线。
- **The Anatomy of a Triton Attention Kernel** — 2025, arXiv（2511.11581）— 最新 Triton 注意力 kernel 解剖。
  [arXiv](https://arxiv.org/abs/2511.11581)

---

## 4. 开源项目盘点（名称 + star 量级 + 活跃度 + 维护方）

> star 为调研当天 GitHub 抓取近似值；"量级"供横向比较。

| 项目 | star 量级 | 活跃度 | 维护方 / 备注 |
|---|---|---|---|
| **vllm-project/vllm** | ~89k | 极高（事实标准） | UC Berkeley 团队 + 社区，PagedAttention/连续批处理事实标准 |
| **sgl-project/sglang** | ~32k | 极高 | 伯克利/社区，RadixAttention、结构化输出，推理模型吞吐领先 |
| **ggml-org/llama.cpp** | ~124k | 极高 | ggml/社区，端侧/CPU 推理事实标准 |
| **ray-project/ray** | ~43k | 极高 | Anyscale，分布式编排底座（Ray Serve LLM 之上跑 vLLM） |
| **ml-explore/mlx** | ~28k | 高 | Apple，Apple Silicon 统一内存推理框架 |
| **volcengine/verl** | ~23k | 高 | 字节跳动 Seed，RLHF/RL 训练与推理框架（doubao 同款） |
| **triton-lang/triton** | ~20k | 高 | 原 OpenAI Triton，现社区治理（PyTorch 生态核心 kernel DSL） |
| **NVIDIA/TensorRT-LLM** | ~14k | 高 | NVIDIA，官方高性能推理引擎 |
| **ai-dynamo/dynamo** | ~7.8k | 中-高（较新） | NVIDIA 开源，多节点分布式推理编排（聚合 TRT-LLM/vLLM） |
| **kvcache-ai/Mooncake** | ~6.3k | 中-高 | Moonshot AI，KVCache 中心化解耦服务 |
| **google/JetStream** | ~450 | 中 | Google，TPU/XLA 推理引擎（GPU 支持在推进） |

---

## 5. 公司落地（逐家）

**NVIDIA（TensorRT-LLM / Dynamo / NIM / Blackwell）**
- 三层栈：**TensorRT-LLM**（单卡/单机高性能引擎）→ **NIM**（预打包微服务容器，一键部署模型）→ **Dynamo**（GTC 2025 开源的多节点分布式推理编排，聚合 TRT-LLM/vLLM/SGLang，提供 KV 路由、disaggregated serving、SLO 自动扩缩）。
- 硬件：**Blackwell GB200 NVL72**（NVLink 域）在 SemiAnalysis InferenceMAX 基准中领先；DeepSeek-R1 MoE 推理性能据称达同规模 AMD MI355X 集群约 **28 倍**。
- 出处：[Dynamo 官方博客](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) · [NVIDIA Blackwell InferenceMAX](https://blogs.nvidia.com/blog/blackwell-inferencemax-benchmark-results/) · [Dynamo 0.4 发布（4x 性能、SLO 扩缩）](https://developer.nvidia.com/blog/dynamo-0-4-delivers-4x-faster-performance-slo-based-autoscaling-and-real-time-observability/)

**Meta（Llama 推理、Gloe）**
- 统一服务层 **Model Runner**：路由、并行编排、内存管理、SLO、batch 形成；在 **异构 TPU+GPU** 上同时跑 TP/CP/EP 三级并行，按网络带宽分层映射（TP 走 NVLink、CP/EP 走 RoCE v2）。
- **KernelEvolve**：Agentic 方式生成硬件优化 CUDA kernel。
- **Gloe**：任务列为 Meta 的异构推理运行时，但公开技术文档稀缺（多轮检索未能定位权威一手出处；Meta 公开可验证的推理叙事以 Model Runner + vLLM/TRT-LLM/SGLang + 异构 TPU/GPU 为主）。此处诚实标注"公开资料有限"。
- 出处：[Meta 推理平台解读（第三方深度整理）](https://github.com/harshuljain13/llm-inference-at-scale/blob/master/content/10_production_stories/09.1_meta_inference_platform/meta_inference_platform.md) · [Meta 转用 Google TPU 的讨论](https://www.atsky.io/thought-leadership-blogs/why-meta-is-turning-to-google-tpus-and-why-nvidia-is-still-market-leader)

**Google（TPU / Gemma / JetStream）**
- **JetStream**：XLA 设备上的吞吐/内存优化推理引擎，TPU 起步（GPU 支持 PR 欢迎），官方用于 GKE 上 TPU 服务 Gemma。
- TPU 路线：v5e/v6 Trillium，Meta 等大厂亦采购 TPU 做异构。
- 出处：[google/JetStream](https://github.com/google/JetStream) · [GKE + TPU + JetStream 服务 Gemma 教程](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-tpu-jetstream)

**OpenAI（Triton / serving）**
- **Triton** 是最重要的公开遗产（kernel DSL，现社区治理）。serving 栈本身高度不透明：靠 Batch API（延迟换折扣）、**蒸馏**（用小模型替换 GPT-4o 级）与系统级优化压成本，并靠"系统优化使推理成本减半"支撑海量请求。
- 出处：[triton-lang/triton](https://github.com/triton-lang/triton) · [OpenAI 通过系统优化削减推理成本（二手报道）](https://vendordeep.com/report/openai-slashes-inference-costs-runs)

**Anthropic**
- Claude 服务栈同样不公开；公开重点是 **MCP（Model Context Protocol）** 与 Agent 工作流（managed agents、MCP tunnels），即"推理平台 + 工具/上下文"层，而非底层 kernel。
- 出处：[Claude Managed Agents / MCP tunnels](https://claude.com/blog/claude-managed-agents-updates)

**DeepSeek（V3/R1 的 MLA、Fire-Flyer 集群、成本公开）**
- **MLA + DeepSeekMoE + FP8 + aux-loss-free 均衡**把训练压到 ~$5.6M（H800 集群）；自建 **Fire-Flyer** 万卡级集群。
- **成本公开**：2025 年 3 月公布理论成本利润率 **545%**（按 R1/V3 定价：输入 cache-hit $0.07/M、输出 ~$1.10/M 量级），直接击穿行业"推理不赚钱"叙事，引发 token 价格战。
- 出处：[DeepSeek-V3 arXiv](https://arxiv.org/abs/2412.19437) · [545% 利润率解读](https://finance.eastmoney.com/a/202503023333900797.html)

**ByteDance（doubao / veRL）**
- **doubao** 大模型对外服务；**veRL**（已开源）是 RLHF/RL 训练+推理框架，据称吞吐最高提升 20 倍，支撑 doubao 的强化学习与推理。
- 出处：[字节 Seed：veRL RLHF 框架开源（最高 20x 吞吐）](https://seed.bytedance.com/zh/blog/%E6%9C%80%E9%AB%98%E6%8F%90%E5%8D%8720%E5%80%8D%E5%90%9E%E5%90%90%E9%87%8F-%E8%B1%86%E5%8C%85%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9B%A2%E9%98%9F%E5%8F%91%E5%B8%83%E5%85%A8%E6%96%B0-rlhf-%E6%A1%86%E6%9E%B6-%E7%8E%B0%E5%B7%B2%E5%BC%80%E6%BA%90) · [volcengine/verl](https://github.com/volcengine/verl)

**Alibaba（Qwen / PAI）**
- **Qwen** 系列（含 Qwen3 MoE、Qwen-Coder）；**PAI（EAS / PAI-PPU）** 平台一键拉起 Qwen3 推理服务，集成 vLLM/SGLang 引擎；据称私有化部署 Qwen3-Coder 推理成本比公有 API 低 60%。
- 出处：[PAI-PPU 一键拉起 Qwen3 推理](https://help.aliyun.com/en/pai/use-cases/one-button-pull-up-qwen3-inference-service-in-pai-ppu) · [PAI 私有化部署 Qwen3-Coder 降本 60%](https://developer.aliyun.com/article/1734028)

**Apple（Foundation Models / MLX）**
- 开源 **Foundation Models 框架**（并接入 Claude/Gemini）；**MLX** 面向 Apple Silicon 统一内存，是端侧/本机推理主力；Apple Intelligence 走"端侧 + Private Cloud Compute"混合。
- 出处：[Apple 开源 Foundation Models 框架（二手报道）](http://rits.shanghai.nyu.edu/ai/apple-open-sources-its-foundation-models-framework-adds-claude-and-gemini/) · [ml-explore/mlx](https://github.com/ml-explore/mlx)

**Moonshot（Kimi / Mooncake）**
- **Kimi** 长上下文服务；**Mooncake** 是 Kimi 背后的 KVCache 中心化解耦服务平台，获 FAST 2025 Best Paper。
- 出处：[kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) · [FAST 2025 最佳论文报道（清华）](https://www.tsinghua.edu.cn/en/info/1245/14138.htm)

**初创（Together / Fireworks / Baseten / Groq / Cerebras / Modular / Anyscale）**
- **Together AI / Fireworks / Baseten**：GPU 云 + 推理平台，围绕 vLLM/SGLang 做托管、定价与调优（Artificial Analysis 等持续横向评测其延迟/成本/可用性）。
- **Groq**：自研 LPU（SRAM 密集）主打极低延迟；**Cerebras**：晶圆级 Wafer-Scale Engine 主打超高 tokens/s。
- **Modular（Mojo/MAX）**：MAX 引擎跨 NVIDIA/AMD/Apple 统一推理，宣传在 AMD MI355 上 14 天达到 SOTA 性能。
- **Anyscale（Ray）**：Ray Serve LLM 之上跑 vLLM，提供 Wide-EP、disaggregated serving 与自动扩缩。
- 出处：[推理厂商横向评测（dataku）](https://dataku.ai/blog/ai-inference-market-25-providers-ranked) · [Modular 25.6 跨厂商 GPU](https://www.modular.com/blog/modular-25-6-unifying-the-latest-gpus-from-nvidia-amd-and-apple) · [Ray Serve LLM Wide-EP](https://www.anyscale.com/blog/ray-serve-llm-anyscale-apis-wide-ep-disaggregated-serving-vllm) · [Groq/Cerebras 速度对比](https://deploybase.ai/articles/fastest-llm-api)

---

## 6. 趋势判断（未来 12–24 个月）

1. **test-time compute 主导推理成本**：推理模型（o3/R1/Gemini 类）把更多算力从训练搬到推理，推理成为主要能耗与成本来源（[Joule 论文：AI 推理能耗与 test-time scaling](https://www.cell.com/joule/fulltext/S2542-4351(26)00114-5)）。
2. **token 单价继续暴跌，但总支出反升（Jevons 悖论）**：18 个月内顶级模型成本降 ~280 倍（$20/M → $0.07/M），廉价智能反致用量爆发、"算力通胀"。
   [Silicon Canals：280 倍成本下降](https://siliconcanals.com/the-cost-of-running-a-top-tier-ai-query-dropped-280-fold-in-roughly-18-months-from-20-per-million-tokens-in-late-2022-to-seven-cents-by-late-2024-a-price-collapse-with-no-real-parallel-in/)
3. **机架级/多节点成为默认单位**：NVL72 这类 NVLink 域 + disaggregated serving 让"一个模型跨数百 GPU"常态化；Dynamo 类编排层标准化。
4. **异构设备统一调度**：TPU+GPU+ASIC（+端侧）混合是必然（Meta 已做），"一套 serving 跑所有芯片"成为显性诉求（Modular MAX、JetStream 的 GPU 化）。
5. **Agent/工具化推理**：MCP、function calling、长 Agent 会话带来**长上下文 KV 复用与状态持久化**需求，KVCache 池化/缓存成为核心杠杆。
6. **多模态与扩散模型推理**兴起：VLM/视频/音频的 serving 引擎（vLLM Omni 等）开始补课，但远未成熟。

---

## 7. 已饱和点（赛道已拥挤）

- **纯 GPU 托管/推理 API**：Together/Fireworks/Baseten/各云厂商 + OpenRouter/Martian 网关，价格战激烈，差异化难。
- **单卡/单机 LLM 推理引擎**：vLLM、SGLang、TRT-LLM 三强格局已定，新做一个"通用 LLM 引擎"基本没有空间。
- **通用 MoE/量化/投机解码论文**：FlashAttention/PagedAttention/投机解码被大量重复改进，边际创新有限。
- **RLHF/RL 框架**：veRL、OpenRLHF、TRL、Ray 等已成熟，再做通用 RLHF 框架价值不大。
- **NVIDIA 专有加速**：TRT-LLM + NIM + Dynamo 生态内卷，第三方在纯 NVIDIA 单卡优化上难与官方竞争。

---

## 8. 被忽视的空白与机会

1. **异构设备统一 serving（真开源）**：Meta/Google 内部有，开源侧没有一个"一套代码跑 TPU+GPU+ASIC 且性能不崩"的成熟引擎（JetStream 尚弱、MAX 闭源）。
2. **跨引擎 KVCache 标准化与池化**：KV cache 压缩、跨实例/跨请求复用、以及预填/解码间的 KV 传输协议（Mooncake transfer engine）缺乏统一标准与中立实现。
3. **多模态/扩散模型推理系统**：VLM、视频、音频、扩散模型（图/视频生成）的 serving 优化是明显短板（vLLM Omni 刚起步）。
4. **推理的可观测性 / 成本治理 / SLO 自动扩缩**：Dynamo 开始做，但中立、跨引擎、面向"成本/延迟/质量"三维调优的开源工具稀缺。
5. **长上下文 + Agent 会话的状态与缓存层**：Agent 长会话的 KV/上下文持久化、压缩、跨会话复用是显性缺口。
6. **端侧/边缘统一 API**：MLX（Apple）、llama.cpp（CPU）、ExecuTorch（端侧）各搞各的，缺统一运行时与模型分发。
7. **推理成本基准与可复现评测**：SemiAnalysis/Artificial Analysis 是闭源/收费，开源可复现的成本-延迟-质量基准缺失。

---

## 9. 具体候选切入点（3–5 个，由行业判断支撑）

1. **异构 KV-cache 中心化服务层**（受 Mooncake FAST'25 + Meta 异构 + Dynamo KV 路由支撑）：做一个中立、跨引擎（vLLM/SGLang/TRT-LLM）、跨设备（GPU/TPU）的 KV cache 池化 + 传输协议标准库，切入长上下文/Agent 复用这一未标准化地带。
2. **多模态 & 扩散模型的推理引擎插件/运行时**（受 vLLM Omni 起步、VLM/视频 serving 短板支撑）：为 VLM/音频/视频/扩散模型提供预填-解码/调度/量化优化，避开已饱和的纯文本 LLM 引擎。
3. **推理 SLO 自动扩缩 + 成本治理可观测层**（受 Dynamo 0.4 SLO 扩缩 + "invoice disagrees"痛点支撑）：跨引擎、跨云的中立控制面，做成本/延迟/质量三维调优与 FinOps。
4. **异构设备统一 serving 运行时**（受 Meta 转 TPU + Modular MAX + JetStream 支撑）：以 XLA/MLIR 为底座做"一套 API 跑 NVIDIA+AMD+TPU+ASIC"的开源运行时，差异化于闭源 MAX 与 NVIDIA 绑定的 TRT-LLM。
5. **端侧统一推理运行时 + 模型分发**（受 MLX/llama.cpp 碎片化 + Apple 端侧战略支撑）：统一 Apple Silicon/CPU/NPU 的运行时与量化模型仓库，服务 Agent 端侧落地。

---

## 附：关键出处汇总（Citations）

- NVIDIA Dynamo: https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/
- NVIDIA Blackwell InferenceMAX: https://blogs.nvidia.com/blog/blackwell-inferencemax-benchmark-results/
- SemiAnalysis GTC 2025: https://newsletter.semianalysis.com/p/nvidia-gtc-2025-built-for-reasoning-vera-rubin-kyber-cpo-dynamo-inference-jensen-math-feynman
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- Mooncake: https://arxiv.org/abs/2407.00079
- vLLM: https://github.com/vllm-project/vllm
- SGLang: https://github.com/sgl-project/sglang
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
- Triton: https://github.com/triton-lang/triton
- JetStream: https://github.com/google/JetStream
- MLX: https://github.com/ml-explore/mlx
- verl: https://github.com/volcengine/verl
- Ray: https://github.com/ray-project/ray
- llama.cpp: https://github.com/ggml-org/llama.cpp
- Mooncake GitHub: https://github.com/kvcache-ai/Mooncake
- Dynamo GitHub: https://github.com/ai-dynamo/dynamo
- 推理厂商评测: https://dataku.ai/blog/ai-inference-market-25-providers-ranked
- 成本下降 280 倍: https://siliconcanals.com/the-cost-of-running-a-top-tier-ai-query-dropped-280-fold-in-roughly-18-months-from-20-per-million-tokens-in-late-2022-to-seven-cents-by-late-2024-a-price-collapse-with-no-real-parallel-in/
- DeepSeek 545% 利润率: https://finance.eastmoney.com/a/202503023333900797.html
- Modular 25.6: https://www.modular.com/blog/modular-25-6-unifying-the-latest-gpus-from-nvidia-amd-and-apple
- Ray Serve LLM Wide-EP: https://www.anyscale.com/blog/ray-serve-llm-anyscale-apis-wide-ep-disaggregated-serving-vllm
