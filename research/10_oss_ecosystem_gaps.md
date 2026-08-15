# 开源推理 Infra 生态盘点 + 什么样的 OSS 项目能赢

> 调研日期：2026-08-15。所有 star/贡献者数据来自当日对 GitHub 页面与 shields.io badge 的实时抓取（量级可信，精确值以仓库为准）。结论与来源 URL 均附在文中。

---

## 0. TL;DR（一分钟结论）

1. **推理引擎已形成双寡头：vLLM 与 SGLang**。二者都已经"上岸"——vLLM 背后是 Neural Magic→Red Hat 收购 + 2026 年并入 Linux Foundation 旗下的 PyTorch Foundation；SGLang 团队 2026 年 5 月成立公司 RadixArk，拿到 Accel 领投、NVIDIA/AMD/Intel 参与的 1 亿美元种子轮。**通用 Python 推理引擎这条赛道对个人/小团队已经关闭。**
2. **llama.cpp / ollama / MLX / exllamav2 证明了"单机/边缘/特定硬件"是独立于数据中心的另一条价值线**，它们由 1–2 人起步并活到今天，因为 vLLM/SGLang 的根本定位（多卡、多租户、数据中心 GPU）与它们不重叠。
3. **被巨头吸收的领域**（别去做）：通用 attention 内核（FlashInfer 被 vLLM/SGLang 吸收）、通用结构化输出（outlines 的功能被 MLC-AI 的 xgrammar 取代，xgrammar 已被 vLLM/SGLang/NVIDIA NIM 共同采用）。
4. **最被低估、且大项目不会轻易吃掉的空白**集中在四个方向：① 异构/多厂商调度层；② NPU/端侧（含 Apple ANE、Windows Copilot+ NPU）；③ 推理性能回归测试/成本可观测（perf CI）；④ 带正确性保证的结构化生成与验证。详见第 8 节。

---

## 1. 现状总览：一张地图

推理 Infra 可以按"执行发生在哪、面向谁"切成三层：

| 层 | 代表项目 | 本质 |
|---|---|---|
| **数据中心/多卡服务引擎** | vLLM、SGLang、TensorRT-LLM、TGI、LMDeploy | 追求吞吐 × 延迟 × 多租户 × 多 GPU |
| **单机/边缘/本地运行时** | llama.cpp、ollama、MLX、exllamav2、mistral.rs、candle、burn、exo | 追求"任何设备跑起来"、低门槛、低资源 |
| **底层算子/编译器/框架** | Triton、TVM、FlashInfer、xgrammar、transformers | 被上层消费的"建材" |

三条关键结构事实：

- **上层在收敛**：数据中心层基本只剩 vLLM vs SGLang 两个赢家（TRT-LLM 是 NVIDIA 自留地，TGI 明显掉队）。
- **底层在被吸收**：FlashInfer（attention 内核库）进了 vLLM 和 SGLang；outlines 的 JSON-schema 结构化生成被 xgrammar 取代，xgrammar 又成为 vLLM/SGLang/NVIDIA NIM 的默认后端。
- **单机层是长尾**：llama.cpp、ollama、MLX 各自卡住一个"巨头不愿碰"的生态位（CPU/边缘、一键本地、Apple 芯片），活得很好但天花板明确。

---

## 2. 关键技术（带出处）

### 2.1 连续批处理 + 分页 KV Cache（PagedAttention）
**解决什么瓶颈**：早期服务按请求静态分配 KV cache，显存利用率 <40%，吞吐被显存卡死。PagedAttention 把 KV cache 按"页"管理、按需映射，消除碎片，批量吞吐提升 2–4×。
**卡点在哪**：page 管理引入分配/抢占开销，preemption + 异步调度仍有竞态 bug（见第 5 节 vLLM issue）。
- 论文：*Efficient Memory Management for LLM Serving with PagedAttention*（SOSP 2023，arXiv 2309.06180）https://arxiv.org/abs/2309.06180

### 2.2 前缀缓存 / RadixAttention
**解决什么瓶颈**：多轮对话、agent、多请求共享 system prompt 时，重复计算前缀 token 的 KV。RadixAttention 用 radix 树自动复用前缀 KV，SGLang 报告最高 10× 吞吐提升。
- 论文：*SGLang: Efficient Execution of Structured Language Model Programs*（SOSP 2024，arXiv 2312.07104）https://arxiv.org/abs/2312.07104

### 2.3 FlashAttention 系列
**解决什么瓶颈**：标准 attention 的 O(n²) 显存读写（HBM IO 是真正的墙）。分块 + 重计算把 attention 变成 IO-bound 下的最优实现，成为所有引擎的默认内核。
- FlashAttention（NeurIPS 2022，arXiv 2205.14135）；FlashAttention-2（arXiv 2307.08691）；FlashAttention-3（Hopper 上 1.5–2×，arXiv 2407.08608）https://arxiv.org/abs/2205.14135

### 2.4 编译器/DSL：Triton 与 TVM
**解决什么瓶颈**：手写 CUDA 门槛高、不可移植。Triton 用 Python DSL + tile 抽象自动生成 GPU kernel，让"写算子"从 CUDA 专家下沉到普通工程师——这是 vLLM/SGLang 能快速迭代算子（FP8/FP4/Marlin）的根本原因。TVM 走"图级 + 自动调优 + 多后端"路线，IREE/MLIR 生态同源。
- Triton（MAPL 2019，arXiv 1904.10544）https://arxiv.org/abs/1904.10544
- TVM（OSDI 2018，arXiv 1802.04799）https://arxiv.org/abs/1802.04799
**卡点**：Triton 是 NVIDIA 中心（AMD ROCm 的 Triton 支持至今是 bug 大户，见第 5 节）；TVM 生态因 OctoML 商业化受挫而长期缺资源，但 xgrammar（MLC-AI，陈天奇团队）证明了 TVM 血统仍有后劲。

### 2.5 量化内核：AWQ / GPTQ / Marlin / FP8/FP4
**解决什么瓶颈**：显存带宽是 decode 阶段的墙。4-bit（GPTQ/AWQ/Marlin）、8-bit（FP8）、4-bit（FP4/MXFP4）压缩权重，让更大模型装进更少显存，Marlin 达到接近 FP16 的 decode 速度。
- AWQ（MLSys 2024，arXiv 2306.00978）https://arxiv.org/abs/2306.00978
- GPTQ（ICLR 2023，arXiv 2210.17323）https://arxiv.org/abs/2210.17323
- Marlin（HPCA 2024，arXiv 2401.09225）https://arxiv.org/abs/2401.09225

### 2.6 注意力内核库 FlashInfer
**解决什么瓶颈**：不同引擎重复造 attention 内核、难适配新架构。FlashInfer 提供可定制 attention 后端，已被 vLLM/SGLang 采纳为内核选项——这是"独立内核项目被吸收"的教科书案例。
- *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving*（MLSys 2025 最佳论文，arXiv 2501.01005）https://arxiv.org/abs/2501.01005

### 2.7 预填/解码分离（Disaggregation）
**解决什么瓶颈**：prefill（compute-bound）与 decode（memory-bound）混跑互相干扰，goodput 下降。把两者拆到不同 GPU 池、用 KV 传输连接（DistServe、Mooncake、NVIDIA Dynamo）。
- *DistServe*（OSDI 2024，arXiv 2401.09670）https://arxiv.org/abs/2401.09670
- Mooncake（Kimi，arXiv 2407.00079）

### 2.8 投机解码 / 多 token 预测（MTP）
**解决什么瓶颈**：decode 逐 token 串行，延迟高。用 draft 模型 / Medusa 头 / DeepSeek MTP 一次预测多个 token 再验证。DeepSeek MTP 让 SGLang 在高并发下吞吐提升约 1.7–1.9×。
- Speculative Decoding（ICML 2023，arXiv 2211.17192）；Medusa（ICML 2024，arXiv 2401.10774）
- SGLang MTP 落地博客 https://www.lmsys.org/blog/2025-07-17-mtp/

### 2.9 结构化生成：outlines → xgrammar
**解决什么瓶颈**：让 LLM 输出严格符合 JSON schema / 正则 / 语法（agent/函数调用刚需）。outlines 开创了"编译到有限状态机 + logits 掩码"路线，但 xgrammar 以更快、更可移植（MLIR 后端）的实现成为事实标准，已被 vLLM、SGLang、NVIDIA NIM 采用。
- xgrammar（MLC-AI）https://github.com/mlc-ai/xgrammar

---

## 3. 论文清单（名称 + 年份 + venue，一句话核心）

| 论文 | 年份 / Venue | 一句话核心 |
|---|---|---|
| PagedAttention (vLLM) | 2023 / SOSP | 分页 KV cache 消除显存碎片，批量吞吐 2–4× |
| SGLang | 2024 / SOSP | radix 树自动前缀缓存 + DSL，多轮/agent 吞吐大幅提升 |
| FlashAttention | 2022 / NeurIPS | 分块+重计算，attention 不再被 HBM 带宽卡死 |
| Triton | 2019 / MAPL | Python tile DSL 自动生成 GPU kernel，算子开发民主化 |
| TVM | 2018 / OSDI | 端到端编译栈 + 自动调优 + 多硬件后端 |
| GPTQ | 2023 / ICLR | 逐层量化 + 重建，把 LLM 压到 4-bit |
| AWQ | 2024 / MLSys | 激活感知量化，保护关键权重通道 |
| Marlin | 2024 / HPCA | 4-bit 矩阵乘内核，decode 逼近 FP16 |
| FlashInfer | 2025 / MLSys（最佳论文） | 可定制 attention 内核引擎，被主流服务框架吸收 |
| DistServe | 2024 / OSDI | 预填/解码分离，按 goodput 优化 |
| Medusa | 2024 / ICML | 多头并行 draft，投机解码加速 |
| Speculative Decoding | 2023 / ICML | draft-verify 范式，投机解码奠基 |
| Mooncake | 2024 / arXiv | 以 KV cache 为中心的分布式服务架构 |

---

## 4. 开源项目盘点（名称 + star 量级 + 贡献者 + 活跃度 + 维护方）

> star/贡献者数据抓取于 2026-08-15（GitHub 页面 + shields.io badge，量级）。

| 项目 | star 量级 | 贡献者 | 活跃度 | 维护方 / 背后支持 |
|---|---|---|---|---|
| **ollama** | ~178k | ~450 | 每日 | 独立公司 Ollama Inc |
| **transformers** | ~164k | ~440 | 每日 | Hugging Face |
| **llama.cpp** | ~124k | ~450 | 每日 | Georgi Gerganov + ggml-org（曾获 GitHub 前 CEO Nat Friedman 投资） |
| **vLLM** | ~89k | ~455 | 每日 | Neural Magic→Red Hat（2024 收购）；2026 并入 PyTorch Foundation / Linux Foundation |
| **exo** | ~47k | ~98 | 停滞（末次提交 2026-06） | exo-explore 独立团队 |
| **SGLang** | ~32k | ~454 | 每日 | 团队成立 RadixArk，Accel 领投 + NVIDIA/AMD/Intel 参投 1 亿美元 |
| **MLX** | ~28k | ~312 | 每日 | Apple（ml-explore） |
| **candle** | ~21k | ~272 | 每日 | Hugging Face（Rust） |
| **Triton** | ~20k | ~409 | 每日 | OpenAI 起源 → PyTorch Foundation（triton-lang） |
| **outlines** | ~16k | ~179 | 活跃 | dottxt（.txt / Rémi Louf） |
| **burn** | ~16k | ~305 | 每日 | tracel-ai |
| **TensorRT-LLM** | ~14k | ~421 | 每日 | NVIDIA |
| **TVM** | ~14k | ~366 | 每周 | Apache（MLC-AI 生态，陈天奇） |
| **TGI** | ~11k | ~145 | 明显放缓（末次提交 2026-03） | Hugging Face（被战略降优先级） |
| **LMDeploy** | ~8k | ~146 | 每日 | InternLM（上海 AI 实验室/商汤系） |
| **mistral.rs** | ~7.6k | ~91 | 每日 | Eric Buehler（个人主导） |
| **ExLlamaV2** | ~4.6k | ~53 | 放缓（末次提交 2026-03） | turboderp（个人） |
| **aphrodite-engine** | ~1.8k | ~42 | 每周 | PygmalionAI 社区（vLLM fork） |

**结构解读**：
- **star 与"生产价值"不成正比**：ollama（178k）> vLLM（89k），但数据中心生产用的是 vLLM/SGLang。ollama 赢在"一键本地"的 DX，不是赢在性能。
- **活跃度是更硬的信号**：TGI（末次 2026-03）、exo（2026-06）、ExLlamaV2（2026-03）已经掉队或减速；aphrodite 仅剩少量维护者。
- **贡献者数高度集中**：头部引擎都在 ~450 人量级，说明"赢家通吃"——社区贡献被少数引擎虹吸。
- **个人项目仍能活**：llama.cpp、mistral.rs、ExLlamaV2 是 1–2 人主导，但它们卡的是"巨头不做的生态位"。

---

## 5. 真实抱怨与未解决痛点（GitHub issue / HN / Reddit / 论坛）

这是判断"空白在哪"最有价值的证据。以下均来自真实 issue/讨论：

### vLLM（用户量最大，问题也最多）
- **版本升级导致性能雪崩**：*Qwen3-VL-32B-FP8 throughput collapses on v0.25.1 (v0.24.0 fine)*（issue #49259）——用户升级引擎反而掉速，无回滚保证。
- **调度/抢占竞态**：*Preemption + async scheduling race can corrupt prompt-token accounting and crash Prometheus counters*（#36755）；*performance regression caused by frequently preempting and resuming a request*（#25538）。
- **长会话性能退化**：*Performance degradation with increasing number of requests in long-running vLLM inference sessions*（#16985）。
- **底层内核 bug 直接 crash**：*cuMemcpyBatchAsync segfaults or hangs on large/repeated compact scatter submissions*（#49276）。
- **AMD/ROCm 是二等公民**（对用户是持续痛点）：*TP=2 deadlock on dual AMD R9700 (RDNA4)*（#40980）、*gfx1151 segfault loading Qwen3-VL AWQ*（#37151）、*Triton MXFP4 MoE device capability check breaks RDNA3.5*（#40301）、*decode throughput regression from CUDA graph memory reservation on ROCm*（#48453）。
- 结论：**vLLM 的护城河（快速迭代）同时也是它的债**——近 5 万 issue，回归风险高，且 NVIDIA 优先、AMD 长期滞后。

### llama.cpp
- **并发是硬伤**：*Why llama-server is so limited to concurrent requests even using -cb and -np?*（discussion #13935）——llama-server 不适合多租户高并发，这是它永远无法替代 vLLM 的原因。
- **单模型上比 vLLM 慢 40%**：*Eval bug: Llama.cpp 40% slower than VLLM + high CPU usage when running Qwen Coder Next*（#19345）。
- **CPU 内存带宽是物理墙**、NUMA 拆分复杂（discussion #12303）。

### ollama
- **"本地神器，生产毒药"**：社区一致结论是 ollama 到多并发/多租户/K8s 就崩（*Why Ollama Breaks at Scale*、K8s 单 GPU 死锁案例）。
- **请求串行化**：*mlxrunner: requests are serialized; support batched/concurrent decode*（ollama #17666）。
- 定位：**它不是推理引擎，是模型下载器 + 一键运行器**，性能/并发天花板被广泛抱怨。

### TensorRT-LLM
- 构建复杂、文档难、**NVIDIA 锁死**、移植性差；社区普遍认为它是"性能极致但 DX 差"的自留地，第三方很难贡献。

### 跨项目通病
- **缺少中立的性能回归测试**：从 vLLM 掉速、llama.cpp 慢 40% 可以看到，没有任何中立工具能持续追踪"升级某个引擎版本后，tok/s、$/token、延迟是否退化"。这是空白（见第 8 节）。
- **ROCm/AMD 系统性体验差**：vLLM、Triton 在 ROCm 上 bug 频发，社区反复抱怨"开源但对非 NVIDIA 不友好"。

---

## 6. 公司落地（谁在用、谁在养）

| 项目 | 商业化 / 落地证据 |
|---|---|
| vLLM | Red Hat（Neural Magic 收购）产品化；AWS、Anyscale、Databricks 等提供托管；Red Hat 官方发文"why vLLM is the best choice" https://developers.redhat.com/articles/2025/10/30/why-vllm-best-choice-ai-inference-today |
| SGLang | RadixArk 公司化，Accel 领投 1 亿美元，NVIDIA/AMD/Intel 参投（BusinessWire 2026-05-05）https://www.businesswire.com/news/home/20260505077157 |
| TensorRT-LLM | NVIDIA NIM / Triton Inference Server 的默认引擎，DGX 一体机 |
| TGI | Hugging Face Inference Endpoints（但已战略降级，末次提交 2026-03） |
| LMDeploy | 商汤/上海 AI 实验室内部 + 开源，主打 TurboMind 低延迟 |
| MLX | Apple 官方 ML 框架，驱动本地 Apple Silicon 推理 |
| llama.cpp | ggml-org；LM Studio、Jan、多数本地 UI 的底层引擎 |
| ollama | Ollama Inc，企业版商业化（Ollama for Enterprise） |
| exo | 家用设备 P2P 集群实验，尚未见严肃生产采用 |
| xgrammar | MLC-AI 维护，被 vLLM/SGLang/NVIDIA NIM 采纳为默认结构化后端 |

**关键信号**：数据中心推理的"谁赢"已经由**资本下注**回答——vLLM（Red Hat）与 SGLang（RadixArk）各拿到大厂背书，赛道对新人关闭。

---

## 7. 趋势判断

1. **双寡头 + 基金会化**：vLLM 并入 PyTorch Foundation（Linux Foundation 官方通稿明确列出 PyTorch/vLLM/DeepSpeed/Ray 同属该基金会，2026-02-24）https://www.linuxfoundation.org/press/pytorch-foundation-announces-new-members-as-agentic-ai-demand-grows ；SGLang 走独立公司 + 大厂战投。中性治理 vs 资本化是两条路线。
2. **MoE / 专家并行是主战场**：DeepSeek V3/R1 让 EP（expert parallelism）成为标配，也是 SGLang 反超 vLLM 的转折点。
3. **分离式（disaggregated）服务 + KV 传输**：DistServe/Mooncake/Dynamo 路线，面向超大规模与低成本长尾。
4. **MTP/投机解码**成为所有引擎的默认加速开关。
5. **异构多厂商是下一个战场**：ZML（Zig+MLIR，宣称"any model, any hardware"）、llm-d 等新项目瞄准跨 NVIDIA/AMD/Intel 的统一运行时 https://techcrunch.com/2026/07/08/hot-french-startup-zml-releases-free-product-to-speed-inference-across-lots-of-ai-chips/
6. **端侧/NPU**：SLM 上手机/PC NPU（Apple ANE、Qualcomm Hexagon、AMD XDNA、Intel NPU）是确定方向，但开源工具链极度匮乏。
7. **Rust/嵌入式/WASM**：candle/burn/mistral.rs 持续存在但尚未出现统治级项目，属于"有需求、无赢家"。

---

## 8. 已饱和点 & 被忽视的空白与机会

### 已饱和 / 会被吸收（不要碰）
- **通用 Python 多卡推理引擎**：vLLM/SGLang 双寡头 + 大厂资本，个人团队无胜算。
- **通用 attention/量化内核**：FlashInfer、Marlin 已被吸收；你写的内核会被上游 merge。
- **通用 JSON-schema 结构化生成**：outlines 已证明"可被 xgrammar 取代"，xgrammar 被所有引擎采纳——这是"单点功能被平台吸收"的样板。
- **前缀缓存 / 调度器 / 连续批处理**：已是引擎核心，不是独立生意。
- **"又一个本地聊天 UI"**：LM Studio/Jan/ollama 已覆盖，无差异化空间。

### 被忽视的空白（关键洞察）

**空白 A：异构/多厂商统一调度层**。vLLM/SGLang 是 NVIDIA 优先，ROCm 系统性 bug 频发（第 5 节有大量证据）；且 RadixArk 拿了 NVIDIA 的钱、vLLM 在 PyTorch Foundation（NVIDIA 深度绑定）——**它们没有中立动机去服务 AMD/Intel/Apple**。一个"把 vLLM/SGLang/llama.cpp 当后端、按成本/可用性路由"的中立层，或一个真正跨厂商的运行时，是真实且尚未被赢家锁定的空白。ZML/llm-d 刚起步。

**空白 B：端侧/NPU 运行时（尤其 Apple ANE、Windows Copilot+ NPU）**。vLLM 不做端侧；llama.cpp 覆盖 CPU/GPU 但不覆盖 NPU；MLX 是 Apple Metal 不是 ANE；ANE 至今无开源运行时（只有 Orion 等实验项目）。SLM 爆发但 NPU 工具链几乎空白。

**空白 C：推理性能回归测试 / 成本可观测（perf CI）**。没有中立工具持续追踪"引擎版本升级后 tok/s、$/token、TTFT 是否退化"——而退化真实发生（vLLM 掉速、llama.cpp 慢 40%）。引擎厂商不会客观 benchmark 竞争对手。这是纯软件/测试工程，个人可做，且所有推理团队都需要。

**空白 D：带正确性保证的结构化生成与验证**。xgrammar 覆盖了 80% 的 JSON 快乐路径，但"生成结果的形式化验证、流式 JSON 的严格保证、多步 agent trace 的约束正确性、VLM/推理模型的 token 级约束、以及结构化生成的 benchmark/正确性测试"仍是空白。

---

## 9. 具体候选切入点（3–5 个，说明为何大项目不会轻易吃掉）

> 判据：① 巨头无动机做（利益冲突）或 ② 巨头有动机但看不起（规模太小/不性感）或 ③ 技术护城河是"非核心路径"，个人能独立交付。

### 候选 1：推理性能回归检测与成本基准服务（对应空白 C）
- **做什么**：开源 harness + 托管服务，持续对主流引擎（vLLM/SGLang/llama.cpp/TensorRT-LLM）在固定模型/硬件上跑基准，输出版本间的 tok/s、TTFT、$/token 回归报告，接入 CI 在升级前自动拦截退化。
- **为何大项目不会吃**：引擎厂商是利益相关方，不会客观地把自己和对手放在一起打分；而且这是"测试/观测"层，不在 vLLM/SGLang 的核心路径上，他们更愿意把精力放在 MoE/EP 而不是 benchmark CI。
- **门槛**：低（软件工程 + 数据工程 + 少量 GPU），个人可起步，靠中立性建立信任。

### 候选 2：异构/多厂商推理路由与成本优化层（对应空白 A）
- **做什么**：把多个引擎/多厂商硬件抽象成统一 API，按成本、延迟、可用性动态路由（例如把 decode 丢给 AMD、prefill 丢给 NVIDIA，或把长尾请求降级到 CPU/llama.cpp）。
- **为何大项目不会吃**：vLLM（PyTorch/NVIDIA 绑定）和 SGLang（拿 NVIDIA/AMD 的钱）都没有中立动机做"帮用户跨厂商省钱的调度层"，这直接触及它们的硬件伙伴利益；而 ZML/llm-d 还很早期。
- **门槛**：中（需要理解多引擎语义 + 基准），但可先做"路由 + 观测"而非自研内核。

### 候选 3：Apple ANE / NPU 的 LLM 运行时（对应空白 B）
- **做什么**：一个面向 Apple Neural Engine（或 Qualcomm Hexagon/AMD XDNA）的极简 LLM 推理运行时，专注 1–8B SLM，做量化 + 内存布局 + 投机解码。
- **为何大项目不会吃**：vLLM 定位数据中心，做 ANE 是"另一个项目"；MLX 官方只走 Metal/GPU 路线、ANE 至今无公开运行时；llama.cpp 的 ggml 后端不覆盖 NPU。这是一块"巨头有硬件但不开软件"的真空。
- **门槛**：高（需逆向/低层调试能力），但一旦打通就是独占位，且 SLM 端侧需求在暴涨。

### 候选 4：带验证的结构化生成层（对应空白 D）
- **做什么**：在 xgrammar 之上做"验证 + 流式保证 + 多步 agent 约束正确性 + VLM 结构化输出 + 结构化生成 benchmark"。不是重新造 xgrammar，而是补它不做的正确性/测试/高级格式。
- **为何大项目不会吃**：xgrammar/outlines 只解决"生成时约束"，不解决"生成后验证、跨步骤保证、正确性测试"；这块更偏 PL/形式化/测试工程，不是引擎团队的兴趣所在，个人可独立做出差异化。
- **门槛**：低–中（需要 PL 背景 + 对 tokenizer/FSM 的理解），适合 2–3 人。

### 候选 5：面向消费级多卡/大显存分片的"穷人版分离式服务"（可选）
- **做什么**：把 DistServe/Mooncake 的分离式思想下放到 2–8 张消费卡/多台 PC（exo 想做的但停滞了），做 prefill/decode 分离 + KV 卸载到 CPU/SSD。
- **为何大项目不会吃**：vLLM/SGLang 的分离式面向 A100/H100 集群，消费级分片"不赚钱也不性感"；exo 热度高（47k star）但维护停滞（末次提交 2026-06），说明需求真实、供给掉队。
- **门槛**：中–高，风险在于 exo 若复活会正面竞争；建议作为候选 2 的子方向而非主攻。

---

## 10. 附：成功 OSS infra 的共性特征（用来反推"你能做什么"）

1. **单一锋利价值主张**：vLLM=「易用 + 吞吐」、llama.cpp=「任何 CPU 跑起来」、ollama=「一条命令本地跑」。一句话能说清，且这句话解决一个高频痛点。
2. **Python 可用**：vLLM/SGLang 赢在 Python-native + Triton，贡献门槛低；TRT-LLM 输在构建复杂、闭源味重。
3. **快速迭代 + 及时响应新模型**：SGLang 靠 DeepSeek EP 支持弯道超车；vLLM 靠每日发布。慢 = 死。
4. **强文档 + 可复现 benchmark**：vLLM/SGLang 都有公开 benchmark 与博客，这是建立信任的杠杆。
5. **社区 + 中性治理**：最终赢家都走向基金会/大厂战投；个人项目靠"单点极致"卡生态位，而非做大而全。
6. **骑上新模型/新硬件的浪**：SGLang 骑 DeepSeek，llama.cpp 骑 GGUF，MLX 骑 Apple Silicon。**追浪能力 > 追功能**。

---

## 主要来源 URL（citation 汇总）

- PagedAttention / vLLM：https://arxiv.org/abs/2309.06180
- SGLang：https://arxiv.org/abs/2312.07104
- FlashAttention：https://arxiv.org/abs/2205.14135
- Triton：https://arxiv.org/abs/1904.10544
- TVM：https://arxiv.org/abs/1802.04799
- GPTQ：https://arxiv.org/abs/2210.17323
- AWQ：https://arxiv.org/abs/2306.00978
- Marlin：https://arxiv.org/abs/2401.09225
- FlashInfer：https://arxiv.org/abs/2501.01005
- DistServe：https://arxiv.org/abs/2401.09670
- xgrammar：https://github.com/mlc-ai/xgrammar
- RadixArk / SGLang 融资：https://www.businesswire.com/news/home/20260505077157
- Red Hat 收购 Neural Magic：https://techcrunch.com/2024/11/12/red-hat-acquires-ai-optimization-startup-neural-magic/
- PyTorch Foundation 通稿（含 vLLM）：https://www.linuxfoundation.org/press/pytorch-foundation-announces-new-members-as-agentic-ai-demand-grows
- Red Hat "why vLLM"：https://developers.redhat.com/articles/2025/10/30/why-vllm-best-choice-ai-inference-today
- SGLang MTP：https://www.lmsys.org/blog/2025-07-17-mtp/
- ZML 异构推理（TechCrunch）：https://techcrunch.com/2026/07/08/hot-french-startup-zml-releases-free-product-to-speed-inference-across-lots-of-ai-chips/
- vLLM 性能雪崩 issue：https://github.com/vllm-project/vllm/issues/49259
- vLLM ROCm TP deadlock：https://github.com/vllm-project/vllm/issues/40980
- llama.cpp 并发讨论：https://github.com/ggml-org/llama.cpp/discussions/13935
- ollama 请求串行化：https://github.com/ollama/ollama/issues/17666
- TGI 支持 xgrammar 的 issue（说明其吸收关系）：https://github.com/huggingface/text-generation-inference/issues/2900
