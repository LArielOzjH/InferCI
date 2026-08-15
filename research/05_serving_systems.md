# LLM Serving 系统深度调研

> 调研时间：2026 年。方向：LLM Inference / Serving Systems。
> 覆盖：continuous batching、PagedAttention、RadixAttention、chunked prefill、KV cache 淘汰与压缩、调度、各推理引擎、MoE serving、多 LoRA、约束解码、PD 分离、长上下文、显存/offload。
> 所有关键事实尽量附出处 URL；GitHub star 数为量级估计（调研时点，会持续变化）。

---

## 1. 现状总览

LLM 推理服务经历了从「单请求静态 batch」到「token 级动态调度」再到「PD 分离 + KV 存储池化」的三级跳。核心矛盾始终只有一个：**decoder-only 的自回归推理在硬件上是两种完全不同的负载——prefill（计算密集 / compute-bound）和 decode（显存带宽密集 / memory-bound），且每步都要读写整个 KV cache**。所有关键技术的本质都是在围绕这个矛盾做文章：

- **吞吐瓶颈**：decode 阶段每个 token 的 FLOPs 很小，但要把全部参数权重和 KV cache 从 HBM 读一遍，GPU 算力大量闲置 → 靠「把更多请求塞进一个 batch」来摊薄带宽成本（continuous batching）。
- **显存瓶颈**：KV cache 与 batch size × 序列长度成正比，长上下文/大并发下显存先于算力耗尽 → 靠「分页内存管理（PagedAttention）、压缩（量化/稀疏/淘汰）、offload（CPU/NVMe）来缓解」。
- **延迟瓶颈**：prefill 会「挤占」decode 的算力，导致正在解码的用户卡顿 → 靠「chunked prefill 把 prefill 切碎」或「PD 分离把两类负载放到不同 GPU 池」来隔离。
- **成本瓶颈**：MoE 降低激活参数（算力），但专家权重必须常驻显存；多模型/多租户场景下显存碎片化 → 靠「MLA、专家/权重卸载、多模型池化」来省钱。

目前主流生产引擎（vLLM、SGLang、TensorRT-LLM）已把**单模型、单节点、高吞吐**这条路径榨得比较干，竞争前沿已转移到：**KV cache 作为一等公民的分布式管理、PD 分离下的调度器、多模型/多租户 GPU 池化、长上下文成本、SLO 感知（goodput 而非裸吞吐）**。这正是「还有真实痛点、且个人/小团队可切入」的地方（见 §8、§9）。

---

## 2. 关键技术（带出处）

### 2.1 Continuous Batching（Orca，OSDI 2022）
- **为什么**：早期系统（如 FasterTransformer、早期 TGI）是 *request-level / static batching*——一个 batch 里的请求必须**全部结束**才能释放槽位，短输出被长输出拖住（barrier），GPU 利用率低。decode 又是 memory-bound，batch 越大越划算。
- **解决了什么**：Orca 提出 **iteration-level（token-level）调度**，每一步迭代都重新组 batch：新请求进来、结束的请求退出，中间态（K/V）留在 GPU 上继续。相当于把「批处理」变成「流水线」。吞吐可提升 2–4 倍，是现代所有引擎的默认能力（table stakes）。
- **当前卡点**：单靠 continuous batching 已到极限，瓶颈转移到 KV cache 管理（paging/淘汰）、prefill-decode 干扰（chunked prefill）、以及 SLO 感知调度。
- 出处：https://www.usenix.org/conference/osdi22/presentation/yu ；论文《Orca: A Distributed Serving System for Transformer-Based Generative Models》(OSDI 2022)。

### 2.2 PagedAttention（vLLM，SOSP 2023）
- **为什么**：KV cache 按「请求预留 max_len」分配，会造成**内部碎片**（短请求占大坑）和**外部碎片**，实测浪费 60–80% 显存。
- **解决了什么**：把 KV cache 切成**固定大小的 block（page）**，用类似 OS 虚拟内存的 block table 做逻辑→物理映射，按需分配、不连续存储。消除了预留浪费，并让**多个请求/采样分支共享同一份 KV block**（beam search、parallel sampling 写时复制）。
- **当前卡点**：块粒度仍有碎片；chunked prefill 需要跨 block 的拷贝；跨模型/跨引擎没有统一 block 格式（见 §8）。
- 出处：论文 https://arxiv.org/abs/2309.06180 ；官方 blog https://blog.vllm.ai/2023/06/20/vllm.html ；vLLM 仓库 https://github.com/vllm-project/vllm 。

### 2.3 RadixAttention / 前缀缓存（SGLang，NeurIPS 2024）
- **为什么**：真实负载里大量请求共享前缀（system prompt、few-shot、多轮对话历史、agent 工具描述）。重复 prefill 同一段前缀是纯浪费。
- **解决了什么**：SGLang 用 **radix tree（基数树）** 把已计算的 KV cache 按 token 前缀组织，新请求自动匹配最长公共前缀并复用，只 prefill 增量部分。多轮对话、少样本、agent 场景吞吐可提升数倍。
- **当前卡点**：树节点的**淘汰策略（eviction）** 简单（LRU），在长上下文/海量前缀下命中率与显存占用难以平衡；跨节点、跨实例的分布式前缀共享仍不成熟（需 LMCache/Mooncake 这类外部 KV 层）。
- 出处：论文 https://arxiv.org/abs/2312.07104 ；文档 https://docs.sglang.ai/ ；RadixAttention 概念 https://mintlify.wiki/sgl-project/sglang/concepts/radix-attention 。

### 2.4 Chunked Prefill（Sarathi / Sarathi-Serve，OSDI 2024）
- **为什么**：prefill 是 compute-bound，decode 是 memory-bound，二者在同批次混合时，一个长 prefill 会「暂停」所有正在 decode 的请求（pipeline bubble），造成尾延迟抖动；完全分离又会降低吞吐。
- **解决了什么**：Sarathi 把 prefill **切成 decode 步长大小的 chunk**（约 512 token），与 decode 步骤交错调度，让 GPU 计算/带宽混合利用，在不牺牲吞吐的前提下压平尾延迟。Sarathi-Serve 进一步用 stall-free 调度 + 智能 chunk 大小。
- **当前卡点**：chunked prefill 会引入更多 KV block 分配与拷贝；chunk 大小、prefill/decode 配比、与 PD 分离的调度策略组合仍靠启发式，缺少系统化理论。
- 出处：Sarathi-Serve https://www.usenix.org/conference/osdi24/presentation/agrawal ；arXiv https://arxiv.org/abs/2403.02310 ；Sarathi 原论文 arXiv 2308.16369。

### 2.5 KV cache 淘汰与压缩
- **淘汰（稀疏/eviction）——H2O（NeurIPS 2023）**：观察到注意力高度稀疏，少数「heavy hitter」token 贡献大部分注意力分数。H2O 用动态注意力分数做 oracle，逐层保留 heavy hitter + 近期 token，其余淘汰，可把 KV 砍到 20% 而精度损失很小。卡点：有损、对长上下文 recall 有影响、逐层选择实现复杂。出处：https://github.com/FMInference/H2O ；https://arxiv.org/abs/2306.14048 。
- **量化——KVQuant（NeurIPS 2024）**：KV 占显存大头，用**逐 channel + 非均匀量化（per-channel、non-uniform、对 RoPE 前异常值单独处理）** 把 KV 压到 1–4 bit，支持到 1M+ 上下文。卡点：异常值处理与算子融合使工程复杂，低比特有精度风险。出处：https://github.com/SqueezeAILab/KVQuant ；https://arxiv.org/abs/2401.18079 。
- **稀疏注意力——Quest（ICML 2024）/ MInference（Microsoft）/ DeepSeek NSA**：长上下文里按 query 动态挑重要 block/chunk 做注意力，prefill 可加速、KV 可稀疏。MInference 用离线识别的注意力模式做动态稀疏（1M token 推理加速约 10×）；Quest 用 query 感知的 chunk 稀疏。NSA 把稀疏注意力做到硬件对齐+原生训练。卡点：稀疏模式与模型强相关，通用性、精度权衡。出处：Quest https://github.com/mit-han-lab/quest ；MInference https://arxiv.org/abs/2407.02490 ；NSA（DeepSeek）https://arxiv.org/abs/2502.11089 。

### 2.6 调度：优先级 / 抢占 / 公平
- 各引擎的调度器都从「FCFS + 简单 token 配额」演进到**支持优先级、抢占、公平**。vLLM 从 2024 起陆续合入 priority scheduling（issue #6077、PR #5958），V1 引擎在 2025 合入 V1 优先级调度（PR #19057），并有「高优先级请求抢占低优先级」的 PR（#45561）。
- **抢占实现**：多基于 chunked prefill / token-slice 的细粒度切分 + KV 换出（swap）到 CPU，而非杀死请求。
- **卡点**：优先级 + 抢占 + 公平 + SLO 的组合没有系统化；抢占的 KV 换出成本高；公平性（多租户）研究薄弱。DistServe 提出的「goodput（满足 SLO 的有效吞吐）」视角尚未成为引擎默认调度目标。
- 出处：vLLM issue https://github.com/vllm-project/vllm/issues/6077 ；V1 优先级 PR https://github.com/vllm-project/vllm/pull/19057 ；IBM token-slice 抢占 https://research.ibm.com/publications/the-token-slice-implementing-preemptive-scheduling-via-chunked-decoding 。

### 2.7 主流引擎（细节见 §4）
- **TensorRT-LLM（NVIDIA）**：vendor 级手写 CUDA kernel + **in-flight batching** + Triton backend，NVIDIA 卡上原始吞吐最强；代价是编译引擎、模型覆盖滞后、仅 NVIDIA。
- **TGI（HuggingFace）**：Rust router + Python 模型分片，连续 batch、量化、PEFT 支持好；但优化迭代速度渐落后于 vLLM/SGLang，社区出现「迁移到 vLLM/SGLang」的指南。
- **LMDeploy（InternLM/上海AI Lab）**：TurboMind 引擎的 **persistent batch** + PagedAttention + 4-bit AWQ/KV int8，中文模型与国产卡生态好。
- **MLC-LLM（MLC-AI/TVM）**：编译器路线，一处编译多处部署（CUDA/Vulkan/Metal/WebGPU/移动端），可移植性最强；服务器峰值性能不如手写 kernel。
- **llama.cpp（ggml）**：GGUF 量化 + CPU/GPU/Apple Silicon，**单用户本地**王者；不是多租户高并发服务器，无 PD 分离、无多模型池化。

### 2.8 MoE serving（DeepSeek V3/R1）
- **为什么**：MoE 把激活参数从 671B 降到 37B，**计算量大减**，但**全部专家权重必须常驻显存**，且需要 all-to-all 专家分发通信。DeepSeek 还引入 **MLA（Multi-head Latent Attention）**：把 K/V 压到低秩 latent 向量，KV cache 大幅压缩（约 28.5 倍 vs 标准 MHA），再用 **FP8（DeepGEMM）** 降带宽/显存，用 **auxiliary-loss-free 负载均衡 + Multi-Token Prediction（MTP）** 提升效率与吞吐。
- **解决了什么**：让 671B MoE 能以显著更低成本服务；成为 vLLM/SGLang 适配的标杆模型（DeepSeek-V3.2 已进 vLLM recipes）。
- **当前卡点**：专家权重显存（多卡 tensor parallel + expert parallel 权衡）、all-to-all 通信、冷门专家的「权重卸载/缓存」（见 CrossPool）。
- 出处：DeepSeek-V3 技术报告 https://arxiv.org/abs/2412.19437 ；DeepGEMM https://github.com/deepseek-ai/DeepGEMM ；vLLM recipe https://recipes.vllm.ai/deepseek-ai/DeepSeek-V3.2 ；SGLang DeepSeek 支持 https://deepwiki.com/sgl-project/sglang/8-deepseek-models-and-moe 。

### 2.9 多 LoRA 并发（S-LoRA / Punica）
- **为什么**：多租户 SaaS 里每个客户一个 LoRA adapter，共享同一个 base model。若每个 adapter 一份 base 权重，显存爆炸。
- **解决了什么**：base 权重常驻一份，adapter 按需加载。Punica（MLSys 2024）用 **SGMV（segmented gather matrix multiply）** 高效地把不同 adapter 的 LoRA 计算 batch 化；S-LoRA（MLSys 2024）用 **unified paging** 把 adapter 权重也分页管理，支持数千个并发 adapter。
- **当前卡点**：数千 adapter 的权重显存、跨 adapter 调度的公平性、与 MoE/量化/约束解码的组合；S-LoRA/Punica 原仓库多已停更（见 §4）。
- 出处：Punica https://arxiv.org/abs/2310.18547 ；S-LoRA https://arxiv.org/abs/2311.03285 ；https://proceedings.mlsys.org/paper_files/paper/2024/hash/906419cd502575b617cc489a1a696a67-Abstract-Conference.html 。

### 2.10 结构化输出 / 约束解码
- **为什么**：agent/工具调用需要「保证合法 JSON/schema/grammar」，靠 prompt 约束不可靠，靠 rejection 重采样浪费。
- **解决了什么**：**XGrammar（MLC-AI）** 用上下文无关文法（CFG）+ 下推自动机 + 自适应 token mask + 持久化执行栈，把约束解码做成高性能可移植库，已被 SGLang/NVIDIA NIM 集成；llama.cpp 有 GBNF 语法，Outlines 有 index 方案。约束解码现在基本能「零精度代价 + 可接受开销」。
- **当前卡点**：复杂语法（跨 token 边界、多 schema 联合）仍有开销；约束解码与 speculative decoding、MoE、多 LoRA 的组合尚未系统化。
- 出处：XGrammar https://github.com/mlc-ai/xgrammar ；NVIDIA NIM 结构化生成 https://docs.nvidia.com/nim/large-language-models/1.14.0/structured-generation.html 。

### 2.11 输出长度预测与 batch 调度研究
- **为什么**：理想的调度（SJF、优先级、抢占、fair share）都依赖「每个请求最终输出多长」，但自回归模型输出长度未知。
- **解决了什么**：用**小 proxy 模型预测输出长度**，做 SJF 类调度以降低平均等待时间/尾延迟（《Efficient Interactive LLM Serving with Proxy Model-based Sequence Length Prediction》）；ALISE 用 speculative scheduling 按预测长度与预算提前排程。
- **当前卡点**：预测精度与开销的权衡；这些是研究原型，未成为主流引擎默认能力。
- 出处：Proxy model 论文 https://arxiv.org/abs/2404.08509 ；ALISE https://arxiv.org/abs/2410.23537 。

### 2.12 GPU 显存效率与 Offload（FlexGen / vLLM offload / LMCache）
- **FlexGen（2023）**：单 GPU 高吞吐，把**权重 + KV 全 offload 到 CPU DRAM + NVMe**，用块调度重叠计算与 I/O。卡点：延迟很高，已被更成熟的量化+paging 取代。
- **vLLM CPU/KV offload**：把 KV cache 或权重换出到 CPU/NVMe，长上下文/超参模型可在单卡跑。出处：https://docs.vllm.ai/en/v0.25.0/features/kv_offloading_usage/ 。
- **LMCache（arXiv 2510.09665）**：把 KV cache 当作**多层存储对象（GPU/CPU DRAM/NVMe/远程）** 统一管理，支持跨实例前缀共享、PD 分离的 KV 传输、故障恢复，是「KV cache 存储引擎」路线的代表。卡点：传输带宽、跨层淘汰策略、与各引擎 block 格式的对接。
- 出处：LMCache https://github.com/LMCache/LMCache ；论文 https://arxiv.org/abs/2510.09665 ；FlexGen https://arxiv.org/abs/2303.06865 。

### 2.13 PD 分离对 scheduler 的影响（重点）
- **为什么分离**：prefill（compute-bound）和 decode（memory-bound）资源需求冲突，同池混跑会互相干扰（prefill 抢算力→decode 延迟尖峰）。Splitwise（微软）、DistServe（OSDI 2024）、Mooncake（Moonshot，FAST 2025 最佳论文）都把二者拆到不同 GPU 池/节点，prefill 池产出 KV 通过高速传输（NVLink/RDMA）交给 decode 池。
- **对调度器的影响（本质）**：调度问题从「单池内 prefill/decode 交错」变成**跨两个池的联合优化**——prefill 池何时把 KV 传给哪个 decode 实例、decode 池 KV 放不下时如何淘汰/回退、优先级/抢占如何跨池一致、长 prefill 如何路由。DistServe 把目标从「吞吐」改为「goodput（满足 SLO 的有效吞吐）」，用 placement 算法把请求按资源特征放到最合适的实例。
- **当前卡点**：KV 传输带宽/延迟是硬约束；跨池的优先级/抢占/公平语义无统一方案；「PD 分离 + chunked prefill + MoE + 多模型」的组合调度几乎空白。
- 出处：Splitwise https://arxiv.org/abs/2311.18677 ；DistServe https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin ；Mooncake https://arxiv.org/abs/2407.00079 / https://github.com/kvcache-ai/Mooncake 。

### 2.14 统一 KV cache 管理 & 多模型 serving
- **问题**：不同引擎/不同模型/不同实例各自维护 KV block，无法跨引擎复用或统一换出；多模型常驻时显存碎片化、冷启动慢。
- **进展**：Prism（arXiv 2505.04021）用 GPU memory ballooning 做多模型显存复用；CrossPool（arXiv 2606.24506）对冷门 MoE 做 KV + 权重 disaggregation；阿里 Aegaeon 报告多模型池化省 82% GPU；kvcached 项目做 GPU 上多模型 KV 缓存。
- **卡点**：**没有跨引擎统一的 KV cache 序列化/格式标准**；多模型权重冷热换入换出的 SLO 感知策略缺失；多为研究原型。
- 出处：Prism https://arxiv.org/abs/2505.04021 ；CrossPool https://arxiv.org/abs/2606.24506 ；阿里池化报道 https://www.theregister.com/2025/10/21/alibaba_aegaeon_gpu_scheduling_improvements 。

---

## 3. 论文清单（名称 + 年份 + venue，一句话核心）

| 论文 | 年份 / venue | 一句话核心 |
|---|---|---|
| Orca: A Distributed Serving System for Transformer-Based Generative Models | 2022 / OSDI | 提出 iteration-level continuous batching，token 级动态组批打破 request 级 barrier。 |
| Efficient Memory Management for LLM Serving with PagedAttention (vLLM) | 2023 / SOSP | KV cache 分页管理 + 写时复制共享，消除显存碎片浪费。 |
| SGLang: Efficient Execution of Structured Language Model Programs | 2024 / NeurIPS | RadixAttention 基数树实现自动前缀缓存 + 结构化编程前端。 |
| Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills | 2023 / arXiv | 把 prefill 切块与 decode 交错，消除 prefill 造成的 decode 停顿。 |
| Taming Throughput-Latency Tradeoff with Sarathi-Serve | 2024 / OSDI | stall-free 调度 + 智能 chunk，系统化解吞吐-尾延迟矛盾。 |
| H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs | 2023 / NeurIPS | 按注意力 heavy hitter 动态淘汰 KV，大幅压缩缓存。 |
| KVQuant: Towards 10M Context Length LLM Inference with KV Cache Quantization | 2024 / NeurIPS | 逐 channel 非均匀量化把 KV 压到 1–4 bit。 |
| DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving | 2024 / OSDI | PD 分离 + goodput 目标 + 请求 placement。 |
| Splitwise: Efficient Generative LLM Inference Using Phase Splitting | 2023 / arXiv (Microsoft) | 异构硬件按 prefill/decode 阶段拆分。 |
| Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving | 2025 / FAST（最佳论文） | Kimi 生产级 KV 为中心的解耦架构 + 传输引擎。 |
| Punica: Multi-Tenant LoRA Serving | 2024 / MLSys | SGMV 内核批量计算多租户 LoRA。 |
| S-LoRA: Serving Thousands of Concurrent LoRA Adapters | 2024 / MLSys | unified paging 管理数千 adapter 权重。 |
| DeepSeek-V3 Technical Report | 2024 / arXiv | MoE + MLA + FP8 + MTP，671B/37B 激活的极致成本设计。 |
| FlexGen: High-Throughput Generative Inference with a Single GPU | 2023 / arXiv | 权重+KV offload 到 CPU/NVMe 的单卡高吞吐。 |
| Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference | 2024 / ICML | query 感知 chunk 稀疏降低长上下文注意力成本。 |
| MInference 1.0: Accelerating Pre-filling for Long-Context LLMs | 2024 / arXiv (Microsoft) | 动态稀疏注意力模式加速百万 token 级 prefill。 |
| LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference | 2025 / arXiv | KV 多层存储引擎，跨实例前缀共享与 PD 传输。 |
| Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning | 2025 / arXiv | 多模型显存弹性复用。 |
| CrossPool: Efficient Multi-LLM Serving for Cold MoE Models | 2026 / arXiv | 冷门 MoE 的 KV+权重解耦池化。 |

---

## 4. 开源项目盘点（名称 + star 量级 + 活跃度 + 维护方）

> star 为调研时点的量级估计（约），会持续增长。

| 项目 | star 量级 | 活跃度 | 维护方 |
|---|---|---|---|
| vLLM | ~60k | 极高，V1 引擎 2025 落地 | 社区（UC Berkeley 起源，现基金会/多公司共建） |
| SGLang | ~20k | 极高，主力挑战者 | 社区（Stanford/Berkeley 起源，Anthropic/LMSys 等共建） |
| llama.cpp | ~90–100k | 极高 | ggml-org / Georgi Gerganov |
| TensorRT-LLM | ~11k | 高 | NVIDIA |
| LMDeploy | ~11k | 中高 | InternLM / 上海人工智能实验室 |
| MLC-LLM | ~21k | 中 | MLC-AI（TVM 团队） |
| TGI (text-generation-inference) | ~11k | 中（渐被 vLLM/SGLang 分流） | Hugging Face |
| LMCache | ~2k | 高 | LMCache 团队（社区/企业共建） |
| Mooncake | ~5k | 中高 | Moonshot AI（kvcache-ai） |
| Punica | ~1k | 已停更（archive） | 原作者（可扩展 LoRA 作者） |
| S-LoRA | ~1k | 已停更 | 原作者 |
| Quest | ~1k | 低（研究） | MIT Han Lab |

维护方/仓库出处：vLLM https://github.com/vllm-project/vllm ；SGLang https://github.com/sgl-project/sglang ；llama.cpp https://github.com/ggml-org/llama.cpp ；TensorRT-LLM https://github.com/NVIDIA/TensorRT-LLM ；LMDeploy https://github.com/InternLM/lmdeploy ；MLC-LLM https://github.com/mlc-ai/mlc-llm ；TGI https://github.com/huggingface/text-generation-inference ；LMCache https://github.com/LMCache/LMCache ；Mooncake https://github.com/kvcache-ai/Mooncake 。

---

## 5. 公司落地

- **Moonshot AI / Kimi**：Mooncake KV 为中心的解耦架构用于 Kimi 生产，已开源（FAST 2025 最佳论文）。出处 https://arxiv.org/abs/2407.00079 。
- **Anthropic**：深度参与并主导 SGLang 的优化（尤其 MoE/大 batch 场景），是 SGLang 生产化的重要推动者。
- **NVIDIA**：TensorRT-LLM + Triton + NIM 全家桶，Blackwell 上针对 DeepSeek-V3.2 的优化技术博客。出处 https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/ec909660ff02923a7856e38543d009e5c9021613/docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md 。
- **Hugging Face**：Inference Endpoints 默认引擎 TGI，逐步引入 vLLM/SGLang 选项。
- **Microsoft**：Splitwise（PD 分离）、MInference（长上下文稀疏）研究与 Azure 落地。
- **阿里云**：Aegaeon 多模型 GPU 池化，宣称省 82% GPU。出处 https://www.theregister.com/2025/10/21/alibaba_aegaeon_gpu_scheduling_improvements 。
- **DeepSeek**：V3/R1/V3.2 的 MoE+MLA+FP8+MTP 设计直接改变了 serving 成本曲线，DeepGEMM/FlashMLA 开源反哺社区。出处 https://github.com/deepseek-ai/DeepGEMM 。

---

## 6. 趋势判断

1. **优化目标从「吞吐」转向「goodput / SLO / 成本」**：DistServe 的 goodput 理念逐渐成为共识，调度与资源配置按 SLO 而非 token/s 衡量。
2. **KV cache 成为一等公民的数据对象**：LMCache/Mooncake 把 KV 存储化、可跨实例传输/持久化，未来可能出现 KV cache 的「存储格式标准」与独立存储层。
3. **PD 分离从「论文」走向「默认架构」**：vLLM/SGLang 都原生支持 disaggregation，Mooncake 提供传输引擎；下一步是跨池统一调度。
4. **模型侧协同设计反哺 serving**：MLA、稀疏注意力（NSA）、MTP、MoE 稀疏激活让「模型架构」本身成为最大的 serving 优化杠杆。
5. **多模型/多租户 GPU 池化**：Prism、CrossPool、阿里 Aegaeon 指向「一卡多模型、按需换入换出」，解决长尾模型冷启动与显存碎片。
6. **量化 + 稀疏 + offload 组合成默认**：FP8/INT4/KV 量化 + 稀疏注意力 + CPU/NVMe offload 成为长上下文与成本敏感场景的标准配置。
7. **生态收敛**：vLLM 与 SGLang 双雄主导，TGI 相对弱化，TensorRT-LLM 守 NVIDIA 高端卡；小引擎向「细分硬件/端侧」退守。

---

## 7. 已饱和点（不建议个人/小团队硬碰）

- **单模型单节点连续批处理吞吐**：vLLM/SGLang/TensorRT-LLM 已榨干大部分，kernel 级竞争是大厂/硬核 CUDA 团队的战场。
- **基础 PagedAttention / 单节点前缀缓存**：所有引擎标配，无差异化空间。
- **权重量化（AWQ/GPTQ/FP8）与基础 KV 量化**：成熟且被引擎内置。
- **基础 JSON/schema 约束解码**：XGrammar/Outlines/llama.cpp 已解决主要问题。
- **「再做一个小而快的推理引擎」**：除非绑定特定硬件/端侧（RISC-V、NPU、边缘）或特定语言运行时，否则正面竞争无意义。
- **简单 PD 分离**：DistServe/Splitwise/Mooncake 已有工程化，纯「分池」不是空白。

---

## 8. 被忽视的空白与机会（真实痛点 + 个人/小团队可行）

1. **跨引擎统一 KV cache 格式与迁移/checkpoint**：vLLM/SGLang/TensorRT-LLM 的 KV block 布局、dtype、head 排列各不相同，无法在引擎间复用、也无法把 KV cache 序列化后跨节点迁移/持久化（checkpoint 续聊）。这直接阻碍「统一 KV cache 管理」。做一个 **KV cache 标准化序列化/转换层 + 适配器**（对个人可行：读各引擎源码做 block 映射 + 格式转换），价值清晰。
2. **SLO 感知的调度器基准与仿真**：现有调度（优先级/抢占/公平/chunked prefill/PD 路由）都是启发式，缺一个**可复现的调度策略评测平台/模拟器**（输入负载、输出 goodput/尾延迟/显存曲线）。inference-sim 存在但覆盖有限；做成「调度策略的对战基准」门槛不高、痛点真实。
3. **输出长度预测作为 serving 一等原语**：用小 proxy 模型预测输出长度并接入现有引擎调度（SJF/优先级/抢占），现有多为研究原型、未工程化。可做成 vLLM/SGLang 的插件或独立 predictor 服务。
4. **长上下文 serving 成本的可观测/剖析工具**：缺一个能逐请求报告「KV cache 占用、前缀命中率、显存压力、稀疏/淘汰造成的精度回退」的开源 profiler/监控。做成 vLLM/SGLang 的 sidecar 或 dashboard，对大量跑长上下文的团队是刚需。
5. **多模型显存池化 / 权重冷热换入换出的实用控制面**：Prism/CrossPool/阿里 Aegaeon 都是研究或闭源；一个面向小团队的「多模型 warm/cold swap + SLO 感知淘汰」的轻量控制面（结合 NVMe 权重 offload + kvcached 思路）仍是空白，且可用普通多卡/单机验证。

---

## 9. 具体候选切入点（3–5 个，个人/小团队可行）

1. **KV cache 互操作层（KV-cache interchange）**：实现跨 vLLM ↔ SGLang 的 KV cache 序列化、block 映射与迁移/续聊/checkpoint 标准 + 开源适配器。先解决「同架构不同引擎」的 KV 迁移（续聊、热迁移、PD 解耦中把 prefill 池 KV 无损交给 decode 池），再扩展多格式。
2. **开源调度策略评测模拟器**：一个离散事件模拟器，输入真实负载 trace + 引擎参数，输出 goodput/尾延迟/显存/抢占次数，内置 FCFS/priority/preemption/fair-share/chunked-prefill/PD-routing 策略对比，作为「调度研究的标准床」。可被学术与工业同时采用。
3. **输出长度预测插件服务**：训练/蒸馏一个极小 predictor（≤100M 参数，低延迟），以 sidecar 或引擎插件形式预测输出长度，驱动 SJF 调度与抢占，实测降低尾延迟；对 vLLM/SGLang 各做一键接入。
4. **长上下文成本剖析 sidecar（KV-Auditor）**：非侵入式采集 vLLM/SGLang 的每请求 KV 占用、前缀命中、eviction 造成的精度回退，输出成本归因（哪个请求/前缀最烧显存），帮助团队定位长上下文成本。
5. **小团队多模型池化控制面**：面向 1–8 卡的小规模多模型部署，实现「权重到 NVMe offload + SLO 感知换入换出 + 前缀缓存预热」，让单卡按需跑多个模型，降低冷启动与显存碎片（结合 LMCache/kvcached 生态而非重造）。

> 建议优先级：**切入点 1（KV 互操作层）** 痛点最真实、与「统一 KV cache 管理」这一主线直接挂钩、且个人/小团队可逐步落地；**切入点 2（调度评测模拟器）** 最容易出可复用成果并形成社区影响力；两者可组合（用模拟器验证 KV 迁移策略）。

---

## 10. 关键出处汇总（URL）

- Orca (OSDI 2022): https://www.usenix.org/conference/osdi22/presentation/yu
- vLLM / PagedAttention: https://arxiv.org/abs/2309.06180 ；https://github.com/vllm-project/vllm
- SGLang / RadixAttention: https://arxiv.org/abs/2312.07104 ；https://github.com/sgl-project/sglang
- Sarathi-Serve (OSDI 2024): https://www.usenix.org/conference/osdi24/presentation/agrawal ；https://arxiv.org/abs/2403.02310
- H2O: https://github.com/FMInference/H2O ；https://arxiv.org/abs/2306.14048
- KVQuant: https://github.com/SqueezeAILab/KVQuant ；https://arxiv.org/abs/2401.18079
- Quest: https://github.com/mit-han-lab/quest ；https://arxiv.org/abs/2406.10774
- MInference: https://arxiv.org/abs/2407.02490
- DeepSeek-V3: https://arxiv.org/abs/2412.19437 ；https://github.com/deepseek-ai/DeepGEMM
- Punica: https://arxiv.org/abs/2310.18547
- S-LoRA: https://arxiv.org/abs/2311.03285
- XGrammar: https://github.com/mlc-ai/xgrammar
- 输出长度预测: https://arxiv.org/abs/2404.08509 ；ALISE https://arxiv.org/abs/2410.23537
- FlexGen: https://arxiv.org/abs/2303.06865
- LMCache: https://github.com/LMCache/LMCache ；https://arxiv.org/abs/2510.09665
- DistServe (OSDI 2024): https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
- Splitwise: https://arxiv.org/abs/2311.18677
- Mooncake: https://arxiv.org/abs/2407.00079 ；https://github.com/kvcache-ai/Mooncake
- vLLM V1: https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
- vLLM KV offload: https://docs.vllm.ai/en/v0.25.0/features/kv_offloading_usage/
- Prism: https://arxiv.org/abs/2505.04021
- CrossPool: https://arxiv.org/abs/2606.24506
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
- TGI 架构: https://huggingface.co/docs/text-generation-inference/en/architecture
- LMDeploy: https://github.com/InternLM/lmdeploy
- MLC-LLM: https://github.com/mlc-ai/mlc-llm
- llama.cpp: https://github.com/ggml-org/llama.cpp
- vLLM 优先级调度: https://github.com/vllm-project/vllm/issues/6077 ；https://github.com/vllm-project/vllm/pull/19057
- 阿里 Aegaeon 池化: https://www.theregister.com/2025/10/21/alibaba_aegaeon_gpu_scheduling_improvements
