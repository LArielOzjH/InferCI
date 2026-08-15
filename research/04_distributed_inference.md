# 分布式推理（Distributed Inference）深度调研

> 调研日期：2026-01；方向：AI Infra / LLM Serving。所有关键事实尽量附出处 URL。

---

## 0. TL;DR（一句话本质）

分布式推理的本质是：**当单个 GPU 装不下模型权重 / KV cache / 激活，或单机的算力与显存无法同时满足 prefill（算力密集）与 decode（访存密集）两种截然不同的负载时，把计算、显存、通信在多个设备/节点间重新编排**。它的技术主线从"把权重切开放下（张量/流水线并行）"，演进到"把序列切开放下（序列并行/上下文并行）"，再到"把专家按稀疏激活切开放下（专家并行）"，当前正在走向"把 prefill 和 decode 当作两种不同资源池分离调度（PD 分离）"，而 KV cache 的跨节点传输成为新的第一性瓶颈。

---

## 1. 现状总览

1. **单卡内存墙是起点**：现代 LLM（如 70B、671B MoE）权重远超单卡 HBM（80GB），必须并行切分才能推理。这催生了张量并行（TP）、流水线并行（PP）、专家并行（EP）三条主线。
2. **TP 是"标准答案"但被 NVLink 锁死在节点内**：Megatron-LM 提出的 TP 是今天所有框架的默认切分方式，但每层都要 all-reduce，跨节点通信开销使其难以扩展到单机之外（[Megatron-LM, 2021](https://arxiv.org/abs/2104.04473)）。
3. **MoE 让"参数多、算力省"成为可能，但把瓶颈从权重搬到了 all-to-all**：DeepSeek-V3（671B 总参、每 token 仅激活 37B）证明稀疏激活 + 专家并行可行，代价是跨节点的 MoE dispatch/combine 通信成为新的主导开销（[DeepSeek-V3, 2024](https://arxiv.org/abs/2412.19437)）。
4. **prefill/decode 分离（PD disaggregation）是当前最大趋势**：Splitwise 提出分相、DistServe 在 OSDI'24 给出 goodput 最优解、Mooncake（Kimi）用"KV cache 为中心"的架构把 PD 分离工业化、NVIDIA Dynamo/TensorRT-LLM 与 SGLang 都已原生支持。PD 分离的代价是 prefill→decode 之间要**跨节点搬运 KV cache**，这成为当前最尖锐的工程瓶颈。
5. **KV cache 传输"无标准、碎片化"**：Mooncake 传输协议（RDMA）核心闭源；TensorRT-LLM、SGLang、LMCache 各有一套 KV 传输机制；仅有一个早期 IETF 草案和一篇"Tensor-Centric KV Cache Transfer Protocol"论文在尝试标准化，尚未形成生态共识。

---

## 2. 关键技术（带出处）

### 2.1 张量并行（Tensor Parallelism）—— 为什么存在 + 卡点
- **为什么**：单层权重矩阵（如 FFN/Attention 的 W）横向切到多卡，每卡算一块，再 all-reduce 结果。它解决"单卡装不下单层权重/激活"的问题，通信是 layer 级、粒度最细。
- **出处**：Megatron-LM（[arXiv:2104.04473](https://arxiv.org/abs/2104.04473)）；PyTorch 以 **DTensor** 提供可组合的分布式张量抽象，配合 **FSDP2** 实现 TP+DP 的"2D 并行"（[Ray Train 2D Parallel 文档](https://docs.ray.io/en/latest/train/examples/pytorch/tensor_parallel_dtensor/README.html)；[PyTorch DTensor/FSDP2](https://github.com/pytorch/pytorch)）。
- **卡点**：每层两次 all-reduce（forward 一次、backward 一次），对带宽极度敏感，通常只能锁在 NVLink 域内（≤8 卡）。**跨节点 TP 不划算**，因此 TP 基本被当作"节点内原语"，跨节点交给 PP/EP。

### 2.2 流水线并行（Pipeline Parallelism）
- **为什么**：当模型按"层"维度跨节点切分，微批（micro-batch）流水线执行；它解决"模型太大放不进一个节点、且想跨节点扩展"的问题，通信只在 stage 边界、量小。
- **演进**：GPipe 微批流水线 → PipeDream/DeepSpeed 的 1F1B 调度消除显存峰值 → vLLM/DeepSpeed 的推理 PP；DeepSeek-V3 的 **DualPipe** 把前向/反向气泡进一步压掉；DynaPipe 提出动态层重分配进一步减少气泡（[DynaPipe, NeurIPS 2025](https://neurips.cc/virtual/2025/loc/san-diego/poster/119240)）；DeepSpeed Inference 论文（[arXiv:2207.00032](https://arxiv.org/abs/2207.00032)）。
- **卡点**：**pipeline bubble**（上游等下游的空转）；异构设备（不同型号 GPU）负载不均。vLLM 已支持 PP 以跑异构/超大模型（[vLLM Parallelism 文档](https://docs.vllm.ai/en/v0.21.0/serving/parallelism_scaling/)）。

### 2.3 序列并行 / 上下文并行（Sequence Parallelism / Context Parallel）
- **为什么**：长上下文（百万 token）时 KV cache 与 attention 计算本身超过单卡显存，需要把"序列维度"切开分散到多卡，同时保持 attention 全局性。
- **三条路线**：
  - **Ring Attention**：块状 attention + 沿环形拓扑传递 KV，上下文随设备数线性扩展（[arXiv:2310.01889](https://arxiv.org/abs/2310.01889)；[GitHub lhao499/RingAttention](https://github.com/lhao499/RingAttention)）。
  - **DistFlashAttn**：分布式 memory-efficient attention，兼顾长上下文训练（[arXiv:2310.03294](https://arxiv.org/abs/2310.03294)）。
  - **DeepSpeed-Ulysses**：按 head 维切分、all-to-all 换 sequence 维（[arXiv:2309.14509](https://arxiv.org/abs/2309.14509)）。Megatron 也有沿 TP 维的 sequence parallel（切 LayerNorm/Dropout）。
- **卡点**：ring 方式有大量无效等待、all-to-all 方式通信量大；LoongServe 提出"弹性序列并行"让上下文并行可动态伸缩（[arXiv:2404.09526](https://arxiv.org/abs/2404.09526)）。vLLM/SGLang 现在把 ring + Ulysses 统一为可插拔 context parallel。

### 2.4 专家并行（Expert Parallelism, EP）
- **为什么**：MoE 模型参数量巨大（DeepSeek-V3 671B），但每 token 只激活 top-k 专家（V3 是 8/256）。若每个专家参数只驻留一张卡，权重不再需要全量复制，显存压力骤降；代价是 token 要在专家所在设备间 **all-to-all dispatch/combine**。
- **出处**：
  - DeepSeek-MoE：细粒度专家切分 + 共享专家隔离（[arXiv:2401.06066](https://arxiv.org/abs/2401.06066)）。
  - DeepSeek-V2：MLA（Multi-head Latent Attention）把 KV 压缩成低秩潜向量，大幅减小 KV cache（[arXiv:2405.04434](https://arxiv.org/abs/2405.04434)）。
  - DeepSeek-V3：256 专家、FP8 训练、**低精度通信**（dispatch/combine 用 FP8 减少跨节点流量）、通信-计算重叠（[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)）。
  - Megatron-Core MoE：token dispatcher 抽象 + Hybrid Expert Parallel（DP+TP+EP 混合，见 [NVIDIA 技术博客](https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/)；[Megatron Core MoE README](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)）。
- **卡点**：**all-to-all 是跨节点 MoE 推理的第一瓶颈**；专家负载不均（hot expert）、专家放置策略、通信与计算重叠程度决定了吞吐上限。

### 2.5 prefill/decode 分离（PD Disaggregation）—— 当前最大趋势
- **为什么**：prefill 是**算力密集**（矩阵乘、长 prompt 并行），decode 是**访存密集**（逐 token、带宽受限）；两者挤在同一 GPU 上会互相拖垮延迟/吞吐。把两类负载放到不同资源池，分别优化，可同时提升 goodput 和尾延迟。
- **出处**：
  - Splitwise：相分离 + 不同硬件配比（[arXiv:2311.18677](https://arxiv.org/abs/2311.18677)）。
  - DistServe：goodput 最优的 PD 分离，OSDI'24（[论文 PDF](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)；[演讲页](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)）。
  - Mooncake（Kimi/Moonshot）：**KVCache-centric** 架构，prefill 边算边 layer-wise 把 KV 通过 RDMA transfer engine 传给 decode，FAST'25（[arXiv:2407.00079](https://arxiv.org/abs/2407.00079)；[FAST25 论文](https://www.usenix.org/system/files/fast25-qin.pdf)）。
  - NVIDIA Dynamo：数据中心级分布式推理框架，原生 PD 分离 + 路由（[ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo)；[Dynamo 文档](https://docs.nvidia.com/dynamo/)）；TensorRT-LLM 提供 disaggregated serving 与 KV cache transfer（[TRT-LLM disagg-serving](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/disagg-serving.md)）。
  - SGLang：基于 RadixAttention 的 PD 分离 + Mooncake/LMCache 传输（[SGLang 论文 2312.07104](https://arxiv.org/abs/2312.07104)；[SGLang disaggregation](https://docs.dynamo.nvidia.com/dynamo/dev/knowledge-base/modular-components/backends/sg-lang/disaggregation)）。

### 2.6 KV cache 跨节点传输与重叠
- **为什么**：PD 分离后，prefill 产出的 KV 必须送到 decode 节点；KV 体积巨大（长上下文可达 GB 级），传输延迟直接进入端到端延迟，必须压缩 + 与计算重叠。
- **关键做法**：
  - Mooncake 的 **layer-wise 重叠**：prefill 逐层算，边算边把已完成层的 KV 传给 decode，隐藏传输延迟（[arXiv:2407.00079](https://arxiv.org/abs/2407.00079)）。
  - SGLang 的 layer-wise KV transfer PR（[sgl-project/sglang#34905](https://github.com/sgl-project/sglang/pull/34905)）。
  - LMCache 提供 NIXL / Mooncake 等多种 transfer channel，做 GPU 池化与跨节点 KV 复用（[LMCache 文档](https://github.com/LMCache/LMCache/blob/main/docs/source/kv_cache/storage_backends/mooncake.rst)；[LMCache × Mooncake](https://blog.lmcache.ai/en/2025/05/08/lmcache-x-mooncake-unite-to-pioneer-kvcache-centric-llm-serving-system/)）。
  - "Tensor-Centric KV Cache Transfer Protocol" 论文（LMCache，[arXiv:2510.09665](https://arxiv.org/abs/2510.09665)）提出张量级 KV 传输协议。
  - IETF 早期草案：KVCache over MoQT（[draft-shi-moq-kvcache](https://datatracker.ietf.org/doc/html/draft-shi-moq-kvcache-01)）。

### 2.7 编排层：Ray / DeepSpeed-MII / vLLM-Ray / Dynamo
- **为什么**：多机推理需要统一调度、容错、弹性伸缩。Ray 提供通用分布式运行时；vLLM 通过 Ray Serve 集成；DeepSpeed-MII 曾是 DeepSpeed 的低延迟推理封装，**现已基本被 DeepSpeed 合并/停更**（[DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII)）；NVIDIA Dynamo 是新一代 Rust 内核 + NATS 消息总线的统一调度框架，可混合调度 vLLM/SGLang/TRT-LLM 后端。

### 2.8 Exo（点对点 / 异构设备）
- **为什么**：把日常设备（手机/笔记本/多台 Mac）用 P2P 拓扑组网，用 ring attention 切分模型，无中心服务器，面向隐私与消费级设备（[exo-explore/exo](https://github.com/exo-explore/exo)）。
- **卡点**：WAN 带宽/抖动大、异构算力负载不均、token 生成速度受限，偏"DIY/边缘"而非数据中心。

### 2.9 MoE offload（专家卸载）
- **为什么**：MoE 冷专家可以放 CPU/SSD，热专家放 GPU，按需预取，缓解显存墙。
- **出处**：Pre-gated MoE 提出 fetch-on-demand 与按需预取（[arXiv:2308.12066](https://arxiv.org/abs/2308.12066)）；vLLM 社区在推进 MoE 专家 offload/prefetch（[vllm#33869](https://github.com/vllm-project/vllm/issues/33869)、[vllm#38256](https://github.com/vllm-project/vllm/issues/38256)）；ollama 已支持 MoE CPU offload（[ollama#15988](https://github.com/ollama/ollama/pull/15988)）。

### 2.10 跨节点通信开销（all-reduce / all-to-all / 激活量化传输）
- **三类通信**：TP 的 all-reduce（层内、量大但可同机 NVLink）、EP/SP-Ulysses 的 all-to-all（跨节点、MoE 主开销）、PD 的 KV 传输（单向、延迟敏感）。
- **降开销手段**：**低精度通信**（DeepSeek-V3 用 FP8 dispatch/combine，[技术报告](https://arxiv.org/abs/2412.19437)）；**通信-计算重叠**（[deepseek-ai/open-infra-index](https://github.com/deepseek-ai/open-infra-index)）；RDMA/GDR（GPU Direct RDMA）绕 CPU。AllReduce 有跨层分类综述（[IEEE survey](https://ieeexplore.ieee.org/abstract/document/11541474)）。

---

## 3. 论文清单（名称 + 年份 + venue + 一句话核心）

| 论文 | 年份 | Venue | 一句话核心 |
|---|---|---|---|
| Megatron-LM: Efficient Large-Scale LM Training on GPU Clusters ([2104.04473](https://arxiv.org/abs/2104.04473)) | 2021 | arXiv/SC21 | 提出模型并行（TP）切分 attention/FFN 权重，成为所有框架的节点内并行原语 |
| GPipe ([1811.06965](https://arxiv.org/abs/1811.06965)) | 2019 | NeurIPS | 微批流水线并行，允许超单卡模型，代价是 pipeline bubble |
| PipeDream: Generalized Pipeline Parallelism ([1806.06962](https://arxiv.org/abs/1806.06962)) | 2019 | SOSP | 1F1B 调度消除显存峰值，是 PP 调度的基准 |
| DeepSpeed Inference ([2207.00032](https://arxiv.org/abs/2207.00032)) | 2022 | arXiv | 大规模 Transformer 推理的并行 + 内存优化体系 |
| Ring Attention with Blockwise Transformers ([2310.01889](https://arxiv.org/abs/2310.01889)) | 2023 | arXiv | 块状 attention + 环形 KV 传递，上下文随设备数线性扩展 |
| DistFlashAttn ([2310.03294](https://arxiv.org/abs/2310.03294)) | 2023 | arXiv | 分布式 memory-efficient attention，长上下文训练 |
| DeepSpeed-Ulysses ([2309.14509](https://arxiv.org/abs/2309.14509)) | 2023 | arXiv | 按 head 维切分 + all-to-all 换 sequence 维的序列并行 |
| DeepSeek-MoE ([2401.06066](https://arxiv.org/abs/2401.06066)) | 2024 | arXiv | 细粒度专家切分 + 共享专家隔离，MoE 推理基础 |
| DeepSeek-V2 ([2405.04434](https://arxiv.org/abs/2405.04434)) | 2024 | arXiv | MLA 低秩潜向量压缩 KV，大幅降低 KV cache |
| DeepSeek-V3 Technical Report ([2412.19437](https://arxiv.org/abs/2412.19437)) | 2024 | arXiv | 671B MoE + FP8 训练 + 低精度 all-to-all + DualPipe |
| Splitwise ([2311.18677](https://arxiv.org/abs/2311.18677)) | 2023 | arXiv | prefill/decode 相分离 + 差异化硬件配比 |
| DistServe ([OSDI'24](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)) | 2024 | OSDI | goodput 最优的 PD 分离资源分配 |
| Sarathi-Serve | 2024 | OSDI | chunked-prefill 驯服吞吐-延迟权衡 |
| Mooncake: A KVCache-centric Disaggregated Architecture ([2407.00079](https://arxiv.org/abs/2407.00079)) | 2024 | FAST'25 | KV cache 为中心 + RDMA 传输引擎 + layer-wise 重叠 |
| Preble: Efficient Distributed Prompt Scheduling ([2407.00023](https://arxiv.org/abs/2407.00023)) | 2024 | ICLR'25 | 分布式 prompt 调度，PD 分离下的请求放置 |
| LoongServe ([2404.09526](https://arxiv.org/abs/2404.09526)) | 2024 | arXiv/SOSP'24 | 弹性序列并行 + PD 分离，长上下文服务 |
| InfiniGen ([2406.19707](https://arxiv.org/abs/2406.19707)) | 2024 | arXiv | KV cache 动态管理与 offload 的生成式推理 |
| Pre-gated MoE ([2308.12066](https://arxiv.org/abs/2308.12066)) | 2023 | ISCA'24 | fetch-on-demand + 按需预取的 MoE 专家卸载 |
| SGLang: Efficient LLM Programming ([2312.07104](https://arxiv.org/abs/2312.07104)) | 2023 | arXiv | RadixAttention 前缀 KV 复用 + 高效执行 |
| vLLM: PagedAttention ([2309.06180](https://arxiv.org/abs/2309.06180)) | 2023 | SOSP | PagedAttention 分页 KV，高吞吐推理 |
| Tensor-Centric KV Cache Transfer Protocol ([2510.09665](https://arxiv.org/abs/2510.09665)) | 2025 | arXiv | LMCache 提出的张量级 KV 传输协议 |
| DynaPipe | 2025 | NeurIPS | 动态层重分配减少 PP bubble |

---

## 4. 开源项目盘点（名称 + star 量级 + 活跃度 + 维护方）

| 项目 | Star 量级 | 活跃度 | 维护方 / 说明 |
|---|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | 数万（事实标准，>40K） | 极高，日更 | UC Berkeley 起源，社区 + NVIDIA/多厂共建；PagedAttention、PP、PD 分离 |
| [SGLang](https://github.com/sgl-project/sglang) | ~30.5K（[star-history](https://www.star-history.com/sgl-project/sglang/)） | 极高 | 伯克利/斯坦福系；RadixAttention、PD 分离、DeepSeek 优化 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | ~14K（[star-history](https://www.star-history.com/nvidia/tensorrt-llm/)） | 高 | NVIDIA；极致性能 + disaggregated serving |
| [Mooncake](https://github.com/kvcache-ai/Mooncake) | ~10K（开源首小时 +1.2K） | 高 | Moonshot AI + 清华/阿里/华为等；transfer engine 开源，KV store 部分闭源 |
| [LMCache](https://github.com/LMCache/LMCache) | ~5K | 高 | 芝加哥大学/社区；KV cache 复用层，NIXL/Mooncake 传输 |
| [Dynamo](https://github.com/ai-dynamo/dynamo) | 数千（快速上涨） | 极高 | NVIDIA；Rust 内核 + NATS，统一调度多后端 |
| [exo](https://github.com/exo-explore/exo) | ~22K | 中高 | Exo Labs；P2P 异构设备集群，ring attention |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | 数万 | 高 | 微软；ZeRO/PP/Ulysses；DeepSpeed-MII 已基本停更 |
| [Ray](https://github.com/ray-project/ray) | 数万 | 极高 | Anyscale；分布式运行时 + Ray Serve/DTensor |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | 数万 | 高 | NVIDIA；TP/EP 参考实现（Megatron-Core） |
| [RingAttention](https://github.com/lhao499/RingAttention) | ~1K | 中 | 作者维护；ring attention 参考实现 |

---

## 5. 公司落地

- **Moonshot AI（Kimi）**：Mooncake 是 Kimi 的线上服务底座，KV-cache-centric PD 分离，宣称显著降本（[arXiv:2407.00079](https://arxiv.org/abs/2407.00079)；[FAST25](https://www.usenix.org/system/files/fast25-qin.pdf)）。
- **DeepSeek**：V3/R1 线上推理系统开源，**prefill/decode 分离 + EP 专家并行 + 通信-计算重叠**，官方披露理论成本利润率 545%（[deepseek-ai/open-infra-index](https://github.com/deepseek-ai/open-infra-index)；[DeepSeek 推理系统博客](https://github.com/deepseek-ai/open-infra-index)）。
- **NVIDIA**：Dynamo + TensorRT-LLM 提供企业级 PD 分离与 KV 传输，面向数据中心 GPU 集群（[ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo)）。
- **各云厂商 / vLLM、SGLang 生态**：PD 分离成为主流默认能力；LMCache 在多家企业用于 KV 复用降本。

---

## 6. 趋势判断

1. **PD 分离从论文走向默认能力**：vLLM/SGLang/TRT-LLM/Dynamo 均已原生支持，进入"工程化收敛"阶段。
2. **KV cache 成为一等网络公民**：从"显存里的一块缓存"变成"可跨节点传输、可池化、可复用的网络对象"，催生 KV cache fabric / 池化。
3. **通信-计算重叠成为默认设计原则**（DualPipe、Mooncake layer-wise、DeepSeek overlap），"把通信藏进计算"。
4. **低精度通信普及**：FP8/更低精度的激活、KV 量化传输成为降本标配。
5. **MoE + 专家并行成为跨节点新瓶颈**，多机 MoE 调度/专家放置/offload 成为新战场。
6. **KV 传输协议标准化起步**（IETF 草案、Tensor-Centric 协议），但尚未收敛，窗口期存在。

---

## 7. 已饱和点（不建议作为切入点）

- **节点内张量并行**：Megatron→DTensor 已标准化，各框架一致，创新空间小。
- **密集模型流水线并行调度**：GPipe/1F1B/Zero-Bubble/DynaPipe 大量论文，趋于收敛。
- **连续批处理 + PagedAttention**：vLLM/SGLang 成熟默认能力。
- **PD 分离"架构概念"本身**：Splitwise/DistServe/Mooncake 已把 idea 讲透并工业化。
- **单模型静态切分 / 静态负载均衡**：成熟。

---

## 8. 被忽视的空白与机会

1. **KV cache 传输无开源标准**：Mooncake 传输协议闭源，TRT-LLM/SGLang/LMCache 各一套，互操作缺失，IETF 草案极早期。
2. **低延迟跨节点 KV 传输**：RDMA setup 延迟 + GB 级数据量，需 GDR + 量化 + 重叠的联合优化，缺统一 benchmark。
3. **多机 MoE 调度**：all-to-all 热点、专家放置、hot-expert 复制、通信-计算重叠，缺开源、可插拔的 EP 感知调度器。
4. **异构/边缘 P2P 的 KV 池化**：Exo 方向偏玩具，缺面向 WAN 带宽/隐私约束的实用 KV 共享层。
5. **KV 量化的精度-延迟权衡**：激进量化省带宽但伤质量，缺面向"传输"而非"存储"的 KV 量化标准。
6. **PD 分离的公平 benchmark**：现有评测碎片化，缺统一的可比性工具，导致 Mooncake/TRT-LLM/SGLang 各说各话。

---

## 9. 具体候选切入点（3-5 个可落地）

### 切入点 A：开源、后端无关的 KV-cache 传输协议 + 互操作层
- **做什么**：定义一套开放的、可版本协商的 KV 传输协议（参考 Tensor-Centric 协议 + IETF 草案），实现一个 Rust/Go 传输库（RDMA/GDR + NVMe-oF + TCP 后端可插拔），并给 vLLM/SGLang/TRT-LLM 写适配器。
- **为什么现在**：Mooncake 闭源、生态碎片化，是"标准真空期"。
- **验证指标**：跨节点 KV 传输延迟/吞吐 vs Mooncake RDMA；多框架互操作 demo。

### 切入点 B：面向传输的 KV 量化压缩（非存储导向）
- **做什么**：针对"prefill→decode 一次性传输"场景，设计 per-head FP8/int4 缩放 + 稀疏化的 KV 压缩，与 layer-wise 重叠结合，量化开销计入延迟预算。
- **为什么现在**：现有 KV 量化多为存储/复用优化，缺"传输专用"的精度-延迟最优解。
- **验证指标**：端到端 TTFT/TPOT 影响、PPL 退化、压缩率 vs 传输延迟收益。

### 切入点 C：开源 EP-aware 多机 MoE 调度器（可插拔）
- **做什么**：一个可插入 vLLM/SGLang 的调度器，做专家级负载均衡、hot-expert 复制、all-to-all 与计算重叠、冷专家 CPU offload 预取，支持跨节点 EP。
- **为什么现在**：DeepSeek 证明可行但内部实现闭源，社区缺统一开源的 EP 推理调度层。
- **验证指标**：MoE all-to-all 时间占比、专家负载方差、吞吐/尾延迟。

### 切入点 D：PD 分离的统一 benchmark 与剖析工具
- **做什么**：开源一套可复现的 PD 分离评测套件（真实工作负载 + 合成 KV 传输模型），统一度量 Mooncake/TRT-LLM/SGLang/Dynamo 的延迟-吞吐-成本帕累托，并做 KV 传输路径剖析。
- **为什么现在**：各厂商各说各话，缺中立标尺。
- **验证指标**：社区采用度、可复现的跨框架对比结论。

### 切入点 E：WAN 约束下的 P2P/边缘 KV 池化（Exo 方向升级）
- **做什么**：在 Exo 风格 P2P 拓扑上，增加面向带宽受限网络的 KV cache 增量同步、隐私隔离与去重，让家庭/边缘设备组网真正可用。
- **为什么现在**：Exo 已聚集 22K star 用户但实用性不足，KV 是 WAN 下最大瓶颈。
- **验证指标**：WAN 下 TTFT/TPOT、带宽节省率、多设备协同正确性。

---

## 10. 关键结论

分布式推理的竞争焦点已从"**怎么把模型放下**（TP/PP/EP）"转移到"**怎么把中间状态（KV cache）在节点间低成本、低延迟地流动**"。PD 分离是当前最确定的趋势，但它暴露出的 **KV 传输无标准、低延迟难、多机 MoE 调度难** 三个空白，正是尚未被巨头锁死的可切入机会窗口。
