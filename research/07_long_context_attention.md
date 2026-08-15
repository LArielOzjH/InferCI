# 长上下文与高效注意力：推理侧深度调研

> 调研方向：长上下文（Long-Context）与高效注意力（Efficient Attention），聚焦**推理/服务（Serving）侧**。
> 调研时间：2026 年中。核心事实均附出处 URL；star 数为写作时的数量级估计，会随时间漂移。

---

## 1. 现状总览

### 1.1 为什么是「长上下文 + 高效注意力」这个技术

标准 Transformer 的 softmax 注意力有两个根本代价，且随序列长度 L 增长而放大：

1. **计算代价 O(L²)**：prefill（预填充）阶段要一次性算整条序列的注意力分数矩阵，L 到 128K/1M 时计算量爆炸，直接卡住首 token 延迟（TTFT）。
2. **显存代价 O(L)**：每层都要缓存 K/V，KV cache 随 L 线性膨胀；decode 阶段每个 token 都要读一遍全部 KV，变成**显存带宽受限（memory-bound）**，吞吐被带宽卡死。

因此「长上下文」与「高效注意力」本质是同一个问题的一体两面：**要让模型真正用得起 100K–1M 上下文，必须同时削减注意力的计算量与 KV 显存/带宽**。这条线的所有技术（线性注意力/SSM、稀疏注意力、KV 压缩、上下文并行、RoPE 缩放）都是在不同层面拆解这两个 O(L²)/O(L) 代价。

### 1.2 当前主线与卡点

- **架构侧**：Mamba/Mamba-2、RWKV、RetNet、GLA、HGRN 等线性注意力/状态空间模型把注意力改成 O(1) 状态递推，推理显存几乎不随 L 增长，但**长程召回/上下文检索质量仍弱于全注意力**，因此出现「混合模型」（Jamba、Samba、Griffin）折中。
- **工程侧**：稀疏注意力（MInference、DeepSeek NSA、Moonshot MoBA）在保留 softmax 质量的同时跳过大量 KV；KV 压缩（H2O、StreamingLLM、SnapKV、PyramidKV）直接砍 KV cache 显存，是**目前生产落地最现实的一条**。
- **系统侧**：Ring Attention / DeepSpeed Ulysses 等上下文并行把序列切分到多卡，是突破单卡显存、跑到 1M+ 的硬手段；chunked prefill 与 prefill/decode 分离是服务调度的标配。
- **位置编码**：LongRoPE/YaRN 用极低成本把已有模型上下文窗口外推，是「伪长上下文」时代的产物，现已被原生长上下文训练取代，但仍广泛用于低成本扩窗。

**核心卡点**：
1. **「宣称的」与「有效的」上下文差距巨大**：RULER/NoLiMa 等评测显示，模型在 128K 广告窗口下有效利用率很低，且「Lost in the Middle」、长链推理退化普遍存在。
2. **稀疏注意力的真实加速难**：朴素稀疏在 GPU 上往往不快于 FlashAttention 稠密实现，必须**与 FlashAttention 的分块/tiling 对齐、硬件对齐**（NSA 的核心论点）才有收益。
3. **KV 压缩是有损且无召回保证**：压缩策略与下游任务（尤其长推理、多跳检索）强相关，难以做到「通用且保精度」。
4. **decode 阶段仍是带宽瓶颈**：稀疏/量化/混合内核在 decode 侧的成熟度远低于 prefill 侧。

---

## 2. 关键技术（带出处）

### 2.1 线性注意力 / 状态空间模型（SSM）

- **Mamba**（Selective SSM）：把 S4 的时不变状态改为输入相关（selective），用硬件感知的 scan 算法实现线性时间训练与 O(1) 逐 token 推理。出处：[arXiv 2312.00752](https://arxiv.org/abs/2312.00752)。
- **Mamba-2 / SSD**：提出「结构化状态空间对偶」（SSD），证明 SSM 与注意力可通过半可分矩阵（semiseparable matrices）统一，得到比 Mamba-1 快 2–8 倍的张量核友好算法，训练吞吐反超高度优化的 Transformer。出处：[arXiv 2405.21060](https://arxiv.org/abs/2405.21060)。
- **RetNet**：用 retention 机制支持并行/递归/分块三种等价形式，训练并行、推理 O(1)。出处：[arXiv 2307.08621](https://arxiv.org/abs/2307.08621)。
- **GLA**（Gated Linear Attention）：给线性注意力加数据相关门控 + 硬件高效分块训练。出处：[arXiv 2312.06635](https://arxiv.org/abs/2312.06635)。
- **HGRN**（Hierarchically Gated Recurrent Network）：用分层门控 RNN 近似注意力，线性复杂度。出处：[arXiv 2311.04823](https://arxiv.org/abs/2311.04823)。
- **RWKV-7「Goose」**：引入 state tuning（状态微调），在 O(1) 递推下逼近甚至超过同尺寸 Transformer 质量；RWKV-8/9 继续沿「线性注意力 + 状态」演进。出处：[RWKV wiki](https://github.com/RWKV/RWKV-wiki) / [State Tuning arXiv 2504.05097](https://arxiv.org/abs/2504.05097)。

> 本质：把 O(L²) 的成对注意力换成 O(1) 的「状态压缩 + 递推」，代价是**状态是有损压缩**，长程精确检索天然弱于显式注意力——这是「混合模型」出现的原因。

### 2.2 混合模型（Hybrid：全注意力 + 线性/SSM 层）

- **Jamba**（AI21）：Mamba 层 + Transformer 注意力层 + MoE，256K 上下文，是首个大规模混合架构。出处：[HF Jamba-v0.1](https://huggingface.co/ai21labs/Jamba-v0.1)。
- **Samba**（Microsoft）：混合 Mamba 与滑动窗口注意力，3.8B 训到 **1M 上下文**，声称无限长度外推。出处：[arXiv 2406.07522](https://arxiv.org/abs/2406.07522) / [ICLR 2025](https://mlanthology.org/iclr/2025/ren2025iclr-samba/)。
- **Griffin**（DeepMind）：门控线性递归层 + 局部注意力。出处：[arXiv 2402.19427](https://arxiv.org/abs/2402.19427)。
- **Zamba**（Zyphra）：共享注意力 + Mamba 的小型混合模型，主打效率。出处：[The Neural Base](https://theneuralbase.com/transformer-architecture/learn/advanced/hybrid-models-jamba-zamba/)。

> 趋势：混合架构是当前「质量 vs 成本」的最优折中，但带来**双内核（attention + SSM）调度的工程复杂度**。

### 2.3 Ring Attention / 上下文并行（Context Parallelism）

- **Ring Attention with Blockwise Transformers**（Liu et al.）：把序列切成块在设备间组成环形通信，块级并行计算注意力，实现「near-infinite context」。出处：[arXiv 2310.01889](https://arxiv.org/abs/2310.01889) / [GitHub](https://github.com/lhao499/RingAttention)。
- **DeepSpeed Ulysses**：按 head 维度切分 + all-to-all 通信，训练超长序列。出处：[arXiv 2309.14509](https://arxiv.org/abs/2309.14509)。
- **Megatron-LM Context Parallel**：NVIDIA 0.5+ 版本引入的一等并行维度。出处：[AI Wiki](https://aiwiki.ai/wiki/ring_attention)。
- **Mnemosyne**：针对**百万级上下文服务**的并行策略（环形/条纹注意力的 head-of-line 阻塞问题）。出处：[arXiv 2409.17264](https://arxiv.org/abs/2409.17264)。

### 2.4 稀疏注意力（Sparse Attention）

- **MInference**（Microsoft）：利用长上下文注意力的动态稀疏性（三种模式：A-shape/Vertical/Slash），prefill 最高 **10x 加速**、可跑 **1M token**（A100）。出处：[arXiv 2407.02490](https://arxiv.org/abs/2407.02490) / [GitHub](https://github.com/microsoft/MInference)。
- **DeepSeek NSA**（Native Sparse Attention）：**原生可训练、硬件对齐**的稀疏注意力，三路（token 压缩 + 块选择 + 滑动窗口），64K 上下文下解码提速最高 11.6x，端到端训练。出处：[arXiv 2502.11089](https://arxiv.org/abs/2502.11089) / [ACL 2025](https://aclanthology.org/2025.acl-long.1126.pdf)。
- **Moonshot MoBA**（Mixture of Block Attention）：把上下文分块，每 query 只选 top-k 块做注意力（gating 机制），与 MoE 思路同构。出处：[arXiv 2502.13189](https://arxiv.org/abs/2502.13189)。
- **NVIDIA Skip Softmax Attention**：TensorRT-LLM 中跳过对注意力贡献微小的 softmax 计算来加速长上下文。出处：[NVIDIA Blog](https://developer.nvidia.com/blog/accelerating-long-context-inference-with-skip-softmax-in-nvidia-tensorrt-llm/)。
- **IndexCache**（THUDM）：跨层复用稀疏注意力索引减少在线选择开销。出处：[GitHub](https://github.com/THUDM/IndexCache)。

> 本质：attention 分数天然稀疏，**skip 掉不重要的 KV** 同时保持 softmax 的检索精度；真正的难点是把稀疏模式做成 GPU 硬件友好的 block 结构并融合进 FlashAttention。

### 2.5 KV Cache 压缩（生产落地最成熟的一条）

- **StreamingLLM**：发现「attention sink」（开头几个 token 吸收大量注意力），保留 sink + 滚动窗口即可无限流式生成，支持到 **4M token**。出处：[ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html) / [arXiv 2309.17453](https://arxiv.org/abs/2309.17453)。
- **H2O**（Heavy-Hitter Oracle）：动态保留「重击者」token + 最近 token，贪心但有理论性质，5x 吞吐提升。出处：[arXiv 2306.14048](https://arxiv.org/abs/2306.14048)。
- **SnapKV**：观察 prompt 末尾注意力投票识别重要 KV，压缩 prompt KV，几乎不损精度。出处：[arXiv 2404.14469](https://arxiv.org/abs/2404.14469)。
- **PyramidKV**：不同层分配**金字塔形**预算（浅层多、深层少），更贴合各层重要性分布。出处：[arXiv 2406.02069](https://arxiv.org/abs/2406.02069)。
- **KV-Compress**：分页 + 每 head 可变压缩率。出处：[arXiv 2410.00161](https://arxiv.org/abs/2410.00161)。
- **CacheBlend**：面向 RAG，对缓存的历史 KV 做「选择性重算」再融合，避免重复 prefill。出处：[arXiv 2405.16444](https://arxiv.org/abs/2405.16444) / [EuroSys 2024](https://uchi-jcl.github.io/group-website/publication/cacheblend/)。
- **DeepSeek MLA**（Multi-head Latent Attention）：把 KV 低秩压缩成 latent 向量，KV cache 减少约 90%+，是 DeepSeek-V2/V3/R1 廉价长上下文的核心。出处：[arXiv 2405.04434](https://arxiv.org/abs/2405.04434) / [TransMLA 2502.07864](https://arxiv.org/abs/2502.07864)。

> 本质：KV 是长上下文推理显存的最大头；压缩 KV = 更多并发/更大 batch = 更低的每 token 成本。**MLA 是「训练时就把 KV 变小」的范式，其余多是「推理时后验裁剪」范式**。

### 2.6 RoPE 缩放（低成本扩窗）

- **YaRN**：NTK-aware 插值 + 温度因子 + 注意力缩放，小数据量即可扩窗。出处：[arXiv 2309.00071](https://arxiv.org/abs/2309.00071)。
- **LongRoPE**：非均匀位置插值 + 渐进扩展，把 LLaMA 扩到 **2M token**。出处：[arXiv 2402.13753](https://arxiv.org/abs/2402.13753)。
- **LongRoPE2**：近无损扩窗，只训极小参数量。出处：[arXiv 2502.20082](https://arxiv.org/abs/2502.20082)。

> 本质：位置编码决定模型「见过的最长位置」，RoPE 缩放是用小代价把已有权重外推到更长位置；现在主流厂商已改为**原生长上下文预训练**，RoPE 缩放退居为微调/廉价扩窗工具。

### 2.7 Chunked Prefill 与异步调度

- **vLLM PagedAttention**：KV 分页管理，消除碎片化，是长上下文服务的基础设施。出处：[arXiv 2309.06180](https://arxiv.org/abs/2309.06180)。
- **Sarathi-Serve**：chunked prefill（把长 prefill 切成小块与 decode 交错）+ stall-free 调度，调和吞吐-延迟权衡。出处：[OSDI 2024](https://arxiv.org/abs/2403.02310)。
- **DistServe**：prefill 与 decode **分离部署**（PD disaggregation），各用不同硬件/并行度。出处：[OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) / [arXiv 2401.09670](https://arxiv.org/abs/2401.09670)。

> 本质：长上下文下 prefill 是计算密集、decode 是带宽密集，两类负载相互干扰；chunked/disaggregation 是把它们解耦，SGLang/vLLM 均已原生支持。

### 2.8 长上下文评测

- **Needle-in-a-Haystack**（Kamradt, 2023）：最早的长上下文检索压力测试，但被证明过于简单（单 needle 靠位置编码即可作弊）。
- **RULER**（Hsieh et al., 2024）：合成多类任务（多 needle、聚合、QA），揭示模型「真实上下文大小」远低于广告窗口。出处：[arXiv 2404.06654](https://arxiv.org/abs/2404.06654)。
- **LongBench / LongBench v2**（Bai et al., 2023/2024）：中英文多任务长上下文基准。出处：[arXiv 2308.14508](https://arxiv.org/abs/2308.14508)。
- **NoLiMa**（ICML 2025）：超越「字面匹配」的长上下文评测，考察真正的聚合/推理。出处：[ICML 2025](https://mlanthology.org/icml/2025/modarressi2025icml-nolima/)。
- **LOFT**（Google, 2024）：面向检索/RAG 的长上下文基准。出处：[arXiv 2406.13121](https://arxiv.org/abs/2406.13121)。

### 2.9 1M 上下文 Serving 成本（关键量化）

粗算（全注意力，FP16，无压缩）：
- **KV cache 显存** ≈ 2 × layers × hidden × 2 bytes × L。以 70B 级模型（80 层、8192 维）为例，单 token KV ≈ 2×80×8192×2 ≈ 2.6MB/token；**1M token ≈ 2.6TB KV**，需要几十张 80GB 卡。7B 级（32 层、4096 维）单 token ≈ 0.5MB，1M ≈ 0.5TB。
- **prefill 计算**：O(L²) 注意力 FLOPs，1M 序列的稠密注意力需要 FlashAttention 级别的内核 + 上下文并行才能跑。
- **decode 带宽**：每步读全部 KV，1M KV 单步就是几百 MB 读，纯带宽瓶颈。

因此 1M 上下文服务在工程上基本由三条路收敛：**(1) MLA/低秩 KV**（DeepSeek 路线，KV 降 10x）、**(2) 稀疏注意力**（NSA/Minference，跳过 KV）、**(3) 上下文并行 + 混合/线性架构**（Samba/Gemini 路线）。参考：[Advertised vs Effective Context Windows](https://benchlm.ai/blog/posts/context-window-comparison) / [Long-Context LLM Infrastructure](https://introl.com/ja/blog/long-context-llm-infrastructure-million-token-windows-guide)。

---

## 3. 论文清单（名称 + 年份 + venue + 一句话核心）

| 论文 | 年份 / Venue | 一句话核心 |
|---|---|---|
| Mamba: Linear-Time Sequence Modeling with Selective SSMs | 2023 / arXiv (COLM 2024) | 选择性状态 + 硬件感知 scan，线性时间训练、O(1) 推理。 |
| Transformers are SSMs (Mamba-2) | 2024 / ICML | SSD 对偶统一 SSM 与注意力，比 Mamba-1 快 2–8x。 |
| RetNet | 2023 / arXiv | retention 三态等价，训练并行、推理 O(1)。 |
| Gated Linear Attention (GLA) | 2023 / arXiv | 数据相关门控 + 硬件高效分块线性注意力。 |
| HGRN | 2023 / arXiv (NeurIPS) | 分层门控 RNN 逼近注意力，线性复杂度。 |
| RWKV-7 "Goose" | 2025 / arXiv | state tuning 让 O(1) 递推逼近 Transformer 质量。 |
| Griffin | 2024 / arXiv (DeepMind) | 门控线性递归 + 局部注意力混合。 |
| Jamba | 2024 / arXiv (AI21) | Mamba + Transformer + MoE，256K 上下文。 |
| Samba | 2024 / ICLR 2025 | 3.8B 混合模型训到 1M 上下文。 |
| Ring Attention with Blockwise Transformers | 2023 / arXiv | 环形块级注意力实现 near-infinite context。 |
| DeepSpeed Ulysses | 2023 / arXiv | 按 head 切分 + all-to-all 的序列并行。 |
| MInference | 2024 / NeurIPS Spotlight | 动态稀疏注意力，prefill 10x 加速、1M token。 |
| DeepSeek NSA | 2025 / ACL 2025 | 硬件对齐、原生可训练稀疏注意力。 |
| Moonshot MoBA | 2025 / arXiv | Mixture-of-Block-Attention，块级 gating 稀疏。 |
| StreamingLLM | 2024 / ICLR | attention sink + 滚动窗口，4M token 流式。 |
| H2O | 2023 / NeurIPS | 重击者 + 最近 token 的 KV 淘汰。 |
| SnapKV | 2024 / arXiv (NeurIPS) | 注意力投票压缩 prompt KV。 |
| PyramidKV | 2024 / arXiv | 金字塔形分层 KV 预算。 |
| CacheBlend | 2024 / EuroSys | RAG 缓存 KV 选择重算 + 融合。 |
| YaRN | 2023 / arXiv | NTK 插值 + 温度缩放扩窗。 |
| LongRoPE | 2024 / arXiv | 非均匀插值扩窗到 2M token。 |
| RULER | 2024 / COLM | 揭示真实上下文大小远低于广告窗口。 |
| NoLiMa | 2025 / ICML | 超越字面匹配的长上下文评测。 |
| Sarathi-Serve | 2024 / OSDI | chunked prefill + stall-free 调度。 |
| DistServe | 2024 / OSDI | prefill/decode 分离服务。 |
| vLLM PagedAttention | 2023 / SOSP | KV 分页管理，长上下文服务基础设施。 |
| DeepSeek-V2 (MLA) | 2024 / arXiv | 低秩 latent KV，KV 减 ~90%。 |

---

## 4. 开源项目盘点（名称 + star 量级 + 活跃度 + 维护方）

> star 数为数量级估计，随时间变化；「活跃度」基于写作时点。

| 项目 | star 量级 | 活跃度 | 维护方 | 定位 |
|---|---|---|---|---|
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | 40k+ | 极活跃 | vLLM 社区（UC Berkeley 起） | 长上下文服务事实标准，PagedAttention/MLA/稀疏集成 |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | 15k+ | 极活跃 | SGLang 社区（LMSYS/伯克利） | RadixAttention、chunked prefill、NSA 优化路线 |
| [dao-ailab/flash-attention](https://github.com/dao-ailab/flash-attention) | 15k+ | 活跃 | Tri Dao 团队 | FA2/FA3 内核，稀疏注意力融合底座 |
| [state-spaces/mamba](https://github.com/state-spaces/mamba) | ~14k | 中等 | Albert Gu / Tri Dao | Mamba/Mamba-2 官方实现 |
| [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) | ~12k | 活跃 | RWKV 社区（Peng Bo） | RWKV 全系列线性模型 |
| [microsoft/MInference](https://github.com/microsoft/MInference) | ~3k | 中等 | Microsoft | 动态稀疏注意力加速长上下文 prefill |
| [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) | ~3k | 活跃 | DeepSeek | MLA 高效内核（Hopper） |
| [mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm) | ~1k | 低 | MIT HAN Lab | 无限流式 KV 窗口 |
| [FMB-code/H2O](https://github.com/FMB-code/H2O) | ~1k | 低 | Rice 大学 | 重击者 KV 淘汰 |
| [Zefan-Cai/KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory) | ~1k | 中 | 社区（清华系） | 统一 KV 压缩方法框架 |
| [lhao499/RingAttention](https://github.com/lhao499/RingAttention) | ~500 | 低 | Hao Liu (Berkeley) | 环形注意力参考实现 |
| [THUDM/IndexCache](https://github.com/THUDM/IndexCache) | 数百 | 新 | 清华 THUDM | 跨层复用稀疏索引 |

---

## 5. 公司落地

- **Google / Gemini**：Gemini 2.5 Pro 提供 1M（预览时 2M）上下文，走长上下文 + 高效注意力 + 大规模并行服务的闭源路线（[对比](https://artificialanalysis.ai/models/comparisons/deepseek-v3-1-terminus-vs-gemini-2-5-pro)）。
- **DeepSeek**：MLA（V2/V3/R1）+ NSA（V3 系列集成）是「廉价长上下文 + 稀疏注意力」的标杆，FlashMLA 内核开源，128K 上下文；SGLang/vLLM 均有 MLA/NSA 集成路线（[SGLang NSA issue](https://github.com/sgl-project/sglang/issues/11989)、[vLLM 稀疏 MLA](https://github.com/vllm-project/vllm/issues/38006)）。
- **Moonshot（Kimi）**：MoBA 块注意力 + 长上下文产品（Kimi 系列以长文本出名）。
- **Microsoft**：MInference（动态稀疏）、Samba（1M 混合模型），并把稀疏注意力集成到推理栈。
- **AI21**：Jamba 混合架构商用。
- **NVIDIA**：TensorRT-LLM 推出 Skip Softmax Attention 加速长上下文（[Blog](https://developer.nvidia.com/blog/accelerating-long-context-inference-with-skip-softmax-in-nvidia-tensorrt-llm/)）；Megatron-LM 提供上下文并行。
- **Zyphra**：Zamba 混合模型主打推理效率。

---

## 6. 趋势判断

1. **原生稀疏注意力（NSA 范式）成为下一代默认**：从「推理时后验裁剪」转向「训练时原生稀疏 + 硬件对齐」，因为只有 natively trainable 才能让稀疏模式稳定且可被内核高效执行。
2. **KV 压缩从「研究 trick」走向「生产可开关」**：SnapKV/PyramidKV 类方法被收编进 vLLM/SGLang 作为可配置 eviction 策略，配「召回回归」门禁。
3. **混合架构与「质量-成本」分级并存**：全注意力（质量上限）+ 混合/线性（成本下限）会长期共存，模型按场景选型。
4. **MLA 低秩 KV 成为标配**：训练侧直接压缩 KV 比推理侧裁剪更稳，TransMLA 让 GQA 模型也能迁移。
5. **上下文并行向 decode 侧下沉**：长上下文 decode 的带宽瓶颈推动 decode context parallel（如 SGLang issue #12196）。
6. **评测从 Needle 升级到聚合推理**：NoLiMa/LOFT 代表「有效上下文」的真正度量，会倒逼厂商不再只堆窗口长度。

---

## 7. 已饱和点（Saturated）

- **RoPE 缩放扩窗**：LongRoPE/YaRN 技术路线已被原生长上下文训练取代，纯插值扩窗的红利基本榨干。
- **「堆窗口长度」营销**：广告 128K/1M 与有效上下文差距已被 RULER/NoLiMa 戳破，单纯加大窗口不再是卖点。
- **Needle-in-a-Haystack 单点检索评测**：过于简单，已不作为有区分度的指标。
- **单卡注意力内核（FlashAttention 稠密路径）**：FA2/FA3 已把 dense attention 做到接近硬件极限，进一步优化空间小。
- **静态/均匀 KV 裁剪**：简单固定预算的 KV 淘汰研究趋于饱和，需转向自适应 + 召回保证。

---

## 8. 被忽视的空白与机会

1. **decode 侧的稀疏/压缩内核**：多数稀疏注意力（MInference/NSA）优化 prefill，但 decode 才是长上下文服务的成本大头（带宽受限），decode 稀疏化 + 融合 PagedAttention 的内核仍是空白。
2. **KV 压缩的「召回保证」与可观测性**：压缩方法的精度回归缺少统一门禁；「质量-成本」换算（RULER 分/美元）没有标准工具。
3. **混合注意力的统一内核运行时**：attention + SSM 双路径的调度、负载均衡、kernel 融合缺乏成熟的自动分发框架。
4. **上下文并行 decode**：>1M 的 decode 服务、CP 的 KV 通信优化（Mnemosyne 方向）仍未产品化。
5. **长上下文评测与 serving 的闭环**：把 NoLiMa/LOFT 作为 KV 压缩/稀疏部署的 CI 回归，目前基本没有系统做。
6. **稀疏注意力索引的在线开销**：块选择本身有开销（IndexCache 才刚开始解决跨层复用）。

---

## 9. 具体候选切入点（可做）

1. **FlashAttention 融合的 decode 侧原生稀疏 KV 内核**：把 NSA 的块选择 + 稀疏 PagedAttention 下沉到 decode 阶段，用 Triton/CUDA 做一个硬件对齐的稀疏 decode 内核，目标是把 decode 的 KV 带宽开销降 5–10x。度量：decode tokens/s 与 RULER 精度。锚点：NSA + FA3/FA4 + PagedAttention。
2. **生产级自适应 KV 压缩插件（带召回保证）**：在 vLLM/SGLang 上做一个 scheduler-aware 的 KV eviction 插件，整合 SnapKV/PyramidKV/attention-sink，支持 per-request 预算 + 热点重算，并把 RULER/NoLiMa 做成回归门禁；输出「压缩率 vs 精度」曲线。锚点：KVCache-Factory + CacheBlend。
3. **混合注意力统一内核运行时 + 成本模型**：写一个跨 dense/稀疏/线性(SSM) 路径的自动分发器（按层/head/工作负载选择内核），并附一个「质量-成本」成本模型；解决双内核负载均衡与 kernel 碎片化。锚点：Jamba/Samba + FA + Mamba。
4. **长上下文 serving 成本可观测工具**：开源一个针对 128K–1M 服务的成本/显存/吞吐测量与预测工具，横跨 dense、MLA、稀疏、KV 压缩四档，输出「每 token 成本 + 有效上下文质量」，做成 LLM 选型/容量规划的标尺。锚点：benchlm/introl 类分析的产品化。
5. **上下文并行 decode 优化**：针对 >1M 上下文的 decode，优化 CP 的 KV 末块通信与跨卡注意力（Mnemosyne 方向的工程化），或在 SGLang 上实现高效 decode CP。锚点：Mnemosyne + Ring Attention。

---

## 附：主要出处链接

- Mamba: https://arxiv.org/abs/2312.00752
- Mamba-2: https://arxiv.org/abs/2405.21060
- RetNet: https://arxiv.org/abs/2307.08621
- GLA: https://arxiv.org/abs/2312.06635
- HGRN: https://arxiv.org/abs/2311.04823
- RWKV state tuning: https://arxiv.org/abs/2504.05097
- Samba: https://arxiv.org/abs/2406.07522
- Jamba: https://huggingface.co/ai21labs/Jamba-v0.1
- Ring Attention: https://arxiv.org/abs/2310.01889
- DeepSpeed Ulysses: https://arxiv.org/abs/2309.14509
- Mnemosyne: https://arxiv.org/abs/2409.17264
- MInference: https://arxiv.org/abs/2407.02490 / https://github.com/microsoft/MInference
- NSA: https://arxiv.org/abs/2502.11089 / https://aclanthology.org/2025.acl-long.1126.pdf
- MoBA: https://arxiv.org/abs/2502.13189
- Skip Softmax (TensorRT-LLM): https://developer.nvidia.com/blog/accelerating-long-context-inference-with-skip-softmax-in-nvidia-tensorrt-llm/
- IndexCache: https://github.com/THUDM/IndexCache
- StreamingLLM: https://arxiv.org/abs/2309.17453
- H2O: https://arxiv.org/abs/2306.14048
- SnapKV: https://arxiv.org/abs/2404.14469
- PyramidKV: https://arxiv.org/abs/2406.02069
- KV-Compress: https://arxiv.org/abs/2410.00161
- CacheBlend: https://arxiv.org/abs/2405.16444
- MLA/DeepSeek-V2: https://arxiv.org/abs/2405.04434
- TransMLA: https://arxiv.org/abs/2502.07864
- YaRN: https://arxiv.org/abs/2309.00071
- LongRoPE: https://arxiv.org/abs/2402.13753
- LongRoPE2: https://arxiv.org/abs/2502.20082
- RULER: https://arxiv.org/abs/2404.06654
- NoLiMa: https://mlanthology.org/icml/2025/modarressi2025icml-nolima/
- LongBench: https://arxiv.org/abs/2308.14508
- LOFT: https://arxiv.org/abs/2406.13121
- Sarathi-Serve: https://arxiv.org/abs/2403.02310
- DistServe: https://arxiv.org/abs/2401.09670
- vLLM PagedAttention: https://arxiv.org/abs/2309.06180
- KVCache-Factory: https://github.com/Zefan-Cai/KVCache-Factory
- FlashMLA: https://github.com/deepseek-ai/FlashMLA
- FlashAttention: https://github.com/dao-ailab/flash-attention
