# 剪枝与稀疏化（Pruning & Sparsity）—— 推理侧深度调研

> 调研时间：2026-08（基于实时联网检索 arXiv / 顶会 / 技术博客 / GitHub / 行业报道）
> 聚焦：**推理侧** 的权重稀疏、激活稀疏、稀疏注意力、稀疏内核与开源/落地生态。

---

## 1. 现状总览

### 1.1 这个技术解决什么瓶颈

大模型推理的两大硬约束是 **算力（FLOPs）** 与 **访存（memory bandwidth）**。Transformer 里每个 token 都要经过全部层、全部 attention head、全部 FFN 神经元，即使很多计算对最终输出贡献极小。稀疏化的核心命题是：

> **只计算"有用"的那部分** —— 把权重里接近 0 的元素、对当前输入不激活的神经元、对当前 query 不相关的 KV，在推理时跳过，从而把 FLOPs 和访存同时降下来，而不是像量化那样只降精度。

它和量化是正交的：量化把"每个数"变小，稀疏把"数的个数"变少。二者联合（2:4 稀疏 + INT4）能叠加收益（[Sparse-Marlin](https://github.com/IST-DASLab/Sparse-Marlin)）。

### 1.2 稀疏的三条主线

| 主线 | 对象 | 是否硬件友好 | 代表工作 |
|---|---|---|---|
| **权重稀疏（静态）** | 权重矩阵里的 0 | 需结构化(2:4/N:M/block)才提速 | SparseGPT、Wanda、SparseLLM |
| **激活/动态稀疏（输入相关）** | 每次前向实际激活的神经元/head | 依赖硬件与预测器 | Deja Vu、TEAL、ProSparse、PowerInfer |
| **注意力稀疏** | attention 矩阵 / KV 参与范围 | 天然块状，最易落地 | MInference、NSA、MoBA、FlashAttention block-sparse |

### 1.3 当前卡点：为什么"雷声大雨点小"

1. **非结构化稀疏没有真实速度收益**。GPU/TPU 的 tensor core 只能对"稠密 tile"或"2:4 等规则模式"加速；随机 80% 置 0 的矩阵在 dense GEMM 里依然逐元素计算（权重已加载，0 也参与 MAC），只有极度内存受限时才靠"跳过载入"省一点。社区反复讨论这一点的典型例子是 SparseGPT 官方 repo 的 issue：[How should I verify the speedup effect](https://github.com/IST-DASLab/sparsegpt/issues/15)。
2. **结构化稀疏（2:4）硬件绑定 NVIDIA**。2:4 是 NVIDIA Ampere+ tensor core 的专有模式，靠 [cuSPARSELt](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/)/CUTLASS 落地；AMD、昇腾、TPU、以及很多推理芯片不支持，导致"稀疏模型"不可移植。
3. **精度损失与恢复成本**。激进剪枝（>50%）会掉点，要么用 LoRA 微调恢复（[LLM-Pruner](https://arxiv.org/abs/2305.11627)），要么继续预训练（[Sheared LLaMA](https://arxiv.org/abs/2310.06694)），工程链路长、成本接近从头训小模型。
4. **工程复杂度高**。稀疏需要自定义布局/重排/元数据、专用 kernel，serving 框架（vLLM/SGLang）支持是"打补丁"式的，不同稀疏格式互不兼容。
5. **生态震荡**。最激进的稀疏创业公司 Neural Magic 的 SparseML/DeepSparse 已 **archived**（转向 llm-compressor + vLLM），本身就是"独立稀疏栈难以为继"的信号。

结论：**稀疏在"注意力/长上下文"这条线上已经真落地（DeepSeek、Moonshot 量产），在"权重剪枝"这条线上仍然主要停留在论文与 demo。**

---

## 2. 关键技术（含出处 URL）

### 2.1 权重稀疏：非结构化 → 结构化（2:4 / N:M / block）

- **SparseGPT**（ICML 2023, [arXiv:2301.00774](https://arxiv.org/abs/2301.00774)）：首个把 LLM 一次性剪到 50–60% 稀疏且精度基本不掉的工作。核心是逐层求解近似稀疏重构（基于 Hessian 逆的 OBS 思想 + 免再训练的列稀疏），支持 **2:4 / 4:8 等 N:M 半结构化**与 block 模式，可在 175B 级模型上跑。它证明了"稀疏化大模型在算法上可行"，但速度收益依赖 cuSPARSELt。
- **Wanda**（ICLR 2024, [arXiv:2306.11695](https://arxiv.org/abs/2306.11695)）：剪枝准则 = `|W| · ||X||₂`（权重 × 激活范数），**无需任何更新/梯度**，比 SparseGPT 更快更省内存，1 分钟内剪 LLaMA-7B。是最常用的轻量基线。后续 Wanda++ 用区域梯度改进（[arXiv:2503.04992](https://arxiv.org/html/2503.04992)）。
- **SparseLLM**（NeurIPS 2024, [arXiv:2402.17934](https://neurips.cc/virtual/2024/poster/93617)）：把全局剪枝目标分解成"每层子问题 + 全局掩码交替优化"，解决了 SparseGPT/Wanda 逐层局部贪婪导致的次优，是后训练全局剪枝的代表。
- **Compressing LLMs**（Frantar & Alistarh, ICLR 2024）：系统对比剪枝 vs 量化，指出**稀疏化的真实加速高度依赖硬件 2:4 支持**，而量化（INT4）在任何 GPU 上都有收益——解释了权重稀疏落地难的关键论据。

**半结构化 2:4 的硬件基础**：NVIDIA Ampere（A100）+ tensor core 原生支持 2:4 稀疏，理论 2× 峰值；[cuSPARSELt](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/) 提供 SpMM 接口；[TensorRT Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)（原 AMMO）提供 sparsification + structured pruning 的工业级流程。但 AMD ROCm 无等价、跨厂商不可移植。

### 2.2 结构化剪枝（head / channel / layer 粒度）

- **LLM-Pruner**（NeurIPS 2023, [arXiv:2305.11627](https://arxiv.org/abs/2305.11627)）：把结构化剪枝做成"依赖图分组 → 按组重要性打分 → 剪掉耦合单元 → LoRA 微调恢复"，输出的是**真正能提速的稠密小模型**（head/神经元/层被整块删除），而非稀疏权重。
- **ShortGPT**（2024, [arXiv:2403.03853](https://arxiv.org/abs/2403.03853)）：发现 LLM 的**层冗余**远超预期，用 Block Influence (BI) 指标直接删 ~25% 的层，部分模型删 10 层质量几乎不变，为"深度剪枝"提供依据。
- **Sheared LLaMA**（ICLR 2024, [arXiv:2310.06694](https://arxiv.org/abs/2310.06694)）：结构化剪枝 + **继续预训练**的完整范式——"shear（剪）→ 动态批量加载恢复"，用极低成本从大模型得到强小模型（Sheared-LLaMA 系列）。是"剪枝不是终点，恢复训练才是"的标杆。

### 2.3 稀疏注意力（注意力/KV 稀疏）—— 目前最成功的落地线

- **MInference**（微软, 2024, [arXiv:2407.02490](https://arxiv.org/abs/2407.02490)）：**免训练**的动态稀疏注意力，离线识别三种模式（A-shape / Vertical-Slash / Block-Sparse），预填充最高 **10× 加速、单卡 A100 跑 1M token**。证明长上下文是稀疏最甜的场景。
- **DeepSeek NSA**（[arXiv:2502.11089](https://arxiv.org/abs/2502.11089)）：**硬件对齐、原生可训练**的稀疏注意力——分块压缩 token + 块选（top-k）+ 滑动窗口，与 tensor core 的 block 计算对齐，训练/推理统一，长序列 **11.6× 加速**。梁文锋署名。
- **Moonshot MoBA**（NeurIPS 2025, [arXiv:2502.13189](https://arxiv.org/abs/2502.13189)）：把 **MoE 路由思想搬到注意力**——query 只在若干 KV block 上做 full attention，其余 block 只算均值/跳过，长上下文可扩展，已用于 Kimi。
- **DeepSeek Sparse Attention (DSA)**（DeepSeek-V3.2, 2025）：NSA 思路的工程化量产，[SGLang Day-0 支持](https://lmsys.org/blog/2025-09-29-deepseek-V32/)，API 成本降 ~50%（[报道](https://yourstory.com/ai-story/deepseek-sparse-attention-api-price-cut-ai)）。稀疏注意力的**标准化/落地速度远超权重稀疏**。
- **FlashAttention 稀疏化**：Tri Dao 早期 [Block-Sparse Attention](https://github.com/openai/blocksparse) 演示了 block-sparse 在 Triton 上的可行性；FlashAttention 本体（[2205.14135](https://arxiv.org/abs/2205.14135)）是 exact attention，但其 tile 化设计正是后来各种 block/top-k 稀疏内核的底座。注意：FA 官方对稀疏注意力的支持一直不完整/未纳入主线，是个生态空档。

### 2.4 稀疏内核

- **cuSPARSELt**（NVIDIA）：2:4 SpMM 官方库，Ampere+ 硬件稀疏的唯一成熟入口（[NVIDIA 博客](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/)）。
- **Marlin**（IST-DASLab, [GitHub](https://github.com/IST-DASLab/marlin)）：FP16×INT4 的 LLM 推理 kernel，近理想 ~4× 加速（batch 16–32）；已被 [vLLM 合入](https://github.com/vllm-project/vllm/pull/2497)。**Sparse-Marlin**（[GitHub](https://github.com/IST-DASLab/Sparse-Marlin)）把 **2:4 稀疏 + 4-bit 量化**叠加到同一 kernel，是"稀疏+量化联合"的内核级范本（但 star 少、维护弱）。
- **STen**（IST-DASLab, [ACM TACO](https://dl.acm.org/doi/full/10.1145/3815424)）：PyTorch 里"以稀疏格式写 kernel"的编译器/运行时，降低稀疏内核开发门槛。

### 2.5 动态 / 输入相关稀疏（激活稀疏）

- **Deja Vu**（ICML 2023, [arXiv:2310.17157](https://arxiv.org/abs/2310.17157)）：**上下文稀疏**——用小型 predictor 预测哪些 head/神经元对当前输入重要，verifier 兜底，推理时按输入跳过（~2× 加速，质量微降）。
- **PowerInfer**（SOSP 2024, [arXiv:2312.12456](https://arxiv.org/abs/2312.12456)）：利用 LLM 激活的**幂律分布**（少数"热神经元"高频激活），把热神经元放 GPU、冷神经元放 CPU，消费级单卡跑大模型，11× token 生成速度。
- **ProSparse / PowerInfer-2**（2024, [arXiv:2402.13516](https://huggingface.co/SparseLLM/prosparse-llama-2-7b)）：用 ReLU 化 + 稀疏正则**训练出内在高激活稀疏**的模型（>80% 稀疏），PowerInfer-2 在 GPU 上也能吃到激活稀疏收益。
- **TEAL**（ICLR 2025, [arXiv:2408.14690](https://arxiv.org/abs/2408.14690)）：**免训练**激活稀疏——对现有模型逐层 magnitude 剪激活，40–50% 稀疏基本不掉点，40% 速度提升。把"动态稀疏"降到了"跑一次校准"的成本。

### 2.6 稀疏 + MoE

- MoE 本质就是**结构化的条件计算稀疏**（每个 token 只走 top-k 专家，[Switch Transformer](https://arxiv.org/abs/2101.03961) 开先河）。
- 新趋势是"稀疏注意力 × MoE"合流：**MoBA 用 MoE 式门控做块注意力**，DeepSeek-V3 用 MoE 叠 DSA。稀疏已经从"压缩技巧"变成**架构级条件计算原语**。
- 热点细分：细粒度/自适应专家选择（[XMoE, ACL 2024](https://aclanthology.org/2024.findings-acl.694/)）、密集→动态-k 专家转换（[NeurIPS 2024](https://mlanthology.org/neurips/2024/szatkowski2024neurips-exploiting/)）、以及"MoE 推理的双重惩罚"（[arXiv:2603.08960](https://ar5iv.labs.arxiv.org/html/2603.08960)）这类对"专家越多越快吗"的反思。

---

## 3. 论文清单（名称 + 年份 + venue + 一句话核心）

| 论文 | 年份/Venue | 一句话核心 |
|---|---|---|
| SparseGPT | 2023 / ICML | 一次性 50–60% 稀疏化百亿参数模型，支持 2:4/4:8，算法可行但速度靠硬件 |
| Wanda | 2024 / ICLR | `|W|·‖X‖₂` 免更新剪枝，1 分钟级、最常用基线 |
| LLM-Pruner | 2023 / NeurIPS | 依赖图分组结构化剪枝 + LoRA 恢复，输出真能提速的稠密小模型 |
| ShortGPT | 2024 / arXiv | 层冗余被低估，可删 ~25% 层，BI 指标指导深度剪枝 |
| Sheared LLaMA | 2024 / ICLR | 结构化剪枝 + 继续预训练 = 低成本强小模型 |
| SparseLLM | 2024 / NeurIPS | 全局剪枝目标分解为逐层子问题 + 交替优化，超局部贪婪法 |
| Compressing LLMs | 2024 / ICLR | 系统论证"稀疏加速依赖 2:4 硬件、量化则通用"，解释稀疏落地难 |
| Deja Vu | 2023 / ICML | predictor+verifier 实现输入相关上下文稀疏，~2× 推理加速 |
| PowerInfer | 2024 / SOSP | 热/冷神经元分置 GPU/CPU，消费级单卡跑大模型 |
| ProSparse / PowerInfer-2 | 2024 / arXiv | 训出内在高激活稀疏模型，让 GPU 也吃到激活稀疏收益 |
| TEAL | 2025 / ICLR | 免训练逐层 magnitude 剪激活，40–50% 稀疏不掉点 |
| MInference | 2024 / arXiv | 免训练三种动态稀疏模式，预填充 10× 加速、1M token |
| DeepSeek NSA | 2025 / arXiv | 硬件对齐、原生可训练稀疏注意力，长序列 11.6× 加速 |
| MoBA | 2025 / NeurIPS | MoE 式门控块注意力，长上下文可扩展 |
| Switch Transformer | 2021 / JMLR | MoE 条件计算稀疏的开山之作（top-k 专家路由） |

---

## 4. 开源项目盘点（名称 + star 量级 + 活跃度 + 维护方）

> star 为 2026-08 实时抓取 GitHub API 的数值；PowerInfer/DeepSpeed 因限流未取到，标注为估算量级。

| 项目 | star 量级 | 活跃度 | 维护方 / 状态 |
|---|---|---|---|
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | ~89k | 极活跃 | vLLM 社区；稀疏支持（Marlin/2:4）持续合入 |
| [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) | ~3.7k | 活跃 | Neural Magic→vLLM；量化+剪枝统一入口 |
| [neuralmagic/deepsparse](https://github.com/neuralmagic/deepsparse) | ~3.2k | **已 archived** | Neural Magic（现 Red Hat）弃维护 |
| [neuralmagic/sparseml](https://github.com/neuralmagic/sparseml) | ~2.1k | **已 archived** | Neural Magic 弃维护 |
| [mit-han-lab/torchsparse](https://github.com/mit-han-lab/torchsparse) | ~1.5k | 低-中 | MIT Han Lab；3D 稀疏卷积/点云 |
| [microsoft/MInference](https://github.com/microsoft/MInference) | ~1.2k | 中 | 微软；长上下文动态稀疏 |
| [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) | ~1.1k | 中 | IST-DASLab；INT4 kernel，已入 vLLM |
| [horseee/LLM-Pruner](https://github.com/horseee/LLM-Pruner) | ~1.1k | 低 | 作者维护，研究用 |
| [IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt) | ~0.9k | 低 | IST-DASLab；研究用 |
| [princeton-nlp/LLM-Shearing](https://github.com/princeton-nlp/LLM-Shearing) | ~0.6k | 低 | Princeton NLP |
| [IST-DASLab/Sparse-Marlin](https://github.com/IST-DASLab/Sparse-Marlin) | ~0.1k | 低 | 稀疏+量化 kernel，实验级 |
| [SG-FlashAttention/PowerInfer](https://github.com/SG-FlashAttention/PowerInfer) | ~7k（估） | 中 | SJTU IPADS |

**关键信号**：Neural Magic 的独立稀疏栈（SparseML/DeepSparse）**双双 archived**，资源全部并入 llm-compressor + vLLM；nm-vllm（Neural Magic 的 vLLM 分支）也随收购并入主线。**独立的"稀疏 serving"栈没有活下来，稀疏作为特性活进了 vLLM/SGLang。**

---

## 5. 公司落地

- **Neural Magic → Red Hat**：早期最激进的"无 GPU 稀疏推理"公司，SparseML（训练侧稀疏）/DeepSparse（CPU 稀疏 serving）曾量产；2024 年被 Red Hat 收购后转向 llm-compressor（并入 vllm-project），原产品 archived。Red Hat 在 HF 发布 `2of4` 稀疏 + W4A16 量化模型（如 [Sparse-Llama-3.1-8B](https://huggingface.co/RedHatAI/Sparse-Llama-3.1-8B-evolcodealpaca-2of4-quantized.w4a16)），走"稀疏+量化联合"路线。
- **NVIDIA**：硬件侧唯一成体系的稀疏玩家——Ampere/Blackwell 2:4 tensor core + cuSPARSELt + [TensorRT Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)（sparsity + structured pruning 一体化）。稀疏是 Blackwell 卖点之一，但生态仍绑定 CUDA。
- **DeepSeek**：从 NSA 论文到 DeepSeek-V3.2 的 DSA 量产，稀疏注意力**真正进了 API 服务**，成本降 ~50%，并推动 SGLang 等框架 Day-0 支持——是"稀疏注意力标准化"的最大推手。
- **Moonshot AI**：MoBA 用于 Kimi 长上下文，稀疏注意力落地在自家产品线。
- **微软**：MInference 开源 + 论文，主打长上下文预填充加速（免训练、即插即用）。
- **SJTU IPADS**：PowerInfer/ProSparse 面向消费级硬件，学术开源路线。

---

## 6. 趋势判断

1. **重心从"权重稀疏"移到"注意力/激活稀疏"**。权重剪枝受 2:4 硬件锁死，而稀疏注意力（NSA/DSA/MoBA）是块状、算法与硬件天然对齐、且长上下文需求真实存在——这是唯一已经量产、有明确成本收益的稀疏线。
2. **稀疏从"后处理压缩"变成"架构原语"**。NSA/MoBA/DSA 都是"原生可训练"稀疏，不再是对稠密模型打补丁；稀疏 = 条件计算，与 MoE 合流成统一的"按需计算"范式。
3. **稀疏+量化联合压缩**成为权重线最后的现实出路（2:4 + INT4/FP8 同 kernel，Sparse-Marlin 方向），单一稀疏难存活。
4. **生态在收敛到 vLLM/SGLang**：独立稀疏栈（DeepSparse）已死，稀疏以"kernel + 格式 + 配置"的形式寄生在主流 serving 框架里。
5. **硬件碎片化短期无解**：2:4 是 NVIDIA 专有，AMD/昇腾/TPU/ASIC 各有偏好，稀疏模型跨硬件不可移植，抑制了标准化。

---

## 7. 已饱和点（不建议再卷）

- **非结构化权重量化后的"精度可达性"研究**：SparseGPT/Wanda 之后的局部 vs 全局、准则微调，精度故事已讲完，边际收益小。
- **纯 paper 的"剪 X% 不掉点"**：只报 perplexity/榜单分数、不报端到端 latency 的工作，社区已审美疲劳（甚至出现 [The Benchmark Illusion](https://arxiv.org/html/2606.17609) 这类对"剪枝模型会做选择题但不会答问答题"的批评）。
- **静态权重 2:4 的算法层**：算法不是瓶颈，内核/硬件/生态才是；再做"更好的 2:4 剪枝"无意义。
- **激活稀疏的"训练出稀疏"路线**（ProSparse 类）：依赖特定激活函数改造模型，与主流模型生态割裂，天花板低。

---

## 8. 被忽视的空白与机会

1. **稀疏注意力内核没有标准化的开源实现**。NSA/DSA/MoBA 各有各的 kernel，FlashAttention 官方对 block/top-k 稀疏支持长期缺位（block-sparse 版半弃维护），vLLM/SGLang 各自打补丁。业界缺一个"**稀疏注意力的 FlashAttention**"——统一 API、覆盖 block-sparse / top-k / 滑窗 / MoBA 门控、多后端（CUDA/Triton/ROCm）的库。
2. **"稀疏感知 serving"空白**。稀疏模型（尤其 2:4 和动态稀疏）在批处理、调度、显存布局上与 dense 模型不同，现有 vLLM/SGLang 是"能跑"而非"跑得好"；缺稀疏感知的 batch packing、负载均衡、预取/淘汰策略。
3. **结构化剪枝（heads/layers）缺自动流水线**。LLM-Pruner/ShortGPT/Sheared LLaMA 各是半成品，缺一条"profile → 打分选层选 head → 剪 → 蒸馏/续训恢复 → 校准 → 真机速度验证"的一体化、可复现工具链（尤其缺"剪完能不能真的快"的自动闭环）。
4. **稀疏+量化联合缺统一工具**。2:4 + W4A16 有 Sparse-Marlin 但近乎无人维护；缺把"联合搜索稀疏掩码与量化位宽 + 误差反馈 + 一键导出 vLLM/SGLang 可加载格式"的产品级工具。
5. **稀疏 MoE 路由优化**。专家冗余、路由开销、跨设备专家调度（"MoE 双重惩罚"）都是真实成本，缺面向推理的"稀疏路由 + 专家剪枝/offload/合并"联合优化。

---

## 9. 具体候选切入点（3–5 个）

### 切入点 A：统一的稀疏注意力内核库（"Sparse-Attention-Kernels"）
**做什么**：做一个 FlashAttention 风格的、覆盖 block-sparse / top-k（NSA/DSA）/ 滑窗 / MoBA 门控的稀疏注意力内核库，统一 Python API + CUDA/Triton 双后端，直接接 PyTorch、并出 vLLM/SGLang 插件。
**为什么有收益**：这是当前最明确、最缺、且有真实速度收益（长上下文预填充 3–10×）的空白；NSA/DSA 已验证算法收益，缺的是标准化内核。
**难度**：中高（内核 + 多模式抽象 + 生态适配）。

### 切入点 B：稀疏感知 serving 调度层
**做什么**：在 vLLM/SGLang 之上做稀疏模型的调度/布局优化——2:4 权重的显存紧凑布局与预取、动态激活稀疏的按层跳过路由、稀疏注意力与 prefill/decode 分段的负载均衡，输出一个可量化的 end-to-end 加速比。
**为什么有收益**：稀疏模型"能加载"≠"能快"，调度层是放大稀疏收益的杠杆，也是大厂 serving 团队会买的方向。
**难度**：中（贴近 vLLM 源码，需真实 A100/H100 验证）。

### 切入点 C：结构化剪枝自动化流水线（"AutoPruner"）
**做什么**：把 heads/channels/layers 剪枝做成一条可复现流水线：依赖图 → 重要性打分（梯度/BI 等）→ 硬件感知选择 → 蒸馏或续训恢复 → 输出稠密小模型，并**自动跑真机 benchmark 形成 quality-latency Pareto 前沿**。
**为什么有收益**：结构化剪枝是唯一"不依赖专用硬件、剪完就是稠密模型"的路径，但现有工具是散的；给企业"从 70B 到 35B 且真快"的一键方案。
**难度**：中（工程整合 + 恢复训练算力）。

### 切入点 D：稀疏+量化联合压缩工具链
**做什么**：基于 llm-compressor 生态做"2:4 稀疏掩码 + W4A16 位宽"联合搜索（误差反馈、逐层敏感度），一键导出到 Sparse-Marlin 类 kernel + vLLM 加载，并自动报告相对 dense-INT4 的额外加速。
**为什么有收益**：权重稀疏单独难落地，但与量化联合是唯一能兑现"硬件 2× + 访存减半"的权重线，且 Red Hat/NVIDIA 已在推。
**难度**：中（内核依赖 + 联合优化算法）。

### 切入点 E：稀疏推理基准（"SparseBench"）
**做什么**：一个公开、可复现的稀疏推理基准——统一测"稀疏化后端到端 latency/吞吐 vs 同参数量 dense 基线"，覆盖 2:4/块稀疏/稀疏注意力/稀疏 MoE × vLLM/SGLang × 多硬件，破除"只报稀疏度不报速度"的营销话术。
**为什么有收益**：行业最大的痛点是"稀疏收益不可信/不可比"，基准本身有社区影响力和标准化价值，成本低、启动快。
**难度**：低-中（基准工程，可先做 2:4 与稀疏注意力子集）。

---

## 附：关键出处速查（URL）

- SparseGPT: https://arxiv.org/abs/2301.00774
- Wanda: https://arxiv.org/abs/2306.11695
- LLM-Pruner: https://arxiv.org/abs/2305.11627
- ShortGPT: https://arxiv.org/abs/2403.03853
- Sheared LLaMA: https://arxiv.org/abs/2310.06694
- SparseLLM: https://neurips.cc/virtual/2024/poster/93617
- MInference: https://arxiv.org/abs/2407.02490
- DeepSeek NSA: https://arxiv.org/abs/2502.11089
- MoBA: https://arxiv.org/abs/2502.13189
- PowerInfer: https://arxiv.org/abs/2312.12456
- Deja Vu: https://arxiv.org/abs/2310.17157
- TEAL: https://arxiv.org/abs/2408.14690
- cuSPARSELt / Ampere 2:4: https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/
- Marlin: https://github.com/IST-DASLab/marlin
- Sparse-Marlin: https://github.com/IST-DASLab/Sparse-Marlin
- llm-compressor: https://github.com/vllm-project/llm-compressor
- SparseML: https://github.com/neuralmagic/sparseml
- DeepSparse: https://github.com/neuralmagic/deepsparse
- TorchSparse: https://github.com/mit-han-lab/torchsparse
- NVIDIA Model Optimizer: https://github.com/NVIDIA/Model-Optimizer
- SGLang × DeepSeek-V3.2 稀疏注意力: https://lmsys.org/blog/2025-09-29-deepseek-V32/
- DeepSeek 稀疏注意力降价报道: https://yourstory.com/ai-story/deepseek-sparse-attention-api-price-cut-ai
- SparseGPT speedup issue: https://github.com/IST-DASLab/sparsegpt/issues/15
