# 大模型推理量化（Quantization）深度调研

> 聚焦推理侧（Inference-side）。调研时间：2025–2026。所有关键事实均附出处 URL；GitHub star 数据来自 GitHub API（2026-08 抓取），标注为"约"。

---

## 1. 现状总览（为什么是量化 + 当前卡点）

### 1.1 这个技术解决什么瓶颈

大模型推理的瓶颈已经从"算力（FLOPs）"转向"内存带宽（memory bandwidth）"和"显存容量（memory capacity）"。在自回归解码（decode）阶段，每生成一个 token 都要把**全部权重**从显存/HBM 里读一遍，却只做一次矩阵乘——这是典型的 memory-bound 场景。量化通过把权重/激活/KV cache 的位宽从 FP16 降到 INT8/FP8/INT4/FP4 甚至 2-bit，直接压缩三样东西：

1. **显存占用**：一个 70B 模型 FP16 约 140GB，INT4 约 35GB，一块消费级/单卡 GPU 即可装下；
2. **内存带宽压力**：位宽减半 → 同样带宽下吞吐近似翻倍；
3. **KV cache 容量**：长上下文场景中 KV cache 往往比权重还大，量化 KV 是解锁超长上下文的关键。

因此量化是"用可接受的一点精度损失，换取 2–4 倍的吞吐提升和 2–4 倍的显存节省"，是目前性价比最高、落地最广的推理优化手段之一。

### 1.2 核心矛盾与当前卡点

量化的本质矛盾是：**LLM 权重和激活中存在少量"离群值"（outlier）**，它们决定了量化范围，导致大多数数值被压到极低精度后噪声巨大。几乎所有主流方法（SmoothQuant 的平滑迁移、AWQ 的激活感知缩放、QuaRot/SpinQuant 的旋转）都是在"想办法消除或绕过离群值"。

当前公认的成熟地带与卡点边界：

- ✅ **成熟/已饱和**：权重 INT4（W4A16，GPTQ/AWQ/GGUF Q4 一系）、权重+激活 FP8（W8A8，H100/Blackwell 原生支持）——精度几乎无损、工具链成熟、已大规模生产。
- ⚠️ **半成熟**：KV cache 4-bit（基本可用）、W4A8/W4A4（有系统级方案但精度与算子未完全统一）。
- ❌ **痛点/空白**：亚 4-bit（2-bit/3-bit）精度与算子、MoE 量化、超长上下文 KV 量化、量化模型的训练/微调（QAT/QFT）、跨硬件量化格式统一、量化精度回归测试基建。

---

## 2. 关键技术（带出处 URL）

### 2.1 PTQ 权重量化（Weight-only / W4A16）

- **GPTQ** — 逐层二阶误差校正（基于 Hessian 的 Optimal Brain Surgeon 近似），在 128 列块内顺序量化并补偿剩余权重的误差。3–4 bit 下几乎无损，是权重量化的奠基方法。
  - 论文：https://arxiv.org/abs/2210.17323
- **AWQ（Activation-aware Weight Quantization）** — 观察"只有 ~1% 的显著权重通道重要"，用激活幅值做 per-channel 缩放来保护这些通道，无需反向传播即可确定缩放。MLSys 2024 Best Paper，4-bit 权重量化精度与速度兼备。
  - 论文：https://arxiv.org/abs/2306.00978
- **OmniQuant** — 同时学习权重/激活的裁剪阈值和"等效变换"，兼容 LLaMA 结构，支持 W4A4/W3A4 等更激进配置；one-shot、无需训练数据之外的重训。
  - 论文：https://arxiv.org/abs/2308.13137
- **SpQR** — 把离群权重单独用稀疏高精度存储、其余量化，3–4 bit 近无损；但稀疏混合格式导致实际推理加速困难，主要价值在压缩率而非吞吐。
  - 论文：https://arxiv.org/abs/2306.03078

### 2.2 KV cache 量化

- **SmoothQuant** — 最早系统解决"激活离群值"的方法：通过 per-channel 缩放把激活的离群幅度"迁移"到权重，实现 W8A8 且训练后可用；它的思想被 KV 量化和 W8A8 广泛继承。
  - 论文：https://arxiv.org/abs/2211.10438
- **KVQuant** — 针对 KV cache：Key 用 per-channel + 非均匀量化（non-uniform），Value 用 per-token，并显式处理 attention sink（前几个 token），可把 KV 压到 1–2 bit 并支撑千万级上下文。
  - 论文：https://arxiv.org/abs/2401.18079 ；仓库：https://github.com/SqueezeAILab/KVQuant
- **KIVI** — 免调参（tuning-free）非对称 2-bit KV 量化：Key per-channel、Value per-group，流式/按 token 更新，无需校准数据，适合增量 decode。
  - 论文：https://arxiv.org/abs/2402.02750 ；仓库：https://github.com/jy-yuan/KIVI

### 2.3 激活量化 FP8 / INT8

- **FP8（E4M3/E5M2）W8A8** 已成为数据中心 GPU 推理的"甜点位"：Hopper（H100）起通过 Transformer Engine 原生支持 FP8 张量核心，精度损失远小于 INT8（动态范围更宽，无需复杂裁剪）。
  - NVIDIA Hopper FP8：https://theneuralbase.com/quantization-fundamentals/learn/beginner/nvidia-hopper-fp8-native-support/
  - vLLM FP8 W8A8 文档：https://docs.vllm.ai/en/latest/features/quantization/fp8.html
- **INT8 激活**（W8A8）在 CPU/老 GPU 上仍是主力，但需要 SmoothQuant 式平滑 + 动态范围校准，工程更重，精度上限低于 FP8。
- **动态激活量化（dynamic activation quantization）** 仍是痛点：激活分布随输入/层变化大，静态量化误差不稳定；主流要么用 FP8 规避，要么做 per-tensor/per-token 动态缩放。

### 2.4 极端低位 FP4 / MX 格式（Microscaling）

- **OCP Microscaling Formats (MX) v1.0** 是 Open Compute Project 制定的开放标准，定义了带 block-scaled 的低精度格式：MXFP4（E2M1）、MXFP6（E3M2）、MXFP8（E4M3）、MXINT8；用 32 元素共享一个 scale，兼顾精度与存储。
  - 规范：https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- **NVIDIA Blackwell FP4（NVFP4）**：第五代 Tensor Core 新增 FP4 张量核心，FP4 吞吐约为 FP8 的 **2 倍**；Transformer Engine 2 支持在线 block-scale，NVFP4 与 OCP MXFP4 兼容。Blackwell 是 FP4 从研究走向量产的关键硬件。
  - NVIDIA Blackwell 架构介绍（含 FP4 / Transformer Engine 2）：https://gigagpu.com/nvidia-blackwell-architecture-ai-overview/
  - Fireworks FP4 落地报告（FireAttention V4，B200 上 >250 tokens/s）：https://fireworks.ai/blog/fireattention-v4-fp4-b200
- 训练侧已出现 MicroMix 等混合精度 MX 格式研究（同时兼顾 Blackwell FP4 张量核心利用率）。

### 2.5 亚 4-bit：QuaRot / AQLM / QuIP#（2-bit 路线）

- **QuaRot** — 用随机 Hadamard 旋转把权重/激活的离群值"摊平"，实现真正的端到端 4-bit（含激活和 KV），并把离群消除技术推广到多结构。NeurIPS 2024。
  - 论文：https://arxiv.org/abs/2404.00456 ；仓库：https://github.com/spcl/QuaRot
- **AQLM（Additive Quantization）** — 把每组权重分解为多个码本（codebook）之和（多码本加性量化），2–2.5 bit 达到接近 3–4 bit 的精度，但推理算子复杂、依赖特定硬件/反量化开销。
  - 论文：https://arxiv.org/abs/2401.06118 ；仓库：https://github.com/Vahe1994/AQLM
- **QuIP#** — 用随机正交变换做 incoherence 处理 + E8 格码本，2-bit 下精度领先；与 AQLM 同为"极致压缩"研究代表。
  - 论文：https://arxiv.org/abs/2402.04396 ；仓库：https://github.com/Cornell-RelaxML/quip-sharp
- **SpinQuant** — 学习旋转矩阵（而非随机 Hadamard）以最小化离群值，可迁移到多模态/非 LLaMA 结构，被多家厂商吸收进 PTQ 工具链。
  - 论文：https://arxiv.org/abs/2405.16406
- **BitNet b1.58** — 权重三值化 {-1, 0, 1}（1.58 bit）的极端路线，1-bit LLM 在吞吐/能耗上潜力巨大，但需要"从预训练就按 1-bit 训练"，与现有 FP16 预训练模型不兼容。
  - 论文：https://arxiv.org/abs/2402.17764

### 2.6 系统级 co-design（量化 + 算子 + 调度）

- **QServe（W4A8KV4）** — 首次证明"亚 4-bit 权重 + 8-bit 激活 + 4-bit KV"在 GPU 上能真正跑赢 4-bit 系统，给出系统化 co-design 的标杆。MLSys 2025。
  - 论文：https://arxiv.org/abs/2405.04532

### 2.7 校准数据敏感性与 accuracy 评估

- **校准数据（calibration set）是 PTQ 的隐性变量**：AWQ/GPTQ/OmniQuant 的缩放或 Hessian 都依赖一小段校准语料（通常 128 个样本）。校准语料与推理分布不匹配会导致过拟合式"校准内好、泛化差"；领域专用校准（代码/多语言/长文本）是实践中的常见坑。
  - llama.cpp 2-bit 讨论中明确指出校准数据集对领域模型的影响：https://github.com/ggml-org/llama.cpp/pull/4897
- **评估标准事实是 EleutherAI 的 lm-eval-harness**（~13.6k stars）：量化后模型普遍用 wikitext perplexity + 零样本/少样本任务（HellaSwag/ARC/MMLU 等）做回归对比。但它本身是"模型评测"框架，不是"量化专用回归测试"框架。
  - 仓库：https://github.com/EleutherAI/lm-evaluation-harness
  - 精度对比示例（FP8 vs INT8）：https://theneuralbase.com/quantization-fundamentals/learn/intermediate/fp8-vs-int8-quality-comparison/

### 2.8 硬件支持一览

| 硬件 | 原生低精度 | 说明 |
|---|---|---|
| NVIDIA A100 (Ampere) | INT8/INT4 Tensor Core | 无 FP8，INT8 是主力 |
| NVIDIA H100 (Hopper) | **FP8**（Transformer Engine, E4M3/E5M2） | FP8 推理成为生产标配 |
| NVIDIA B200/GB200 (Blackwell) | **FP4 (NVFP4)** + FP8 + FP6 | FP4 吞吐约 2×FP8，Transformer Engine 2 |
| AMD MI300X/MI350 | FP8 / MXFP4（MI350 起支持 MX） | 生态晚于 NVIDIA |
| 消费级 GPU / Apple Silicon | INT8/INT4/INT2（无 FP8 Tensor Core 加速，走 GGUF 路线） | llama.cpp 主战场 |

---

## 3. 论文清单（名称 + 年份 + venue + 一句话核心）

| 论文 | 年份 | Venue | 一句话核心 |
|---|---|---|---|
| GPTQ（Frantar et al.） | 2022/2023 | ICLR 2023 | 逐层二阶误差校正，奠定权重量化 W3/W4 基准 |
| SmoothQuant（Xiao et al.） | 2022/2023 | ICML 2023 | 把激活离群值"平滑"迁移到权重，实现 W8A8 |
| AWQ（Lin et al.） | 2023/2024 | MLSys 2024 (Best Paper) | 激活感知 per-channel 缩放，保护显著通道 |
| SpQR（Dettmers et al.） | 2023/2024 | ICLR 2024 | 稀疏+量化混合，近无损但难加速 |
| OmniQuant（Shao et al.） | 2023/2024 | ICLR 2024 | 可学习裁剪+等效变换，W4A4 免重训 |
| KVQuant（Hooper et al.） | 2024 | NeurIPS 2024 | per-channel K + 非均匀量化 + attention sink，1-bit KV |
| KIVI（Liu et al.） | 2024 | ICML 2024 | 免调参非对称 2-bit KV，流式更新 |
| QuaRot（Ashkboos et al.） | 2024 | NeurIPS 2024 | 随机 Hadamard 旋转消除离群，端到端 4-bit |
| AQLM（Egiazarian et al.） | 2024 | ICML 2024 | 多码本加性量化，2–2.5 bit |
| QuIP#（Tseng et al.） | 2024 | arXiv | incoherence + E8 格码本，2-bit 精度领先 |
| SpinQuant（Liu et al.） | 2024 | ICLR 2025 | 学习旋转替代随机旋转，通用性更强 |
| QServe（Lin et al.） | 2024/2025 | MLSys 2025 | W4A8KV4 系统 co-design，亚 4-bit 真提速 |
| BitNet b1.58（Ma et al.） | 2024 | arXiv | 1.58-bit 三值权重，需从头训练 |
| LLM-QAT（Liu et al.） | 2023 | arXiv | 数据自由蒸馏做 QAT，量化后精度回填 |
| L4Q（Jeon et al.） | 2024 | arXiv | 参数高效量化感知微调（QFT） |

---

## 4. 开源项目盘点（名称 + star 量级 + 活跃度 + 维护方）

| 项目 | Star（约） | 活跃度 | 维护方/定位 |
|---|---|---|---|
| [llama.cpp (GGUF)](https://github.com/ggml-org/llama.cpp) | ~12.4 万 | 极活跃 | ggml-org（Georgi Gerganov），C/C++ 本地推理；GGUF 的 Q2_K–Q8_0、i-quants（IQ2_XXS 等）成为事实标准 |
| [vLLM](https://github.com/vllm-project/vllm) | ~8.9 万 | 极活跃 | vLLM 社区（UC Berkeley 起源），支持 FP8/W8A8、AWQ/GPTQ/GGUF、MXFP4/NVFP4 内核 |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | ~1.4 万 | 活跃 | EleutherAI，量化精度回归的事实评估工具 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | ~1.4 万 | 活跃 | NVIDIA，FP8/FP4/INT4，Blackwell 最优路径 |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | ~8400 | 活跃 | bitsandbytes-foundation，NF4+双重量化、QLoRA、8-bit |
| [LMDeploy](https://github.com/InternLM/lmdeploy) | ~8000 | 活跃 | 上海 AI Lab/InternLM，4-bit AWQ/GPTQ + KV 量化 + 服务化 |
| [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) | ~5100 | **已归档** | GPTQ 的 HF 集成，事实停更（功能并入 vLLM/transformers） |
| [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) | ~2300 | **已归档** | AWQ 集成，事实停更 |
| [AQLM](https://github.com/Vahe1994/AQLM) | ~1300 | 维护中 | Yandex 系，2-bit 参考实现 |
| [QuaRot](https://github.com/spcl/QuaRot) | ~530 | 维护中 | ETH SPCL，旋转量化参考实现 |
| [KVQuant](https://github.com/SqueezeAILab/KVQuant) | ~430 | 维护中 | Berkeley SqueezeAILab，KV 量化参考实现 |

> 观察：权重量化的"格式层"（GGUF k-quants）和"引擎层"（vLLM/llama.cpp/TensorRT-LLM）高度活跃且已收敛；而 AutoGPTQ/AutoAWQ 的归档标志着"独立量化转换工具"正在被并入推理引擎本身（vLLM 内建 GPTQ/AWQ/FP8/MX 支持），这是重要的生态信号。

---

## 5. 公司落地

- **NVIDIA**：Hopper 用 FP8、Blackwell 主推 FP4（NVFP4）+ Transformer Engine 2；官方博客给出 DeepSeek R1 在 B200 上的 FP4/FP8 吞吐优化实践，TensorRT-LLM + TensorRT Model Optimizer 是官方量化管线。
  - https://nvidia.github.io/TensorRT-LLM/1.3.0rc21/blogs/tech_blog/blog03_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.html
  - https://developer.nvidia.com/blog/model-quantization-post-training-quantization-using-nvidia-model-optimizer/
- **Fireworks AI**：FireAttention V4 在 B200 上用 FP4 达到 >250 tokens/s，宣称行业领先的延迟/成本，是 FP4 生产化的典型第三方。
  - https://fireworks.ai/blog/fireattention-v4-fp4-b200
- **DeepSeek**：V3/R1 采用 FP8 训练（业界首次大规模 FP8 MoE 训练），推理侧配套 FP8/W8A8 部署，推动 FP8 成为共识。
- **Meta / Hugging Face / Red Hat**：llama.cpp GGUF 生态 + vLLM 生产部署（Red Hat 与 vLLM 合作推出 LLM Compressor 做压缩+部署一体）。
  - https://developers.redhat.com/articles/2024/08/14/llm-compressor-here-faster-inference-vllm
- **AMD**：MI350 系列开始支持 OCP MX 格式（MXFP4/MXFP8），Quark 提供 Microscaling 量化，追赶 Blackwell FP4 生态。
  - https://quark.docs.amd.com/release-0.11/onnx/tutorial_microscaling_quantization.html

---

## 6. 趋势判断

1. **FP8 成为数据中心推理的默认"基线精度"**，FP4/MXFP4 是下一波（由 Blackwell 硬件驱动，从"能用"走向"好用"）。
2. **量化能力正在从独立工具下沉进推理引擎**：AutoGPTQ/AutoAWQ 归档、vLLM/TensorRT-LLM/llama.cpp 内建多种格式，用户不再关心"用什么工具量化"，而关心"引擎支持什么格式"。
3. **KV cache 量化从学术走向工程默认**：长上下文 + MoE 时代 KV 占显存比例上升，4-bit KV 正在成为标配，2-bit 是研究热点。
4. **旋转类（rotation）方法成为通用离群消除基座**：QuaRot/SpinQuant 的 Hadamard/学习旋转被并入各厂商 PTQ 管线，从 LLaMA 扩展到多模态。
5. **系统 co-design 取代"单点量化"**：量化位宽选择必须和算子内核、批大小、KV 管理、硬件代数联合优化（QServe/Blackwell 是标志）。
6. **量化训练（QAT/QFT）与 PTQ 融合**：纯 PTQ 在亚 4-bit 上到瓶颈，学界转向低成本 QAT/量化后微调（L4Q、EfficientQAT）回收精度。

---

## 7. 已饱和点（成熟/不建议作为新切入点）

- **常见权重的 INT4（W4A16）**：GPTQ/AWQ/GGUF Q4_K 三足鼎立，工具、模型库、算子全面成熟，精度接近无损。
- **W8A8 FP8 推理**：硬件原生（H100/Blackwell）、vLLM/TensorRT-LLM 开箱即用，DeepSeek 等已大规模生产，格局定型。
- **4-bit KV cache**：vLLM/LMDeploy 等已工程化，学术增量有限。
- **单格式权重量化转换工具**（独立 AutoGPTQ/AutoAWQ 一类的"转模型"工具）：已被引擎内建能力取代，AutoGPTQ/AutoAWQ 归档即为信号。
- **标准 LLaMA 系稠密模型的 3–4 bit PTQ 算法本身**：GPTQ/AWQ/OmniQuant/SpQR 之后，稠密权重 3–4 bit 精度改进边际收益极低。

---

## 8. 被忽视的空白与机会

1. **MoE 量化**：expert 数量多、每个 expert 激活稀疏、校准数据难以覆盖所有 expert，GPTQ 系方法在 MoE 上明显退化；现有方案多为"逐 expert 独立量化"的权宜之计，缺路由感知（routing-aware）与共享专家感知的专用量化。
2. **超长上下文 KV 量化**：KVQuant/KIVI 解决了 4→2-bit 的静态精度，但动态增量、多轮对话、RAG 检索场景下 KV 分布漂移导致精度回归，attention sink 处理仍是启发式。
3. **亚 4-bit（2-bit/3-bit）的"能用"工程**：AQLM/QuIP# 精度够但算子慢/反量化开销大；缺同时满足"精度 + 低反量化开销 + 通用结构"的 2-bit 生产方案。
4. **动态激活量化**：静态激活量化在分布漂移（长尾输入、多模态、长上下文）下不稳；FP8 是"绕开"而非"解决"。
5. **量化模型的训练/微调（QAT/QFT）**：QLoRA 只覆盖 NF4 权重；针对 INT4/FP8/FP4/MX 全格式的、参数高效、低成本的量化后微调是空白；QAT 计算成本高、工具链割裂。
6. **跨硬件量化格式统一**：GGUF（本地 CPU/GPU）与 GPTQ/MXFP4/NVFP4（数据中心 GPU）互不兼容，模型发布方需为每个后端重新量化/打包，缺一个"格式无关"的中间表示与转换标准。
7. **量化精度回归测试基建**：现状是研究者各自跑 lm-eval 的 perplexity，无统一的"量化专用评测套件"（校准敏感性、分布漂移、长尾任务、多语言、代码）与 CI 门禁；量化引入的隐性精度回归常在生产才暴露。

---

## 9. 具体候选切入点（3–5 个，被忽视但可做）

1. **MoE 的"路由/共享专家感知"量化工具与评测**：设计针对 MoE 的校准集构造（按 expert 激活分布采样）+ 逐 expert 位宽自动分配（热门 expert 高精度、冷门 expert 低精度），并输出一个 MoE 量化精度/吞吐基准套件。现状：各框架逐 expert 独立量化，无公开的 expert 级感知与基准。
   - 佐证（GPTQ 在 MoE 上为何更难）：https://theneuralbase.com/mixture-of-experts/learn/advanced/gptq-on-moe/

2. **量化模型的参数高效微调/精度回收框架（QFT，跨格式）**：把 L4Q/EfficientQAT 思路产品化——对已 INT4/FP8/FP4 量化的模型做 LoRA 式量化感知微调来回收亚 4-bit 的精度损失，覆盖 GGUF/GPTQ/MX 多格式，而非只支持 NF4。这是 PTQ 到瓶颈后精度与位宽的"最后 20%"。

3. **量化精度回归测试服务（Quant-Eval CI）**：基于 lm-eval-harness 封装一套量化专用评测：自动探测校准数据敏感性、长尾/多语言/代码任务上的精度漂移、以及"量化前后逐层/逐 task 回归门禁"，作为 CI 插件。解决"量化隐性精度回归在生产才暴露"的空白。

4. **跨硬件量化格式统一与转换（Quant-IR）**：定义一个格式无关的量化中间表示（记录每层的量化方案、scale/zero-point/分组、校准元数据），提供 GGUF↔GPTQ↔MXFP4↔EXL2 的可逆转换与无损重校准，让模型发布方一次量化、多后端发布。直接踩中生态割裂痛点。

5. **超长上下文 KV 量化的动态/自适应位宽**：针对多轮对话与 RAG 的 KV 分布漂移，做"重要 token（attention sink、检索命中的 KV）高精度 + 其余低精度"的自适应混合位宽，并做成 vLLM/LMDeploy 可插拔算子。补齐 KVQuant/KIVI 静态方案在动态场景的空白。

---

## 参考链接汇总（Citations）

- GPTQ: https://arxiv.org/abs/2210.17323
- AWQ: https://arxiv.org/abs/2306.00978
- OmniQuant: https://arxiv.org/abs/2308.13137
- SpQR: https://arxiv.org/abs/2306.03078
- SmoothQuant: https://arxiv.org/abs/2211.10438
- KVQuant: https://arxiv.org/abs/2401.18079
- KIVI: https://arxiv.org/abs/2402.02750
- QuaRot: https://arxiv.org/abs/2404.00456
- AQLM: https://arxiv.org/abs/2401.06118
- QuIP#: https://arxiv.org/abs/2402.04396
- SpinQuant: https://arxiv.org/abs/2405.16406
- QServe: https://arxiv.org/abs/2405.04532
- BitNet b1.58: https://arxiv.org/abs/2402.17764
- LLM-QAT: https://arxiv.org/abs/2305.17888
- L4Q: https://arxiv.org/abs/2402.04902
- OCP Microscaling Formats (MX) spec: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- NVIDIA Hopper FP8: https://theneuralbase.com/quantization-fundamentals/learn/beginner/nvidia-hopper-fp8-native-support/
- FP8 vs INT8 对比: https://theneuralbase.com/quantization-fundamentals/learn/intermediate/fp8-vs-int8-quality-comparison/
- NVIDIA Blackwell 架构: https://gigagpu.com/nvidia-blackwell-architecture-ai-overview/
- Fireworks FireAttention V4 (FP4 B200): https://fireworks.ai/blog/fireattention-v4-fp4-b200
- NVIDIA TRT-LLM DeepSeek R1 on Blackwell: https://nvidia.github.io/TensorRT-LLM/1.3.0rc21/blogs/tech_blog/blog03_Optimizing_DeepSeek_R1_Throughput_on_NVIDIA_Blackwell_GPUs.html
- NVIDIA Model Optimizer PTQ: https://developer.nvidia.com/blog/model-quantization-post-training-quantization-using-nvidia-model-optimizer/
- vLLM FP8 W8A8: https://docs.vllm.ai/en/latest/features/quantization/fp8.html
- vLLM MXFP4/NVFP4 issue: https://github.com/vllm-project/vllm/issues/35528
- Red Hat LLM Compressor: https://developers.redhat.com/articles/2024/08/14/llm-compressor-here-faster-inference-vllm
- AMD Quark Microscaling: https://quark.docs.amd.com/release-0.11/onnx/tutorial_microscaling_quantization.html
- GPTQ on MoE: https://theneuralbase.com/mixture-of-experts/learn/advanced/gptq-on-moe/
- llama.cpp 校准讨论: https://github.com/ggml-org/llama.cpp/pull/4897
- 开源仓库: https://github.com/ggml-org/llama.cpp | https://github.com/vllm-project/vllm | https://github.com/NVIDIA/TensorRT-LLM | https://github.com/EleutherAI/lm-evaluation-harness | https://github.com/bitsandbytes-foundation/bitsandbytes | https://github.com/InternLM/lmdeploy | https://github.com/AutoGPTQ/AutoGPTQ | https://github.com/casper-hansen/AutoAWQ | https://github.com/Vahe1994/AQLM | https://github.com/spcl/QuaRot | https://github.com/SqueezeAILab/KVQuant | https://github.com/jy-yuan/KIVI
