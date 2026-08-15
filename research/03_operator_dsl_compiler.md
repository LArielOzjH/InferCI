# 算子开发 / DSL / 编译器 —— 推理侧深度调研

> 调研方向：推理（serving）侧的算子开发、领域特定语言（DSL）与编译器栈。
> 调研时间：2026-08。所有关键事实尽量附出处 URL；GitHub star 为快照数量级（`≈` 为近似值）。

---

## 1. 现状总览（为什么是这个技术，解决了什么瓶颈）

推理侧的算子/编译器栈本质上是在回答一个矛盾：**模型的数学定义是"形状/精度/结构"高度异构且快速演进的，而硬件（GPU/NPU/TPU）的峰值性能只能通过高度手写、硬件专用的 kernel 榨出来。** 编译器与 DSL 就是中间那层"抽象 + 自动/半自动映射"的胶水。

关键瓶颈的历史脉络：

1. **内存墙（memory wall）**：注意力、线性层在 decode 阶段是 memory-bound（受 KV cache 读取带宽限制），在 prefill/训练阶段是 compute-bound。FlashAttention 系列证明"IO-aware 的算子重排"比硬件算力本身更能决定端到端速度。
2. **手工 CUDA 的生产力墙**：CUTLASS/CuTe 是性能天花板，但模板元编程心智负担极高，一个 SM90 的 GEMM kernel 动辄上千行、数千个编译期参数；Triton/ThunderKittens/TileLang 想用"更高层的 tile 抽象"把生产力与性能同时拿下来。
3. **碎片化墙**：NVIDIA(CUDA/CUTLASS)、AMD(ROCm/hipBLASLt/AITER)、Intel(XPU/oneAPI)、Apple(Metal/MLX)、华为(Ascend/CANN)、高通(Hexagon) 各自一套算子库，跨厂商移植靠人肉重写。MLIR/TVM/Triton 后端都在试图统一这一层。
4. **serving 侧的特殊性**：推理不是"一次编译跑到底"，而是**长尾形状 + 动态批 + KV cache + 低延迟**的实时场景。这带来训练没有的痛点：JIT 编译/自动调优的实时开销、shape 爆炸导致的重复编译、内存受限下的调度选择（split-K、persistent、paged attention）。

一句话总结现状：**NVIDIA 侧的 GEMM 与注意力 kernel 已被 CUTLASS/CuTe + FlashAttention/FlashInfer 基本"吃透"，竞争焦点正在从"单卡单 kernel 更快"转向 (a) 跨厂商可移植、(b) 编译/调优的实时开销、(c) 新结构算子（MLA/稀疏/MoE/低精度）的快速落地。**

---

## 2. 关键技术（带出处）

### 2.1 Triton 及其局限
Triton 用 "block/tile 级编程 + 类 C 的 pointer/`tl.load`/`tl.store` 抽象"替代 CUDA 线程级编程，让算子以 Python 写、编译到各后端（NVIDIA/AMD/Intel/Qualcomm/Ascend 等有社区或官方后端）。它是 **PyTorch Inductor 的默认 GPU 后端**，也是 vLLM/SGLang 大量 attention/MoE 内核的载体。
- 出处：[triton-lang/triton](https://github.com/triton-lang/triton)（≈20k stars）；[Triton 论文 arXiv:1801.01946](https://arxiv.org/abs/1801.01946)。

**局限（这是真正的机会来源，逐条有出处）：**
- **JIT 编译 + autotune 的实时开销**：serving 里遇到新 shape 会触发 kernel 编译/重编译，vLLM 直接有 issue 报 "Triton kernel JIT compilation during inference"（[vLLM #43009](https://github.com/vllm-project/vllm/issues/43009)），以及针对多模态 attention 区间重编译的修复 PR（[vLLM #48736](https://app.semanticdiff.com/gh/vllm-project/vllm/pull/48736/overview)）。
- **Hopper/Blackwell 新硬件特性覆盖不全**：TMA（Tensor Memory Accelerator）、persistent kernel、MXFP4 原生支持仍然"坑"多——TMA store 非确定、persistent kernel 调优死循环、MXFP4 需要 "persistent + TMA-compliant" 才能跑（[Triton #8548](https://github.com/triton-lang/triton/issues/8548)、[#6638](https://github.com/triton-lang/triton/issues/6638)、[#4332](https://github.com/triton-lang/triton/issues/4332)）。
- **抽象层不够低**：Triton 不暴露 warp specialization、异步流水、寄存器级控制，峰值性能追不上 CuTe/CUTLASS，这是 ThunderKittens/TileLang 的立身点。

### 2.2 TVM / Relax / Unity
TVM 是"计算与调度分离 + 自动调优"的老牌编译器（OSDI 2018）。**Relax/Unity** 是其面向动态 shape、控制流、可组合的图级 IR，目标是把 TVM 从"静态图+手工 schedule"升级到能承载 LLM 推理的端到端栈（与 MLIR 对齐，支持 TVMScript 脚本化算子）。
- 出处：[apache/tvm](https://github.com/apache/tvm)（≈13.7k stars）；[Relax: Composable Abstractions arXiv:2311.02903](https://arxiv.org/abs/2311.02903)；[TVM Unity 讨论区](https://discuss.tvm.apache.org/c/development/unity/14)。

**现状判断**：TVM 在 LLM 推理主战场（vLLM/SGLang/TRT-LLM）的份额被 PyTorch Inductor + Triton 明显挤压，但在**非 NVIDIA 硬件、边缘/端侧、厂商定制**场景仍有不可替代地位（如 Relax IR 与 NNAPI/Ascend 集成）。

### 2.3 MLIR / Torch-MLIR
MLIR 是编译器基础设施（dialect 化的多层 IR，[arXiv:2002.11054](https://arxiv.org/abs/2002.11054)）。Torch-MLIR 把 PyTorch 图 lower 到 Linalg/TOSA 等 dialect，再由各厂商后端接力。它是**跨硬件"共享中间表示"的事实标准底座**，但本身不产出高性能 kernel，仍依赖下游（如 IREE、厂商 codegen）。
- 出处：[llvm/torch-mlir](https://github.com/llvm/torch-mlir)（≈1.9k stars）；[TorchToLinalg 示例 PR](https://github.com/llvm/torch-mlir/pull/3732)。

### 2.4 CUTLASS / CuTe
NVIDIA 官方的 GEMM/卷积/注意力模板库。**CUTLASS 3.x 引入 CuTe（CUDA Templates）** 作为 tile/atom/layout 的编译期代数 DSL，是当前 Hopper/Blackwell 上 peak 性能的事实标准；4.x 全面支持 SM90/SM100，新增 FP4/MXFP、分组 GEMM（grouped GEMM，MoE/MLA 的核心）、稀疏 2:4。
- 出处：[NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)（≈10.3k stars）；[CUTLASS 文档](https://docs.nvidia.com/cutlass/4.5.2/)；[CuTe DSL 教程](https://deepwiki.com/NVIDIA/cutlass/4.6-cute-dsl-gemm-examples-and-tutorials)。
- **意义**：SGLang/FlashInfer 的 MLA、DeepSeek 生态内核大量用 CuTe 重写，CuTe 正在成为"高性能 attention 内核的汇编层"。

### 2.5 FlashAttention 家族
- **FA1**（NeurIPS 2022）：tiling + online softmax，避免把完整注意力矩阵写回 HBM。[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)。
- **FA2**：更好的并行与 work partition，减少非 matmul FLOP，2-4x。[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)。
- **FA3**（NeurIPS 2024）：利用 Hopper 的 TMA + warp specialization（producer/consumer 异步流水），配合 FP8 低精度，H100 上 1.5-2x。[arXiv:2407.08608](https://arxiv.org/abs/2407.08608)、[PyTorch 博客](https://pytorch.org/blog/flashattention-3/)。
- **Flash-Decoding**：长上下文 decode 时按 KV 分块并行再归约，把 decode 阶段的并行度打满。[Stanford CRFM 博客](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)。
- **FlashInfer**：面向 LLM serving 的 GPU kernel 库，把 attention/MoE/量化 GEMM 做成"定制化、可组合"的算子，覆盖 prefill/decode、GQA、paged KV、FP8，是 vLLM/SGLang 的核心后端。[arXiv:2501.01005](https://arxiv.org/abs/2501.01005)、[flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)（≈6.2k stars）。
- **FlexAttention**：PyTorch 提供的"用 `score_mod`/`mask_mod` 表达任意 attention 变体 + torch.compile 生成高性能 kernel"的机制，把 masking/bias/稀疏结构交给编译器而不是手写 kernel。[PyTorch 博客](https://pytorch.org/blog/flexattention/)。
- **DeepSeek FlashMLA**：为 Hopper（H800/H100）优化的 MLA 解码内核，memory-bound 3000 GB/s、compute-bound 580 TFLOPS，block size 64、BF16。[deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA)（≈12k stars）。

### 2.6 ThunderKittens（Stanford Hazy Research）
"Simple, Fast, and Adorable" 的嵌入式 DSL，核心观点是 **tile 才是 GPU 编程的正确抽象**（而非 thread），在 C++ 里用寄存器 tile + 共享内存 tile 表达内核，配合 warp 级模板。其 blog 展示了在 H100 上对 FlashAttention-2 的超越。
- 出处：[GPUs Go Brrr 博客](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)、[HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens)（≈3.6k stars）。
- **定位**：生产力介于 CUDA 与 Triton 之间、性能逼近手写 CUDA；研究属性强，生态规模小。

### 2.7 XLA 的 Pallas / MOSAIC
Pallas 是 JAX 里的 kernel 级编程 API（TPU 为主，GPU 经 Mosaic GPU 后端），让用户在 XLA 图里写 block 级算子；Mosaic 是 Pallas 的 TPU 编译器。它证明了"同一套 XLA 抽象能覆盖 TPU 与 GPU"，但对 NVIDIA peak 性能仍不如 CUTLASS。
- 出处：[JAX Pallas 文档](https://docs.jax.dev/en/latest/pallas/quickstart.html)、[mosaic_gpu pipeline 源码](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/pipeline.py)。

### 2.8 Halide / Exo / Slapo（"用户可调度"路线）
- **Halide**（PLDI 2013）：把"算法"与"调度"分离，让用户显式控制并行/局部性/重计算，是 TVM 调度的思想源头。[论文](https://halide-lang.org/)。
- **Exo**（PLDI 2022）："exocompilation"，编译器显式暴露底层硬件能力给用户做**可验证的等价变换**，追求"不牺牲硬件能力的可编程性"。[arXiv:2201.09533](https://arxiv.org/abs/2201.09533)、[exo-lang/exo](https://github.com/exo-lang/exo)。
- **Slapo**（ASPLOS 2024）：**面向大模型的"渐进式调度语言"**，以"模型张量级"做 schedule（算子融合/替换/并行），正好命中 LLM 训练/推理的图级优化。[arXiv:2302.08005](https://arxiv.org/abs/2302.08005)。

### 2.9 自动调优（Triton autotuner / AutoTVM / Ansor）
- **AutoTVM**（OSDI 2018）：模板化 + 学习式 cost model 搜调度参数。[arXiv:1805.08166](https://arxiv.org/abs/1805.08166)。
- **Ansor**（OSDI 2020）：`AutoTVM` 的进化，自动生成搜索空间。[arXiv:2006.06762](https://arxiv.org/abs/2006.06762)。
- **Triton autotuner**：`@triton.autotune` 对 `BLOCK_M/BLOCK_N/num_warps/num_stages` 做网格搜索，是 serving 性能的重要来源，也是 JIT 延迟的重要来源。
- 工业实践：IBM 做"Triton autotuning 结果跨平台记住/复用"以给 vLLM 做可移植性（[IBM Research / Ray Summit 2024](https://research.ibm.com/publications/achieving-platform-portability-for-vllm-by-using-triton-autotuning-and-remembering-it)）；AMD/Intel 各自维护 Triton 调优指南（[AMD Triton 调优](https://github.com/ROCm/triton/wiki/General-Guide-of-AMD-Triton-Performance-Optimization)、[Intel XPU 调优 issue](https://github.com/intel/intel-xpu-backend-for-triton/issues/536)）。

### 2.10 跨厂商可移植
- **AMD ROCm**：hipBLASLt/hipSPARSELt + CK（Composable Kernel）+ **AITER**（AMD 官方针对 LLM 的算子树，含 paged attention、FP8、稀疏 attention），FlashInfer 已有 ROCm 后端。[AITER 博客](https://rocm.blogs.amd.com/artificial-intelligence/flashinfer-release2/README.html)、[hipSPARSELt 博客](https://rocm.blogs.amd.com/artificial-intelligence/introduce_hipsparselt/README.html)。
- **Intel**：intel-xpu-backend-for-triton（社区/Intel 维护）、oneAPI/oneDNN；Gaudi 走另一条（SYCL + TPC 编程）。
- **Apple MLX**：Metal 之上的统一内存数组框架，用 JIT 的 Metal shader 做算子，是"端侧 LLM 推理"的标杆。[ml-explore/mlx](https://github.com/ml-explore/mlx)（≈27k stars）。
- **WebGPU**：onnxruntime 已实现 WebGPU 的 MatMulNBits/SubGroupMatrix（[PR #23729](https://app.semanticdiff.com/gh/microsoft/onnxruntime/pull/23729/overview)），浏览器端 LLM 用 WGSL/wgpu；Triton 有 Emscripten/WASM 提案（[triton #3631](https://github.com/triton-lang/triton/issues/3631)）。

### 2.11 推理侧重点内核（attention/GQA/MLA/长上下文/KV-cache/MoE/低精度/稀疏）
- **GQA**：grouped-query attention 是主流省 KV 方案；FlashInfer 明确给 GQA decode 加 `use_tensor_cores`（[PR #317](https://github.com/flashinfer-ai/flashinfer/pull/317)）。
- **MLA**：DeepSeek-V2 提出（[arXiv:2405.04434](https://arxiv.org/abs/2405.04434)），把 KV 压缩成低秩 latent；推理内核成为 FlashMLA/FlashInfer/SGLang/CUTLASS 的竞技场。DeepSeek V3.2 进一步推 **sparse MLA**（DSA），vLLM 有 `TRITON_MLA_SPARSE` 后端（[vLLM #38476](https://github.com/vllm-project/vllm/pull/38476)）。
- **长上下文 / KV-cache-aware**：paged attention（[vLLM/SOSP 2023](https://arxiv.org/abs/2309.06180)）+ FP8 KV cache（AITER SKV、TurboQuant 等）让 KV 量化与分页成为内核设计的一等公民。
- **MoE 路由内核**：grouped GEMM + top-k 路由 + 分派/归约融合；DeepSeek **DeepGEMM**（FP8 细粒度缩放 GEMM，约 300 行核心 CUDA）是代表作。[deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)。
- **新数据类型 GEMM**：MX（Microscaling）FP4/FP8 由 OCP 提出（[arXiv:2310.10537](https://arxiv.org/abs/2310.10537)），NVIDIA Blackwell 引入 NVFP4/MXFP4；CUTLASS 4.x 支持 Blackwell FP4 grouped GEMM（[CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)）。
- **稀疏内核**：2:4/N:M 结构化稀疏（cuSPARSELt、hipSPARSELt、SGLang sparse MLA）。
- **推理 vs 训练的内核差异**：decode 是 memory-bound、batch 小、latency 敏感（KV 读取带宽决定一切，FA/FlashDecoding/FlashMLA 都围绕"把带宽吃满"）；训练是 compute-bound、大 batch、吞吐敏感、需要反向与激活重计算。这导致同一 attention 在内核上**需要完全不同的并行策略与调度**（split-K vs 单次遍历、persistent vs 非 persistent）。

---

## 3. 论文清单（名称 + 年份 + venue，一句话核心）

| 论文 | 年份 / venue | 一句话核心 |
|---|---|---|
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | 2022 / NeurIPS | tiling + online softmax，避免完整注意力矩阵落 HBM。 |
| FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | 2023 / arXiv | 更优并行与工作划分，减少非 matmul FLOP。 |
| FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | 2024 / NeurIPS | Hopper TMA + warp specialization + FP8，H100 上 1.5-2x。 |
| FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving | 2025 / arXiv | serving 定制化、可组合的 attention 内核引擎。 |
| Efficient Memory Management for LLM Serving with PagedAttention | 2023 / SOSP | 分页 KV cache，消除内存碎片，支撑 vLLM。 |
| DeepSeek-V2: A Strong, Economical, and Efficient MoE LLM | 2024 / arXiv | 提出 MLA，低秩 KV 压缩降推理成本。 |
| Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations | 2019 / MAPL | tile 级 DSL + 编译器，替代手写 CUDA 的中间层。 |
| Relax: Composable Abstractions for End-to-End Dynamic Machine Learning | 2023 / arXiv | 面向动态 shape 的图级 IR，TVM Unity 底座。 |
| MLIR: Scaling Compiler Infrastructure for Domain Specific Computation | 2020 / arXiv (CGO'21) | 多层 dialect IR，跨领域编译器基础设施。 |
| TVM: An Automated End-to-End Optimizing Compiler for Deep Learning | 2018 / OSDI | 计算-调度分离 + 自动调优端到端编译器。 |
| Learning to Optimize Tensor Programs (AutoTVM) | 2018 / OSDI | 学习式 cost model + 调度参数搜索。 |
| Ansor: Generating High-Performance Tensor Programs for Deep Learning | 2020 / OSDI | 自动生成搜索空间，超越 AutoTVM。 |
| Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation | 2013 / PLDI | 算法/调度分离，显式控制并行与局部性。 |
| Exocompilation for Productive Programming of Hardware Accelerators | 2022 / PLDI | 用户可调度的等价变换，不牺牲硬件能力。 |
| Slapo: A Schedule Language for Progressive Optimization of LLM Training | 2024 / ASPLOS | 模型张量级的渐进式调度，算子融合/替换/并行。 |
| Microscaling Data Formats for Deep Learning (MX) | 2023 / arXiv | FP4/FP8 的 block 共享 scale，新精度标准。 |
| ThunderKittens: Simple, Fast, and Adorable AI Kernels | 2024 / arXiv (blog) | 以 tile 为原语的嵌入式 DSL，逼近手写 CUDA。 |

> 补充：Flash-Decoding（Stanford CRFM 博客 2023，非正式论文）、FlexAttention（PyTorch 博客 2024，非论文）、CUTLASS/CuTe（NVIDIA 技术库，无单一论文）。

---

## 4. 开源项目盘点（名称 + star 量级 + 活跃度 + 维护方）

| 项目 | star 量级 | 活跃度 | 维护方 |
|---|---|---|---|
| [triton-lang/triton](https://github.com/triton-lang/triton) | ≈19.9k | 极高（周级迭代） | OpenAI + 社区 |
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | ≈10.3k | 极高 | NVIDIA |
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | ≈24.7k | 高 | Dao-AILab（Stanford/Princeton） |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | ≈6.2k | 极高 | 学术 + 社区（vLLM/SGLang 生态） |
| [apache/tvm](https://github.com/apache/tvm) | ≈13.7k | 高 | Apache（社区） |
| [llvm/torch-mlir](https://github.com/llvm/torch-mlir) | ≈1.9k | 高 | LLVM 社区（Nod.ai/AMD 等） |
| [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens) | ≈3.6k | 低（研究） | Stanford Hazy |
| [exo-lang/exo](https://github.com/exo-lang/exo) | ≈2k | 中（研究） | MIT CSAIL |
| [ml-explore/mlx](https://github.com/ml-explore/mlx) | ≈27k | 高 | Apple |
| [google/jax](https://github.com/jax-ml/jax) | ≈32k | 极高 | Google（JAX 迁至 jax-ml 组织） |
| [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) | ≈12k | 中 | DeepSeek |
| [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | ≈10k | 中 | DeepSeek |
| [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | ≈15k | 中 | DeepSeek |
| [tile-ai/tilelang](https://github.com/tile-ai/tilelang) | ≈10k+ | 高 | Tile-AI（DeepSeek 生态） |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | ≈55k | 极高 | vLLM 社区（多家公司共建） |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | ≈35k | 极高 | SGLang（伯克利 + 社区） |
| [pytorch/pytorch](https://github.com/pytorch/pytorch) | ≈102k | 极高 | PyTorch（Meta + 社区） |
| [pytorch/FBGEMM](https://github.com/pytorch/FBGEMM) | ≈3k | 中 | Meta |
| [ROCm/aiter](https://github.com/ROCm/aiter) | ≈1-2k | 高 | AMD |
| [intel/intel-xpu-backend-for-triton](https://github.com/intel/intel-xpu-backend-for-triton) | ≈1k | 高 | Intel + 社区 |
| [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | ≈16k | 极高 | Microsoft |

> star 数值为调研时快照；`≈` 表示数量级近似（部分仓库因 API 限流未逐一精确核对）。

---

## 5. 公司落地

- **NVIDIA**：CUTLASS/CuTe 是自家 TensorRT-LLM 与生态内核的底座；cuDNN/cuBLASLt/cuSPARSELt 提供闭源最优实现。FA3/Blackwell FP4 由 NVIDIA 与社区共同推进。
- **Meta（PyTorch）**：Inductor（默认 Triton 后端）+ FlexAttention + torch.compile，把"图编译 + 内核生成"产品化到 torch 栈；FBGEMM 负责量化/稀疏 CPU/GPU 算子。
- **Google**：JAX/XLA + Pallas/Mosaic，TPU 为第一公民；XLA 是 TensorFlow/JAX 共享后端，代表"图级编译器 + 统一 IR"路线。
- **AMD**：ROCm 全家桶（hipBLASLt、CK、AITER、ROCm Triton、ROCm FlashInfer），以"兼容 CUDA 生态 + 自建内核树"双线追赶 NVIDIA。
- **Intel**：XPU backend for Triton、oneAPI/oneDNN、Gaudi（TPC 编程），主打数据中心与 XPU 可移植。
- **Apple**：MLX/Metal，端侧（Mac/iOS）LLM 推理，以统一内存 + JIT Metal shader 独树一帜。
- **DeepSeek**：FlashMLA、DeepGEMM、DeepEP、3FS 等"开源周"内核全家桶，并拥抱 **TileLang** 作为跨硬件 tile DSL（华为 Ascend Day0 适配），示范了"用高层 DSL 写生产级内核"。[TileLang 报道](https://qbitai.com/2025/09/338386.html)。
- **SGLang / vLLM**：作为推理引擎，是 attention/MoE 内核的"集成与分发中心"——同一逻辑维护 Triton、FlashInfer、CUTLASS、TRT-LLM、AITER 多后端（[SGLang attention backend 文档](https://docs.sglang.io/docs/advanced_features/attention_backend)）。
- **华为 Ascend**：CANN + 社区 Triton-Ascend / TileLang-Ascend，试图让 NPU 进入 Triton/TileLang 生态。

---

## 6. 趋势判断

1. **"高层 tile DSL"正在成为内核开发的主战场**：Triton → ThunderKittens → TileLang 一路把抽象提高，同时用 MLIR 做统一 lowering。手写 CUDA 退守"最后一个 10% 性能"与"新硬件特性首发"。
2. **CuTe/CUTLASS 成为高性能 attention/GEMM 的"事实汇编层"**：FlashInfer、SGLang、DeepSeek 都用 CuTe 重写关键内核，Triton 之上再叠一层"手写 CuTe 兜底"。
3. **注意力内核从"通用 FA"走向"结构特化"**：MLA、sparse MLA（DSA）、GQA、FP8 KV、paged、MTP 各一套，内核随模型结构演进的速度已快于编译器泛化的速度——这解释了 FlashMLA/FlashInfer 的价值。
4. **低精度全面 FP4/MX 化**：FP8 已成主流，FP4/MXFP4/NVFP4 在 Blackwell 落地，GEMM/注意力内核都要重写以吃满新 tensor core。
5. **serving 侧的"编译/调优开销"成为新瓶颈**：shape 爆炸 + JIT + autotune 网格搜索，把"编译时"变成了运行时问题；AOT 预编译、config 复用、编译缓存是下一波优化点。
6. **跨厂商收敛但远未统一**：MLIR 统一了"表示"，但没有统一"内核"；每家仍是自建内核树。统一的高性能算子层是空白。

---

## 7. 已饱和点（不建议再投入）

- **NVIDIA 单卡 GEMM/attention 的纯性能内卷**：CUTLASS/CuTe + FA3 + 厂商闭源库已把主流形状吃透，个人/小团队很难再压出显著增益。
- **"又一个 Triton 竞品 DSL"**：若无明确硬件/场景差异化，纯粹做新 DSL 语言本身已过饱和（Triton/ThunderKittens/TileLang/Exo/Halide 已覆盖）。
- **通用静态图编译器（对标 TVM/MLIR 的基础设施）**：作为研究"重造轮子"意义有限，除非绑定具体新硬件。
- **通用 attention 的 FA2/FA3 复刻**：上游（Dao-AILab/FlashInfer）迭代极快，复刻很快过期。

---

## 8. 被忽视的空白与机会

1. **跨硬件的高性能注意力内核"标准/抽象层"缺失**：GQA/MLA/decode 在 NVIDIA/AMD/Intel/Ascend 各有一套（FlashMLA 是 Hopper-only、AITER 是 ROCm-only、TRT-LLM 是 NVIDIA-only）。缺一个"一次表达、多后端 lower"的 attention 内核 DSL/IR（FlexAttention 只覆盖 PyTorch 生态，且不覆盖 MLA/decode 特化）。
2. **MLA 内核开源度不足**：FlashMLA 只给 Hopper；sparse MLA（V3.2 DSA）主要靠 TileLang/vendor 后端或闭源；跨硬件（MI300/Blackwell/Ascend）的、带 paged KV 的 MLA decode 参考实现仍是空白。
3. **Triton 在 serving 的实时编译开销无人系统解决**：JIT 编译 + autotune 导致首 token/首次请求抖动（[vLLM #43009](https://github.com/vllm-project/vllm/issues/43009)）；缺"离线预编译 + 运行时零编译 + config 冻结"的一体化方案。
4. **自动调优基建薄弱**：现有 autotuner 是"每次网格搜索"，缺跨 shape/跨硬件的 config 迁移、学习式 cost model、调优结果的服务化存储与共享（IBM 只做了雏形）。
5. **异构 NPU 的算子层真空**：Ascend/Qualcomm/新兴国产 NPU 缺 Triton 级别的可用抽象（Triton-Ascend/TileLang-Ascend 刚起步）；这是国产算力落地 LLM 的最大工程缺口。
6. **推理 vs 训练的统一内核表达**：同一模型两套内核（memory-bound decode vs compute-bound train），缺"一份 tile 规格 + 编译期按工作负载切换调度策略"的机制。

---

## 9. 具体候选切入点（3-5 个）

**切入点 A：跨后端的 MLA/GQA/decode 注意力内核 DSL（"attention 的 FlexAttention + CuTe"）**
设计一个 tile 级注意力规格语言，把"头结构（MHA/GQA/MLA/sparse-MLA）、精度（FP8/FP4）、KV 布局（paged/量化）"声明式表达，再分别 lower 到 Triton / CuTe / AITER / TileLang。产出：一个可复用的 MLA decode 参考内核 + 跨 NVIDIA/AMD/Ascend 的 benchmark 矩阵。命中空白 #1/#2。

**切入点 B：serving 级"零 JIT"编译/调优层（AOT 预编译 + config 冻结 + 缓存服务）**
对 Triton/Inductor 做"shape→kernel→autotune config"的离线穷举/缓存，运行时命中即用、miss 走兜底内核，消除首请求抖动。可与 vLLM/SGLang 集成。命中空白 #3/#4。

**切入点 C：学习式/迁移式自动调优器（amortized autotuner）**
用轻量 cost model + 跨 shape/跨硬件迁移，替代 Triton autotuner 的网格搜索；提供"调优结果数据库 + 查询 API"，把 IBM 雏形产品化。命中空白 #4。

**切入点 D：异构 NPU 的 Triton-class 算子层（MLIR-based tile lowering）**
在 MLIR 上实现一套 tile 方言 + 调度原语，lower 到 Ascend C/Qualcomm Hexagon，让 NPU 拿到"类 Triton"的生产力；优先打通 MLA + FP8 GEMM 两个内核。命中空白 #5。

**切入点 E：推理/训练共用的内核规格 + 编译期工作负载切换**
定义一份算子 tile 规格（以 GEMM/attention 为试点），编译期根据"memory-bound（decode）vs compute-bound（train/prefill）"自动切换 split-K、persistent、streaming 等调度，产出单一源码、双模式内核。命中空白 #6。

---

### 主要出处索引（真实 URL）
- Triton：https://github.com/triton-lang/triton ；https://arxiv.org/abs/1801.01946
- Triton serving JIT 开销：https://github.com/vllm-project/vllm/issues/43009
- Triton TMA/MXFP4 局限：https://github.com/triton-lang/triton/issues/8548
- TVM：https://github.com/apache/tvm ；Relax：https://arxiv.org/abs/2311.02903
- Torch-MLIR：https://github.com/llvm/torch-mlir
- CUTLASS：https://github.com/NVIDIA/cutlass ；https://docs.nvidia.com/cutlass/4.5.2/
- FlashAttention-3：https://arxiv.org/abs/2407.08608 ；https://pytorch.org/blog/flashattention-3/
- FlexAttention：https://pytorch.org/blog/flexattention/
- FlashInfer：https://github.com/flashinfer-ai/flashinfer ；https://arxiv.org/abs/2501.01005
- FlashMLA：https://github.com/deepseek-ai/FlashMLA
- DeepGEMM：https://github.com/deepseek-ai/DeepGEMM
- ThunderKittens：https://hazyresearch.stanford.edu/blog/2024-05-12-tk
- Pallas：https://docs.jax.dev/en/latest/pallas/quickstart.html
- Slapo：https://arxiv.org/abs/2302.08005 ；Exo：https://arxiv.org/abs/2201.09533
- MX formats：https://arxiv.org/abs/2310.10537
- MLX：https://github.com/ml-explore/mlx
- AITER/ROCm：https://rocm.blogs.amd.com/artificial-intelligence/flashinfer-release2/README.html
- IBM Triton autotune 复用：https://research.ibm.com/publications/achieving-platform-portability-for-vllm-by-using-triton-autotuning-and-remembering-it
- TileLang/Ascend：https://qbitai.com/2025/09/338386.html
