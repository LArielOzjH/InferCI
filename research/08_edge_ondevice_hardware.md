# 边缘 / 端侧 / 异构硬件推理 深潜调研

> 调研范围：端侧 LLM / 小模型推理的运行时代、量化、NPU、浏览器(WebGPU/WASM)、投机解码、功耗、长上下文、端侧 RAG、异构调度。
> 调研方式：多轮英文关键词联网检索（arXiv 顶会论文、公司技术博客、GitHub 仓库、行业报道、专家 blog），关键事实附出处 URL。Star 数量为调研时点近似量级。

---

## 0. 一句话本质

端侧 LLM 推理的**根本瓶颈不是算力，而是内存带宽与功耗**。自回归解码（逐 token 生成）是 memory-bound 的：每生成一个 token 都要把整个权重矩阵和 KV cache 读一遍。因此这个方向的所有关键工作，本质都在回答同一个问题——**如何用更少的字节、更少的功耗、更少的往返，把模型"塞进"一部手机/一块 NPU/一个浏览器标签页里跑起来**。量化压缩权重、投机解码摊薄带宽、KV cache 压缩支撑长上下文、异构调度平衡功耗与延迟，都是这一本质问题的不同侧面。

---

## 1. 现状总览

**范式已经确立**：2023 年 llama.cpp 证明"一台消费级笔记本/手机 CPU 就能跑 7B"，2024–2025 年行业从"能不能跑"转向"跑得好不好、省不省电、隐私与成本"。两条路线并存：

1. **可移植运行时 + 厂商 delegate**：ExecuTorch、ONNX Runtime、LiteRT（原 TFLite）、MLC/TVM、IREE。一次编译/转换，下探到各厂商 NPU/GPU/CPU 后端。
2. **手写高性能内核**：llama.cpp（每后端手写 GGML 内核）、MLX（Apple 专属 Metal JIT）。性能极致但移植成本高。

**硬件栈正在收敛但也高度碎片化**：手机侧 Qualcomm Hexagon NPU（QNN/QAI）、MediaTek APU/NeuroPilot、Google Tensor、Apple Neural Engine；PC 侧 Intel NPU、AMD Ryzen AI、Qualcomm Copilot+、Apple M 系统一内存。每一家都有自己的 SDK、算子覆盖度、量化格式（INT8/INT4/FP16 各不相同），**没有一个真正跨厂商的统一 NPU 运行时**——这是当前最大的结构性缺口（见 §8）。

**"杀手应用"正在从聊天 demo 转向端侧 Agent + 检索**：小模型（Phi/Gemma/Qwen/SmolLM/MobileLLM）质量提升后，配合工具调用、端侧 RAG、个性化微调，才真正成立"为什么要端侧"（延迟、隐私、离线、成本）。

**当前卡点**（共性）：
- NPU 对 LLM 的**算子覆盖不全**（尤其动态 shape、softmax、KV 管理），大模型 decode 往往还是 CPU/GPU 更快，NPU 强在 prefill 与视觉。
- **<4-bit 量化精度损失**、且各 NPU 对 INT4/INT8 per-channel 支持不一致。
- **长上下文**下 KV cache 内存线性膨胀，手机 8–16GB 内存很快耗尽。
- **功耗/热**：持续 decode 会热降频，缺乏以"能量/每 token"为一等公民的调度器。
- **浏览器** WebGPU 性能仍比原生低一个量级，且浏览器无法访问 NPU。

---

## 2. 关键技术（含出处 URL）

### 2.1 运行时代与格式

- **llama.cpp / GGML / GGUF**：C/C++ 无依赖推理引擎 + 量化模型格式。核心价值是让 LLM 在纯 CPU/RAM 上通过 2–8bit 量化 + mmap 跑起来，后端覆盖 CPU(AVX/NEON)/CUDA/Metal/Vulkan/SYCL/OpenCL 等。量化演进 K-quants → IQ-quants → MX quants，并持续探索 KV cache 量化（TurboQuant）、NPU 支持（如 Intel NPU/Arc 讨论）。
  - https://github.com/ggml-org/llama.cpp
  - https://github.com/ggml-org/ggml
  - https://github.com/ggml-org/llama.cpp/discussions/15883 （Intel NPU 支持讨论）
  - https://github.com/ggml-org/llama.cpp/discussions/14294 （同时用 NPU/GPU/CPU 的内存带宽问题）
  - https://github.com/ggml-org/llama.cpp/discussions/17393 （量化类型文档）

- **Apple MLX**：Apple 的统一内存数组框架，惰性求值 + Metal JIT + 可组合函数变换（JAX 风格），专吃 M 系统一内存带宽（400–800 GB/s，远超独立 GPU 走 PCIe）。瓶颈：仅 Apple 生态。
  - https://github.com/ml-explore/mlx
  - https://github.com/ml-explore/mlx/discussions/3209 （M3 Ultra 系统化推理基准）

- **ExecuTorch / PyTorch Edge**：PyTorch 原生端侧运行时，AOT 导出 + delegate 机制（XNNPACK CPU、Core ML/ANE、Qualcomm QNN/HTP、Vulkan、ARM Ethos-U）。2025-10 发布 1.0，主打"一次导出、多后端"。
  - https://pytorch.org/blog/introducing-executorch-1-0/
  - https://www.qualcomm.com/developer/blog/2025/10/bringing-edge-ai-performance-to-pytorch-developers-with-executorch-1-0
  - https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/ethos-u-and-beyond-how-executorch-1-0-powers-ai-at-the-edge
  - https://github.com/pytorch/executorch

- **ONNX Runtime（+ GenAI）**：中立交换格式 + 可插拔 Execution Provider（CPU/CUDA/DirectML/QNN/NNAPI/CoreML/OpenVINO），ORT GenAI 直接跑量化 LLM。QNN EP 在 Copilot+ PC（Qualcomm Windows）上由微软持续推送更新。
  - https://github.com/microsoft/onnxruntime
  - https://mintlify.wiki/microsoft/onnxruntime-genai/acceleration/qnn
  - https://windowsforum.com/threads/kb5067994-qnn-execution-provider-update-for-copilot-on-qualcomm-windows-11.381916/

- **MediaPipe LLM Inference API / Google AI Edge / LiteRT**：Google 端侧栈。LLM Inference API 用 XNNPACK 在移动 CPU/GPU 上跑 Gemma 等；AI Edge Torch 把 PyTorch 转 LiteRT；LiteRT 新增 NPU accelerator（Qualcomm AI Engine Direct、MediaTek NeuroPilot、GPU），把手机 NPU 抬为 LLM"一等公民"。浏览器侧也用 MediaPipe 跑 7B+（WebGPU）。
  - https://research.google/blog/unlocking-7b-language-models-in-your-browser-a-deep-dive-with-google-ai-edges-mediapipe/
  - https://developers.googleblog.com/en/large-language-models-on-device-with-mediapipe-and-tensorflow-lite/
  - https://github.com/google-ai-edge/mediapipe
  - https://github.com/google-ai-edge/ai-edge-torch
  - https://developers.google.com/edge/litert/android/npu/qualcomm

- **MLC-LLM / WebLLM**：Apache TVM 编译栈 + WebGPU，跨 iOS/Android/浏览器/桌面。
  - https://github.com/mlc-ai/mlc-llm
  - https://github.com/mlc-ai/web-llm

- **IREE（MLIR）**：基于 MLIR 的可重定向编译器/运行时，Vulkan/SPIR-V 作为跨 GPU 的统一后端（AMD 力推）。跨 NPU 统一的理想 IR 底座。
  - https://github.com/iree-org/iree
  - https://www.phoronix.com/news/AMD-Vulkan-SPIR-V-Wide-AI

### 2.2 NPU 与移动 SoC

- **Qualcomm Hexagon NPU**：三套 API 并存（QNN / Qualcomm AI Engine Direct "QAI" / Hexagon SDK），经 LiteRT、ORT QNN EP、ExecuTorch delegate 暴露。Snapdragon 8 Elite Gen 5（2025）继续堆 NPU TOPS。
  - https://developers.google.com/edge/litert/android/npu/qualcomm
  - https://github.com/qualcomm/qidk

- **MediaTek NeuroPilot / Dimensity APU**：NeuroPilot 工具链 + LiteRT NeuroPilot accelerator，主攻电视/手机/物联网海量设备。
  - https://www.mediatek.com/tek-talk-blogs/mediatek-npus-neuropilot-and-litert-are-ready-to-bring-power-ai-in-millions-of-devices

- **Apple Neural Engine + Core ML + Foundation Models 框架**：ANE 经 Core ML 调用；WWDC25（2025-06）开源 Foundation Models framework，抽象"端侧 + 私有云计算 + 第三方（Claude/Gemini）"，iPhone 可跑 70B 级（分片/量化）本地模型。Apple Intelligence 端侧 3B 模型论文公开了训练与量化细节。
  - https://developer.apple.com/videos/play/wwdc2025/286/
  - https://arxiv.org/abs/2407.21075 （Apple Intelligence Foundation Language Models）
  - https://9to5mac.com/2025/11/20/apple-shows-how-much-faster-the-m5-runs-local-llms-compared-to-the-m4/ （M5 跑 MLX 比 M4 快 ~27%）

### 2.3 小模型

- **Phi-3/3.5-mini、Phi-4-mini(3.8B)**（Microsoft）、**Gemma 2 2B / Gemma 3 1B**（Google）、**Qwen2.5/3 0.5B–4B**（阿里）、**SmolLM/SmolLM2/3**（HuggingFace）、**MobileLLM**（Meta）、**Llama 3.2 1B/3B**（Meta）。趋势：蒸馏 + 长上下文 + 工具调用，让 <4B 模型质量逼近更大模型。
  - https://ai.google.dev/gemma/docs/integrations/mobile

### 2.4 量化

- **权重量化**：GPTQ、AWQ、GGUF K/IQ-quants、**SpinQuant**（ICLR 2025，学习旋转）、QuaRot、AQLM、EfficientQAT、W4A16/W4A8/W8A8、MX 微缩放格式。
- **KV cache 量化**：TurboQuant（llama.cpp 讨论）、SubKV、Cache Me If You Must（自适应 KV 量化）。
- **关键卡点**：<4-bit 权重与激活精度损失，且 NPU 对 INT4/per-channel 支持不一致。
  - https://mlanthology.org/iclr/2025/liu2025iclr-spinquant/
  - https://huggingface.co/papers/2501.19392 （Cache Me If You Must）
  - https://github.com/ggml-org/llama.cpp/discussions/20969 （TurboQuant）

### 2.5 WASM / WebGPU（浏览器端）

- **WebLLM**（TVM/MLC → WebGPU compute shader）、**transformers.js**（ONNX Runtime Web，WebGPU + WASM SIMD）。零安装跨平台，但性能比原生低约一个量级、受浏览器内存与 4GB WASM 堆限制、无法访问 NPU。
  - https://github.com/mlc-ai/web-llm
  - https://github.com/huggingface/transformers.js
  - https://localmode.dev/blog/compare/webllm-vs-transformers-js

### 2.6 端侧投机解码

- 用 draft 小模型或 self-speculative（量化 KV 的 QuantSpec）预测多个 token 再由大模型验证，摊薄 memory-bound decode 带宽。
  - https://ieeexplore.ieee.org/document/10812936 （EdgeLLM，IEEE TMC）
  - https://proceedings.mlr.press/v267/tiwari25b.html （QuantSpec，ICML 2025）
  - https://ieeexplore.ieee.org/abstract/document/11432593 （Device-Server Collaborative Speculative Decoding）

### 2.7 功耗与异构调度

- **PowerInfer-2**（arXiv 2406.06282）：把智能手机 CPU 大小核 + GPU + NPU 协同，神经元/激活稀疏感知，跑 47B Mixtral。
- **HeteroLLM**（arXiv 2501.14794）：移动 SoC 异构 AI 加速器协同。
- 量化直接降能耗：Ollama 框架量化模型的能耗/延迟/精度系统评测。
  - https://arxiv.org/abs/2406.06282
  - https://hackernoon.com/lite/the-conductor-in-your-pocket-how-powerinfer-2-orchestrates-smartphone-hardware-for-llm-inference
  - https://github.com/SJTU-IPADS/PowerInfer
  - https://arxiv.org/abs/2501.14794 （HeteroLLM）
  - https://ar5iv.labs.arxiv.org/html/2504.03360 （量化 LLM 能耗评测）

### 2.8 端侧长上下文

- KV cache 压缩/驱逐：SubKV、Locret、Mustafar（KV 剪枝）、Cache Me If You Must。目标：在 8–16GB 手机上撑住 128K 上下文。
  - https://oar.a-star.edu.sg/communities-collections/articles/22330 （SubKV）
  - https://researchportal.hkust.edu.hk/en/publications/locret-enhancing-eviction-in-long-context-llm-inference-with-trai/
  - https://huggingface.co/papers/2505.22913 （Mustafar）

### 2.9 端侧 RAG

- Qdrant Edge（嵌入式向量搜索）、RAGdb（零依赖可嵌入式多模态 RAG）、NPU-first 的 ONNX Runtime + FAISS/BM25 私有无监督检索。
  - https://qdrant.tech/blog/qdrant-edge/
  - https://ar5iv.labs.arxiv.org/html/2602.22217 （RAGdb）
  - https://github.com/krtarunsingh/on-device-npu-rag

### 2.10 端侧微调 / 个性化

- DP-FedLoRA（差分隐私联邦微调）、Prada（黑盒私有适配）。让"本地模型学你的数据"成立。
  - https://arxiv.org/abs/2509.09097

---

## 3. 论文清单（名称 + 年份 + venue + 一句话核心）

- **PowerInfer-2** — 2024, arXiv（2406.06282）— 在智能手机上协同 CPU 大小核 + GPU + NPU，以激活稀疏感知策略跑 47B 级模型。
- **Apple Intelligence Foundation Language Models** — 2024, arXiv（2407.21075）— 公开 Apple 端侧 3B 模型训练/量化/适配细节，奠定 iPhone 端侧智能。
- **EdgeLLM: Fast On-Device LLM Inference With Speculative Decoding** — 2024, IEEE TMC — 端侧投机解码加速的早期系统化方案。
- **QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache** — 2025, ICML — 用量化 KV cache 自投机，免额外 draft 模型。
- **SpinQuant: LLM Quantization with Learned Rotations** — 2025, ICLR — 学习旋转使 <4-bit 量化更稳健。
- **Cache Me If You Must: Adaptive Key-Value Quantization** — 2025, ICML（PMLR）— 自适应 KV 量化支撑长上下文。
- **SubKV** — 2025（A*STAR）— 面向亚十亿参数端侧模型的 KV cache 量化。
- **Locret** — 2024（HKUST）— 训练保留头做长上下文 KV 驱逐。
- **HeteroLLM** — 2025, arXiv（2501.14794）— 移动 SoC 异构 AI 加速器（NPU/GPU/CPU）协同推理。
- **Sustainable LLM Inference for Edge AI（Ollama 量化能耗）** — 2025, arXiv（2504.03360）— 系统评测量化 LLM 的能耗/延迟/精度权衡。

---

## 4. 开源项目盘点（名称 + star 量级 + 活跃度 + 维护方）

> star 为调研时点近似值（GitHub 实时抓取）。

- **llama.cpp** — ~12.4 万 star，极活跃，ggml-org（社区，创始人 Georgi Gerganov）。
- **MLX** — ~2.8 万 star，活跃，Apple（ml-explore）。
- **Ollama** — ~17.9 万 star，极活跃，Ollama Inc.（本地运行 LLM 的用户入口，底层常用 llama.cpp）。
- **MediaPipe** — ~3.7 万 star，活跃，Google AI Edge。
- **MLC-LLM** — ~2.3 万 star，活跃，MLC-AI 社区/CMU。
- **ONNX Runtime** — ~2.1 万 star，极活跃，Microsoft。
- **WebLLM** — ~1.9 万 star，活跃，MLC-AI（浏览器 WebGPU）。
- **transformers.js** — ~1.6 万 star，活跃，HuggingFace。
- **ggml** — ~1.5 万 star，活跃，ggml-org（张量内核库）。
- **PowerInfer** — ~9.7k star，较不活跃（研究原型），SJTU IPADS。
- **ExecuTorch** — ~4.9k star，活跃，PyTorch/Meta。
- **IREE** — ~3.9k star，活跃，IREE 社区（Google/AMD）。
- **ai-edge-torch** — ~1.1k star，活跃，Google AI Edge。

---

## 5. 公司落地

- **Apple**：ANE + Core ML + Foundation Models framework + MLX；Apple Intelligence 端侧 3B；M5 加速 MLX（比 M4 快 ~27%）。主打"端侧 + 私有云"分级。
- **Google**：LiteRT/MediaPipe/AI Edge Torch + Gemma（2B/1B）+ Gemini Nano（Pixel）；把 Qualcomm QAI 与 MediaTek NeuroPilot 接进 LiteRT，推动"NPU 一等公民"。
- **Microsoft**：ONNX Runtime + GenAI；Copilot+ PC 的 QNN EP 更新；Phi 系列小模型（Phi-3.5-mini、Phi-4-mini）。
- **Qualcomm**：QNN/QAI + Hexagon NPU；联合 Meta 推 ExecuTorch 1.0；骁龙 AI 手机/Copilot+ PC。
- **MediaTek**：NeuroPilot + Dimensity 9400/9500；LiteRT NeuroPilot 覆盖海量设备。
- **Meta**：ExecuTorch/PyTorch Edge + Llama 3.2 1B/3B + MobileLLM。
- **NVIDIA（边缘）**：TensorRT-LLM / TensorRT-Edge-LLM 跑 Jetson Orin/Thor 等嵌入式 GPU。
- **Samsung**：Exynos NPU + 端侧 Galaxy AI（Gemini Nano 等）。
- **HuggingFace**：SmolLM 系列 + transformers.js + 量化生态（llama-stack）。

---

## 6. 趋势判断

1. **"可移植运行时 + 厂商 delegate"成为事实标准**：ExecuTorch/ORT/LiteRT 三足鼎立，手写内核（llama.cpp/MLX）继续在最热路径保持性能优势。
2. **NPU 从"视觉/语音专用"走向 LLM 一等公民**，但受限于算子覆盖、INT4/INT8 精度与片上 SRAM，短期内 decode 主力仍是 CPU/GPU，NPU 主导 prefill 与轻量模型。
3. **统一内存架构**（Apple M、Qualcomm Oryon、Copilot+ PC）把"手机/笔记本/桌面"边界抹平，端侧可跑模型规模上移。
4. **小模型 + 端侧 Agent + 工具调用 + 检索**是下一阶段价值点，而非裸聊天。
5. **投机解码 + KV cache 压缩**成为标准优化，直击 memory-bound decode。
6. **浏览器（WebGPU/WASM）成为新分发渠道**，隐私/零安装诉求推动，但性能与 NPU 访问仍是短板。
7. **功耗/热成为一等公民**：持续 decode 的热降频与"能量/每 token"指标开始进入产品决策。
8. **端侧微调/个性化 + 联邦学习**（DP-FedLoRA 等）让"本地模型学你的数据"成为差异化。

---

## 7. 已饱和点

- **4-bit 权重量化**（GPTQ/AWQ/GGUF K-quants）已高度成熟，进一步压缩收益边际递减、精度代价陡增。
- **GGUF/llama.cpp 在旗舰机/笔记本 CPU 上跑 7B 聊天**已是"成品级"，差异化空间小。
- **WebGPU 的 WebLLM/transformers.js 聊天 demo** 高度商品化，性能提升受限于浏览器与硬件暴露。
- **NPU 上的 8-bit/4-bit 视觉模型（检测/分割/分类）** 已是成熟赛道。
- **"7B 上旗舰手机"的演示性指标**（能不能跑）已不是卖点，转而是延迟/功耗/长上下文/工具能力。

---

## 8. 被忽视的空白与机会

- **跨厂商 NPU 统一运行时**：QNN/ANE/NeuroPilot/Intel NPU/AMD Ryzen AI 各自为政，缺一个"一次导出、按能力自动降级"的统一抽象（IREE/MLIR 是最有希望的底座，但 NPU 后端的算子覆盖远未补齐）。
- **NPU 原生的投机解码与 KV cache**：投机解码/KV 压缩多跑在 CPU/GPU，NPU 上的低功耗 draft + 量化 KV 尚未成体系。
- **端侧长上下文（128K+）**：KV cache 内存线性膨胀仍未在消费级硬件上"可生产化"解决。
- **端侧 RAG 的一体化工程**：嵌入（NPU 加速）+ 混合检索（BM25+向量）+ 重排 + 权限/隐私，缺少端到端、面向产品的开源栈。
- **以"能量/每 token"为核心的电源/热感知调度器**：PowerInfer-2/HeteroLLM 是研究原型，未产品化；缺跨硬件、跨模型的统一调度层。
- **浏览器 NPU 访问与 WebGPU 高级特性**：W3C/浏览器厂商尚未打通，隐私应用受限于此。
- **端侧基准与可观测性**：缺一个标准化的"能耗/热/内存/延迟"多运行时基准 + 模型选择路由器（按设备状态自动选模型+量化档位）。
- **端侧微调/持续学习**：个人化与隐私的交叉点，工程化远未成熟。

---

## 9. 具体候选切入点（3–5 个）

1. **跨厂商 NPU 统一 LLM 运行时（能力注册 + 自动降级）**：基于 MLIR/TOSA（或复用 ExecuTorch delegate / ORT EP / LiteRT accelerator）做一个"NPU 能力描述层"，输入算子覆盖 + 量化格式 + 片上内存，运行时自动把模型切成"NPU prefill / GPU-CPU decode / CPU fallback"；对不支持的算子透明回退 CPU/GPU。价值点：开发者一套代码覆盖 Hexagon/ANE/NeuroPilot/Intel/AMD。落地抓手：先做 Qualcomm QNN + Intel NPU + MediaTek 三家的能力矩阵与动态切分。

2. **NPU 常驻的低功耗投机解码 + 量化 KV cache**：把 draft 模型与量化 KV 放到 NPU/片上 SRAM，主模型在 GPU/CPU 验证，目标"长上下文 + 低功耗持续 decode"。可复用 QuantSpec / EdgeLLM 思路，做成可插拔模块挂到 llama.cpp / ORT GenAI。价值点：手机上长时间对话的能耗与延迟。

3. **端侧隐私 RAG 引擎（NPU-first 的一体化检索栈）**：嵌入模型跑 NPU（INT8）+ BM25/向量混合检索（SQLite-vec / FAISS / Qdrant Edge）+ 交叉编码重排 + 权限过滤，端到端离线。价值点：医疗/法律/笔记等隐私敏感场景的"本地知识库"。落地抓手：先打通 MediaPipe/LiteRT 的 NPU 嵌入 + 本地混合检索，做出可审计的检索质量与功耗报告。

4. **电源/热感知的异构推理调度器（"能量/每 token"为一等公民）**：在 PowerInfer-2/HeteroLLM 思想之上做产品化运行时，实时根据电量、温度、前台/后台、网络把模型分片调度到大小核/GPU/NPU，并自动选择量化档位。价值点：手机端"不发热、不耗电"的长时可用性。落地抓手：先做多后端 token 生成的能量/热采样模型，再做调度策略。

5. **端侧 LLM 基准 + 自动模型选择路由器**：标准化"能耗/热/内存/延迟"多运行时（llama.cpp/MLX/ORT/ExecuTorch/LiteRT）基准，输出设备画像，再按任务动态选模型 + 量化档位（本地小模型 vs 云端大模型的分级路由）。价值点：被忽视的可观测性 + 模型运维层，可作为开源基建产品。

---

### 附：关键出处 URL 汇总

- https://github.com/ggml-org/llama.cpp
- https://github.com/ggml-org/ggml
- https://github.com/ml-explore/mlx
- https://github.com/pytorch/executorch
- https://pytorch.org/blog/introducing-executorch-1-0/
- https://www.qualcomm.com/developer/blog/2025/10/bringing-edge-ai-performance-to-pytorch-developers-with-executorch-1-0
- https://github.com/microsoft/onnxruntime
- https://github.com/google-ai-edge/mediapipe
- https://github.com/google-ai-edge/ai-edge-torch
- https://developers.googleblog.com/en/large-language-models-on-device-with-mediapipe-and-tensorflow-lite/
- https://research.google/blog/unlocking-7b-language-models-in-your-browser-a-deep-dive-with-google-ai-edges-mediapipe/
- https://developers.google.com/edge/litert/android/npu/qualcomm
- https://www.mediatek.com/tek-talk-blogs/mediatek-npus-neuropilot-and-litert-are-ready-to-bring-power-ai-in-millions-of-devices
- https://github.com/mlc-ai/mlc-llm
- https://github.com/mlc-ai/web-llm
- https://github.com/huggingface/transformers.js
- https://github.com/iree-org/iree
- https://www.phoronix.com/news/AMD-Vulkan-SPIR-V-Wide-AI
- https://github.com/SJTU-IPADS/PowerInfer
- https://github.com/ollama/ollama
- https://arxiv.org/abs/2406.06282
- https://arxiv.org/abs/2407.21075
- https://arxiv.org/abs/2501.14794
- https://arxiv.org/abs/2504.03360
- https://arxiv.org/abs/2509.09097
- https://mlanthology.org/iclr/2025/liu2025iclr-spinquant/
- https://proceedings.mlr.press/v267/tiwari25b.html
- https://huggingface.co/papers/2501.19392
- https://huggingface.co/papers/2505.22913
- https://ieeexplore.ieee.org/document/10812936
- https://ieeexplore.ieee.org/abstract/document/11432593
- https://oar.a-star.edu.sg/communities-collections/articles/22330
- https://researchportal.hkust.edu.hk/en/publications/locret-enhancing-eviction-in-long-context-llm-inference-with-trai/
- https://qdrant.tech/blog/qdrant-edge/
- https://ar5iv.labs.arxiv.org/html/2602.22217
- https://developer.apple.com/videos/play/wwdc2025/286/
- https://9to5mac.com/2025/11/20/apple-shows-how-much-faster-the-m5-runs-local-llms-compared-to-the-m4/
- https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/ethos-u-and-beyond-how-executorch-1-0-powers-ai-at-the-edge
