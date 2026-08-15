# 开源 LLM 推理 Infra —— 现状总览、缺口地图与候选方向

> 综合 10 个深潜子代理返回的研究摘要（量化 / 剪枝稀疏 / 算子编译 / 分布式推理 / Serving / 投机解码 / 长上下文 / 边缘端侧 / 行业格局 / OSS 生态）生成。
> 生成口径：每一处结论均回溯到摘要中的具体事实（论文、项目、star 数、收购/融资事件）。

---

## 一、State of the Art（现状总览）

2025–2026 年 LLM 推理 Infra 的现状可一句话概括：**自回归 decode 是「访存带宽受限」而非「算力受限」**，因此所有主线优化最终都落在内存、带宽与批处理效率上。技术栈已分化为四层——内核/算子、单机引擎、分布式编排、异构端侧——而演进逻辑正从「单点压缩」走向「系统 co-design」再到「架构原语原生化」。

**内核层**：注意力已被 FlashAttention（FA1/2/3）与 CUTLASS/CuTe 基本吃透，竞争焦点转向跨厂商可移植与 MLA/稀疏/MoE/FP4 等新结构内核。Triton（约 19.9k★）是主流 tile DSL，但存在 JIT 编译延迟、TMA/MXFP4 支持不全、抽象不够低三大公开短板；ThunderKittens/TileLang 在 tile 抽象上逼近手写 CUDA，TileLang 已适配华为 Ascend；MLIR 统一了表示却未统一内核，推理与训练内核仍分裂。

**引擎层**：vLLM 与 SGLang 形成双寡头且均已资本化——vLLM 经 Neural Magic→Red Hat 收购并入 PyTorch Foundation，SGLang 团队成立 RadixArk 获 Accel 领投 1 亿美元、NVIDIA/AMD/Intel 参投。TensorRT-LLM 守 NVIDIA 最优路径，TGI 停滞（2026-03 起不活跃），llama.cpp/ollama/MLX 证明单机/边缘是独立价值线。PagedAttention（分页）→RadixAttention（前缀缓存）→KV 量化/稀疏/淘汰→LMCache（存储化）这条线，把 KV cache 从显存管理细节升级为**一等公民数据对象**。prefill（算力密集）/decode（访存密集）冲突催生 PD 分离（DistServe/Splitwise/Mooncake），已成各框架默认架构，并让 KV 跨节点传输成为第一性瓶颈。

**模型侧协同是最大优化杠杆**：MLA 低秩 KV 压缩约降 90% cache；MoE 以稀疏激活换吞吐；MTP/投机解码随预训练一体化，取代独立 draft 小模型（Medusa/EAGLE）；DeepSeek-V3 的 671B MoE+MLA+FP8 以 545% 毛利重置成本预期。量化上 FP8 已成数据中心默认基线、FP4/MXFP4（Blackwell 吞吐约 2×FP8）是下一波，能力整体并入引擎（vLLM/llama.cpp/TRT-LLM），AutoGPTQ/AutoAWQ/DeepSparse/SparseML 等独立工具相继归档。稀疏的重心从权重（2:4 被 NVIDIA cuSPARSELt 锁死、跨硬件不可移植）转向注意力/激活，DeepSeek NSA/Moonshot MoBA/MInference 已在长上下文兑现 3–11× 加速并进入 API。

**端侧**走「可移植运行时 + 厂商 delegate」（ExecuTorch/ORT/LiteRT/IREE）与手写内核（llama.cpp/MLX）双路线，NPU（Hexagon/ANE/NeuroPilot）走向 LLM 一等公民但 decode 主力仍是 CPU/GPU，跨厂商 NPU 统一运行时与浏览器 NPU 访问是公认空白。

**宏观**上 token 价格 18 个月降约 280×（Jevons 悖论下总支出反升），test-time compute 把工作推向推理，NVIDIA Dynamo（Rust 内核+NATS）与 NVL72 把机架级多节点 serving 设为默认部署单元。真正对个人/小团队开放、且尚未被资本锁死的空白集中在：**跨引擎 KV 互操作与传输协议、中性性能/成本回归 CI、NPU/端侧运行时、量化精度回归、投机解码统一 draft 基建、长上下文召回门禁**——通用引擎与通用内核已对个人关闭。

---

## 二、Gap Map（跨方向缺口地图）

> 标注【饱和/半饱和/空白】并给出「为什么是真实缺口」。饱和=已被资本/事实标准锁死；半饱和=有论文/原型但无产品级落地；空白=无人系统解决。

### 量化
1. **MoE 量化（路由/共享专家感知）——【空白】** 为什么真实：GPTQ/AWQ/FP8 均为稠密假设，MoE 的稀疏激活专家与高频共享专家被同等对待；DeepSeek 671B 已证明 MoE 是成本最优形态，却无路由感知量化工具与专用基准。
2. **亚 4-bit（2–3 bit）生产算子——【半饱和】** 为什么真实：AQLM/QuIP#/KVQuant 算法存在、QServe 已到 W4A8KV4，但缺低反量化开销的生产算子，Blackwell FP4 之外无 2-bit 真提速。
3. **量化模型微调（QFT）——【空白】** 为什么真实：QLoRA 只覆盖 NF4，INT4/FP8/FP4/MX 的微调路径空缺；QAT/QFT 与 PTQ 融合被点名但无工具。
4. **跨硬件格式统一 + 量化精度回归 CI——【空白】** 为什么真实：GGUF/GPTQ/AWQ/EXL2/MXFP4 各一套，缺格式无关中间表示；量化回归隐性、生产才暴露，lm-evaluation-harness 是评测工具而非回归门禁。

### 剪枝/稀疏
5. **稀疏注意力内核标准化——【空白】** 为什么真实：NSA/MoBA/MInference 已量产兑现 3–11×，但缺统一 API 与多后端开源内核，FlashAttention block-sparse 之外无标准实现。
6. **稀疏感知 serving（调度/布局/负载均衡）——【空白】** 为什么真实：稀疏模型「能跑但不快」，serving 层无稀疏感知调度，2:4 绑定 NVIDIA 后跨硬件短期无解。
7. **结构化剪枝端到端流水线 + 稀疏量化联合工具链——【半饱和】** 为什么真实：LLM-Pruner/Sheared LLaMA 算法可行、Sparse-Marlin 叠加 2:4+INT4 存在，但缺一键导出与真机速度验证闭环，DeepSparse/SparseML 已归档。

### 算子/DSL/编译器
8. **跨硬件高性能注意力内核标准——【空白】** 为什么真实：各厂商各一套（CUDA/ROCm/Ascend），MLIR 收敛表示但内核层未统一，异构 NPU 缺 Triton 级算子层。
9. **MLA 内核开源——【半饱和】** 为什么真实：FlashMLA 仅 Hopper、sparse MLA 缺跨硬件参考实现，MLA 已成标准架构却无中立实现。
10. **Triton serving 实时 JIT/autotune 开销 + 迁移式 autotuner——【空白】** 为什么真实：serving 编译/调优开销成新瓶颈，缺跨 shape/硬件的学习式 cost model，AOT/缓存无人系统解决。

### 分布式推理
11. **KV cache 跨节点传输开源标准——【空白】** 为什么真实：Mooncake transfer engine 开源但 KV store 与协议闭源，「Tensor-Centric KV Cache Transfer Protocol」刚起步未收敛，各框架碎片化。
12. **多机 MoE 的 EP 感知可插拔调度器——【空白】** 为什么真实：跨节点 all-to-all 成新瓶颈，缺开源、可插拔的专家并行感知调度器，FP8 通信已普及但调度仍各厂商私有。
13. **PD 分离统一 benchmark——【空白】** 为什么真实：PD 已成默认架构但「各厂商各说各话」，goodput/SLO 定义不统一，缺可复现模拟器。

### Serving
14. **跨引擎 KV 互操作/迁移/checkpoint——【空白】** 为什么真实：PagedAttention 与 RadixAttention 格式不兼容，跨引擎、跨池、跨会话 KV 复用无从谈起；长上下文 Agent 会话缓存层缺持久化标准。
15. **SLO 感知调度评测基准与模拟器——【空白】** 为什么真实：优化目标已从吞吐转向 goodput/SLO，但缺可复现评测，优先级/抢占/公平语义在 PD 分离下无统一方案。
16. **长上下文逐请求成本剖析/监控——【空白】** 为什么真实：量化+稀疏+offload 组合成标配但成本不可观测，稀疏注意力精度回退缺可观测性。
17. **多模型显存池化控制面——【半饱和】** 为什么真实：Prism/CrossPool/Aegaeon 存在但冷热权重换入换出缺实用控制面，小团队多模型池化无成品。

### 投机解码
18. **统一 draft 基建 / draft-as-a-service——【空白】** 为什么真实：MTP 已成主流 draft 但部署碎片化，缺跨引擎可插拔 draft 接口与验证树接口；SpecForge/FastDraft 只覆盖训练侧。
19. **batch 自适应投机开关 + 负结果基准——【半饱和/空白】** 为什么真实：batch 收益衰减无诚实评测（Spec-Bench/SPEED-Bench 刚起步），「何时不划算」的成本模型基准无人做。
20. **长上下文 × MoE 投机（专家预路由/KV 预取）+ PD 分离下 draft 放置——【空白】** 为什么真实：投机解码与 MoE、长上下文、PD 分离耦合滞后，draft 跨节点放置与流水无人系统解决。

### 长上下文/高效注意力
21. **decode 侧稀疏/压缩内核——【半饱和】** 为什么真实：MInference/NSA 主要优化 prefill，decode 带宽仍主导，decode 侧稀疏内核未与 FlashAttention 融合。
22. **KV 压缩召回门禁 / quality-per-dollar——【空白】** 为什么真实：KV 压缩「有损无召回保证」，NoLiMa/LOFT 已转向聚合/推理评测却无人做成部署 CI；稀疏 index 在线选择开销（IndexCache）仅早期。
23. **混合注意力内核运行时与负载均衡——【空白】** 为什么真实：Jamba/Samba/Griffin 混合 dense/SSM/sparse 路径，但缺统一内核运行时与跨路径调度。

### 边缘/端侧
24. **跨厂商 NPU 统一运行时（能力注册 + 自动降级）——【空白】** 为什么真实：各厂商 delegate 闭源且能力不可查，MLX 只走 Metal、llama.cpp 无视 NPU；NPU 原生投机解码与量化 KV 未成体系。
25. **端侧 128K 长上下文 + 能量/热感知调度——【空白】** 为什么真实：端侧 KV 内存膨胀未产品化解决，每 token 能量/功耗感知调度器缺开源实现，浏览器无 NPU 访问。

### 行业/OSS 生态
26. **异构设备统一 serving（TPU+GPU+ASIC）——【空白】** 为什么真实：Meta 已跑 TP/CP/EP 跨设备，NVIDIA-first 引擎无优化 AMD/Intel/Apple 的激励，ZML/llm-d 早期、HexGen-2 仅研究。
27. **中性推理性能回归与成本观测 CI——【空白】** 为什么真实：引擎 ship 回归（vLLM 版本升级掉速）且不 benchmark 竞品，Artificial Analysis/SemiAnalysis 是付费黑盒，开源可复现替代缺失。
28. **ROCm/AMD 可靠性层——【空白】** 为什么真实：vLLM/Triton 在 ROCm 上 bug 频出、ROCm 是二等公民，稳定非 NVIDIA 后端被严重 underserved。
29. **多模态/扩散模型 serving——【半饱和】** 为什么真实：vLLM Omni 刚起步，VLM/视频/音频 serving 优化严重不足，是下一个增量战场但门槛已抬升。

---

## 三、Candidate Directions（候选开源项目方向）

> 选取标准：个人/小团队可启动、有单一锋利价值主张、不与 vLLM/SGLang 正面竞争、非饱和。

### 1. KVBus —— 跨引擎 KV cache 互操作与传输协议
- **name**：KVBus（跨引擎 KV cache 互操作与传输协议）
- **what**：定义一个中立的 KV cache 序列化/传输协议 + vLLM/SGLang/TRT-LLM 的 adapter，实现跨引擎 KV 迁移、checkpoint、跨节点传输与 PD 分离互操作；叠加「面向传输而非存储」的 KV 量化（精度-延迟标准）。
- **why_now**：PD 分离已成默认架构、KV 传输是第一性瓶颈；Mooncake 协议闭源、Tensor-Centric KV Cache Transfer Protocol 刚起步未收敛；PagedAttention 与 RadixAttention 格式不兼容。
- **underserved**：NVIDIA/Moonshot 各自为政、双寡头无互操作激励，缺中性第三方协议。
- **differentiation**：做「KV 领域的 ONNX」——协议 + adapter 层，不绑定引擎/厂商，个人可赢（协议与适配而非重写内核）。
- **feasibility**：中。协议设计 + 序列化 + 适配层是软件工程；难点在量化 KV 的精度-延迟权衡与 RDMA/GDR 集成；可先以跨引擎 checkpoint 互操作做 MVP。
- **risks**：引擎内部 KV 格式无公开 API（需上游合作或逆向）；双寡头可能自研协议反吞并；协议不被采纳则生态价值归零。

### 2. InferCI —— 跨引擎推理性能回归与成本观测
- **name**：InferCI（中性推理性能回归 + 成本观测 CI）
- **what**：持续的 tok/s、TTFT、ITL、$/token、精度（perplexity/benchmark）全矩阵 CI，覆盖 vLLM/SGLang/TRT-LLM/llama.cpp 各版本与 GPU 型号，自动报警性能/精度回归，输出可复现成本模型。
- **why_now**：vLLM 版本升级掉速（issue #49259）、ROCm 二等公民等隐性回归生产才暴露；引擎不自 benchmark 竞品；Artificial Analysis/SemiAnalysis 是付费黑盒。
- **underserved**：无中立、可复现、跨引擎的性能+精度+成本联合回归基准。
- **differentiation**：中立治理 + 开源可复现，把「谁更快」变成 CI 门禁，单一锋利价值主张（契合 OSS 生态摘要的成功共性）。
- **feasibility**：高。纯工程 + 云 GPU 编排，可先从少量模型+引擎矩阵起步。
- **risks**：云 GPU 成本持续；引擎快速迭代使 baseline 漂移；ROCm/多厂商硬件难获取导致覆盖不全。

### 3. QuantCI —— 量化精度回归 CI 与格式无关中间表示
- **name**：QuantCI（量化精度回归 CI + 格式无关 IR）
- **what**：跨格式（GGUF/GPTQ/AWQ/EXL2/MXFP4）格式无关中间表示 + 精度回归测试基建，把 lm-evaluation-harness 接成量化 CI，自动检测某格式/位宽/模型组合的精度漂移，覆盖长上下文/多轮负载。
- **why_now**：AutoGPTQ/AutoAWQ 已归档、能力并入引擎，但缺跨格式/跨硬件精度回归；量化回归隐性、生产才暴露；跨硬件格式统一缺中间表示。
- **underserved**：无量化专用 CI 与格式无关 IR；lm-eval 是评测工具而非回归门禁。
- **differentiation**：格式无关 IR + 自动回归，做「量化的安全网」，不重复造量化算法。
- **feasibility**：高。IR 设计与评测编排是软件工程，可复用 lm-evaluation-harness + KVCache-Factory。
- **risks**：格式作者无统一动机；长上下文/多轮精度覆盖难；IR 可能过于理想化而无人采用。

### 4. DraftHub —— 统一投机解码 draft 基建与 batch 自适应决策器
- **name**：DraftHub（统一 draft 基建 + batch 自适应投机决策器）
- **what**：draft-as-a-service：统一的 draft 加载/验证树/树注意力接口，跨 vLLM/SGLang 可插拔；加 batch 自适应开关（何时开投机、选哪个 draft、batch 多大划算），并输出负结果/成本模型基准。
- **why_now**：MTP 随预训练一体化成为主流 draft 但部署碎片化；batch 收益衰减无诚实评测；SpecForge/FastDraft 只覆盖训练侧，运行侧统一决策层缺失。
- **underserved**：缺统一 draft 基建与 batch 自适应决策器；「何时不划算」的负结果 benchmark 无人做。
- **differentiation**：做决策层 + 可插拔接口，不训练新 draft 模型，补足诚实评测。
- **feasibility**：中。需对接各引擎 spec decode 插件点；决策器可先离线 profile 驱动。
- **risks**：引擎 spec decode 接口变动频繁；收益依赖模型/负载难量化；可能被引擎内建吸收。

### 5. RecallGate —— 长上下文 KV 压缩召回门禁与成本可观测
- **name**：RecallGate（KV 压缩召回门禁 + quality-per-dollar 可观测）
- **what**：为 KV 压缩（SnapKV/H2O/PyramidKV/KV 量化）加召回回归门禁（NoLiMa/LOFT 作 CI），输出 quality-per-dollar 指标；逐请求剖析长上下文成本（每 token 的 KV 读取/命中/淘汰）。
- **why_now**：KV 压缩是生产最成熟路径但「有损无召回保证」；NoLiMa/LOFT 已转向聚合/推理评测却无人做成部署 CI；长上下文逐请求剖析工具缺失。
- **underserved**：召回保证 / 可观测 / quality-per-dollar 三层均空白。
- **differentiation**：把评测从论文指标变成部署门禁 + 计费视角，中立插件式接入 vLLM/SGLang。
- **feasibility**：高。评测编排 + 剖析钩子是工程，复用 KVCache-Factory 与 RULER/LongBench。
- **risks**：召回「正确性」对推理类任务难定义；依赖引擎 KV 接口稳定性；指标不被采纳则沦为论文工具。

### 6. MoEQuant —— 路由感知 MoE 量化
- **name**：MoEQuant（路由感知 + 共享专家感知的 MoE 量化）
- **what**：针对 MoE 的量化方法与工具：区分路由专家/共享专家/门控网络的不同位宽策略，专家稀疏激活感知的逐专家量化，配套专用基准，并与 EP 调度联动做位宽分配。
- **why_now**：MoE（DeepSeek 671B）是成本最优形态，但 GPTQ/AWQ/FP8 均为稠密假设；MoE 量化缺路由/共享专家感知方法与专用基准，是最难量化目标。
- **underserved**：共享专家（高频激活）与路由专家（稀疏激活）被同等对待，无 MoE 结构感知量化工具。
- **differentiation**：不做通用量化，聚焦 MoE 结构感知 + 与 EP 调度联动。
- **feasibility**：中。方法研究 + 基于 vLLM MoE 支持的工具，需真机验证。
- **risks**：MoE 模型获取/推理成本高；DeepSeek 原生 FP8 已部分覆盖；可能被 vLLM/SGLang 内建吸收。

### 7. PD-Bench —— 统一 prefill/decode 分离基准与模拟器
- **name**：PD-Bench（PD 分离统一 benchmark + 可复现模拟器）
- **what**：统一的 PD 分离评测：goodput、TTFT/TPOT 分布、KV 传输延迟、跨池调度公平性；配一个可配置模拟器，不烧真 GPU 即可复现调度策略对比。
- **why_now**：PD 分离成默认架构但「各厂商各说各话」；SLO 感知调度缺可复现评测与模拟器；goodput 定义不统一。
- **underserved**：无中立 PD 分离基准，引擎自报数字不可比。
- **differentiation**：中立 + 可复现模拟器（先模拟后真机），标准化 goodput/SLO 定义。
- **feasibility**：高。模拟器 + 指标定义是纯软件，真机可选。
- **risks**：模拟器保真度受质疑；各家不承认第三方数字；可能被引擎团队主导的社区边缘化。

### 8. NPURT —— 端侧跨厂商 NPU 能力注册 + 自动降级运行时
- **name**：NPURT（跨厂商 NPU 能力注册 + 自动降级 + 功耗感知调度）
- **what**：在可移植运行时（ExecuTorch/ORT/LiteRT）之上加一层「NPU 能力注册表 + 自动降级路由」：探测 Hexagon/ANE/NeuroPilot 的算子覆盖与精度，不可加速层自动回落 CPU/GPU，配每 token 功耗感知调度。
- **why_now**：NPU 走向 LLM 一等公民但跨厂商统一运行时缺失；能力注册+自动降级是空白；MLX 只走 Metal、llama.cpp 无视 NPU；端侧 SLM 缺开源 NPU 运行时。
- **underserved**：各厂商 delegate 闭源且能力不可查，无统一的「能跑什么、多快、多省电」注册与降级。
- **differentiation**：不做内核（硬），做「能力注册 + 降级 + 功耗调度」控制面（软），解耦厂商 SDK 差异。
- **feasibility**：中。依赖各厂商 delegate 的 int8/int4 算子覆盖探测；先以 Android（QNN/NeuroPilot）+ iOS（ANE/CoreML）单点切入。
- **risks**：厂商 SDK 闭源、能力探测黑盒；NPU 精度/算子上限可能撑不起 LLM；设备碎片化测试矩阵巨大。

---

## 附：已饱和/关闭区域（不建议个人切入）

- 权重 INT4 PTQ（GPTQ/AWQ/GGUF Q4）、W8A8 FP8 基线 —— 饱和（能力已并入引擎）。
- 通用单机 dense serving 引擎（vLLM/SGLang 双寡头，均已资本化）—— 饱和。
- PagedAttention / continuous batching / RadixAttention —— 饱和（事实标准）。
- 通用注意力内核（FlashAttention/CUTLASS/CuTe）与 2:4 硬件绑定稀疏 —— 饱和/关闭（NVIDIA 锁死）。
- RoPE 缩放（YaRN/LongRoPE/LongRoPE2）—— 饱和。
- 独立量化/稀疏转换工具（AutoGPTQ/AutoAWQ/DeepSparse/SparseML）—— 已归档关闭。
- 独立小模型 draft（Medusa/EAGLE）—— 半饱和（被 MTP 预训练一体化取代）。
- TGI —— 停滞，被 vLLM/SGLang 分流。
- 通用内核与通用引擎 —— 对个人/小团队已关闭。
