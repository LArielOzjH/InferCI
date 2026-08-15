# 推理优化 AI Infra 深度调研 —— 开源项目切入点

> 目标：系统地调研「推理优化」方向的 AI Infra（量化/剪枝稀疏/算子 DSL/分布式/serving/投机解码/长上下文/端侧硬件），找到对个人/小团队**真正可赢**的开源项目切入点。
>
> 方法：用 multi-agent workflow 展开 —— 10 个并行研究子代理深潜 10 个方向（联网查论文/公司博客/GitHub/行业报道），1 个综合代理产出「现状总览 + 缺口地图 + 候选方向」，1 个红队代理对候选方向做「能不能活下来」的终审并排序。
>
> 产出物：`` 目录下 12 份详细报告（约 2800 行），本文是它们的索引与最终结论。

---

## 一、总论（一句话结论）

**通用内核与通用引擎对个人/小团队已经关闭。** vLLM / SGLang 双寡头均已资本化（Red Hat / PyTorch Foundation vs RadixArk 的 $100M 融资，NVIDIA/AMD/Intel 参投），注意力内核被 FlashAttention/CUTLASS/CuTe 吃透，量化/稀疏/投机解码作为「特性」被引擎逐一吸收（AutoGPTQ / AutoAWQ / DeepSparse / SparseML / FlashInfer 相继归档或并入）。

因此小团队唯一有**不对称优势**的战场不是「做得更快」（那是在 incumbent 的 moat 里正面对抗），而是「做得**中立、可复现、被信任**」。因为 vLLM/SGLang/TRT-LLM **永远不会公平地 benchmark 竞品、永远不会自我批判自己 ship 的 KV 压缩精度回退、永远不会公开自己的成本回归**——这是结构性利益冲突，不是意愿问题，所以这个空白**持久且不可自供**。

**推荐第一切入点：InferCI（中性推理性能回归 + 成本观测 CI），并以 RecallGate 的「长上下文 quality-per-dollar」作为第一个差异化质量垂直。**

---

## 二、为什么是这个结论（三关过滤出来的）

调研从 10 个方向收敛出 8 个候选，再用「五问判生死」红队过滤：

1. **缺口真实且持久吗？**（是「没人做」还是「没人需要」）
2. **小团队相对 incumbent 真能赢吗？**（在核心赛道对抗，还是在它们无动机/无能力/无利益去做的侧翼）
3. **谁会用它？痛点是什么？time-to-value 多快？**
4. **护城河是什么？**（代码易复制 / 方法易吸收 / 标准易被接管 / 信任·中立性·不可自供）
5. **生态风险：会被大项目顺手吸收吗？**

三个「必死陷阱」直接否决/降级了一批方向：

| 陷阱 | 命中方向 | 为什么死 |
|---|---|---|
| 标准陷阱 | KVBus（跨引擎 KV 协议）、QuantCI 的「格式无关 IR」 | 标准只有在双端点都采纳时才有价值，双寡头是竞争者无互操作动机；「ONNX 是标准，不是生意」 |
| 研究前沿陷阱 | MoEQuant（路由感知 MoE 量化） | 方法一旦有效立刻被 vLLM/SGLang/TRT-LLM 吸收为 kernel+config；在 research frontier 跟 funded lab 抢 = 必输 |
| 内部吸收陷阱 | DraftHub（统一投机解码 draft 基建） | MTP 随预训练一体化、自适应投机已被引擎内建，你在别人 moat 里抢活 |

---

## 三、最终排序（红队终审）

| 排名 | 方向 | 分数 | 红队判定 | 一句话 |
|---|---|---|---|---|
| 🥇 1 | **InferCI** — 中性推理性能回归 + 成本观测 CI | 8.5 | 通过 ✅ | 唯一一个 incumbent 结构性无法自供的空白，中立性即 moat |
| 🥈 2 | **RecallGate** — KV 压缩召回门禁 + quality-per-dollar | 8.0 | 通过 ✅ | 长上下文/Agent 下「省完显存还对不对、每美元买多少质量」 |
| 🥉 3 | NPURT — 端侧跨厂商 NPU 能力注册 + 自动降级 | 6.5 | 存疑 ⚠️ | 真缺口真持久，但买方模糊 + NPU 扛 decode 前提未证实，长期大 bet |
| 4 | KVBus — 跨引擎 KV 互操作/传输协议 | 6.0 | 存疑 ⚠️ | 「KV 的 ONNX」，无财团背书的标准归零 |

**否决 ❌**：DraftHub（被内建吸收）、MoEQuant（研究前沿）。
**合并 🔀**：QuantCI（精度回归 → 并入 InferCI 精度模块）、PD-Bench（SLO/模拟器 → 并入 InferCI 路线图）。

---

## 四、第一切入点详解：InferCI

**一句话 pitch**：把「谁更快、谁更便宜」从黑盒报告变成可复现的 CI 门禁——引擎每次升级，先过我这关。

**做什么**：持续输出 tok/s、TTFT、ITL、TPOT、$/token、精度 的全矩阵 CI，覆盖 vLLM/SGLang/TRT-LLM/llama.cpp 各版本 × GPU 型号，自动报警性能/精度回归，输出可复现的成本模型。

**为什么能赢（四层判断）**：

1. **结构性无法自供**——引擎永远不会公平 benchmark 竞品、不会公开成本回归、不会自我批判 KV 压缩质量（利益冲突，非意愿问题）→ 缺口持久且不被吸收。
2. **小团队有不对称优势**——不在 serving/内核上跟它们抢，而是在其上做 instrumentation + 云 GPU 编排 + 可复现基线，纯工程、无内核竞争、TTV 最快（几周出首版矩阵）。
3. **护城河是信任/中立性/历史语料**，而非代码或方法——「公平仲裁者」地位只能靠长期中立行为累积，incumbent 因冲突无法复制，付费分析机构（SemiAnalysis/Artificial Analysis）因黑盒不愿自证。
4. **是家族入口不是单点**——InferCI 是「信任层」母体，RecallGate / QuantCI / PD-Bench 都是其模块。

**90 天路线图**：
- **D1–14**：锁定 3 模型（Llama-3.1-8B/70B、Qwen2.5）× 2 引擎（vLLM、SGLang）× 1 GPU（H100），搭可复现 harness（Docker/种子/prompt 集固定），产出 tok/s / TTFT / ITL / TPOT 基线。
- **D15–30**：加 $/token 成本模型 + 自动回归告警；复现 vLLM 最近 4–6 个 release 的真实「升级掉速」点；发布首个公开 dashboard + 一篇「版本 pin 建议」。
- **D31–60**：扩展引擎矩阵（+TRT-LLM、llama.cpp）+ GPU 型号（+A100/L40S）+ 精度维度（量化格式 vs 基线精度回退），QuantCI 作为精度模块并入。
- **D61–90**：建 ROCm/AMD 有限覆盖 + 长上下文评测钩子（RULER/LongBench，为 RecallGate 铺路）；开源 CI 模板；建立中立治理，沉淀信任。

**主要风险**：云 GPU 成本持续；ROCm/多厂商硬件难获取致覆盖不全；引擎迭代快 baseline 漂移需持续维护；变现偏「社区权威资产」（需 hosted CI / 认证版本 / 咨询兑现）。

---

## 五、备选方向速览

**RecallGate（第 2，可作第一差异化垂直）**：给 KV 压缩（SnapKV/H2O/PyramidKV/量化）装质量门禁——复用 RULER/LongBench/NoLiMa/LOFT 做召回回归 CI，定义 quality-per-dollar 指标，逐请求剖析 KV 读取/命中/淘汰成本。硬骨头是「召回正确性」对推理/聚合任务难形式化（执行风险 > 市场风险），故建议作为 InferCI 的长上下文模块先行。

**NPURT（第 3，长期 bet）**：端侧跨厂商 NPU 的「能力黄页 + 自动降级」运行时（探测 Hexagon/ANE/NeuroPilot 的算子覆盖/精度/功耗，不可加速层自动回落 CPU/GPU）。真缺口真持久、吸收风险低，但端侧免费 OSS 惯例致变现难、且 decode 仍主要靠 CPU/GPU 的前提未证实，不适合第一切入点。

---

## 六、调研文件索引

**10 个方向深潜报告**（``）：
- `01_quantization.md` — 量化（PTQ/KV/FP8/FP4-MX/2bit/硬件/生态）
- `02_pruning_sparsity.md` — 剪枝与稀疏（2:4/稀疏注意力 NSA·MoBA·MInference/内核）
- `03_operator_dsl_compiler.md` — 算子/DSL/编译器（Triton/TVM/CUTLASS/FlashAttention/ThunderKittens/MLA）
- `04_distributed_inference.md` — 分布式推理（TP/PP/EP/序列并行/PD 分离/Mooncake）
- `05_serving_systems.md` — Serving 系统（PagedAttention/RadixAttention/连续批/KV 管理/调度）
- `06_speculative_decoding.md` — 投机解码（EAGLE/Medusa/MTP/树验证/自投机）
- `07_long_context_attention.md` — 长上下文与高效注意力（SSM/ring/稀疏注意力/KV 压缩）
- `08_edge_ondevice_hardware.md` — 边缘/端侧/异构硬件（GGUF/MLX/ExecuTorch/NPU）
- `09_industry_landscape.md` — 行业格局与趋势（NVIDIA/Meta/DeepSeek/初创/成本趋势/专家判断）
- `10_oss_ecosystem_gaps.md` — OSS 生态盘点与「什么样的 OSS 能赢」

**综合与终审**：
- `synthesis_state_of_art.md` — 现状总览 + 缺口地图（30 条，标注饱和/半饱和/空白）+ 8 候选方向
- `synthesis_final_ranking.md` — 红队终审：五问判生死 + 逐方向检验 + 排序 + 最终推荐

---

## 七、给「想打磨一个好开源项目」的一句话

> **不做更快的内核，不做更大的引擎，做谁都做不了、谁都不敢做、谁做了都不被信的那件事——给推理一个可复现的、中立的、被信任的观测与质量/成本门禁。**

先以最刚需的「vLLM 版本 pin 回归 + $/token」楔入，再用 RecallGate 的 quality-per-dollar 打 Agent/长上下文这个增长最快的买方，逐步长成「推理可观测性与质量信任层」这一整条侧翼。
