# wLLM: World and Multimodal Model Serving — 工程规格 v1.0

> 权威来源：`ref/goal_and_plan_new_wLLM.md`（2026-07-24 用户令）。本文件是其工程化压缩版 + 集群适配。
> 取代 TESSERA_SPEC.md（其 IR/探针/planner 设计并入本规格；**Tessera 仅保留为 planner 算法名**）。
> 命名纪律：项目一切交付物（代码/README/注释/文档/测试）**不出现两个上游参考项目的名字**；
> 内部一律称 `upstream-a`（agent-guided 部署 harness 参考）与 `upstream-b`（kernel/静态图推理引擎参考, Apache-2.0）。
> 法律底线：凡实际派生/改写 upstream-b 代码的文件，保留 Apache-2.0 许可证头与 NOTICE（许可证合规不等于品牌宣传）。

## 1. 定位与命名

- **wLLM**（World and Multimodal Model Serving）：面向 world model / video model / MLLM / WAM 的统一自动部署优化框架。README 首屏必须声明 "wLLM is not an LLM-only engine"。
- 五大组件：**wGraph**（类型化有状态层级 IR）· **Tessera Planner**（规则+求解器+成本模型+agent 混合搜索）· **wRuntime**（分布式执行/流式调度/恢复）· **wKernels**（高性能算子与静态图后端，整合改造 upstream-b）· **wBench**（正确性与性能验证）。
- 用户面：`wllm doctor / init / optimize / serve`；Python `Application.from_callable`。
- 目标函数：Π* = argmin_{Π∈L(P,C)} E_w[αL + βG + γM + δK]（L 延迟、G gap/deadline 违约、M 显存、K GPU-秒；L(P,C) 为语义合法方案集）。核心贡献 = **Verified Semantic Deployment Synthesis**。

## 2. 两个上游参考的取材（整合不照搬；均不具名）

| | upstream-a（harness） | upstream-b（kernel 引擎, Apache-2.0） |
|---|---|---|
| 继承思路 | reference→IR→合法变体→正确性验证→真实测量；chunk-periodic 状态分类；TTFO/持续率/平滑度三指标分离 | 稳定小 API、显式硬件 dispatch、声明式 WEIGHT_SPEC、calibration cache、静态 CUDA Graph、C ABI、冷/暖/热三相契约、deployment fingerprint |
| 在 wLLM 中的落点 | wGraph 语义 + wBench 验证环思想 + Tessera 测量门控 | wKernels L3 后端（§11：修改完善后整合其代码，保留许可头）+ wRuntime 冷暖热契约与 fingerprint |
| 明确不继承 | 用户须写 frontend/adapter/共享内存契约；"≥10 变体测完"硬规则；单一 conda 环境 | 每模型 800-1200 行手工前端成为用户前置成本 |

## 3. 四级接入（产品核心：门槛递进）

- **L0 Opaque**：既有 server / WorldFoundry runner / 官方 CLI 子进程 → 优化 placement/replica/流水/路由/批处理/环境生命周期。目标覆盖绝大多数 runnable entry。`wllm optimize --model worldfoundry:<id> --workload w.yaml --hardware auto`。
- **L1 Pipeline**：`Application.from_callable(run, example_inputs=...)`（<30 行集成）→ 组件共置/分离、组件级流水、state placement、跨 chunk overlap。
- **L2 Model-aware**：可见 transformer block/diffusion step/KV/MoE/causal VAE → TP/SP/CP/PP/EP、CUDA Graph、attention backend、KV 量化、step-pipeline。
- **L3 Kernel-native**：wKernels（upstream-b 改造）后端：pointer-only forward、WEIGHT_SPEC、calibration、整图 capture、C ABI。**可选的最后一步**，绝不是用户前置要求。

## 4. wGraph（IR）

- Region：Sequential / Parallel / Autoregressive / Diffusion / Flow / ChunkRollout / HierarchicalRollout / MultiAgent / Feedback / **Environment / Composition**。循环语义一等公民：AR h_{t+1},y_t=f(h_t,y_{<t},c)；Diffusion z_{k-1}=Φ(z_k,c,k)；WM rollout s_{t+1}=F(s_t,a_t)；WAM a_t=π(o_t,h_t), h_{t+1}=U(h_t,o^real_{t+1})。
- State 类型：ImmutableSession / RecomputableFeature / KV / Recurrent / RollingContext / Stochastic / FeedbackCritical / DeadlineBound / MultiAgent；字段 scope/ordered/recomputable/migratable/forkable/owner/max_staleness_ms/deadline_policy/memory_bytes/**verified**（探针实证后才可为 true，planner 只消费 verified）。
- Stream 类型：Token/Latent/Frame/Audio/Action/Control/MultiView；字段 chunk_size/rate_hz/variable_rate/timestamped/**bounded_queue/backpressure(block|drop_oldest|coalesce|reject)**/deadline_ms/sync_group。
- Quality contract：exact（默认）| bounded_degradation（显式 `--allow-approximate` 才启用近似变换）。

## 5. 语义恢复（四路融合，contract 落盘可审计）

静态分析（call graph/loop/shape/cache/sampler/mask）→ 动态 tracing（module hooks/CUDA event/NVTX/torch.profiler/可选 FX）→ agent 提议（非标准语义命名假设+置信度）→ **反事实探针裁决**（reset/recompute/reorder/delay/fork/migration）→ `contracts/<model>.yaml`（sidecar，含 probe_evidence 指针）。

## 6. Tessera Planner（预算受控搜索，不设"测完全队列"硬规则）

Step1 真实 baseline（warmup 后：stage latency/显存/利用率/传输/同步/质量）→ Step2 按 region+contract 产合法变换（placement/共置/分离/stage overlap/continuous batching/TP-SP-CP-PP-EP/CUDA Graph/attention backend/KV placement/精度/step reduction/pruning）→ Step3 约束过滤（OOM/架构不支持/不可迁移 state/feedback 过期/带宽/静态 shape/质量 contract）→ Step4 低成本估计（microbench τ 表 + 通信模型 + 显存模型 + 历史 plan 库 + 简单 learned 成本模型）→ Step5 **successive halving 逐级实测**（极短 workload 全测 → 淘汰 → 加长 → Pareto 前沿完整稳定性测试）。
- 用户预算：`wllm optimize --budget 20m|2gpu-hours|thorough`。
- 输出 Plan A(min TTFT/首帧) / B(max 持续率) / C(min GPU 成本) / D(deadline-safe) / E(bounded-quality)。
- Exact / Approximate 两张变换表严格分离。

## 7. wRuntime

- 进程模型：coordinator + 每 GPU(组) 长驻 worker + **每依赖环境独立 subprocess**（不强求单一环境——异构模型各带各的 env）。
- 通信 v1：控制 UDS；小数据 shm ring；GPU 张量 torch.distributed/NCCL；跨环境 shm/CUDA IPC。v2：TensorDescriptor 数据面（same-GPU pointer > CUDA IPC > NCCL P2P > pinned host）。
- **有界队列 + 背压**（每 stream 必填 queue_capacity/overflow/deadline；机器人 action queue=1, reject, stale=never_execute）。
- **冷/暖/热三相契约**（继承 upstream-b）：COLD 加载/选后端/分配 → WARM 校准/编译/capture/shape bucket → HOT 只更新 buffer 内容+replay，禁止 recapture/allocate/rebind。
- Deployment fingerprint：source commit/ckpt hash/config/input schema/硬件/驱动/torch/后端版本/精度/shape bucket/stage DAG/state layout——关键字段变化即失效。
- 自动回退链：optimized plan → last-known-good → reference plan（用户永远有正确路径）。
- State snapshot（KV/rolling world state/session 元数据/action history 周期快照）；不可恢复态崩溃后显式重启 session。

## 8. wBench（五级验证 + 按家族指标）

A 结构（无环/state 唯一 owner/shape/deadline 合法）→ B 数值（fixed seed latent allclose/logits 容差/action trace 相等）→ C 应用质量（MLLM bench/视频质量/WM dynamics/机器人成功率）→ D 压力（动态 shape/长跑/队列溢出/reset/session churn/OOM/worker crash）→ E 故障（杀 worker/输入丢失/超时/ckpt 缺失/state 恢复失败）。生产化：shadow → canary 小流量 → 扩大；机器人新 plan 仅 simulator 验证。
指标族：MLLM(TTFT/inter-token/req/s/首音频包/AV 同步) · Video(首帧/E2E FPS/p95 帧间隔/持续 FPS/VAE 占比/显存/GPU-s per video) · WM(action-to-first-state/rollout steps/s/horizon scaling/branch 吞吐) · WAM(o→a p50-95-99/deadline miss/staleness/idle 占比/成功率)。

## 9. 里程碑（用户日历）

**Alpha（3 天）**：WorldFoundry catalog 读取；runnable entry 皆可 opaque；本地 PyTorch callable；wGraph v0；baseline profiler；exact placement/pipeline 搜索；reference fallback。模型（以可得性审计为准）：Cosmos3-Nano / SANA-Streaming / Wan2.1-2.2 / OpenVLA / Qwen3-Omni（或 Qwen2.5-Omni 替补）。**性能门槛：≥3 模型自动找到优于 baseline 的 plan；代表 workload 中位 E2E ≥1.3×；自动方案 ≥专家 80-90%；exact plan 零功能回退。使用门槛：WF 模型 10 min 首个 baseline；普通 pipeline <30 行接入；不写 frontend/adapter。**
**Beta（2 周）**：AR/diffusion/hybrid、causal streaming、TP/SP/CP/PP、CUDA Graph、shape bucket、多环境 worker、质量 contract、plan cache；深适配 10-12 模型；24h soak / crash 恢复 / OOM 回退 / fingerprint 失效正确。
**1.0（9-12 周）**：Feedback/MultiAgent region、WAM deadline scheduler、approximate（FP8/NVFP4/KV 量化）、可选 L3 native 后端、在线 plan 路由、N+A 双硬件。覆盖 WorldFoundry 大部分模型（范围参照，不需要特别声明）。

## 10. 目录结构（原创）

```
wllm-infra/
├── WLLM_SPEC.md
├── wllm/                      # pip 包（原 tessera/ 规划并入）
│   ├── graph/     regions.py states.py streams.py quality.py program.py
│   ├── capture/   static_scan.py trace.py hypothesis.py probes.py lifter.py
│   ├── contracts/ schema.py store.py
│   ├── planner/   transforms.py rules.py constraints.py cost_model.py search.py pareto.py budget.py
│   ├── runtime/   coordinator.py worker.py spec.py stage_pipeline.py transport.py
│   │              state_manager.py lifecycle.py fingerprint.py fallback.py
│   ├── backends/  torch_local/ worldfoundry/ subprocess_cli/ vllm/ sglang/ native/(=wKernels 接口)
│   ├── kernels/   # upstream-b 改造整合（保留 Apache-2.0 头）
│   ├── verify/    structural.py numerical.py quality.py stress.py fault.py
│   ├── profiling/ microbench.py report.py
│   ├── serve/     http.py openai_compat.py ws.py cli.py(doctor/init/optimize/serve)
│   └── apps/      # 场景应用（含示例 workload）
├── contracts/<model>.yaml     # sidecar
├── tests/  benchmarks/(results 留盘)  inventory/  slurm/  logs/  docs/  checkpoints/
├── upstream-a/  upstream-b/   # 只读参考镜像
└── worldfoundry-upstream/     # 上游镜像
```

## 11. 集群适配与红线（不变）

sbatch-only、GPU 双上限 72/112、利用率红线（常驻 worker 配 keepalive 或短时限任务化）、复提门控、/verify 过闸；权重 ModelScope 优先；H200：FP8 可用、NVFP4 不可用（Blackwell-only → fallback FP8/BF16 路径必须存在）；env 用共享 FS python 建 .venv（绕 conda 锁），多环境 worker 天然匹配本集群多 env 现实。
