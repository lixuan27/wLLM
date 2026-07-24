# wLLM: Agent-Operable Multimodal Deployment Optimizer — 工程规格 v2.0

> 权威来源：`ref/goal_wLLM_724{,b,c}.md` + `ref/goal_and_plan_wLLM_724{,b}.md`（2026-07-25 用户总任务更新令）。
> v1.0（World and Multimodal Model Serving）的五大组件（wGraph/Tessera/wRuntime/wKernels/wBench）全部保留，
> v2.0 在其上确立**产品定位与控制面**：wLLM 不是又一个推理引擎，而是
> **由 Agent 一句话触发、但不依赖 Agent 正确性的多模态部署优化基础设施**。
>
> 命名纪律（红线）：项目一切交付物（代码/README/注释/文档/测试/配置）**不出现任何上游参考项目名词**。
> 内部只用中性代号：`upstream-a`（agent-guided 部署 harness 参考）、`upstream-b`（kernel/静态图引擎参考, Apache-2.0）、
> `upstream-c`（模型目录/评测底座参考）。外部引擎一律经环境变量绑定（`docs/ENGINES.md`），
> push 前必须 `scripts/release_gate.sh --all` 全树 PASS。
> 法律底线：凡实际派生/改写 upstream-b 代码的文件保留 Apache-2.0 许可证头与 NOTICE（合规≠品牌宣传）。

## 1. 定位（v2 更新的核心）

- **一句话交互**：用户对 coding agent 说「用 wLLM 优化这个项目，4×H200，优先首帧延迟，质量不降」。
  Agent 只做四件事：解析意图为 typed spec → 调 `wllm` CLI/API → 展示结果 → （可选）提交 PR。
- **三条产品公理**：
  1. *Agent 只表达意图*——判断优化是否合法、是否更快、质量是否保住的，必须是 wLLM 的
     profiler / planner / verifier，不是 agent 的经验之谈；同一套 CLI 在 CI 无 agent 时必须可独立运行。
  2. *Registry first, compiler second, agent synthesis last*——先匹配已验证 recipe/后端；
     匹配不到再做确定性图变换；最后才允许 agent 生成 adapter（且只能写入隔离目录）。
  3. *一切结论来自真实测量*——编译通过≠成功、静态估计≠成功、agent 宣称≠成功；
     没有 baseline 对照 + receipt 的加速不存在。
- 目标函数不变：Π\* = argmin\_{Π∈L(P,C)} E\_w[αL + βG + γM + δK]（L 延迟、G gap/deadline 违约、M 显存、K GPU-秒）。
  核心贡献 = **Verified Semantic Deployment Synthesis**。

## 2. 上游取材与外部引擎纪律

| 代号 | 继承思路 | 明确不继承 |
|---|---|---|
| upstream-a | reference→IR→合法变体→正确性验证→真实测量；chunk-periodic 状态分类；TTFO/持续率/平滑度三指标分离 | 用户手写 frontend/adapter/reference 的高接入成本；"≥10 变体测完"硬规则 |
| upstream-b | 稳定小 API、显式硬件 dispatch、声明式 WEIGHT_SPEC、calibration cache、静态 CUDA Graph、C ABI、冷/暖/热三相契约、deployment fingerprint | 每模型 800–1200 行手工前端成为用户前置成本 |
| upstream-c | 模型 manifest/资产/证据分层；"catalog 声明 ≠ 实际跑通"的支持分层思想 | 把它当 serving 引擎（它不是） |

**外部引擎绑定**（`wllm/engines/` + `docs/ENGINES.md`）：外部 omni 引擎经 `WLLM_OMNI_ENGINE` 解析包名，
stage-config YAML 用 `__WLLM_OMNI_ENGINE__` 占位符 + `render_stage_config()` 渲染；
模型底座经 `WLLM_SUBSTRATE_ROOT / _JOB_MODULE / _CKPT_ENV / _UNIFIED_ENV` 绑定。
未绑定时 **fail closed**（`OmniEngineNotBound` / RuntimeError），绝不静默降级。

## 3. 控制面（v2 新增，Milestone 16+）

### 3.1 Typed OptimizeSpec（Agent Bridge 的唯一产物）

```yaml
project: .
hardware: {accelerator: auto, count: auto}
objective: {primary: p95_first_output, secondary: [sustained_rate, gpu_seconds]}
quality:  {policy: exact | bounded, budget: null | {metric: lpips, max: 0.05}}
contract: {preserve_existing_api: true, required_modalities: [video, audio]}
budget:   quick | balanced | thorough | {gpu_hours: 2}
```

优化器只读 typed spec，不读自然语言。

### 3.2 命令面（CI 可独立运行）

```
wllm inspect .          # 项目发现 → .wllm/manifests/
wllm baseline .         # 冻结参考基线 → .wllm/baselines/
wllm optimize .         # 规划+搜索+实测 → .wllm/plans/ + receipts/
wllm verify <plan>      # 五级验证（wBench）
wllm apply <plan>       # 生成部署物（不覆盖用户源码）
wllm rollback           # 回退链 optimized → last-known-good → reference
wllm report             # 人读报告 + 机器读 JSON
```

### 3.3 产物目录与 Receipt

```
.wllm/
├── manifests/   project|model|hardware|api-contract.json
├── baselines/   冻结基线（含环境 fingerprint）
├── plans/       候选方案（YAML，可重放）
├── receipts/    每个实测方案一张回执
├── generated/   agent/模板生成物（唯一可写区）
└── reports/
```

Receipt 必含：source/ckpt revision、backend+version、硬件/驱动/torch、passes 列表、
性能分布（p50/p95/p99、cold/warm 分开）、质量结果、**authenticity checks**
（优化真的启用：cache 命中>0、并行组正确、graph capture 成功、无 fallback 日志）、
known_limitations、rollback_target。关键字段变化即 fingerprint 失效。

### 3.4 Backend Capability Registry（把经验变成机器可执行）

```yaml
backend: <id>
models: {exact: [...], compatible: [...]}
passes:
  torch_compile: {quality: exact, conflicts: []}
  cfg_parallel:  {quality: exact, requires: {min_gpus: 2, model_uses_cfg: true}}
  fp8:           {quality: bounded, requires: [calibration]}
invariants:
  forbidden_log_patterns: ["Falling back to ..."]   # 命中即结果无效（fail-closed）
```

| skill 式经验 | wLLM 基础设施形式 |
|---|---|
| "建议开 torch.compile" | 版本化 pass |
| "X 与 Y 不能同开" | conflict graph |
| "看到 fallback 日志就无效" | fail-closed invariant |
| "agent 判断哪个最快" | 实测 + receipt |

### 3.5 支持分层（禁止模糊的 supported: true）

`Discovered → Cataloged → Launchable → Parity-verified → Optimized-1GPU → Optimized-multiGPU → Serving-verified → Production`。
每层晋升都要有落盘证据；BETA_REPORT 的 tier ledger 延续此制度。

## 3.6 数据面四支柱（v2.1 新增，全部自研在库、不依赖外部引擎）

对应四类上游能力的**自研优化实现**（不具名纪律不变；外部引擎绑定仍保留为可选互换项）：

| 支柱 | 模块 | 核心语义 | authenticity 信号 |
|---|---|---|---|
| 组合式图运行时 | `wllm/composite/` | ComponentGraph + 请求=Walk（Seq/Par/Loop/Stream）；placement 是数据；session 状态硬隔离可证 reset；有界流背压；跨请求步级合批（签名分组，逐请求 parity） | 每次调用记录 device；`cross_signature_mixes()==0` |
| omni 分阶段引擎 | `wllm/omni/` | AsyncOmni 契约自研实现：stage-config YAML（占位符自解析）、AR 连续合批调度器 + 整请求生成调度器、输出对象兼容 apps 消费形状；模型 stage 可插拔注册，未注册模型 **fail closed**（echo stub 仅显式请求） | `stats().max_step_batch>=2`；steps/completed 计数 |
| 目录/资产控制面 | `wllm/backends/catalog/` | manifest 导入（root 自动发现）+ **内容级资产就绪检查**（缺文件/短文件=代理 stub/JSON 损坏→显式 blockers） | ReadinessReport.blockers 逐条留证 |
| 优化技术执行器 | `wllm/techniques/` | 技术=声明 spec+authenticity 信号；step 残差缓存 / int8 量化模拟；编排器持 exact 参照对照候选：崩溃/未 engage/超质量预算/形状漂移一律拒绝并给理由，输出 receipt 兼容字段 | `steps_reused>0`、`layers_quantized>0`；缺信号=拒绝 |

铁律：技术候选**不能给自己打分**——参照、对照与预算全部由编排器持有；`wllm.omni` 可经 `WLLM_OMNI_ENGINE=wllm.omni` 绑定为 apps 的默认引擎（契约固定，厂商可换）。

## 4. wGraph（IR，承自 v1）

- Region：Sequential / Parallel / Autoregressive / Diffusion / Flow / ChunkRollout / HierarchicalRollout / MultiAgent / Feedback / Environment / Composition。循环语义一等公民。
- State：ImmutableSession / RecomputableFeature / KV / Recurrent / RollingContext / Stochastic / FeedbackCritical / DeadlineBound / MultiAgent；字段 scope/ordered/recomputable/migratable/forkable/owner/max_staleness_ms/deadline_policy/memory_bytes/**verified**（探针实证后 planner 才消费）。
- Stream：Token/Latent/Frame/Audio/Action/Control/MultiView；bounded_queue + backpressure(block|drop_oldest|coalesce|reject) + deadline_ms + sync_group。
- Quality contract：exact（默认）| bounded_degradation（显式 `--allow-approximate` 才启用）。

## 5. 语义恢复（四路融合，contract 落盘可审计）

静态分析 → 动态 tracing（hooks/CUDA event/profiler）→ agent 提议（假设+置信度）→ **反事实探针裁决** → `contracts/<model>.yaml`（含 probe_evidence 指针）。

## 6. Tessera Planner（预算受控搜索）

Step1 真实 baseline → Step2 按 region+contract 产合法变换 → Step3 约束过滤（给拒绝理由）→
Step4 低成本估计（microbench τ 表+通信/显存模型+plan 历史库）→ Step5 successive halving 逐级实测 → Pareto 前沿。
- 四级来源顺序：**已验证 recipe → 确定性变换 → 测量驱动搜索 → agent 生成 adapter**（仅限 `.wllm/generated/`，不得触碰 reference/benchmark/verifier/threshold）。
- Exact / Approximate 两张变换表严格分离；输出 Plan A(min 首帧)/B(max 持续率)/C(min GPU 成本)/D(deadline-safe)/E(bounded-quality)。

## 7. wRuntime（承自 v1）

coordinator + 每 GPU(组) 长驻 worker + 每依赖环境独立 subprocess；控制 UDS、小数据 shm ring、GPU 张量 NCCL；
有界队列+背压；冷/暖/热三相契约（HOT 禁 recapture/allocate/rebind）；deployment fingerprint；
自动回退链 optimized → last-known-good → reference；state snapshot。
**三种集成模式**：A. API 端点代理（默认，业务代码零改动）；B. import 兼容 shim（一处可回滚 import）；C. 嵌入式自定义图（才需要 reference adapter）。

## 8. wBench（五级验证 + 按家族指标）

A 结构 → B 数值（fixed-seed latent/logits/action 容差；tie-aware token gate）→ C 应用质量 →
D 压力（动态 shape/长跑/队列溢出/session churn/OOM/crash）→ E 故障注入。
指标族：MLLM(TTFT/inter-token/req/s/AV 同步) · Video(首帧/E2E FPS/p95 帧间隔/GPU-s per video) ·
WM(action-to-first-state/rollout steps/s/branch 吞吐) · WAM(o→a p50-95-99/deadline miss/staleness/成功率)。
已沉淀 Verifier Laws（BETA_REPORT）：编译扩散按轨迹发散须实证分类；tie 是仲裁不是发散；
reference=ckpt 声明精度；batching 改变结果但分布可不变。

## 9. 质量工程（v2 新增强制）

- **单元/回归**：`tests/`（pytest 兼容 + 独立 `__main__` runner，登录节点禁跑 → CPU sbatch `slurm/wllm_ci_cpu.sbatch`）。
- **BDD 验收**：`tests/features/*.feature`（gherkin）+ `tests/test_bdd_scenarios.py` 步骤实现，
  覆盖 inspect→baseline→optimize→verify→apply→rollback 全链路与 fail-closed 场景。
- **Mutation smoke**：`scripts/mutation_smoke.py` 对控制面核心（registry 约束过滤/receipt 校验/回退链）注入变异，
  要求测试套件杀死率 ≥ 阈值（初始 ≥80%）。
- **Coverage 门控**：控制面新代码行覆盖 ≥85%（`coverage.py`，CPU job 产出 `.wllm/reports/coverage.txt`）。
- **命名门控**：`release_gate.sh`（staged）与 `--all`（全树）双模式，push 前两者必 PASS。
- **/verify 三闸**：提交前、阶段切换、错误修复后。

## 10. 里程碑

**已完成（M1–M15，全部有 SLURM job 证据）**：wGraph v0 + planner v0 + L0/L1 + catalog importer(258/258)；
successive-halving；drift-gated verdicts；双引擎统一；Alpha 3 模型中位 2.75×；
CFG branch-parallel 1.74× bit-exact（2 GPU）；全管线 1.44× bit-exact；Qwen3-VL 2.75×（tie-aware exact）；
OpenVLA 4.59×（native bf16）；E2E app Launchable（704 帧）；Hopper probe tier；两环境 VLA bridge。
**M16**：命名纪律全树清零 + 控制面 v0（OptimizeSpec/inspect/registry/receipts/apply/rollback）+ BDD/mutation/coverage 门控。
**M17（本轮）**：数据面四支柱自研在库（§3.6）：composite 图运行时 / omni 分阶段引擎（AsyncOmni 契约） /
目录资产内容级就绪检查 / 优化技术执行器与 fail-closed 编排器；registry 增 wllm-composite/wllm-omni；
BDD 增技术编排 3 场景；mutation 目标扩至 7 文件。**多 agent 分工**：独立对抗审查 agent 复核四支柱，
确认并修复 3 P0（引擎捏造 latent 表→改为 stage 钩子且构造期 fail-closed；生成 payload 错键→契约键直落；
中毒请求滞留共享 scheduler→abort+批失败退休+感染请求 raise 而非伪造 final；stream 通道跨 session→按 session 隔离并入 reset）
及 F4-F13 批量加固（默认 scheduler 静默替换、stage 顺序/重复 id、Par join/原位变异、Loop 陈旧 until/嵌套 index、
空参照 oracle、缓存连续重用上限+reset 清计数、恒零假证据可证伪化、资产 exact-size 校验）。
风险登记册 `docs/RISK_REGISTER.md`（P0/P1×缓解×缺口×责任角色）。分层诚实：四支柱为 **CPU/contract-verified 基建层**，GPU 实测挂接为后续里程碑。
**M18（本轮，多 agent 分工：2 实现 agent + 主线并行）**：
① **真实证据 receipt 化**（`wllm/control/evidence.py` + `scripts/receipt_wan22_cfgpar.py`）：job 196293 日志
（ref 5761.6ms → 2GPU 并行 4002.4ms = 1.44×，帧级 bit-exact）解析为可晋升 receipt 并在 CI dogfood；
同一 job 的批内 CFG 变体（max_abs 251/255）被同一门正确拒绝——控制面首次接通真实测量数据。
② **MCP 入口**（`wllm/control/mcp.py`，`wllm-mcp`）：stdio JSON-RPC 暴露 inspect/plan/verify/apply/rollback/report
六工具；server 只加传输不加判断，diagnose-only(rc 3) 是真话不是错误。
③ **技术组合器**（`wllm/techniques/composer.py`，agent 实现）：verified singles 的组合搜索——组合合法性靠测量
不靠假设；超可加漂移/变慢/预算违约=interference 且带理由；工厂纯度机制杜绝跨 repeats 状态泄漏。
④ **计划下沉**（`wllm/composite/lowering.py`，agent 实现）：DeploymentPlan → 组件 placement，fail-closed
（未知 node/未分配/双分配/pin 违约/cpu 域错置/parallel_degree>1 诚实拒绝/硬件越界），require() 单一执行闸。
**M19+**：wllm.omni 注册真实模型 runner（qwen3_omni app 于自研引擎端到端）；composite 承接实测 plan 重放；
Claude Code plugin 打包（MCP 已就绪）；24h soak + 故障注入（Beta 收口）。
**1.0（9–12 周）**：Feedback/MultiAgent region、WAM deadline scheduler、approximate（FP8/NVFP4/KV 量化）、
可选 L3 native 后端、在线 plan 路由、双硬件家族。

## 11. 目录结构

```
wllm-infra/
├── WLLM_SPEC.md
├── wllm/
│   ├── graph/ capture/ contracts/ planner/ runtime/ verify/ profiling/
│   ├── control/     spec.py inspect.py registry.py receipt.py apply.py cli.py   # v2 控制面
│   ├── engines/     omni.py            # 外部引擎 env 绑定（不具名）
│   ├── backends/    torch_local/ catalog/ subprocess_cli/ native/
│   ├── kernels_t/   serving/  native/  apps/
│   └── serve/
├── contracts/  tests/(features/)  benchmarks/  inventory/(README only)  slurm/  logs/  docs/
└── upstream-a|b|c/              # 只读参考镜像（gitignored, 永不入库）
```

## 12. 集群适配与红线（不变）

sbatch-only（登录节点连 python 脚本都禁跑 → CI 走 CPU sbatch）、GPU 双上限 72/112、利用率红线、
复提门控、/verify 过闸；权重 ModelScope 优先；H200：FP8 可用、NVFP4 不可用（fallback 路径必须存在）；
.venv 用共享 FS python；多环境 worker 天然匹配本集群多 env 现实。
