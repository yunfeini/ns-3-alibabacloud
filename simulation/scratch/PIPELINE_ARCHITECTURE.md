# PD 分离流水线仿真架构

## 总流程

1. `NormalNetwork.cc` 先读取配置、建立网络、初始化 `cclScheduler` 和 `mnCCL`，并配置普通日志与流完成 CSV。
2. `task_generator.h` 在仿真开始前生成任务序列。单个任务只携带：
   - `prefillLength`
   - `decodeLength`
   - `submitTime`
3. 全局运行参数由 `mnccl::RuntimeConfig` 保存，包括：
   - `need_prefill`
   - `need_decode`
   - `expert_num`
   - 消息大小、decode 计算延迟等。`token_msg_size` 表示每 token 通信字节数，当前默认约 2 bytes/token。
4. 任务下发和任务完成时，都会调用 `cclScheduler::NeedUpdateCallback`。外部调度器可以在这里实时修改全局 `need_prefill`、`need_decode` 和 `expert_num`。
5. 任务进入 PD 分离流水线：
   - P placement
   - P 阶段在全部 P 节点上顺序执行 `allreduce` + `alltoall`
   - 可选：在 P placement 上先执行若干 decode token
   - D placement
   - KV cache transfer 到 D placement
   - 在 D placement 上逐 token 执行剩余 decode，每个 token 为计算延迟 + 1-token `allreduce` + 1-token `alltoall`
   - 任务完成时在 LOG 中打印总 FCT、TTFT、TPOT、decode 端到端时间和关键时间点

P 集群和 D 集群不是互斥资源。只要满足显存和每 GPU 最大专家数量约束，二者可以重叠放置。

## 专家放置逻辑

`expert_num` 表示模型总专家数。代码中不再单独保存 `ep` 参数，专家并行度由当前 placement 的 `need` 和 `expert_num` 共同决定。

`cclScheduler::TryPlaceNeed(...)` 会为 P 或 D placement 选择 `need` 个 GPU，并根据 `expert_num` 估计每个 GPU 承载的专家数量：

- 当 `need >= expert_num` 时，每个 GPU 最多承载 1 个专家；其中 `need > expert_num` 时，同一个 expert rank 会有多个 GPU replica。
- 当 `need < expert_num` 时，多个专家会被分摊到同一批 GPU 上。
- 每个 GPU 最多承载 `kMaxExpertsPerGpu = 9` 个专家。
- 显存占用按 `专家数量 * expert_mem_bytes` 计算。

`cclScheduler::AssignExperts(...)` 会调用专家放置策略并保存 `placement_owner -> expert_id -> replica GPUs` 的映射。

默认专家放置策略是 `RoundRobinExpertPlacementPolicy(...)`。

## 当前调优版专家放置算法

策略名：`domain_balanced_external_min`

该策略用于在已选定的 P/D GPU 集合内决定每个 expert replica 的具体 GPU。策略目标是同时降低跨 L1 交换域访问、避免单个 L1 交换域承载过高专家访问频次，并保持同一专家的 replica 在交换域间不过度偏斜。

设：

- 专家集合为 `E`，隐藏真实访问分布由 `EXPERT_ACCESS_FREQ` 驱动 alltoall 目的专家采样，但该分布不直接暴露给专家放置策略。
- 专家放置使用后验估计 `f_hat_e = expert_access_prior + observed_access_count(e)`。仿真开始时所有专家只有均匀先验；每次 alltoall 实际采样到目的 expert 后，`mnccl::RecordExpertAccess(...)` 更新观测计数。
- 当前选中 GPU 所属 L1 domain 集合为 `D`。
- `x_{e,g} in {0,1}` 表示专家 `e` 是否放置在 GPU `g`。
- `domain(g)` 表示 GPU `g` 所属 L1 domain。
- `L_d = sum_e sum_{g:domain(g)=d} f_hat_e x_{e,g}` 表示 L1 domain `d` 的后验频次加权专家负载。
- `C_e = {domain(g) | x_{e,g}=1}` 表示专家 `e` 覆盖到的 L1 domain 集合。

算法使用增量评分器维护候选放置后的指标：

- `externalAccessCost = sum_e f_hat_e * (|D| - |C_e|)`，惩罚后验高频专家没有覆盖到足够多 L1 domain。
- `replicaDomainImbalance = sum_e f_hat_e * Var({replica_count(e,d) | d in D})`，惩罚同一专家 replica 在不同 L1 domain 上过度不均衡。
- `l1Variance = Var({L_d | d in D})`，惩罚交换域间总访问频次不均衡。
- `maxL1Load = max_d L_d`，控制最热点 L1 交换域。
- `gpuLoadVariance = Var({expert_count(g)})`，控制 GPU 专家槽位分布。

最终归一化目标函数为：

```text
score =
  w_ext * externalNorm
  + w_replica * replicaNorm
  + w_l1_var * l1VarNorm
  + w_l1_max * l1MaxNorm
  + w_gpu_var * gpuVarNorm
```

本轮调优后的默认权重为：

```text
PLACEMENT_EXTERNAL_WEIGHT 4.0
PLACEMENT_REPLICA_BALANCE_WEIGHT 2.0
PLACEMENT_L1_VARIANCE_WEIGHT 1.0
PLACEMENT_L1_MAX_WEIGHT 2.0
PLACEMENT_GPU_VARIANCE_WEIGHT 1.0
PLACEMENT_FORCE_NEW_L1_DOMAIN 1
PLACEMENT_PROBE_WEIGHT 0
```

`PLACEMENT_FORCE_NEW_L1_DOMAIN=1` 表示在填充额外 replica 时，优先选择能让该 expert 覆盖新 L1 domain 的候选；quick-search 结果显示，完全取消这一覆盖约束会让早期 replica 覆盖不足，导致 TTFT/TPOT 劣化。因此当前版本保留覆盖约束，并通过降低 `replica` 权重、提高 `l1_max` 权重来更偏向热点交换域削峰。

最新 probe-aware placement 消融显示，将 route probe cost 直接纳入专家放置会轻微破坏副本域均衡，TTFT 不如当前最佳版本。因此当前版本保留 `probe_path_cost` 监控/研究接口，但默认 `PLACEMENT_PROBE_WEIGHT=0`，probe 只用于同 rank 副本路由。

## 调度策略接口

所有后续要替换策略的位置都用醒目标记：

`SCHEDULING POLICY INTERFACE`

### 任务生成

文件：`task_generator.h`

入口：

- `taskGenerator::PipelineTaskDistributionParams`
- `taskGenerator::GenerateCurrentPipelineTaskDistribution(...)`
- `taskGenerator::SubmitPipelineTasks(...)`
- `taskGenerator::RegisterPipelineWorkload(...)`

当前任务分布为均匀随机生成，任务自身不携带 need、EP、专家放置等信息。

### 全局 Need 更新

文件：`cclscheduler.h`

接口：

- `cclScheduler::SetNeedUpdateCallback(...)`
- `cclScheduler::NotifyNeedEvent(...)`

触发事件：

- `NeedEvent::TaskDispatch`
- `NeedEvent::TaskFinish`

外部调度器应在这个回调里修改全局 `need_prefill`、`need_decode`、`expert_num`。

### PD 节点放置

文件：`cclscheduler.h`

接口：

- `cclScheduler::SetNeedPlacementPolicy(...)`
- `cclScheduler::TryPlaceNeed(...)`

策略输入包含 placement owner、`Prefill`/`Decode` 类型、`need`、`expert_num`、当前显存占用和专家占用快照。

默认策略为 `FillNeedPlacementPolicy(...)`，按节点顺序填充，允许 P/D 在同一 GPU 上重叠，只要容量满足。

### 专家放置

文件：`cclscheduler.h`

接口：

- `cclScheduler::SetExpertPlacementPolicy(...)`
- `cclScheduler::AssignExperts(...)`
- `cclScheduler::GetExpertReplicas(...)`

默认策略为 round-robin。`need > expert_num` 时会自然产生同 rank 多 replica。

调优版策略参数均可在 policy config 中配置，也可由 `run_domain_placement_full_load_compare.py` 通过命令行覆盖：

- `--placement-external-weight`
- `--placement-replica-balance-weight`
- `--placement-l1-variance-weight`
- `--placement-l1-max-weight`
- `--placement-gpu-variance-weight`
- `--placement-probe-weight`
- `--placement-force-new-l1-domain`
- `--pd-ratio` 与 `--pd-total`，例如 `--pd-ratio 3:1 --pd-total 160` 会生成 `NEED_PREFILL=120, NEED_DECODE=40`
- `--expert-num`
- `--expert-per-gpu`

实验脚本中的 `baseline` 现在表示所有策略接口均使用 `default`，包括 need placement、expert placement、same-rank route、local flow schedule、global need update 和 PD split；`incremental` 表示当前保留的最佳版本：启用 `domain_balanced_external_min` 专家放置、`probe_rtt_delta` 路由、`decode_first_prefill_later` 本地队列策略，并显式使用 `PD_SPLIT_POLICY default`。

### PD 分割点

文件：`mnCCL.h`

接口：

- `mnccl::SetPdSplitPolicy(...)`
- `mnccl::PdSplitPolicy`

策略返回“在 P placement 上先执行多少个 decode token”。默认 `NoPrefillSideDecodeSplitPolicy(...)` 返回 0，即 P 完成后立刻 KV 到 D，然后所有 decode 都在 D 上完成。

### 本地 GPU 流调度

文件：`mnCCL.h`

接口：

- `mnccl::SetLocalFlowSchedulePolicy(...)`
- `mnccl::LocalFlowSchedulePolicy`

每个源 GPU 维护本地发送队列，队列元素 `LocalFlowTask` 包含：

- `jobId`
- 集合通信类型
- pipeline stage
- P/D 类型
- 源/目的 GPU
- 消息长度
- 到达时间
- FIFO sequence

默认策略为 `FifoLocalFlowSchedulePolicy(...)`。

### Expert Replica 路由

文件：`mnCCL.h`

接口：

- `mnccl::SetExpertRoutePolicy(...)`
- `mnccl::ExpertRoutePolicy`

当一个 expert rank 有多个 replica 时，该策略选择具体目的 GPU。默认实现为 `RandomExpertRoutePolicy(...)`，优先避免本地 self-send。

## 关键函数说明

`NormalNetwork.cc::main(...)`

完成网络初始化、scheduler/mnCCL 初始化、日志初始化，并调用任务生成入口。

`taskGenerator::RegisterPipelineWorkload(...)`

配置运行参数和所有调度策略接口，生成任务并注册到仿真事件中。

`mnccl::SubmitPipelineTask(...)`

保存任务状态并在指定时间下发。下发时会触发全局 need 更新回调。

`mnccl::TryStartPipelineTask(...)`

为 P 阶段申请 placement，完成专家放置，启动 P 阶段 `allreduce`。

`mnccl::StartPrefillAllReduce(...)`

启动一次 P 阶段全 P 节点 `allreduce`。完成后进入 P 阶段 `alltoall`。

`mnccl::StartPrefillAllToAll(...)`

启动一次 P 阶段基于专家路由的 `alltoall`。

`mnccl::TryStartDecode(...)`

根据 PD split 策略决定是否先在 P placement 上继续执行部分 decode token；到达 split 点后再申请 D placement，并触发 KV cache transfer。

`mnccl::StartKvCacheTransfer(...)`

仿真从 P placement 到 D placement 的 KV cache 传输。KV 完成后释放 P placement。

`mnccl::StartNextDecodeToken(...)`

执行 decode token 状态机。每次迭代对应 1 个 decode token，包含计算延迟、D placement 全组 1-token decode `allreduce` 和 decode `alltoall`。

`mnccl::SubmitAllToAll(...)`

根据专家放置映射生成 alltoall 流。目的 expert rank 由隐藏概率表采样，具体 replica 由路由策略选择；采样到的目的 expert 会写入后验访问统计，供下一次 task 下发时的放置策略使用。

`mnccl::SubmitFlow(...)`

将流放入源 GPU 的本地队列，由本地调度策略决定发送顺序。

`mnccl::TryScheduleLocalGpu(...)`

使用本地调度策略从 GPU 队列中选择一个流并启动。

`mnccl::OnMessageFinish(...)`

处理流完成，释放本地 GPU 队列占用，统计集合通信完成状态，推进 pipeline 状态机，并将流完成记录写入 CSV。

`mnccl::FinishPipelineTask(...)`

释放剩余 placement，触发任务完成 need 更新回调，并在 LOG 中输出任务级指标。
