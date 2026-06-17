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

根据专家放置映射生成 alltoall 流。目的 expert rank 由概率表采样，具体 replica 由路由策略选择。

`mnccl::SubmitFlow(...)`

将流放入源 GPU 的本地队列，由本地调度策略决定发送顺序。

`mnccl::TryScheduleLocalGpu(...)`

使用本地调度策略从 GPU 队列中选择一个流并启动。

`mnccl::OnMessageFinish(...)`

处理流完成，释放本地 GPU 队列占用，统计集合通信完成状态，推进 pipeline 状态机，并将流完成记录写入 CSV。

`mnccl::FinishPipelineTask(...)`

释放剩余 placement，触发任务完成 need 更新回调，并在 LOG 中输出任务级指标。
