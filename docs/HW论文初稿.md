# 面向 MoE Dispatch 的拓扑感知专家调度与可抢占本地队列仿真框架

## 摘要

大规模 MoE 推理中的 dispatch 阶段需要将每个 token 或 batch 的激活发送到 Top-k 专家所在 GPU。专家副本的放置、同 rank 副本路由以及每 GPU 本地发送队列的排序共同决定了 dispatch 端到端时延和 TPOT。已有系统通常将专家放置、网络路由和本地队列调度分开处理，难以在多交换域、热点专家访问和 bursty 流量下形成稳定的端到端收益。

本文提出一组三阶段 MoE dispatch 调度算法：第一，posterior-aware domain-spread expert placement 根据运行时后验专家访问频次，在 L1 交换域之间均衡热点专家副本，并约束外部访问暴露度；第二，async live probe RTT-delta replica routing 通过高、低优先级无负载 probe 的 RTT 差异和 live route cache，在同 rank 专家副本间选择动态路径；第三，preemptive Lyapunov-drift SRPT local scheduling 将大 flow 切分为可抢占 chunk，并基于剩余服务时间和等待债务选择每 GPU 本地发送顺序。为验证该算法组，本文实现了一个基于 ns-3 的 trace-driven dispatch-only 仿真框架，支持专家放置热更新、异步 route probe、本地 flow queue、trace 目标完成驱动、流级完成记录、集群时序监控和专家热图可视化。

在 256 GPU、256 专家、每 GPU 9 专家副本、Top-k 8 的 trace-driven 实验中，完整三算法相对 trace baseline 将平均 TPOT 从 2071.137 us 降低到 622.408 us，降低 69.95%。三算法消融显示，去除专家放置、专家路由和 PLD-SRPT 本地队列调度分别导致 TPOT 增加 24.57%、14.83% 和 54.69%。结果表明，专家放置、动态副本路由和可抢占本地队列调度是 MoE dispatch 优化中的三个互补环节。

## 1. 引言

MoE 模型通过条件计算扩展模型容量，但推理阶段会引入显著的专家 dispatch 通信。对于每个 batch，源 GPU 需要按照路由结果将激活发送到若干专家副本。由于专家访问频次通常呈现强长尾，固定专家放置和默认路由会导致部分交换域或 GPU 发生拥塞，从而推高 dispatch 端到端时延和 TPOT。

MoE dispatch 的困难主要来自三个层次。

第一，专家放置是离散组合优化。每个专家至少需要一个副本，热点专家需要更多副本，但副本过度集中会造成跨 L1 交换域流量和目标 GPU 排队压力。

第二，同 rank 专家通常存在多个副本。仅根据静态拓扑最短路径或随机选择副本，无法反映运行时拥塞。route probe 可以提供动态路径信息，但 probe 频率、样本新鲜度和 route score 需要与流量规模匹配。

第三，源 GPU 本地发送队列会成为真实瓶颈。即使专家放置和路由较好，同一 GPU 上仍可能积累大量不同长度、不同到达时间的 flow。FIFO 或简单 SJF 难以同时优化平均等待时间和避免长 flow 饥饿。

本文的核心观点是：MoE dispatch 优化不能只考虑单一层面的放置或路由，而应将专家放置、专家副本路由和每 GPU 本地队列调度作为一条端到端策略链路。围绕该观点，本文的贡献如下。

1. 提出 posterior-aware domain-spread expert placement。该算法使用运行时专家访问后验，联合优化 L1 交换域负载均衡、外部访问暴露度、同专家副本域间均衡、GPU 槽位均衡和 probe path cost。
2. 提出 async live probe RTT-delta replica routing。该算法用高、低优先级无负载 probe 的 FCT 差值刻画路径拥塞敏感性，并在 route cache 热更新时选择同 rank 专家副本。
3. 提出 preemptive Lyapunov-drift SRPT local scheduling。该算法将 dispatch flow 切成 chunk，按 flow group 聚合剩余服务时间和等待债务，以可抢占 SRPT 的形式降低平均 TPOT。
4. 实现 trace-driven dispatch-only 仿真程序架构。该架构支持真实专家访问 trace、32 卡 baseline 放置复制到 256 卡、持续下发直到 trace 完成、专家放置热更新、异步 route probe、流级完成记录、probe 流量统计和专家热图可视化。

## 2. 问题定义

专家集合为：

```text
E = {0, 1, ..., M - 1}
```

GPU 集合为：

```text
G = {g_1, g_2, ..., g_N}
```

其中当前实验设置为：

```text
M = 256
N = 256
EXPERT_PER_GPU = 9
DISPATCH_TOPK = 8
```

每个 GPU `g` 属于某个 L1 交换域：

```text
d(g) in D
```

一次 dispatch task 中，每张源 GPU 最多产生 `K=DISPATCH_TOPK` 条专家 flow。单个专家 flow 的字节数为：

```text
B_msg =
    DISPATCH_EXPERT_FFN_PARAMS
  * DISPATCH_PRECISION_BYTES
  * DISPATCH_BATCH_SIZE
  * DISPATCH_N
```

当前 DeepSeek V3 1B 近似配置为：

```text
DISPATCH_EXPERT_FFN_PARAMS = 112000
DISPATCH_PRECISION_BYTES   = 1
DISPATCH_BATCH_SIZE        = 16
DISPATCH_N                 = 1
```

trace 文件给出每个专家的目标访问次数 `A_e`。若使用降采样倍率 `z`，则：

```text
A'_e =
  0,             if A_e = 0
  ceil(A_e / z), otherwise
```

仿真持续下发 dispatch task，直到：

```text
forall e in E: observed_access(e) >= A'_e
```

优化目标是最小化平均 dispatch TPOT：

```text
TPOT = finish_time(task) - submit_time(task)
```

在当前 `DISPATCH_N=1` 下，TPOT 等于单次 dispatch 端到端 latency。

## 3. 算法设计

### 3.1 算法一：后验频次驱动的专家放置

放置变量为：

```text
x_{e,g} =
  1, if expert e is placed on GPU g
  0, otherwise
```

容量约束为：

```text
forall e: sum_g x_{e,g} >= 1
forall g: sum_e x_{e,g} <= C_g
```

其中当前 `C_g=EXPERT_PER_GPU=9`。访问频次使用运行时后验：

```text
f_e(t) = EXPERT_ACCESS_PRIOR + observed_access_count_e(t)
```

放置 objective 由多个归一化项构成：

```text
J_place(x) =
    w_ext * ExternalCost(x)
  + w_rep * ReplicaImbalance(x)
  + w_var * L1Variance(x)
  + w_max * L1MaxLoad(x)
  + w_gpu * GpuSlotVariance(x)
  + w_probe * ProbePathCost(x)
```

各项含义如下。

- `ExternalCost`：专家未覆盖某个 L1 domain 时，该 domain 到该专家的访问需要跨域传输；
- `ReplicaImbalance`：同一专家的副本是否过度集中在少数 L1 domain；
- `L1Variance` 与 `L1MaxLoad`：不同 L1 domain 内加权专家访问负载是否均衡；
- `GpuSlotVariance`：GPU 槽位使用是否均匀；
- `ProbePathCost`：放置是否为后续 route probe 提供低代价候选副本。

实际求解采用两阶段启发式。

```text
Algorithm 1: Posterior Domain-Spread Placement

Input:
  experts E, GPUs G, posterior frequency f_e, capacity C_g

Coverage phase:
  sort experts by descending f_e
  for each expert e:
    choose GPU g minimizing:
      (replicas of e in domain d(g),
       L1 load after placing e,
       GPU slot load,
       node id, gpu id)
    place e on g

Surplus phase:
  for each GPU g:
    while load(g) < C_g:
      choose expert e minimizing:
        (replicas of e in domain d(g),
         replica_count(e) / sqrt(f_e),
         L1 load after placing e,
         -f_e,
         expert id)
      place e on g
```

该策略的作用是将热点专家副本分散到更多 L1 domain，同时避免同一个专家在某个 domain 内过度复制。

### 3.2 算法二：异步 probe 的同 rank 副本路由

专家放置后，每个专家 `e` 拥有副本集合：

```text
R_e = {g | x_{e,g}=1}
```

路由算法需要为每个源 GPU `s` 和目的专家 `e` 选择一个副本：

```text
r(s,e) in R_e
```

路由先使用 local-first 规则：

1. 如果源 GPU 本身包含目标专家副本，则路由到本 GPU；
2. 否则如果同主机存在目标专家副本，则路由到同主机 GPU；
3. 否则使用 probe score 选择远端副本。

probe score 为：

```text
Score(s,g) =
    |T_low(s,g) - T_high(s,g)| / T_base(s,g)
  + 0.25 * min(T_low(s,g), T_high(s,g)) / T_base(s,g)
  + w_q * QueuePressure(g) / Q_norm
```

其中 `T_high` 和 `T_low` 分别来自高、低优先级 1 字节 probe 的 FCT。异步 probe 后台按照固定 interval 和 budget 刷新样本表，route cache 在新 task 到达时读取 live probe 结果。

```text
Algorithm 2: Async Live Probe Routing

for each placement update:
  for each source GPU s:
    for each expert e:
      if s hosts e:
        route[s,e] <- s
      else if same host has replica of e:
        route[s,e] <- same-host replica
      else:
        route[s,e] <- argmin_{g in R_e} Score(s,g)

background every probe_interval:
  select stale source-destination pairs
  submit high-priority and low-priority 1-byte probes
  update live sample table after probe completion
```

当前实验中 probe 字节开销极小。20x 完整算法中 probe 字节占总提交流量约 `0.000249%`。

### 3.3 算法三：PLD-SRPT 本地队列调度

每个源 GPU 维护本地发送队列。为了实现可抢占近似，dispatch flow 被切分为 chunk：

```text
LOCAL_FLOW_PREEMPTIVE_CHUNK_BYTES = 262144
```

属于同一原始 flow 的 chunk 共享：

```text
flowGroupKey
flowGroupBytes
flowGroupArrivalTimeNs
```

调度器按 flow group 聚合剩余字节，并估计剩余服务时间：

```text
S_j(t) = standalone_fct(remaining_bytes_j)
```

等待年龄为：

```text
A_j(t) = t - arrival_time_j
```

定义等待债务：

```text
D_j(t) = max(0, A_j(t) - D0 - kappa * S_j(t))
```

PLD-SRPT score 为：

```text
Score_queue(j,t) =
  (V + beta * D_j(t) + 0.5 * D_j(t)^2 / T0)
  / (S_j(t) + O_preempt)
```

每次 GPU 出队时选择 score 最大的 group，并发送该 group 的第一个等待 chunk。

```text
Algorithm 3: PLD-SRPT Local Queue Scheduling

Input:
  local queue Q_g(t)

group chunks by flowGroupKey
for each group j:
  compute remaining bytes B_j
  estimate S_j(t)
  compute waiting debt D_j(t)
  compute Score_queue(j,t)

select group with max Score_queue
tie-break by smaller remaining service, earlier arrival, smaller sequence
dispatch first waiting chunk of selected group
```

当所有 `D_j=0` 时，该策略退化为 SRPT；当长 flow 等待过久，债务项提高其优先级，降低饥饿风险。

当前推荐参数为：

```text
LOCAL_FLOW_SCHEDULE_POLICY pld_srpt
LOCAL_FLOW_PREEMPTIVE_CHUNK_BYTES 262144
LOCAL_FLOW_PLD_SRPT_V_NS 100000
LOCAL_FLOW_PLD_SRPT_T0_NS 100000
LOCAL_FLOW_PLD_SRPT_D0_NS 0
LOCAL_FLOW_PLD_SRPT_KAPPA 2.0
LOCAL_FLOW_PLD_SRPT_BETA 0.05
LOCAL_FLOW_PLD_SRPT_PREEMPT_OVERHEAD_NS 0
```

## 4. 仿真程序架构

本文实现了一个 trace-driven dispatch-only 仿真框架，目标是将算法策略与 ns-3 网络仿真解耦，使专家放置、路由、本地队列调度和实验脚本都可以独立替换。

### 4.1 总体流程

程序流程如下。

```text
load topology and network config
load policy config
load trace device placement and decode access targets
initialize mnCCL state and scheduler state

while trace targets not complete:
  submit new dispatch task
  update posterior expert access statistics
  run expert placement policy
  rebuild route cache
  submit dispatch flows
  local GPU queues schedule flow/chunk order
  ns-3 executes network flows
  collect flow completion and task completion metrics

write logs, CSVs, heatmaps and summary
```

### 4.2 模块划分

主要代码模块如下。

| 模块 | 文件 | 职责 |
|---|---|---|
| 策略配置与 hook | `simulation/scratch/pipeline_policy_config.h` | 解析 policy config，绑定专家放置、路由、本地队列调度、trace 参数 |
| 调度器数据结构 | `simulation/scratch/cclscheduler.h` | 管理 GPU 集合、专家放置、need placement、专家访问频次 |
| 通信与执行 | `simulation/scratch/mnCCL.h` | dispatch task、route cache、async probe、本地 flow queue、流完成统计 |
| 任务生成 | `simulation/scratch/task_generator.h` | fixed workload 和 dispatch-N 分布生成 |
| 主程序 | `simulation/scratch/NormalNetwork.cc` | 读取拓扑和 policy，启动 ns-3 仿真 |
| 热图可视化 | `simulation/plot_expert_heatmap.py` | 从 log 与 policy 生成专家放置/访问热图 |
| 全量对比 | `simulation/run_dispatch_full_compare_visual.py` | 运行 baseline 和完整算法，输出 summary 与图 |
| 三算法消融 | `simulation/run_dispatch_three_algo_ablation_visual.py` | 只对比 baseline、完整算法和三个算法消融 |

### 4.3 Trace 驱动任务生成

trace 来源包括：

- `device.json`：58 层专家放置，当前使用第 0 层；
- `decode_*.csv`：32 个 decode 文件，当前使用第 0 层专家访问次数。

仿真将 trace 访问次数按倍率降采样，并持续下发 dispatch task 直到所有专家达到目标次数。对于 baseline，32 卡 trace device placement 会复制到 256 卡：

```text
template_device_id = selected_gpu_index % 32
```

这样 baseline 能保持 trace 原始放置结构，同时适配 256 GPU 仿真规模。

### 4.4 热更新与策略隔离

每次新 dispatch task 到达时，调度器依次触发：

```text
expert placement
route cache rebuild
dispatch flow submission
```

新的专家放置和路由只影响后续新 task，不修改已经在网络中运行的 flow。本地队列调度在每个 GPU 出队时持续生效。

该设计使三类算法可以独立消融：

```text
all_three
no_expert_placement
no_expert_routing
no_local_pld_srpt
```

### 4.5 监控与可视化

框架输出以下文件：

- `mncc.log`：调度事件、放置、路由、trace 进度、任务完成摘要；
- `mncc_flow_finish.csv`：每条 flow 的完成时间、standalone FCT、实际 FCT；
- `mncc_cluster_timeseries.csv`：队列长度、活跃 GPU、专家槽位利用率等时序指标；
- `summary.csv`：跨方法指标汇总；
- `expert_heatmap_<method>.png`：全过程专家放置/访问热图；
- `expert_placement_latest_<method>.png`：最后一次专家放置热图；
- `three_algo_ablation_tpot.png`：三算法消融 TPOT 对比。

这部分程序架构是本文第二类创新：它不是单次手工仿真脚本，而是可复现实验框架，支持策略组合、消融、参数扫描和可视化诊断。

## 5. 实验设置

实验使用 256 GPU、256 专家、每 GPU 9 专家槽位、Top-k 8。通信量采用 DeepSeek V3 1B 的 FFN 专家层近似：

```text
DISPATCH_EXPERT_FFN_PARAMS = 112000
DISPATCH_PRECISION_BYTES   = 1
DISPATCH_BATCH_SIZE        = 16
DISPATCH_N                 = 1
```

20x trace 消融命令为：

```bash
python3 simulation/run_dispatch_three_algo_ablation_visual.py \
  --output simulation/results/hw_three_algo_ablation_20x \
  --trace-access-divisor 20 \
  --placement-heatmap-snapshot all \
  --access-heat-scale log1p \
  --keep-going
```

对比方法包括：

| 方法 | 含义 |
|---|---|
| `baseline_uniform` | trace device placement + default routing + uniform expert target |
| `baseline_trace` | trace device placement + default routing + original trace target |
| `all_three` | 专家放置 + 专家路由 + PLD-SRPT |
| `no_expert_placement` | 去掉算法一，保留算法二和三 |
| `no_expert_routing` | 去掉算法二，保留算法一和三 |
| `no_local_pld_srpt` | 去掉算法三，保留算法一和二 |

## 6. 实验结果

20x trace 下，所有方法均完成 trace 目标。

| method | avg TPOT us | 相对完整算法 | trace accesses | probe percent |
|---|---:|---:|---:|---:|
| `baseline_uniform` | 1705.909 | +174.08% | 143616 / 143616 | 0 |
| `baseline_trace` | 2071.137 | +232.76% | 143521 / 143521 | 0 |
| `all_three` | 622.408 | ref | 143521 / 143521 | 0.000249% |
| `no_expert_placement` | 775.329 | +24.57% | 143521 / 143521 | 0.000287% |
| `no_expert_routing` | 714.696 | +14.83% | 143521 / 143521 | 0 |
| `no_local_pld_srpt` | 962.798 | +54.69% | 143521 / 143521 | 0.000383% |

相对 `baseline_trace`，完整算法将平均 TPOT 从 2071.137 us 降低到 622.408 us，降低 69.95%。相对 `baseline_uniform`，完整算法降低 63.51%。

三个算法的边际贡献可以由消融结果估计：

- 去除 PLD-SRPT 后，TPOT 从 622.408 us 增加到 962.798 us，说明本地队列调度对源端排队和短 flow 完成顺序影响最大；
- 去除专家放置后，TPOT 增加到 775.329 us，说明后验频次驱动的副本分布减少了热点专家跨域和目标 GPU 压力；
- 去除专家路由后，TPOT 增加到 714.696 us，说明 async live probe 在 20x 更大负载下对副本选择仍有正收益。

数据字节统计也支持专家放置的作用：

```text
all_three data bytes          = 230.65 GB
no_expert_placement data bytes = 248.64 GB
```

默认放置比优化放置多约 7.80% 的数据字节。这说明专家放置不仅改变路径选择，也改变了实际需要经过网络的流量规模。

## 7. 讨论

### 7.1 为什么本地队列调度贡献最大

PLD-SRPT 将大 flow 切成 chunk，使短剩余服务时间的 flow group 能更快完成。对于 trace-driven dispatch，每个 task 会在 256 个源 GPU 上并行产生大量专家 flow。源端队列排序直接影响 task 何时所有 flow 完成。因此即使专家放置和路由已经优化，本地队列仍会成为决定 TPOT 的关键环节。

需要注意的是，启用 chunk 后 `queued_local_flows` 是 chunk 级队列长度，不能与非 chunk case 的 flow 级队列长度直接比较。TPOT 是更可靠的主指标。

### 7.2 为什么专家路由在不同倍率下表现不同

在早期 100x 消融中，去掉专家路由反而更快，说明 route probe 权重与 PLD-SRPT chunking 后的流形态存在耦合；在 20x 更大负载下，算法二恢复正收益。该现象说明算法二的超参数不应固定，而应根据拓扑和负载在线调节。

后续可以引入反事实估计和 SPSA 风格梯度调参：在每次新 task 到达时生成多个反事实 placement 或 route score 参数，利用 posterior 访问、probe 样本和队列压力估计未来 TPOT，再与真实完成反馈共同更新超参数。

### 7.3 仿真架构的价值

本文实现的仿真程序架构使以下实验成为可复现流程：

- trace 放置复制与 trace 访问目标完成；
- 专家放置、路由、本地队列三算法独立消融；
- probe 流量开销统计；
- 专家访问热图与最新放置热图；
- 每 flow standalone FCT 与实际 FCT 记录；
- 跨负载倍率的批量实验。

这使算法设计不依赖单次手工日志分析，而可以直接在可配置脚本中比较策略组、消融项和参数扫描。

## 8. 局限与未来工作

第一，当前专家放置使用启发式两阶段算法，而不是严格全局最优整数规划。后续可以使用反事实估计、SPSA 或 surrogate model 对放置权重进行在线校准。

第二，算法二的 probe score 仍含多个超参数。不同拓扑下 RTT、带宽、oversubscription 和队列尺度不同，固定权重不一定泛化。后续应实现基于真实完成反馈和 probe 反事实样本的在线梯度调参。

第三，PLD-SRPT 当前使用固定 chunk size。chunk 太大会降低抢占效果，chunk 太小会增加 flow 数和仿真开销。后续应将 chunk size 与链路 BDP、flow size 分布和本地队列长度联合调参。

第四，当前仿真聚焦 dispatch-only。虽然这能隔离 MoE expert communication 的核心问题，但真实推理还包含 attention、compute、KV cache 和多层流水线。后续可将本文三算法作为 MoE dispatch 子模块，接入完整推理流水线。

## 9. 结论

本文提出并实现了一组三阶段 MoE dispatch 调度系统：后验频次驱动的专家放置、异步 probe 的同 rank 副本路由、以及可抢占 Lyapunov-drift SRPT 本地队列调度。本文同时构建了 trace-driven dispatch-only 仿真框架，使策略配置、热更新、异步 probe、流级完成记录和专家热图可视化形成完整闭环。

20x trace 实验表明，完整三算法相对 `baseline_trace` 将平均 TPOT 降低 69.95%。消融实验进一步表明，本地队列调度、专家放置和专家路由分别贡献了 54.69%、24.57% 和 14.83% 的相对 TPOT 差异。结果证明，MoE dispatch 优化需要同时处理副本放置、动态路由和源端队列排序；只优化其中一个层面难以获得稳定的端到端收益。
