# HW 策略设计文档

本文档记录当前 HW trace-driven dispatch-only 仿真的策略设计。当前主线已经去除 PD 分离、KV cache、逐 token decode 等额外功能，只研究 MoE dispatch 阶段中专家放置和专家副本路由对 TPOT 与平均端到端时延的影响。每 GPU 本地发送队列调度接口仍保留，但当前 50 负载最佳组合使用默认队列调度，PLD-SRPT 不再写入主线最佳组合。

与本文档配套的形式化模型、公式、伪代码和性质证明见 [HW主要算法论文级说明.md](./HW主要算法论文级说明.md)。

## 1. 当前实验目标

当前仿真工作负载来自 `simulation/examples/HW/data`：

- `device.json`：基线的 58 层专家放置，当前只使用第 0 层；
- `decode_*.csv`：32 个 decode trace 文件，当前只使用第 0 层专家访问次数；
- trace 访问量通过 `TRACE_ACCESS_DIVISOR` 降采样；
- 仿真持续下发 dispatch task，直到所有专家的观测访问次数达到对应 trace 目标。

单条专家 dispatch flow 的通信量为：

```text
msg_size_per_expert =
    DISPATCH_EXPERT_FFN_PARAMS
  * DISPATCH_PRECISION_BYTES
  * DISPATCH_BATCH_SIZE
  * DISPATCH_N
```

当前 DeepSeek V3 1B dispatch 近似参数为：

```text
DISPATCH_EXPERT_FFN_PARAMS = 112000
DISPATCH_PRECISION_BYTES   = 1
DISPATCH_BATCH_SIZE        = 16
DISPATCH_N                 = 1
DISPATCH_TOPK              = 8
EXPERT_NUM                 = 256
DISPATCH_NEED              = 256
EXPERT_PER_GPU             = 9
```

每个 dispatch task 在 256 张 GPU 上运行。每张源 GPU 最多发起 8 条专家 flow。trace 模式中目的专家由剩余访问目标决定，而不是按固定概率随机采样。

## 2. 核心策略模块

当前策略被拆成三个模块，其中前两个是当前最佳组合的核心优化项：

| 算法 | 当前推荐策略 | 主要输入 | 输出 | 触发点 |
|---|---|---|---|---|
| 专家放置 | `probe_balanced_domain_spread` | GPU 集合、L1 domain、posterior 访问频次、容量 | 每张 GPU 的专家副本集合 | 新 dispatch task 下发前 |
| 专家路由 | `probe_rtt_delta` + async live probe | 专家副本集合、probe RTT、队列压力 | `source GPU, expert -> replica GPU` route cache | 放置更新后 |
| 本地队列调度 | `default` | 本 GPU 等待 flow | 下一条要发送的 local flow | GPU 本地发送队列每次出队时 |

当前推荐策略组由 50 负载多拓扑对比确定，写作：

```text
probe_balanced_domain_spread(no_l1_variance score-v2)
+ probe_rtt_delta async live routing
+ default local flow scheduling
```

也就是说，当前主线不再使用新增加的 adaptive/EMA/partition placement，也不把 PLD-SRPT 作为默认最佳组合的一部分。PLD-SRPT 仍作为独立队列模块保留，用于后续 mixed-length 本地队列场景。

## 3. 工作负载与 Baseline

50 负载对比脚本：

```bash
python3 simulation/run_dispatch_ablation_visual.py \
  --output simulation/results/<topology>_ablation_50x \
  --network-config <network-config-for-topology> \
  --trace-access-divisor 50 \
  --methods baseline_uniform baseline_trace no_l1_variance \
  --keep-going
```

默认对比三组方法：

| method | expert placement | route policy | access mode | trace device placement | async probe | live route |
|---|---|---|---|---:|---:|---:|
| `baseline_uniform` | `default` | `default` | `initial` | yes | 0 | 0 |
| `baseline_trace` | `default` | `default` | `initial` | yes | 0 | 0 |
| `no_l1_variance` | `probe_balanced_domain_spread` | `probe_rtt_delta` | `posterior` | no | 1 | 1 |

说明：

- `baseline_uniform`：使用 trace 第 0 层 device 放置，并将所有专家访问目标设置为相同次数；
- `baseline_trace`：使用相同 trace device 放置，但保留 trace 给出的 per-expert 访问次数；
- 两个 baseline 的默认路由不是纯随机。若本 GPU 或同主机已有目标专家副本，先选择本地或同主机；否则在远端候选副本中随机选择；
- `no_l1_variance`：当前最佳 full 组合。使用运行时 posterior 访问频次进行放置，score-v2 权重中将 L1 方差项置 0，随后使用异步 probe 和 live route refresh 为同 rank 专家副本选择路由。

当 `TRACE_USE_DEVICE_PLACEMENT=1` 且 `DISPATCH_NEED=256` 时，32 卡 trace baseline 放置按如下规则复制 8 份：

```text
template_device_id = selected_gpu_index % 32
```

## 4. 算法一：专家放置

当前推荐：

```text
EXPERT_PLACEMENT_POLICY          = probe_balanced_domain_spread
PLACEMENT_ACCESS_MODE            = posterior
PLACEMENT_REPLICA_BALANCE_WEIGHT = 6
PLACEMENT_EXTERNAL_WEIGHT        = 8
PLACEMENT_L1_VARIANCE_WEIGHT     = 0
PLACEMENT_L1_MAX_WEIGHT          = 1.5
PLACEMENT_GPU_VARIANCE_WEIGHT    = 0.5
PLACEMENT_PROBE_WEIGHT           = 7
```

专家放置解决的问题是：在给定 256 张 GPU、每张 GPU 9 个专家槽位、256 个专家的条件下，为每个专家生成至少一个副本，同时为热点专家放置更多副本，并使副本分布适合后续 probe 路由。

### 4.1 访问频次

放置使用的专家访问频次由 `RefreshPlacementAccessFreqForPolicy()` 给出：

```text
initial_e   = configured EXPERT_ACCESS_FREQ[e]
posterior_e = EXPERT_ACCESS_PRIOR + observed_access_count[e]
```

当前最好配置使用 `posterior`，即每次新 task 下发前根据已观测访问次数更新专家频次。这样放置策略不读取未来 trace，只使用运行时后验统计。

### 4.2 容量

每张参与 GPU 的目标专家槽位数为：

```text
if need < expert_num:
    experts_per_gpu = max(expert_per_gpu, ceil(expert_num / need))
else:
    experts_per_gpu = expert_per_gpu
```

同一物理 GPU 如果在 placement request 中重复出现，容量按出现次数累加；同一专家在同一物理 GPU 上去重，不允许重复放置。

### 4.3 放置目标

`probe_balanced_domain_spread` 同时优化以下目标：

- 每个专家至少有一个副本；
- 来自外部 L1 domain 的访问暴露度尽可能小；
- 热点专家获得更多副本；
- 同一专家的副本尽量分散到更多 L1 domain；
- 同一专家的副本在各 L1 domain 间尽量均衡；
- 使用 `l1_max` 控制最热点 L1 交换域，避免单域热点过高；
- GPU 加权访问热度尽量均衡；
- probe path cost 作为轻量评分项，帮助路由获得更好的候选集合。

早期 score-v2 中保留了 `PLACEMENT_L1_VARIANCE_WEIGHT=1`。最新消融显示该项与 `l1_max` 和 `gpu_variance` 高度重叠，在 `topo.txt` 50 负载上去掉后 TPOT 从 `1129957 ns` 降到 `1100133 ns`，因此当前推荐将该权重置 0。实现和日志仍保留 L1 方差观测字段，用于后续拓扑诊断。

该策略不是严格求解全局整数规划，而是在每次 task 到达时使用两阶段启发式：

1. Coverage phase：按 posterior 访问频次从高到低遍历专家，为每个专家放置第一个副本，优先覆盖低负载 L1 domain；
2. Surplus phase：按 GPU 顺序填充剩余槽位，优先为高频且副本不足的专家补副本，同时避免同一专家副本集中在少数 L1 domain。

## 5. 算法二：专家路由

当前推荐：

```text
SAME_RANK_ROUTE_POLICY        = probe_rtt_delta
ASYNC_ROUTE_PROBE_ENABLE      = 1
ASYNC_ROUTE_LIVE_ROUTE        = 1
ASYNC_ROUTE_PROBE_INTERVAL_NS = 20000
ASYNC_ROUTE_PROBE_BUDGET      = 128
ASYNC_ROUTE_PROBE_REFRESH_NS  = 500000
PROBE_BYTES                   = 1
ROUTE_PROBE_MAX_INFLIGHT      = 256
ROUTE_QUEUE_WEIGHT            = 0
```

专家路由的输入是专家放置生成的副本集合。对每个源 GPU 和目的专家，系统构建 route cache：

```text
route[source_gpu][dst_expert] = selected_replica_gpu
```

路由优先级：

1. 若源 GPU 本身包含目标专家副本，直接路由到本 GPU；
2. 否则若同主机包含目标专家副本，路由到同主机 GPU；
3. 否则在远端候选副本中使用 probe score 选择。

probe score 使用高优先级和低优先级 1 字节无负载 probe 的 FCT 差异：

```text
score(src, dst)
  = |T_low - T_high| / T_base
  + 0.25 * min(T_low, T_high) / T_base
  + ROUTE_QUEUE_WEIGHT * queue_pressure(dst) / ROUTE_QUEUE_NORM
```

当前 50 负载最佳组合中 `ROUTE_QUEUE_WEIGHT=0`，因此路由只使用 probe RTT/delta；队列压力项保留为接口，用于把专家路由与每 GPU 本地队列状态进一步耦合。

异步 probe 的作用是让 route cache 构建时可以读取真实探测样本，而不是只依赖静态 RTT 估计。`topo.txt` 50 负载 no-l1-variance run 中：

```text
probe_submitted_flows       = 412160
probe_submitted_bytes       = 412160
probe_submitted_byte_ratio  = 4.4268e-06
probe_submitted_percent     = 0.00044268%
```

probe 的流数量很多，但每条只有 1 字节，因此字节开销可以忽略。

## 6. 算法三：本地队列调度

每个 GPU 维护本地 flow 队列。队列元素包含：

- stage / placement kind / collective op；
- msg size；
- arrival time；
- source / destination；
- job id / PG / sequence；
- flow group key、group total bytes、group arrival time。

当前最佳组合的本地队列调度选择器为：

```text
LOCAL_FLOW_SCHEDULE_POLICY default
```

当前已实现的本地队列选择器包括：

```text
LOCAL_FLOW_SCHEDULE_POLICY default
LOCAL_FLOW_SCHEDULE_POLICY decode_first_prefill_later
LOCAL_FLOW_SCHEDULE_POLICY pld_srpt
```

其中 `pld_srpt` 是会话内本地队列调优得到的一个独立候选，用于长度和到达时间不一的 dispatch/prefill-like flow 排队场景。当前 50 负载多拓扑主线结果没有启用它，因此本文档不再把 `pld_srpt` 写成最佳组合的必选项。

### 6.1 通算分离与可抢占切片

chunk 机制引入的是“通信调度粒度”和“计算任务语义”的分离。计算侧仍把一次 expert dispatch flow 看成原子依赖：只有该 flow 的全部通信字节完成后，对应 expert 计算或 dispatch task 才能继续推进；通信侧则允许把这个大 flow 拆成多个可排队、可插队的 chunk，让本地队列调度器在网络层面近似实现抢占。

因此，chunk 不改变模型并行、专家选择、task 完成条件或 trace 访问计数，只改变发送队列中的服务粒度：

```text
compute flow F = (job, stage, src, dst, expert, total_bytes)

communication chunks C(F) = {c_1, c_2, ..., c_n}
sum_i bytes(c_i) = total_bytes

finish(F) = max_i finish(c_i)
```

所有 chunk 共享同一个 `flowGroupKey`，这使调度器能在通信侧统计 group 剩余字节，同时保持计算侧仍以原始 flow/group 为完成单位。这个表达对应通算分离：计算侧只关心依赖是否完成，通信侧负责在不破坏依赖语义的前提下重排发送顺序。

通过以下参数开启可抢占近似：

```text
LOCAL_FLOW_PREEMPTIVE_CHUNK_BYTES = <chunk bytes>
```

当 flow 大于 chunk bytes 且属于 dispatch/prefill 类 stage 时，提交路径会把它拆成多个 chunk。所有 chunk 保留相同 `flowGroupKey`，调度器按 group 统计剩余字节，从而近似可抢占 SRPT；而上层 job 仍等待该 group 的全部 chunk 完成后才认为原始通信依赖完成。

### 6.2 PLD-SRPT 评分

PLD-SRPT 对每个 flow group 计算：

```text
remaining_ns = standalone_fct(group_remaining_bytes)
age_ns       = now_ns - group_arrival_ns
debt_ns      = max(0, age_ns - D0 - kappa * remaining_ns)

score =
  (V + beta * debt_ns + 0.5 * debt_ns^2 / T0)
  / (remaining_ns + preempt_overhead_ns)
```

选择 score 最大的 group，其首个 chunk 出队。默认配置键：

```text
LOCAL_FLOW_PLD_SRPT_V_NS
LOCAL_FLOW_PLD_SRPT_T0_NS
LOCAL_FLOW_PLD_SRPT_D0_NS
LOCAL_FLOW_PLD_SRPT_KAPPA
LOCAL_FLOW_PLD_SRPT_BETA
LOCAL_FLOW_PLD_SRPT_PREEMPT_OVERHEAD_NS
```

直观解释：

- 分母是预计剩余服务时间，因此短任务天然优先；
- `debt_ns` 是等待时间超过“可接受等待窗口”后的队列债务；
- 二次债务项避免长任务永久饥饿；
- `preempt_overhead_ns` 用于惩罚过细切片导致的频繁抢占。

小规模 dispatch/prefill-like 实验路径：

```text
simulation/results/pld_srpt_try/
```

该实验中，`pld_srpt_varn` 相对 `fifo_chunk_varn`：

```text
avg_TPOT_ns:     132141 -> 67282
avg_latency_ns:  369535 -> 318097
```

平均 TPOT 改善约 49.08%。该结果说明 `pld_srpt` 对 mixed-length 本地队列有潜在价值，但它不是当前 50 负载 trace-driven dispatch-only 最佳组合的一部分。

## 7. 50 负载三拓扑对比结果

本次实验目录：

```text
simulation/results/topo14_ablation_50x
simulation/results/topo22_ablation_50x
simulation/results/topo41_ablation_50x
```

三组方法均完成 trace：

| topology | method | completed tasks | avg TPOT ns | trace accesses | avg queued flows |
|---|---|---:|---:|---:|---:|
| topo14 | `baseline_uniform` | 29 | 1680545 | 57600 / 57600 | 261.05 |
| topo14 | `baseline_trace` | 29 | 1928542 | 57485 / 57485 | 408.74 |
| topo14 | `no_l1_variance` | 29 | 1071901 | 57485 / 57485 | 267.10 |
| topo22 | `baseline_uniform` | 29 | 1634788 | 57600 / 57600 | 247.96 |
| topo22 | `baseline_trace` | 29 | 1971007 | 57485 / 57485 | 406.82 |
| topo22 | `no_l1_variance` | 29 | 1224318 | 57485 / 57485 | 265.81 |
| topo41 | `baseline_uniform` | 29 | 1592911 | 57600 / 57600 | 262.25 |
| topo41 | `baseline_trace` | 29 | 2150106 | 57485 / 57485 | 365.52 |
| topo41 | `no_l1_variance` | 29 | 1305741 | 57485 / 57485 | 265.57 |

相对收益：

```text
topo14 no_l1_variance vs baseline_trace:   44.42%
topo22 no_l1_variance vs baseline_trace:   37.88%
topo41 no_l1_variance vs baseline_trace:   39.27%

topo14 no_l1_variance vs baseline_uniform: 36.22%
topo22 no_l1_variance vs baseline_uniform: 25.11%
topo41 no_l1_variance vs baseline_uniform: 18.03%
```

该结果说明，在 trace 高偏斜访问下，posterior 频次驱动的 domain-spread 放置和 async probe 路由能显著降低 dispatch TPOT。三种拓扑上 `baseline_trace` 均慢于 `baseline_uniform`，说明真实 trace 访问偏斜会放大固定放置和默认路由的排队压力；`no_l1_variance` 在三种拓扑上均明显优于两个 baseline。

## 8. 输出与可视化

50 负载实验输出：

```text
simulation/results/topo14_ablation_50x/summary.csv
simulation/results/topo22_ablation_50x/summary.csv
simulation/results/topo41_ablation_50x/summary.csv
```

关键图：

- `figures/latency_tpot_bar.png`：平均 latency 与 TPOT；
- `figures/tpot_relative_to_baseline_trace.png`：相对 baseline trace 的 TPOT；
- `figures/queue_pressure_bar.png`：本地队列压力；
- `figures/expert_heatmap_<method>.png`：全过程累计的专家放置/访问热图；
- `figures/expert_placement_latest_<method>.png`：最后一次放置的 placement-only 热图。

由于 trace 专家访问长尾很强，热图默认只在可视化层使用：

```text
display = log1p(raw_weighted_access)
```

这不改变仿真、目标访问次数或 summary 指标。

## 9. 关键代码入口

- `simulation/scratch/pipeline_policy_config.h`：策略配置解析、专家放置、专家路由和本地队列选择器；
- `simulation/scratch/cclscheduler.h`：need placement、expert placement 数据结构与调度接口；
- `simulation/scratch/mnCCL.h`：dispatch 提交、route cache、async probe、本地 flow queue、完成统计；
- `simulation/run_dispatch_ablation_visual.py`：当前 50 负载 baseline/full 消融对比、summary 解析和可视化；
- `simulation/run_dispatch_full_compare_visual.py`：历史全量对比、summary 解析和可视化；
- `simulation/run_dispatch_param_sweep_visual.py`：参数扫描；
- `simulation/plot_expert_heatmap.py`：专家放置/访问热图。

## 10. 当前推荐配置

当前 50 负载最佳组合：

```text
EXPERT_PLACEMENT_POLICY          = probe_balanced_domain_spread
SAME_RANK_ROUTE_POLICY           = probe_rtt_delta
PLACEMENT_ACCESS_MODE            = posterior
PLACEMENT_REPLICA_BALANCE_WEIGHT = 6
PLACEMENT_EXTERNAL_WEIGHT        = 8
PLACEMENT_L1_VARIANCE_WEIGHT     = 0
PLACEMENT_L1_MAX_WEIGHT          = 1.5
PLACEMENT_GPU_VARIANCE_WEIGHT    = 0.5
PLACEMENT_PROBE_WEIGHT           = 7
ASYNC_ROUTE_PROBE_ENABLE         = 1
ASYNC_ROUTE_LIVE_ROUTE           = 1
ASYNC_ROUTE_PROBE_INTERVAL_NS    = 20000
ASYNC_ROUTE_PROBE_BUDGET         = 128
ASYNC_ROUTE_PROBE_REFRESH_NS     = 500000
PROBE_BYTES                      = 1
ROUTE_PROBE_MAX_INFLIGHT         = 256
LOCAL_FLOW_SCHEDULE_POLICY       = default
```

保留但不属于当前最佳组合的本地队列候选：

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

因此，复现当前设计文档中的最佳策略组时，应使用 `run_dispatch_ablation_visual.py` 中的 `no_l1_variance` 方法；不要再启用 EMA、新 partition placement 或 PLD-SRPT 作为主线默认项。
