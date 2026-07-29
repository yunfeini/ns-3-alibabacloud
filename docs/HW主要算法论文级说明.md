# HW 主要算法论文级说明

本文档形式化说明当前 HW trace-driven dispatch-only 仿真中的核心策略模块：

1. 专家放置算法：posterior-aware probe-balanced domain-spread placement；
2. 专家路由算法：async live probe RTT-delta replica routing；
3. 本地队列调度接口：默认本地队列调度为当前最佳组合的一部分，PLD-SRPT 作为独立候选保留。

当前 50 负载推荐策略组由 `topo14/topo22/topo41` 多拓扑结果确定，写作“posterior-aware domain-spread placement with score-v2 no L1 variance + async live probe routing + default local scheduling”。也就是说，当前主线不再使用 adaptive/EMA/partition placement，也不把 PLD-SRPT 写成最佳组合的必选项。

## 1. 系统模型

专家集合为：

```text
E = {0, 1, ..., M - 1}
```

其中 `M = EXPERT_NUM`。参与 dispatch 的 GPU multiset 为：

```text
G = {g_1, g_2, ..., g_N}
```

其中 `N = DISPATCH_NEED`。同一物理 GPU 可以在 multiset 中重复出现，表示它承接多个逻辑 placement slot。唯一物理 GPU 集合为：

```text
U = unique(G)
```

每个 GPU `g` 所属 L1 交换域为：

```text
d(g) in D
```

该映射由拓扑中的 network node 通过 `L1GroupOfNetworkNode()` / `L1GroupOfGpu()` 得到。

每个 dispatch task 中，每张源 GPU 最多选择：

```text
K = DISPATCH_TOPK
```

个目的专家。单个专家 flow 的字节数为：

```text
B_msg = DISPATCH_EXPERT_FFN_PARAMS
      * DISPATCH_PRECISION_BYTES
      * DISPATCH_BATCH_SIZE
      * DISPATCH_N
```

当前 DeepSeek V3 1B 近似配置中：

```text
M       = 256
N       = 256
K       = 8
B_msg   = 112000 * 1 * 16 * 1 bytes
```

## 2. Trace Workload Realization

trace 文件给出每个专家的目标访问次数：

```text
A_e >= 0, e in E
```

若启用 `TRACE_ACCESS_DIVISOR = z`，则非零目标按向上取整缩放：

```text
A'_e =
  0,             if A_e = 0
  ceil(A_e / z), otherwise
```

后续记 `A_e` 为缩放后的目标。运行时已观测访问次数为：

```text
N_e(t) = observed accesses to expert e before time t
```

剩余访问量：

```text
R_e(t)     = max(0, A_e - N_e(t))
R_total(t) = sum_e R_e(t)
```

trace 完成条件为：

```text
forall e in E: N_e(t) >= A_e
```

目的专家选择遵循贪心剩余目标规则。对每个 dispatch task、每张源 GPU，在一次 top-k 内不重复选择同一专家：

```text
TraceTargetSelection

for src in G:
  picked <- empty set
  for i in 1..K:
    C <- {e in E : e not in picked and N_e < A_e}
    if C is empty:
      break
    e* <- argmax_{e in C} (A_e - N_e, -e)
    picked <- picked union {e*}
    N_e* <- N_e* + 1
    submit dispatch flow src -> expert e*
```

### 性质 1：有限完成

若每条已提交 flow 最终完成，且 `sum_e A_e` 有限，则 trace workload 在有限条有效 flow 后完成。

证明：每提交一条有效 flow，`R_total` 减少 1。`R_total(0)=sum_e A_e` 有限且非负，因此最多 `sum_e A_e` 条有效 flow 后归零。证毕。

## 3. 算法一：专家放置

### 3.1 问题定义

放置变量：

```text
x_{e,g} =
  1, if expert e is placed on physical GPU g
  0, otherwise
```

每张参与 GPU 的基础槽位数：

```text
B =
  max(EXPERT_PER_GPU, ceil(M / N)), if N < M
  EXPERT_PER_GPU,                   otherwise
```

若物理 GPU `g` 在 multiset `G` 中出现 `occ_G(g)` 次，则容量为：

```text
C_g = min(M, occ_G(g) * B)
```

合法放置满足：

```text
forall e: sum_{g in U} x_{e,g} >= 1
forall g: sum_{e in E} x_{e,g} <= C_g
forall e,g: x_{e,g} in {0,1}
```

第三个约束表示同一专家不能在同一物理 GPU 上重复放置。

### 3.2 在线访问频次

放置不能直接使用未来访问序列。系统使用运行时访问估计：

```text
f_initial(e)   = configured EXPERT_ACCESS_FREQ[e]
f_posterior(e) = alpha + N_e(t)
```

其中：

```text
alpha = EXPERT_ACCESS_PRIOR
```

当前 50 负载最好结果使用：

```text
f_e(t) = f_posterior(e)
```

为避免短 trace 中过早追噪声，系统仍保留 drift gate：

```text
p_initial(e)   = f_initial(e) / sum_j f_initial(j)
p_posterior(e) = f_posterior(e) / sum_j f_posterior(j)

Delta(t) = 0.5 * sum_e |p_initial(e) - p_posterior(e)|
```

当样本数达到 `PLACEMENT_DRIFT_MIN_SAMPLES` 且 `Delta(t)` 超过阈值时，`initial_until_drift` 模式切换到 posterior。

### 3.3 目标函数

L1 domain 负载：

```text
L_d(x) = sum_{g in U: d(g)=d} sum_e f_e x_{e,g}
```

最大 L1 负载与方差：

```text
L_max(x) = max_d L_d(x)
V_L1(x)  = Var({L_d(x): d in D})
```

专家 `e` 在 domain `d` 的副本数：

```text
c_{e,d}(x) = sum_{g in U: d(g)=d} x_{e,g}
```

外部访问暴露度：

```text
X(x) = sum_e f_e * |{d in D : c_{e,d}(x) = 0}|
```

同专家副本域间不均衡度：

```text
R(x) = sum_e f_e * Var({c_{e,d}(x): d in D})
```

GPU 加权访问负载方差：

```text
V_G(x) = Var({sum_e f_e x_{e,g}: g in U})
```

probe path cost 定义为：

```text
C_probe(s,g) =
    lambda_delta * |T_low(s,g) - T_high(s,g)| / T_base(s,g)
  + lambda_rtt   * min(T_low(s,g), T_high(s,g)) / T_base(s,g)
```

其中 `s` 取每个 L1 domain 的代表源 GPU。专家 `e` 的 expected-best probe cost：

```text
P_e(x) = (1 / |S|) * sum_{s in S} min_{g: x_{e,g}=1} C_probe(s,g)
P(x)   = sum_e f_e P_e(x)
```

归一化 objective：

```text
J_place(x) =
    w_x * X_norm(x)
  + w_r * R_norm(x)
  + w_v * V_L1_norm(x)
  + w_m * L_max_norm(x)
  + w_g * V_G_norm(x)
  + w_p * P_norm(x)
```

当前 50 负载最佳组合使用 score-v2 no-l1-variance 权重：

```text
w_r = PLACEMENT_REPLICA_BALANCE_WEIGHT = 6.0
w_x = PLACEMENT_EXTERNAL_WEIGHT = 8.0
w_v = PLACEMENT_L1_VARIANCE_WEIGHT = 0.0
w_m = PLACEMENT_L1_MAX_WEIGHT = 1.5
w_g = PLACEMENT_GPU_VARIANCE_WEIGHT = 0.5
w_p = PLACEMENT_PROBE_WEIGHT = 7.0
```

`V_L1_norm` 仍在日志中保留，但当前推荐权重为 0。消融结果显示，L1 方差与 `L_max_norm`、`V_G_norm` 的作用重叠，并且会在热点 trace 下引入过度均匀化；因此当前最佳组合只使用 L1 最大负载和 GPU 热度方差作为均衡 guardrail。

### 3.4 两阶段启发式

严格求解上述整数规划开销较大，因此当前实现采用 coverage + surplus 两阶段启发式。

Coverage 阶段对专家按 `f_e` 降序排序。候选 GPU 的字典序 key 为：

```text
K_cover(e,g) =
  (
    c_{e,d(g)},
    L_{d(g)} + f_e,
    load_gpu(g),
    node_id(g),
    gpu_id(g)
  )
```

Surplus 阶段在 coverage 后填充剩余槽位。候选专家 key 为：

```text
K_surplus(e,g) =
  (
    c_{e,d(g)},
    replica_count(e) / sqrt(max(1, f_e)),
    L_{d(g)} + f_e,
    -f_e,
    e
  )
```

伪代码：

```text
Algorithm 1: PosteriorProbeBalancedDomainSpreadPlacement

Input:
  E, G, f_e, EXPERT_PER_GPU, L1 domain d(g)

Output:
  replica set R_e for every expert e

1. U <- unique(G)
2. compute C_g for every g in U
3. sort GPUs by (L1 domain, node id, gpu id)
4. sort experts by (-f_e, expert id)
5. initialize all R_e empty

Coverage phase:
6. for e in sorted experts:
7.   C <- {g in U: load(g) < C_g and e not on g}
8.   choose g* = argmin_g K_cover(e,g)
9.   place e on g*

Surplus phase:
10. for g in sorted GPUs:
11.   while load(g) < C_g:
12.     C <- {e in E: e not on g}
13.     if C is empty: break
14.     choose e* = argmin_e K_surplus(e,g)
15.     place e* on g

16. return {R_e}
```

### 3.5 性质

容量合法性：算法每次放置前检查 `load(g) < C_g`，因此任意 GPU 满足：

```text
sum_e x_{e,g} <= C_g
```

单 GPU 不重复专家：候选集合要求 `e not on g`，因此：

```text
x_{e,g} <= 1
```

覆盖性：若总容量满足：

```text
sum_g C_g >= M
```

则 coverage phase 能为每个专家放置至少一个副本。

证明：第 `k < M` 次 coverage 放置前，最多占用 `k` 个槽位。由总容量条件，至少有一个空槽；当前专家还未放置过，因此该空槽 GPU 满足 `e not on g`。故每个专家均能完成一次放置。证毕。

复杂度：设总槽位为 `S=sum_g C_g`，唯一 GPU 数为 `|U|`。Coverage 为 `O(M|U|)`，Surplus 最坏为 `O((S-M)M)`。当前 256 expert、256 GPU、每 GPU 9 槽位时 `S=2304`，可以在 task 到达路径内运行。

## 4. 算法二：专家路由

### 4.1 问题定义

给定专家放置结果：

```text
R_e = {g in U : x_{e,g}=1}
```

路由算法为每个源 GPU `s` 和目的专家 `e` 选择一个副本：

```text
r(s,e) in R_e
```

并构建 route cache：

```text
route[source_gpu][dst_expert] = selected_replica_gpu
```

### 4.2 Local-first 规则

路由先应用确定性本地规则：

1. 若 `s in R_e`，选择 `s`；
2. 否则若 `R_e` 中存在与 `s` 同主机的 GPU，选择同主机副本；
3. 否则进入 probe score。

该规则保证如果源 GPU 本身放置了目标专家，则 dispatch 不产生跨 GPU 网络流。

### 4.3 Probe RTT-Delta Score

对候选副本 `g`，定义：

```text
delta(s,g) = |T_low(s,g) - T_high(s,g)|
base(s,g)  = ProbeRtt_high(s,g)
rtt(s,g)   = min(T_low(s,g), T_high(s,g))
```

其中 `T_high` 和 `T_low` 来自高优先级与低优先级 1 字节 probe 的 FCT。缺失真实 probe 样本时，使用 `ProbeRtt()` 冷启动估计。

候选目标 GPU 的队列压力：

```text
Q(g) = queued_local_flows(g) + active_flow_indicator(g)
```

最终路由评分：

```text
Score_route(s,g) =
    delta(s,g) / base(s,g)
  + 0.25 * rtt(s,g) / base(s,g)
  + w_q * Q(g) / Q_norm
```

当前 50 负载最佳组合中：

```text
w_q = ROUTE_QUEUE_WEIGHT = 0
```

队列压力项作为接口保留，用于把算法二的副本选择与算法三的本地队列状态耦合。

### 4.4 异步 Probe 与 Live Route

异步 probe 持续刷新样本表 `H`：

```text
Algorithm 2: AsyncLiveProbeRttDeltaRouting

Route cache build:
1. for each source GPU s:
2.   for each expert e:
3.     if s in R_e:
4.       route[s][e] <- s
5.     else if exists same-host g in R_e:
6.       route[s][e] <- g
7.     else:
8.       for g in R_e:
9.         read live high/low probe FCT from H if present
10.        otherwise use ProbeRtt() cold-start estimate
11.        compute Score_route(s,g)
12.      route[s][e] <- argmin_g Score_route(s,g)

Background probe:
13. every ASYNC_ROUTE_PROBE_INTERVAL_NS:
14.   candidates <- stale or missing source/replica pairs
15.   submit up to ASYNC_ROUTE_PROBE_BUDGET candidate pairs
16.   respect ROUTE_PROBE_MAX_INFLIGHT
17.   for each pair, send one high-priority and one low-priority probe
18.   update H when probes finish
```

推荐配置：

```text
ASYNC_ROUTE_PROBE_ENABLE      = 1
ASYNC_ROUTE_LIVE_ROUTE        = 1
ASYNC_ROUTE_PROBE_INTERVAL_NS = 20000
ASYNC_ROUTE_PROBE_BUDGET      = 128
ASYNC_ROUTE_PROBE_REFRESH_NS  = 500000
PROBE_BYTES                   = 1
ROUTE_PROBE_MAX_INFLIGHT      = 256
```

### 4.5 性质

本地命中不劣性：当 `s in R_e` 时，算法直接返回 `s`，不产生网络发送。在非负网络时延模型下，选择本地副本不劣于选择远端副本。

退化性：当 `ASYNC_ROUTE_PROBE_ENABLE=0` 或没有 live 样本时，算法使用冷启动 RTT 估计。当 `w_q=0` 时，评分退化为纯 RTT/delta：

```text
Score_route = delta/base + 0.25*rtt/base
```

复杂度：若每个专家平均有 `r` 个候选副本，route cache 覆盖 `S_src` 个源 GPU 和 `M` 个目的专家，则构建复杂度为：

```text
O(S_src * M * r)
```

实际 probe 提交受 `ROUTE_PROBE_MAX_INFLIGHT`、`ASYNC_ROUTE_PROBE_BUDGET` 和刷新间隔限制。

### 4.6 Probe 开销

`topo.txt` 50 负载 no-l1-variance run 中：

```text
data_submitted_flows          = 51956
data_submitted_bytes          = 93105152000
probe_submitted_flows         = 412160
probe_submitted_bytes         = 412160
probe_submitted_flow_ratio    = 0.888054
probe_submitted_byte_ratio    = 4.4268e-06
probe_submitted_percent       = 0.00044268
```

probe 在 flow 数量上占比高，但每条只有 1 字节，因此字节开销约为总提交流量的 `0.00044268%`。

## 5. 本地队列调度接口

### 5.1 问题定义

每个 GPU `g` 维护本地待发送队列：

```text
Q_g(t) = {i}
```

每个 flow 或 chunk `i` 包含：

```text
b_i      = bytes of this chunk
a_i      = arrival time
group(i) = original flow group id
B_i(t)   = remaining bytes of group(i)
```

dispatch-only 仿真中，dispatch flow 可以被看作 prefill-like 流：长度和到达时间不同，目标是降低平均完成等待时间和 TPOT。当前 50 负载最佳组合的本地队列调度配置为：

```text
LOCAL_FLOW_SCHEDULE_POLICY default
```

PLD-SRPT 仍作为队列维度的独立候选保留，但它不是当前 50 负载 trace-driven dispatch-only 最佳组合的一部分。

### 5.2 通算分离与可抢占近似

真实网络 flow 一旦提交后不可任意中断。实现中使用 chunking 近似抢占，其核心思想是将计算侧依赖图和通信侧服务队列解耦。

设计算侧生成的原始通信依赖为：

```text
F_i = (job_i, stage_i, src_i, dst_i, expert_i, B_i)
```

计算侧语义要求 `F_i` 作为一个依赖整体完成后，后续 expert 计算或 dispatch task 才能继续。通信侧将 `F_i` 映射为 chunk 集合：

```text
C_i = {c_{i,1}, c_{i,2}, ..., c_{i,n_i}}
sum_k b_{i,k} = B_i
```

两侧通过 group id 绑定：

```text
group(c_{i,k}) = i
finish(F_i) = max_k finish(c_{i,k})
```

这样，通信侧调度器可以在 `c_{i,k}` 粒度上重排、插队和近似抢占；计算侧仍只观察 `F_i` 是否完成，不感知 chunk 内部顺序。该通算分离表达保证网络调度获得更细粒度控制，同时不破坏原始模型执行语义。

对应实现参数为：

```text
LOCAL_FLOW_PREEMPTIVE_CHUNK_BYTES = q
```

若原始 flow 大小大于 `q`，则拆为：

```text
n_i = ceil(size_i / q)
```

个 chunk。所有 chunk 共享：

```text
flowGroupKey
flowGroupBytes
flowGroupArrivalTimeNs
```

调度器按 group 聚合剩余字节，选择某个 group 后发出该 group 的第一个等待 chunk。上层完成判断仍以 group 为单位，因此 chunking 只改变通信侧排队和服务顺序，不改变计算侧依赖边。

### 5.3 Lyapunov-Drift SRPT Score

对 group `j`，估计剩余服务时间：

```text
S_j(t) = standalone_fct(remaining_bytes_j)
```

等待年龄：

```text
A_j(t) = t - arrival_time_j
```

定义队列债务：

```text
D_j(t) = max(0, A_j(t) - D0 - kappa * S_j(t))
```

PLD-SRPT 选择评分：

```text
Score_queue(j,t) =
  (V + beta * D_j(t) + 0.5 * D_j(t)^2 / T0)
  / (S_j(t) + O_preempt)
```

其中：

- `V` 控制 SRPT 项强度；
- `D0` 是等待宽限；
- `kappa` 表示允许等待时间随服务时间增长；
- `beta` 和二次项表示 Lyapunov debt 压力；
- `T0` 归一化债务增长；
- `O_preempt` 是每次切片抢占的额外代价估计。

伪代码：

```text
Algorithm 3: PreemptiveLyapunovSrptLocalScheduling

Input:
  local queue Q_g(t), current time t

Output:
  selected chunk index

1. group waiting chunks by flowGroupKey
2. for each group j:
3.   B_j <- sum remaining bytes over waiting chunks in group j
4.   a_j <- flowGroupArrivalTimeNs
5.   S_j <- standalone_fct(B_j)
6.   A_j <- max(0, t - a_j)
7.   D_j <- max(0, A_j - D0 - kappa * S_j)
8.   score_j <- (V + beta * D_j + 0.5 * D_j^2 / T0) / (S_j + O_preempt)
9. choose j* with largest score
10. tie-break by smaller S_j, earlier arrival, smaller sequence
11. return first waiting chunk of j*
```

### 5.4 Lyapunov 解释

令队列债务向量为 `D(t)`，Lyapunov 函数为：

```text
L(t) = 0.5 * sum_j D_j(t)^2
```

若某个 group 得到服务，下一时刻它的剩余服务时间下降，未来债务增长也下降。算法中的分子：

```text
V + beta * D_j + 0.5 * D_j^2 / T0
```

可视作“完成短任务带来的平均等待收益”与“降低债务漂移”的合成权重；分母 `S_j + O_preempt` 表示单位服务时间的收益密度。因此它近似选择最大单位时间 drift-plus-penalty 改善的 group。

当所有 `D_j=0` 时：

```text
Score_queue(j,t) = V / (S_j + O_preempt)
```

算法退化为 SRPT。当某个长任务等待过久，`D_j` 的线性和二次项会增加其优先级，从而避免饥饿。

### 5.5 复杂度

设本地队列中有 `n` 个 chunk，聚合后有 `m` 个 group。一次调度：

```text
O(n + m)
```

其中 `n` 用于聚合剩余字节，`m` 用于计算评分。当前实现直接扫描队列，便于和现有仿真逻辑集成。

### 5.6 小规模验证

小规模实验目录：

```text
simulation/results/pld_srpt_try/
```

对比结果：

| method | completed | avg latency ns | avg TPOT ns | data flows | data bytes |
|---|---:|---:|---:|---:|---:|
| `fifo_chunk_varn` | 12 | 369535 | 132141 | 2830 | 741867520 |
| `pld_srpt_varn` | 12 | 318097 | 67282 | 2830 | 741867520 |

在相同数据流量下，PLD-SRPT 将平均 TPOT 从 `132141 ns` 降至 `67282 ns`，改善约 `49.08%`。

该结果说明 `pld_srpt` 对 mixed-length 本地队列有潜在价值。对应配置为：

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

## 6. 策略模块交互

当前主线模块的触发关系为：

```text
new dispatch task arrives
  -> update posterior access estimate
  -> run expert placement
  -> rebuild route cache using async live probe table
  -> submit dispatch flows
  -> each source GPU local queue sends flows by default scheduling
```

新的放置和路由决策只影响后续新 task 和后续提交的 flow，不修改已经在网络中运行的 flow。本地队列接口仍在每个 GPU 出队时生效，但当前最佳组合不启用 PLD-SRPT chunking。

交互关系：

- 专家放置决定候选副本集合 `R_e`，影响本地命中率、L1 外部访问和 route search space；
- 专家路由在候选副本中选择实际目的 GPU，利用 probe RTT/delta 避开拥塞路径；
- 本地队列接口决定同一源 GPU 上多个 flow 的发送顺序，当前最佳组合使用默认调度；
- route score 中的 `ROUTE_QUEUE_WEIGHT` 可以读取候选目的 GPU 队列压力，是路由和队列状态的连接点。当前最好结果中该权重为 0，表示专家路由没有额外加权目的端队列压力。

## 7. 50 负载三拓扑最好结果

实验目录：

```text
simulation/results/topo14_ablation_50x
simulation/results/topo22_ablation_50x
simulation/results/topo41_ablation_50x
```

结果：

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

相对改善：

```text
topo14 no_l1_variance vs baseline_trace:   44.42%
topo22 no_l1_variance vs baseline_trace:   37.88%
topo41 no_l1_variance vs baseline_trace:   39.27%

topo14 no_l1_variance vs baseline_uniform: 36.22%
topo22 no_l1_variance vs baseline_uniform: 25.11%
topo41 no_l1_variance vs baseline_uniform: 18.03%
```

结论：

1. trace 访问偏斜会显著放大固定放置和默认路由的排队压力，三个拓扑上 `baseline_trace` 均慢于 `baseline_uniform`；
2. posterior 频次驱动的 domain-spread 放置能给热点专家更多且更分散的副本；
3. async live probe 路由能在候选副本之间选择更低动态拥塞的路径；
4. L1 方差项在当前 score 中可删去，`l1_max` 与 GPU 热度方差已提供更直接的均衡约束；
5. probe 字节开销约 `0.00044268%`，远小于数据流量。

## 8. 可复现实验与输出

50 负载单拓扑对比模板：

```bash
python3 simulation/run_dispatch_ablation_visual.py \
  --output simulation/results/<topology>_ablation_50x \
  --network-config <network-config-for-topology> \
  --trace-access-divisor 50 \
  --methods baseline_uniform baseline_trace no_l1_variance \
  --keep-going
```

主要输出：

- `summary.csv`：跨方法指标；
- `README.md`：实验配置和结果摘要；
- `figures/latency_tpot_bar.png`：平均 latency / TPOT；
- `figures/tpot_relative_to_baseline_trace.png`：相对 baseline trace；
- `figures/queue_pressure_bar.png`：本地队列压力；
- `figures/expert_heatmap_<method>.png`：全过程累计专家放置与访问热图；
- `figures/expert_placement_latest_<method>.png`：最后一次放置热图。

关键统计字段：

```text
average_latency_ns
average_TPOT_ns
completed_tasks
trace_completed
trace_target_accesses
trace_observed_accesses
queued_local_flows_avg
queued_local_flows_p95
data_submitted_bytes
probe_submitted_bytes
probe_submitted_byte_ratio
probe_submitted_percent
```

由于 trace 访问频次长尾极强，热图默认使用：

```text
display = log1p(raw_weighted_access)
```

该变换仅用于可视化，不改变仿真、目标访问次数或 summary 指标。
