// Minimal in-simulator collective (mnCCL) helpers.
// Header-only to avoid multiple-definition issues in scratch builds.
#ifndef MNCCL_H
#define MNCCL_H

#include <map>
#include <vector>
#include <mutex>
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <functional>
#include <string>
#include <sstream>
#include "ccl_log.h"
#include "common.h"
#include "cclscheduler.h"


namespace mnccl {

// Map JID -> outstanding message count
static std::map<uint32_t, uint32_t> g_outstanding;
// Map JID -> allocated GPUs (vector of <node,gpu_idx>)
static std::map<uint32_t, std::vector<std::pair<uint32_t,uint32_t>>> g_alloc;
// Map encoded key (pg<<48|src<<32|dst<<16|port) -> JID
static std::map<uint64_t, uint32_t> g_key2JID;

struct FlowFinishRecord {
  Ptr<RdmaQueuePair> qp;
  uint64_t msgSize;
  uint64_t finishTimeNs;
  uint64_t actualFct;
  uint64_t standaloneFct;
};

// Map JID -> completed mnCCL messages. Stored for logging when the whole job
// finishes. Protected by g_mutex.
static std::map<uint32_t, std::vector<FlowFinishRecord>> g_JID2FlowFinishes;

// Mutex protecting the above maps
static std::mutex g_mutex;
// Path to write flow completion records.
static std::string g_flow_finish_log_path;

struct CollectiveJob {
  uint32_t jobId;
  CollectiveOp op;
  uint16_t pg;
  uint32_t need;
  double submitTime;
  uint64_t msgSize;
  uint32_t root;
  uint32_t k;
  uint32_t expert_num;
  uint64_t expert_mem_bytes;
};

struct PipelineTask {
  uint32_t taskId;
  double submitTime;
  uint32_t prefillLength;
  uint32_t decodeLength;
};

struct RuntimeConfig {
  uint16_t pg;
  uint32_t need_prefill;
  uint32_t need_decode;
  uint32_t expert_num;
  uint32_t k;
  uint32_t single_token_length;
  uint64_t expert_mem_bytes;
  uint64_t token_msg_size;
  uint64_t base_decode_compute_delay_ns;
};

enum class PipelineStage : uint8_t {
  PrefillAllReduce = 0,
  PrefillAllToAll,
  KvCacheTransfer,
  DecodeAllReduce,
  DecodeAllToAll,
  Generic = 255
};

struct PipelineStageRef {
  uint32_t taskId;
  PipelineStage stage;
};

struct PipelineTaskState {
  PipelineTask task;
  uint32_t prefillPlacementOwner;
  uint32_t decodePlacementOwner;
  uint32_t actualNeedPrefill;
  uint32_t actualNeedDecode;
  uint32_t prefillAllReduceJobId;
  uint32_t prefillAllToAllJobId;
  uint32_t kvCacheJobId;
  uint32_t currentDecodeAllReduceJobId;
  uint32_t currentDecodeAllToAllJobId;
  uint32_t decodeIterations;
  uint32_t decodeIteration;
  uint32_t prefillSideDecodeIterations;
  uint64_t startTimeNs;
  uint64_t prefillFinishNs;
  uint64_t decodeStartNs;
  uint64_t kvCacheFinishNs;
  uint64_t firstTokenFinishNs;
  uint64_t lastTokenFinishNs;
  std::vector<std::pair<uint32_t,uint32_t>> prefillGpus;
  std::vector<std::pair<uint32_t,uint32_t>> decodeGpus;
  bool prefillPlacementReleased;
  bool decodePlacementReleased;
  bool kvCacheTransferred;
};

struct LocalFlowTask {
  uint64_t key;
  uint64_t sequence;
  uint32_t jobId;
  CollectiveOp op;
  PipelineStage stage;
  cclScheduler::PlacementKind pdKind;
  uint16_t pg;
  uint32_t srcNode;
  uint32_t srcGpu;
  uint32_t dstNode;
  uint32_t dstGpu;
  uint16_t srcPort;
  uint64_t msgSize;
  uint64_t arrivalTimeNs;
};

struct ExpertRouteRequest {
  uint32_t jobId;
  uint32_t expertPlacementOwner;
  uint32_t expertId;
  PipelineStage stage;
  cclScheduler::PlacementKind pdKind;
  std::pair<uint32_t,uint32_t> sourceGpu;
  std::vector<std::pair<uint32_t,uint32_t>> candidates;
};

struct PdSplitRequest {
  PipelineTask task;
  RuntimeConfig config;
  uint32_t decodeIterations;
};

using LocalFlowSchedulePolicy =
  std::function<size_t(const std::vector<LocalFlowTask>& queue, uint64_t nowNs)>;

using ExpertRoutePolicy =
  std::function<std::pair<uint32_t,uint32_t>(const ExpertRouteRequest& req)>;

using PdSplitPolicy =
  std::function<uint32_t(const PdSplitRequest& req)>;

//forward declaration
inline void SubmitJob(uint32_t jobId, int need, double sim_time, uint64_t workload);
inline void SubmitColJob(const CollectiveJob& job);
inline void SubmitPipelineTask(const PipelineTask& task);
inline void TryStartPipelineTask(uint32_t taskId);
inline void StartPrefillAllReduce(PipelineTaskState& state);
inline void StartPrefillAllToAll(PipelineTaskState& state);
inline void TryStartDecode(uint32_t taskId);
inline void StartKvCacheTransfer(PipelineTaskState& state);
inline void StartNextDecodeToken(uint32_t taskId);
inline void StartDecodeAllReduce(PipelineTaskState& state);
inline void StartDecodeAllToAll(PipelineTaskState& state);
inline void FinishPipelineTask(uint32_t taskId);
inline void LogPipelineTaskSummary(const PipelineTaskState& state,
                                   uint64_t nowNs,
                                   const RuntimeConfig& cfg,
                                   bool completed);
inline void LogPipelineSummariesAtSimulationEnd();
inline uint64_t DecodeComputeDelayNs(const PipelineTaskState& state);
inline void ConfigureRuntime(const RuntimeConfig& cfg);
inline void SetGlobalPipelineNeeds(uint32_t needPrefill, uint32_t needDecode);
inline void SetModelParallelConfig(uint32_t expertNum);
inline size_t FifoLocalFlowSchedulePolicy(const std::vector<LocalFlowTask>& queue, uint64_t nowNs);
inline void SetLocalFlowSchedulePolicy(LocalFlowSchedulePolicy policy);
inline std::pair<uint32_t,uint32_t> RandomExpertRoutePolicy(const ExpertRouteRequest& req);
inline void SetExpertRoutePolicy(ExpertRoutePolicy policy);
inline uint32_t NoPrefillSideDecodeSplitPolicy(const PdSplitRequest& req);
inline void SetPdSplitPolicy(PdSplitPolicy policy);
inline uint32_t ResolvePrefillSideDecodeIterations(const PipelineTask& task,
                                                   const RuntimeConfig& cfg,
                                                   uint32_t decodeIterations);
inline RuntimeConfig GetRuntimeConfig();
inline RuntimeConfig RefreshRuntimeConfig(cclScheduler::NeedEvent event, const PipelineTask& task);
inline uint64_t PrefillAllReduceMsgSize(const PipelineTask& task);
inline uint64_t PrefillAllToAllMsgSize(const PipelineTask& task);
inline uint64_t KvCacheMsgSize(const PipelineTask& task);
inline uint64_t TokenBytes(uint32_t tokens);
inline uint32_t DecodeIterationsForTask(const PipelineTask& task);
inline uint64_t DecodeIterationMsgSize(const PipelineTaskState& state);
inline void OnCollectiveFinished(uint32_t jobId);
inline uint64_t MakeGpuKey(uint32_t node, uint32_t gpu);
inline void TryScheduleLocalGpu(uint64_t localGpuKey);
inline void DispatchLocalFlow(const LocalFlowTask& task);
inline void SubmitFlow(uint16_t pg,
                       const std::pair<uint32_t,uint32_t>& srcGpu,
                       const std::pair<uint32_t,uint32_t>& dstGpu,
                       uint64_t bytes,
                       std::vector<uint64_t>& keys,
                       uint32_t jobId,
                       CollectiveOp op,
                       PipelineStage stage,
                       cclScheduler::PlacementKind pdKind);
inline void SubmitFlow(uint16_t pg,
                       uint32_t src,
                       uint32_t dst,
                       uint64_t bytes,
                       std::vector<uint64_t>& keys);
inline uint16_t SubmitAllReduce(uint32_t jobId,
                                uint16_t pg,
                                const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                                uint64_t msgSize,
                                bool releaseOnFinish,
                                bool recordFlowFinish,
                                bool logSubmit);
inline uint16_t SubmitAllReduceForStage(uint32_t jobId,
                                        uint16_t pg,
                                        const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                                        uint64_t msgSize,
                                        bool releaseOnFinish,
                                        bool recordFlowFinish,
                                        bool logSubmit,
                                        PipelineStage stage,
                                        cclScheduler::PlacementKind pdKind);
inline uint16_t SubmitBroadcast(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root = 0);
inline uint16_t SubmitReduce(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root = 0);
inline uint16_t SubmitGather(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root = 0);
inline uint16_t SubmitScatter(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root = 0);
inline uint16_t SubmitAllGather(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize);
inline uint16_t SubmitReduceScatter(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize);
inline uint16_t SubmitAllToAll(uint32_t jobId,
                               uint16_t pg,
                               const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                               uint64_t msgSize,
                               uint32_t k = 1,
                               uint32_t expert_num = 1,
                               uint64_t expert_mem_bytes = 0,
                               bool releaseOnFinish = true,
                               bool recordFlowFinish = true,
                               bool logSubmit = true,
                               PipelineStage stage = PipelineStage::Generic,
                               cclScheduler::PlacementKind pdKind = cclScheduler::PlacementKind::Generic,
                               uint32_t expertPlacementOwner = 0);
inline uint16_t SubmitCollective(uint32_t jobId,
                                 CollectiveOp op,
                                 uint16_t pg,
                                 const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                                 uint64_t msgSize,
                                 uint32_t root,
                                 uint32_t k,
                                 uint32_t expert_num,
                                 uint64_t expert_mem_bytes,
                                 bool releaseOnFinish,
                                 bool recordFlowFinish,
                                 bool logSubmit);
inline uint16_t SubmitKvCacheTransfer(uint32_t jobId,
                                      uint16_t pg,
                                      const std::vector<std::pair<uint32_t,uint32_t>>& prefillGpus,
                                      const std::vector<std::pair<uint32_t,uint32_t>>& decodeGpus,
                                      uint64_t msgSize);
inline bool UsePrefillPlacementForNextDecodeToken(const PipelineTaskState& state);
inline const std::vector<std::pair<uint32_t,uint32_t>>& ActiveDecodeGpus(const PipelineTaskState& state);
inline uint32_t ActiveDecodePlacementOwner(const PipelineTaskState& state);
inline cclScheduler::PlacementKind ActiveDecodePlacementKind(const PipelineTaskState& state);
inline const char* PipelineStageName(PipelineStage stage);
inline bool OnMessageFinish(FILE* fout, Ptr<RdmaQueuePair> q, uint64_t msgSize);

inline void SetFlowFinishLogPath(const std::string &path) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_flow_finish_log_path = path;
}

std::vector<std::pair<uint32_t,uint32_t>> participants;
uint32_t JID = 1; // 任务 ID，可以根据实际情况生成唯一 ID
uint32_t TID = 1; // pipeline task ID
uint32_t PID = 0x80000000u; // placement owner ID, independent from collective JID
uint16_t default_pg = 3; // 流优先级
int need = 128; // 需要的 GPU 数量
double sim_time = 0.0001; // 模拟时间，单位秒

// Map jobId -> per-job local ranks (0..n-1)
static std::map<uint32_t, std::vector<uint32_t>> g_job_ranks;
static std::map<uint32_t, PipelineTaskState> g_pipeline_tasks;
static std::map<uint32_t, PipelineStageRef> g_pipeline_stage_by_job;
static std::map<uint32_t, bool> g_release_on_finish;
static std::map<uint32_t, bool> g_record_flow_finish;
static std::map<uint32_t, bool> g_log_collective_submit;
static std::map<uint64_t, std::vector<LocalFlowTask>> g_local_flow_queues;
static std::map<uint64_t, bool> g_local_flow_active;
static std::map<uint64_t, uint64_t> g_flow_key_to_local_gpu;
static uint64_t g_local_flow_sequence = 0;
static LocalFlowSchedulePolicy g_local_flow_schedule_policy = FifoLocalFlowSchedulePolicy;
static ExpertRoutePolicy g_expert_route_policy = RandomExpertRoutePolicy;
static PdSplitPolicy g_pd_split_policy = NoPrefillSideDecodeSplitPolicy;
static RuntimeConfig g_runtime_config{
    default_pg,
    1,
    1,
    1,
    1,
    64,
    0,
    2ULL,
    1};

// Global, unique probability table used by all alltoall jobs. If empty,
// a uniform distribution will be used for the group size when needed.
static std::vector<double> g_prob_table;

inline void SetGlobalProbTable(const std::vector<double>& table) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_prob_table = table;
}

inline void ConfigureRuntime(const RuntimeConfig& cfg) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_runtime_config = cfg;
}

inline void SetGlobalPipelineNeeds(uint32_t needPrefill, uint32_t needDecode) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_runtime_config.need_prefill = needPrefill;
  g_runtime_config.need_decode = needDecode;
}

inline void SetModelParallelConfig(uint32_t expertNum) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_runtime_config.expert_num = expertNum;
}

// ===== SCHEDULING POLICY INTERFACE: per-GPU local send queue scheduling =====
// Queue elements expose PD kind, pipeline stage, message size, and arrival time.
// Default policy is FIFO; replace through SetLocalFlowSchedulePolicy().
inline size_t FifoLocalFlowSchedulePolicy(const std::vector<LocalFlowTask>& /*queue*/, uint64_t /*nowNs*/) {
  return 0;
}

inline void SetLocalFlowSchedulePolicy(LocalFlowSchedulePolicy policy) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_local_flow_schedule_policy = policy ? policy : FifoLocalFlowSchedulePolicy;
}

// ===== SCHEDULING POLICY INTERFACE: same-rank expert replica routing =====
// When an expert rank has multiple replicas, this policy chooses the target
// replica. Default implementation is random and avoids local send if possible.
inline std::pair<uint32_t,uint32_t> RandomExpertRoutePolicy(const ExpertRouteRequest& req) {
  if (req.candidates.empty())
    return req.sourceGpu;

  std::vector<std::pair<uint32_t,uint32_t>> candidates;
  for (const auto& candidate : req.candidates) {
    if (candidate != req.sourceGpu)
      candidates.push_back(candidate);
  }
  if (candidates.empty())
    candidates = req.candidates;

  std::mt19937 rng(static_cast<uint32_t>(
      req.jobId ^ (req.expertPlacementOwner * 2654435761u) ^
      (req.expertId * 2246822519u) ^
      (req.sourceGpu.first * 16777619u) ^
      (req.sourceGpu.second * 2166136261u) ^
      (Simulator::Now().GetNanoSeconds() & 0xffffffffu)));
  std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
  return candidates[dist(rng)];
}

inline void SetExpertRoutePolicy(ExpertRoutePolicy policy) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_expert_route_policy = policy ? policy : RandomExpertRoutePolicy;
}

// ===== SCHEDULING POLICY INTERFACE: PD pipeline split point =====
// Return how many decode-token iterations should run on the prefill placement
// before KV cache is transferred to the decode placement. Default is 0, which
// means classic PD separation: prefill finishes, then KV is transferred, then
// all decode iterations run on the decode placement.
inline uint32_t NoPrefillSideDecodeSplitPolicy(const PdSplitRequest& /*req*/) {
  return 0;
}

inline void SetPdSplitPolicy(PdSplitPolicy policy) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_pd_split_policy = policy ? policy : NoPrefillSideDecodeSplitPolicy;
}

inline uint32_t ResolvePrefillSideDecodeIterations(const PipelineTask& task,
                                                   const RuntimeConfig& cfg,
                                                   uint32_t decodeIterations) {
  PdSplitPolicy policy;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    policy = g_pd_split_policy ? g_pd_split_policy : NoPrefillSideDecodeSplitPolicy;
  }
  uint32_t split = policy(PdSplitRequest{task, cfg, decodeIterations});
  return std::min(split, decodeIterations);
}

inline RuntimeConfig GetRuntimeConfig() {
  std::lock_guard<std::mutex> lk(g_mutex);
  return g_runtime_config;
}

inline RuntimeConfig RefreshRuntimeConfig(cclScheduler::NeedEvent event, const PipelineTask& task) {
  RuntimeConfig cfg = GetRuntimeConfig();
  cclScheduler::NeedState state{
      cfg.need_prefill,
      cfg.need_decode,
      cfg.expert_num};
  cclScheduler::NotifyNeedEvent(event, task.taskId, task.prefillLength, task.decodeLength, state);
  cfg.need_prefill = state.need_prefill;
  cfg.need_decode = state.need_decode;
  cfg.expert_num = state.expert_num;
  ConfigureRuntime(cfg);
  return cfg;
}

inline uint64_t TokenBytes(uint32_t tokens) {
  auto cfg = GetRuntimeConfig();
  return cfg.token_msg_size * static_cast<uint64_t>(std::max(1u, tokens));
}

inline uint64_t PrefillAllReduceMsgSize(const PipelineTask& task) {
  return TokenBytes(task.prefillLength);
}

inline uint64_t PrefillAllToAllMsgSize(const PipelineTask& task) {
  return TokenBytes(task.prefillLength);
}

inline uint64_t KvCacheMsgSize(const PipelineTask& task) {
  return TokenBytes(task.prefillLength);
}

inline uint32_t DecodeIterationsForTask(const PipelineTask& task) {
  return std::max(1u, task.decodeLength);
}

inline uint64_t DecodeIterationMsgSize(const PipelineTaskState& state) {
  (void)state;
  return TokenBytes(1);
}

inline void Init() {
  // Initialization only; tasks should be submitted by the driver (NormalNetwork)
  ccl::CclLog("mnCCL: initialized");
  // schedule one-time scheduler kick to start handling pending queue later
  Simulator::Schedule(Seconds(sim_time*2), [](){ cclScheduler::ScheduleTask(); });
}

inline void SubmitJob(uint32_t jobId, int need, double sim_time, uint64_t workload) {
  // Submit a single job (no recursive behavior). Prefer using SubmitColJob
  // directly from the driver to construct arbitrary job parameters.
  SubmitColJob(CollectiveJob{jobId, CollectiveOp::AllReduce, default_pg, static_cast<uint32_t>(need), sim_time, workload, 0, 1, 1, 0});
}

inline void SubmitColJob(const CollectiveJob& job) {
  Simulator::Schedule(Seconds(job.submitTime), [job](){
    cclScheduler::EnqueuePendingTask(job.jobId, job.op, job.pg, job.need, job.msgSize, job.root, job.k, job.expert_num, job.expert_mem_bytes);
    cclScheduler::ScheduleTask();
  });
}

inline void SubmitPipelineTask(const PipelineTask& task) {
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    PipelineTaskState state{};
    state.task = task;
    state.prefillPlacementOwner = PID++;
    state.decodePlacementOwner = PID++;
    state.decodeIterations = DecodeIterationsForTask(task);
    state.prefillSideDecodeIterations = 0;
    state.prefillPlacementReleased = false;
    state.decodePlacementReleased = false;
    state.kvCacheTransferred = false;
    g_pipeline_tasks[task.taskId] = state;
  }

  Simulator::Schedule(Seconds(task.submitTime), [task](){
    auto cfg = RefreshRuntimeConfig(cclScheduler::NeedEvent::TaskDispatch, task);
    uint32_t decodeIterations = DecodeIterationsForTask(task);
    uint32_t prefillSideDecodeIterations =
        ResolvePrefillSideDecodeIterations(task, cfg, decodeIterations);
    {
      std::lock_guard<std::mutex> lk(g_mutex);
      auto it = g_pipeline_tasks.find(task.taskId);
      if (it != g_pipeline_tasks.end()) {
        it->second.startTimeNs = Simulator::Now().GetNanoSeconds();
        it->second.decodeIterations = decodeIterations;
        it->second.prefillSideDecodeIterations = prefillSideDecodeIterations;
      }
    }
    {
      std::ostringstream ss;
      ss << "mnCCL: pipeline task " << task.taskId
         << " dispatch need_prefill=" << cfg.need_prefill
         << " need_decode=" << cfg.need_decode
         << " expert_num=" << cfg.expert_num
         << " prefill_length=" << task.prefillLength
         << " decode_length=" << task.decodeLength
         << " single_token_length=" << cfg.single_token_length
         << " prefill_side_decode_iterations=" << prefillSideDecodeIterations;
      ccl::CclLog(ss.str());
    }
    TryStartPipelineTask(task.taskId);
  });
}

inline const char* OpName(CollectiveOp op) {
  switch (op) {
    case CollectiveOp::AllReduce: return "allreduce";
    case CollectiveOp::AllToAll: return "alltoall";
    case CollectiveOp::Broadcast: return "broadcast";
    case CollectiveOp::Reduce: return "reduce";
    case CollectiveOp::Gather: return "gather";
    case CollectiveOp::Scatter: return "scatter";
    case CollectiveOp::AllGather: return "allgather";
    case CollectiveOp::ReduceScatter: return "reduce-scatter";
  }
  return "unknown";
}

inline const char* PipelineStageName(PipelineStage stage) {
  switch (stage) {
    case PipelineStage::PrefillAllReduce: return "prefill_allreduce";
    case PipelineStage::PrefillAllToAll: return "prefill_alltoall";
    case PipelineStage::KvCacheTransfer: return "kv_cache";
    case PipelineStage::DecodeAllReduce: return "decode_allreduce";
    case PipelineStage::DecodeAllToAll: return "decode_alltoall";
    case PipelineStage::Generic: return "generic";
  }
  return "unknown";
}

inline uint64_t MakeKey(uint16_t pg, uint32_t src, uint32_t dst, uint16_t port) {
  return ((uint64_t)pg << 48) | ((uint64_t)src << 32) | ((uint64_t)dst << 16) | port;
}

inline uint64_t MakeGpuKey(uint32_t node, uint32_t gpu) {
  return (static_cast<uint64_t>(node) << 32) | gpu;
}

inline uint32_t NetworkNodeId(const std::pair<uint32_t,uint32_t>& gpu) {
  uint32_t stride = std::max(1u, gpus_per_server);
  return gpu.first * stride + gpu.second;
}

inline bool UseLocalFlowQueue(PipelineStage stage) {
  (void)stage;
  return true;
}

inline void DispatchLocalFlow(const LocalFlowTask& task) {
  RdmaClientHelper clientHelper(
      task.pg,
      serverAddress[task.srcNode],
      serverAddress[task.dstNode],
      task.srcPort,
      0,
      task.msgSize,
      has_win ? (global_t == 1 ? maxBdp : pairBdp[n.Get(task.srcNode)][n.Get(task.dstNode)]) : 0,
      global_t == 1 ? maxRtt : pairRtt[task.srcNode][task.dstNode],
      nullptr,
      nullptr,
      1,
      task.srcNode,
      task.dstNode);
  ApplicationContainer apps = clientHelper.Install(n.Get(task.srcNode));
  apps.Start(NanoSeconds(0));
}

inline void TryScheduleLocalGpu(uint64_t localGpuKey) {
  LocalFlowTask task{};
  bool hasTask = false;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_local_flow_active[localGpuKey])
      return;

    auto it = g_local_flow_queues.find(localGpuKey);
    if (it == g_local_flow_queues.end() || it->second.empty())
      return;

    auto policy = g_local_flow_schedule_policy ? g_local_flow_schedule_policy : FifoLocalFlowSchedulePolicy;
    size_t idx = policy(it->second, static_cast<uint64_t>(Simulator::Now().GetNanoSeconds()));
    if (idx >= it->second.size())
      idx = 0;
    task = it->second[idx];
    it->second.erase(it->second.begin() + idx);
    g_local_flow_active[localGpuKey] = true;
    hasTask = true;
  }

  if (hasTask)
    DispatchLocalFlow(task);
}

inline void SubmitFlow(uint16_t pg,
                       uint32_t src,
                       uint32_t dst,
                       uint64_t bytes,
                       std::vector<uint64_t>& keys) {
  if (src == dst || bytes == 0)
    return;

  uint16_t port = portNumber[src][dst]++;
  uint64_t key = MakeKey(pg, src, dst, port);
  uint64_t localGpuKey = MakeGpuKey(src, 0);
  LocalFlowTask task{
      key,
      g_local_flow_sequence++,
      0,
      CollectiveOp::AllToAll,
      PipelineStage::Generic,
      cclScheduler::PlacementKind::Generic,
      pg,
      src,
      0,
      dst,
      0,
      port,
      bytes,
      static_cast<uint64_t>(Simulator::Now().GetNanoSeconds())};

  keys.push_back(key);

  {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_local_flow_queues[localGpuKey].push_back(task);
    g_flow_key_to_local_gpu[key] = localGpuKey;
  }
  TryScheduleLocalGpu(localGpuKey);
}

inline void SubmitFlow(uint16_t pg,
                       const std::pair<uint32_t,uint32_t>& srcGpu,
                       const std::pair<uint32_t,uint32_t>& dstGpu,
                       uint64_t bytes,
                       std::vector<uint64_t>& keys,
                       uint32_t jobId,
                       CollectiveOp op,
                       PipelineStage stage,
                       cclScheduler::PlacementKind pdKind) {
  uint32_t srcNode = NetworkNodeId(srcGpu);
  uint32_t dstNode = NetworkNodeId(dstGpu);
  if (srcNode == dstNode || bytes == 0)
    return;

  uint16_t port = portNumber[srcNode][dstNode]++;
  uint64_t key = MakeKey(pg, srcNode, dstNode, port);
  uint64_t localGpuKey = MakeGpuKey(srcGpu.first, srcGpu.second);
  LocalFlowTask task{
      key,
      g_local_flow_sequence++,
      jobId,
      op,
      stage,
      pdKind,
      pg,
      srcNode,
      srcGpu.second,
      dstNode,
      dstGpu.second,
      port,
      bytes,
      static_cast<uint64_t>(Simulator::Now().GetNanoSeconds())};

  keys.push_back(key);

  if (!UseLocalFlowQueue(stage)) {
    DispatchLocalFlow(task);
    return;
  }

  {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_local_flow_queues[localGpuKey].push_back(task);
    g_flow_key_to_local_gpu[key] = localGpuKey;
  }
  TryScheduleLocalGpu(localGpuKey);
}

inline uint16_t FinishSubmit(uint32_t jobId,
                             CollectiveOp op,
                             const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                             const std::vector<uint64_t>& keys,
                             bool releaseOnFinish = true,
                             bool recordFlowFinish = true,
                             bool logSubmit = true) {
  if (keys.empty()) {
    if (releaseOnFinish)
      cclScheduler::ReleasePlacement(jobId);
    Simulator::ScheduleNow([jobId](){ OnCollectiveFinished(jobId); });
    return jobId;
  }

  {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_outstanding[jobId] = keys.size();
    g_alloc[jobId] = gpus;
    g_release_on_finish[jobId] = releaseOnFinish;
    g_record_flow_finish[jobId] = recordFlowFinish;
    g_log_collective_submit[jobId] = logSubmit;
    for (auto key : keys) {
      g_key2JID[key] = jobId;
    }
  }

  if (logSubmit) {
    std::ostringstream ss;
    ss << "mnCCL: submitted " << OpName(op) << " JID=" << jobId
       << " participants=" << gpus.size() << " sends=" << keys.size();
    ccl::CclLog(ss.str());
  }
  return jobId;
}




// Submit a simple ring all-reduce: each participant sends one message to the
// next participant. `gpus` is vector of <node, gpu_idx>. Returns the pg
// (used to tag the QPs) assigned to this collective.
inline uint16_t SubmitAllReduce(uint32_t jobId,
                                uint16_t pg,
                                const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                                uint64_t msgSize,
                                bool releaseOnFinish,
                                bool recordFlowFinish,
                                bool logSubmit) {
  return SubmitAllReduceForStage(jobId,
                                 pg,
                                 gpus,
                                 msgSize,
                                 releaseOnFinish,
                                 recordFlowFinish,
                                 logSubmit,
                                 PipelineStage::Generic,
                                 cclScheduler::PlacementKind::Generic);
}

inline uint16_t SubmitAllReduceForStage(uint32_t jobId,
                                        uint16_t pg,
                                        const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                                        uint64_t msgSize,
                                        bool releaseOnFinish,
                                        bool recordFlowFinish,
                                        bool logSubmit,
                                        PipelineStage stage,
                                        cclScheduler::PlacementKind pdKind) {
  if (gpus.size() < 2)
    return FinishSubmit(jobId, CollectiveOp::AllReduce, gpus, {}, releaseOnFinish, recordFlowFinish, logSubmit);

  std::vector<uint64_t> keys;
  for (size_t i = 0; i < gpus.size(); ++i) {
    auto src = gpus[i];
    auto dst = gpus[(i + 1) % gpus.size()];
    SubmitFlow(pg, src, dst, msgSize, keys, jobId, CollectiveOp::AllReduce, stage, pdKind);
  }

  return FinishSubmit(jobId, CollectiveOp::AllReduce, gpus, keys, releaseOnFinish, recordFlowFinish, logSubmit);
}

inline uint16_t SubmitBroadcast(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root) {
  if (gpus.size() < 2)
    return jobId;
  root %= gpus.size();

    auto src = gpus[root];
    std::vector<uint64_t> keys;
    for (size_t i = 0; i < gpus.size(); ++i) {
      if (i == root)
        continue;
    SubmitFlow(pg, src, gpus[i], msgSize, keys, jobId, CollectiveOp::Broadcast, PipelineStage::Generic, cclScheduler::PlacementKind::Generic);
  }
  return FinishSubmit(jobId, CollectiveOp::Broadcast, gpus, keys);
}

inline uint16_t SubmitReduce(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root) {
  if (gpus.size() < 2)
    return jobId;
  root %= gpus.size();

  auto dst = gpus[root];
  std::vector<uint64_t> keys;
  for (size_t i = 0; i < gpus.size(); ++i) {
    if (i == root)
      continue;
    SubmitFlow(pg, gpus[i], dst, msgSize, keys, jobId, CollectiveOp::Reduce, PipelineStage::Generic, cclScheduler::PlacementKind::Generic);
  }
  return FinishSubmit(jobId, CollectiveOp::Reduce, gpus, keys);
}

inline uint16_t SubmitGather(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root) {
  if (gpus.size() < 2)
    return jobId;
  root %= gpus.size();

  auto dst = gpus[root];
  uint64_t chunkSize = (msgSize + gpus.size() - 1) / gpus.size();
  std::vector<uint64_t> keys;
  for (size_t i = 0; i < gpus.size(); ++i) {
    if (i == root)
      continue;
    SubmitFlow(pg, gpus[i], dst, chunkSize, keys, jobId, CollectiveOp::Gather, PipelineStage::Generic, cclScheduler::PlacementKind::Generic);
  }
  return FinishSubmit(jobId, CollectiveOp::Gather, gpus, keys);
}

inline uint16_t SubmitScatter(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root) {
  if (gpus.size() < 2)
    return jobId;
  root %= gpus.size();

  auto src = gpus[root];
  uint64_t chunkSize = (msgSize + gpus.size() - 1) / gpus.size();
  std::vector<uint64_t> keys;
  for (size_t i = 0; i < gpus.size(); ++i) {
    if (i == root)
      continue;
    SubmitFlow(pg, src, gpus[i], chunkSize, keys, jobId, CollectiveOp::Scatter, PipelineStage::Generic, cclScheduler::PlacementKind::Generic);
  }
  return FinishSubmit(jobId, CollectiveOp::Scatter, gpus, keys);
}

inline uint16_t SubmitAllGather(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize) {
  if (gpus.size() < 2)
    return jobId;

  uint64_t chunkSize = (msgSize + gpus.size() - 1) / gpus.size();
  std::vector<uint64_t> keys;
  for (size_t i = 0; i < gpus.size(); ++i) {
    auto src = gpus[i];
    auto dst = gpus[(i + 1) % gpus.size()];
    SubmitFlow(pg, src, dst, chunkSize, keys, jobId, CollectiveOp::AllGather, PipelineStage::Generic, cclScheduler::PlacementKind::Generic);
  }
  return FinishSubmit(jobId, CollectiveOp::AllGather, gpus, keys);
}

inline uint16_t SubmitReduceScatter(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize) {
  if (gpus.size() < 2)
    return jobId;

  uint64_t chunkSize = (msgSize + gpus.size() - 1) / gpus.size();
  std::vector<uint64_t> keys;
  for (size_t i = 0; i < gpus.size(); ++i) {
    auto src = gpus[i];
    auto dst = gpus[(i + 1) % gpus.size()];
    SubmitFlow(pg, src, dst, chunkSize, keys, jobId, CollectiveOp::ReduceScatter, PipelineStage::Generic, cclScheduler::PlacementKind::Generic);
  }
  return FinishSubmit(jobId, CollectiveOp::ReduceScatter, gpus, keys);
}

// Submit an all-to-all style job: each participant generates `k` flows
// whose destinations are sampled from a global probability table that is
// shifted/biased by the sender's local rank. The local ranks for the
// job are stored in `g_job_ranks[jobId]`.
inline uint16_t SubmitAllToAll(uint32_t jobId,
                               uint16_t pg,
                               const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                               uint64_t msgSize,
                               uint32_t k,
                               uint32_t expert_num,
                               uint64_t expert_mem_bytes,
                               bool releaseOnFinish,
                               bool recordFlowFinish,
                               bool logSubmit,
                               PipelineStage stage,
                               cclScheduler::PlacementKind pdKind,
                               uint32_t expertPlacementOwner) {
  if (gpus.size() < 2 || (k == 0 && expert_num == 0))
    return FinishSubmit(jobId, CollectiveOp::AllToAll, gpus, {}, releaseOnFinish, recordFlowFinish, logSubmit);

  std::vector<uint64_t> keys;

  // store local ranks 0..n-1 for this job
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_job_ranks[jobId].clear();
    for (uint32_t r = 0; r < gpus.size(); ++r)
      g_job_ranks[jobId].push_back(r);
  }

  // base probability table: use global table if available, otherwise uniform.
  // The table is interpreted over expert ranks, not GPU ids.
  std::vector<double> base;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (!g_prob_table.empty())
      base = g_prob_table;
  }
  if (base.size() < expert_num) {
    base.assign(expert_num, 1.0);
  }

  uint32_t placementOwner = expertPlacementOwner == 0 ? jobId : expertPlacementOwner;
  for (uint32_t sourceExpert = 0; sourceExpert < expert_num; ++sourceExpert) {
    auto sourceReplicas = cclScheduler::GetExpertReplicas(placementOwner, sourceExpert);
    if (sourceReplicas.empty())
      continue;

    for (const auto& sourceGpu : sourceReplicas) {
      // build per-expert probability vector by shifting the base table by expert id
      std::vector<double> probs(expert_num);
      double sum = 0.0;
      for (uint32_t j = 0; j < expert_num; ++j) {
        probs[j] = base[(j + (sourceExpert % base.size())) % base.size()];
        sum += probs[j];
      }
      if (sum <= 0.0) {
        for (uint32_t j = 0; j < expert_num; ++j) probs[j] = 1.0;
        sum = static_cast<double>(expert_num);
      }
      for (uint32_t j = 0; j < expert_num; ++j) probs[j] /= sum;

      std::mt19937 rng(static_cast<uint32_t>(
          jobId ^ (sourceGpu.first * 16777619u) ^
          (sourceGpu.second * 2166136261u) ^ (sourceExpert * 92717u)));
      std::discrete_distribution<int> dist(probs.begin(), probs.end());

      for (uint32_t t = 0; t < k; ++t) {
        int dstExpert = dist(rng);
        int tries = 0;
        auto candidates = cclScheduler::GetExpertReplicas(placementOwner, static_cast<uint32_t>(dstExpert));
        while (candidates.empty() && tries < 3) {
          dstExpert = dist(rng);
          candidates = cclScheduler::GetExpertReplicas(placementOwner, static_cast<uint32_t>(dstExpert));
          ++tries;
        }
        if (candidates.empty())
          continue;
        ExpertRoutePolicy routePolicy;
        {
          std::lock_guard<std::mutex> lk(g_mutex);
          routePolicy = g_expert_route_policy ? g_expert_route_policy : RandomExpertRoutePolicy;
        }
        auto dstGpu = routePolicy(ExpertRouteRequest{
            jobId,
            placementOwner,
            static_cast<uint32_t>(dstExpert),
            stage,
            pdKind,
            sourceGpu,
            candidates});
        SubmitFlow(pg, sourceGpu, dstGpu, msgSize, keys, jobId, CollectiveOp::AllToAll, stage, pdKind);
      }
    }
  }

  return FinishSubmit(jobId, CollectiveOp::AllToAll, gpus, keys, releaseOnFinish, recordFlowFinish, logSubmit);
}

inline uint16_t SubmitCollective(uint32_t jobId,
                                 CollectiveOp op,
                                 uint16_t pg,
                                 const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                                 uint64_t msgSize,
                                 uint32_t root,
                                 uint32_t k,
                                 uint32_t expert_num,
                                 uint64_t expert_mem_bytes,
                                 bool releaseOnFinish,
                                 bool recordFlowFinish,
                                 bool logSubmit) {
  switch (op) {
    case CollectiveOp::AllReduce:
      return SubmitAllReduce(jobId, pg, gpus, msgSize, releaseOnFinish, recordFlowFinish, logSubmit);
    case CollectiveOp::AllToAll:
      return SubmitAllToAll(jobId,
                            pg,
                            gpus,
                            msgSize,
                            k,
                            expert_num,
                            expert_mem_bytes,
                            releaseOnFinish,
                            recordFlowFinish,
                            logSubmit,
                            PipelineStage::Generic,
                            cclScheduler::PlacementKind::Generic,
                            jobId);
    case CollectiveOp::Broadcast:
      return SubmitBroadcast(jobId, pg, gpus, msgSize, root);
    case CollectiveOp::Reduce:
      return SubmitReduce(jobId, pg, gpus, msgSize, root);
    case CollectiveOp::Gather:
      return SubmitGather(jobId, pg, gpus, msgSize, root);
    case CollectiveOp::Scatter:
      return SubmitScatter(jobId, pg, gpus, msgSize, root);
    case CollectiveOp::AllGather:
      return SubmitAllGather(jobId, pg, gpus, msgSize);
    case CollectiveOp::ReduceScatter:
      return SubmitReduceScatter(jobId, pg, gpus, msgSize);
  }
  if (releaseOnFinish)
    cclScheduler::ReleasePlacement(jobId);
  return jobId;
}

inline uint16_t SubmitKvCacheTransfer(uint32_t jobId,
                                      uint16_t pg,
                                      const std::vector<std::pair<uint32_t,uint32_t>>& prefillGpus,
                                      const std::vector<std::pair<uint32_t,uint32_t>>& decodeGpus,
                                      uint64_t msgSize) {
  std::vector<uint64_t> keys;
  if (prefillGpus.empty() || decodeGpus.empty())
    return FinishSubmit(jobId, CollectiveOp::AllToAll, decodeGpus, keys, false, false, false);

  uint32_t flows = std::max(prefillGpus.size(), decodeGpus.size());
  uint64_t bytesPerFlow = (msgSize + flows - 1) / flows;
  for (uint32_t i = 0; i < flows; ++i) {
    auto src = prefillGpus[i % prefillGpus.size()];
    auto dst = decodeGpus[i % decodeGpus.size()];
    SubmitFlow(pg,
               src,
               dst,
               bytesPerFlow,
               keys,
               jobId,
               CollectiveOp::AllToAll,
               PipelineStage::KvCacheTransfer,
               cclScheduler::PlacementKind::Decode);
  }

  return FinishSubmit(jobId, CollectiveOp::AllToAll, decodeGpus, keys, false, false, false);
}

inline uint64_t CalculateStandaloneFct(uint32_t src, uint32_t dst, uint64_t size) {
  uint64_t base_rtt = pairRtt[src][dst], b = pairBw[src][dst];
  uint32_t packet_payload_size =
      get_config_value_ns3<uint64_t>("ns3::RdmaHw::Mtu");
  uint32_t total_bytes = size +
      ((size - 1) / packet_payload_size + 1) *
          (CustomHeader::GetStaticWholeHeaderSize() -
           IntHeader::GetStaticSize());
  return base_rtt + total_bytes * 8000000000lu / b;
}

inline void WriteFlowFinishRecords(uint32_t jobId,
                                   const std::vector<FlowFinishRecord>& records) {
  if (g_flow_finish_log_path.empty())
    return;

  std::ofstream ofs(g_flow_finish_log_path, std::ofstream::app);
  if (!ofs)
    return;

  for (auto &msg : records) {
    uint32_t sid = ip_to_node_id(msg.qp->sip);
    uint32_t did = ip_to_node_id(msg.qp->dip);
    ofs << msg.finishTimeNs << "," << jobId << "," << sid << "," << did
        << "," << msg.qp->m_pg << "," << msg.qp->sport << ","
        << msg.qp->dport << "," << msg.msgSize << ","
        << msg.qp->startTime.GetTimeStep() << "," << msg.actualFct << ","
        << msg.standaloneFct << "\n";
  }
}

inline void TryStartPipelineTask(uint32_t taskId) {
  PipelineTaskState state;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_pipeline_tasks.find(taskId);
    if (it == g_pipeline_tasks.end())
      return;
    state = it->second;
  }

  auto cfg = GetRuntimeConfig();
  uint32_t needPrefill = cfg.need_prefill;
  auto gpus = cclScheduler::TryPlaceNeed(
      state.prefillPlacementOwner,
      cclScheduler::PlacementKind::Prefill,
      needPrefill,
      cfg.expert_num,
      cfg.expert_mem_bytes);

  if (gpus.size() != needPrefill) {
    Simulator::Schedule(MicroSeconds(10), [taskId](){ TryStartPipelineTask(taskId); });
    return;
  }

  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_pipeline_tasks.find(taskId);
    if (it == g_pipeline_tasks.end()) {
      cclScheduler::ReleasePlacement(state.prefillPlacementOwner);
      return;
    }
    it->second.actualNeedPrefill = needPrefill;
    it->second.prefillGpus = gpus;
    state = it->second;
  }

  cclScheduler::AssignExperts(state.prefillPlacementOwner, cfg.expert_num, gpus);
  StartPrefillAllReduce(state);
}

inline void StartPrefillAllReduce(PipelineTaskState& state) {
  uint32_t jobId = JID++;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_pipeline_tasks.find(state.task.taskId);
    if (it == g_pipeline_tasks.end())
      return;
    it->second.prefillAllReduceJobId = jobId;
    g_pipeline_stage_by_job[jobId] = PipelineStageRef{state.task.taskId, PipelineStage::PrefillAllReduce};
  }
  auto cfg = GetRuntimeConfig();
  SubmitAllReduceForStage(jobId,
                          cfg.pg,
                          state.prefillGpus,
                          PrefillAllReduceMsgSize(state.task),
                          false,
                          true,
                          true,
                          PipelineStage::PrefillAllReduce,
                          cclScheduler::PlacementKind::Prefill);
}

inline void StartPrefillAllToAll(PipelineTaskState& state) {
  uint32_t jobId = JID++;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_pipeline_tasks.find(state.task.taskId);
    if (it == g_pipeline_tasks.end())
      return;
    it->second.prefillAllToAllJobId = jobId;
    g_pipeline_stage_by_job[jobId] = PipelineStageRef{state.task.taskId, PipelineStage::PrefillAllToAll};
  }
  auto cfg = GetRuntimeConfig();
  SubmitAllToAll(jobId,
                 cfg.pg,
                 state.prefillGpus,
                 PrefillAllToAllMsgSize(state.task),
                 cfg.k,
                 cfg.expert_num,
                 cfg.expert_mem_bytes,
                 false,
                 true,
                 true,
                 PipelineStage::PrefillAllToAll,
                 cclScheduler::PlacementKind::Prefill,
                 state.prefillPlacementOwner);
}

inline void TryStartDecode(uint32_t taskId) {
  PipelineTaskState state;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_pipeline_tasks.find(taskId);
    if (it == g_pipeline_tasks.end())
      return;
    state = it->second;
  }

  if (state.decodeIteration < state.prefillSideDecodeIterations) {
    {
      std::lock_guard<std::mutex> lk(g_mutex);
      auto it = g_pipeline_tasks.find(taskId);
      if (it == g_pipeline_tasks.end())
        return;
      if (it->second.decodeStartNs == 0)
        it->second.decodeStartNs = Simulator::Now().GetNanoSeconds();
    }
    StartNextDecodeToken(taskId);
    return;
  }

  if (state.decodeIteration >= state.decodeIterations) {
    FinishPipelineTask(taskId);
    return;
  }

  if (!state.decodeGpus.empty()) {
    StartKvCacheTransfer(state);
    return;
  }

  auto cfg = GetRuntimeConfig();
  uint32_t needDecode = cfg.need_decode;
  auto gpus = cclScheduler::TryPlaceNeed(
      state.decodePlacementOwner,
      cclScheduler::PlacementKind::Decode,
      needDecode,
      cfg.expert_num,
      cfg.expert_mem_bytes);

  if (gpus.size() != needDecode) {
    Simulator::Schedule(MicroSeconds(10), [taskId](){ TryStartDecode(taskId); });
    return;
  }

  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_pipeline_tasks.find(taskId);
    if (it == g_pipeline_tasks.end()) {
      cclScheduler::ReleasePlacement(state.decodePlacementOwner);
      return;
    }
    it->second.actualNeedDecode = needDecode;
    it->second.decodeGpus = gpus;
    if (it->second.decodeStartNs == 0)
      it->second.decodeStartNs = Simulator::Now().GetNanoSeconds();
    state = it->second;
  }

  cclScheduler::AssignExperts(state.decodePlacementOwner, cfg.expert_num, gpus);
  StartKvCacheTransfer(state);
}

inline void StartKvCacheTransfer(PipelineTaskState& state) {
  uint32_t jobId = JID++;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_pipeline_tasks.find(state.task.taskId);
    if (it == g_pipeline_tasks.end())
      return;
    it->second.kvCacheJobId = jobId;
    g_pipeline_stage_by_job[jobId] = PipelineStageRef{state.task.taskId, PipelineStage::KvCacheTransfer};
  }
  auto cfg = GetRuntimeConfig();
  SubmitKvCacheTransfer(jobId, cfg.pg, state.prefillGpus, state.decodeGpus, KvCacheMsgSize(state.task));
}

inline uint64_t DecodeComputeDelayNs(const PipelineTaskState& state) {
  const auto& activeGpus = ActiveDecodeGpus(state);
  uint64_t avgRemaining = cclScheduler::GetAverageRemainingMemory(activeGpus);
  uint64_t avgCapacity = cclScheduler::GetAverageCapacity(activeGpus);
  auto cfg = GetRuntimeConfig();
  if (avgCapacity == 0)
    return cfg.base_decode_compute_delay_ns;
  double pressure = 1.0 - static_cast<double>(avgRemaining) / static_cast<double>(avgCapacity);
  if (pressure < 0.0)
    pressure = 0.0;
  if (pressure > 1.0)
    pressure = 1.0;
  return cfg.base_decode_compute_delay_ns +
         static_cast<uint64_t>(cfg.base_decode_compute_delay_ns * pressure);
}

inline bool UsePrefillPlacementForNextDecodeToken(const PipelineTaskState& state) {
  return state.decodeIteration < state.prefillSideDecodeIterations;
}

inline const std::vector<std::pair<uint32_t,uint32_t>>& ActiveDecodeGpus(const PipelineTaskState& state) {
  if (UsePrefillPlacementForNextDecodeToken(state))
    return state.prefillGpus;
  return state.decodeGpus;
}

inline uint32_t ActiveDecodePlacementOwner(const PipelineTaskState& state) {
  return UsePrefillPlacementForNextDecodeToken(state)
         ? state.prefillPlacementOwner
         : state.decodePlacementOwner;
}

inline cclScheduler::PlacementKind ActiveDecodePlacementKind(const PipelineTaskState& state) {
  return UsePrefillPlacementForNextDecodeToken(state)
         ? cclScheduler::PlacementKind::Prefill
         : cclScheduler::PlacementKind::Decode;
}

inline void StartNextDecodeToken(uint32_t taskId) {
  PipelineTaskState state;
  bool shouldFinish = false;
  bool shouldStartKvTransfer = false;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_pipeline_tasks.find(taskId);
    if (it == g_pipeline_tasks.end())
      return;
    if (it->second.decodeIteration >= it->second.decodeIterations) {
      shouldFinish = true;
    } else if (!it->second.kvCacheTransferred &&
               it->second.decodeIteration >= it->second.prefillSideDecodeIterations) {
      state = it->second;
      shouldStartKvTransfer = true;
    } else {
      state = it->second;
    }
  }

  if (shouldFinish) {
    FinishPipelineTask(taskId);
    return;
  }
  if (shouldStartKvTransfer) {
    TryStartDecode(taskId);
    return;
  }

  uint64_t delayNs = DecodeComputeDelayNs(state);
  Simulator::Schedule(NanoSeconds(delayNs), [taskId](){
    PipelineTaskState nextState;
    {
      std::lock_guard<std::mutex> lk(g_mutex);
      auto it = g_pipeline_tasks.find(taskId);
      if (it == g_pipeline_tasks.end())
        return;
      nextState = it->second;
    }
    StartDecodeAllReduce(nextState);
  });
}

inline void StartDecodeAllReduce(PipelineTaskState& state) {
  uint32_t jobId = JID++;
  auto cfg = GetRuntimeConfig();
  const auto& activeGpus = ActiveDecodeGpus(state);
  auto activeKind = ActiveDecodePlacementKind(state);
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_pipeline_tasks.find(state.task.taskId);
    if (it == g_pipeline_tasks.end())
      return;
    it->second.currentDecodeAllReduceJobId = jobId;
    g_pipeline_stage_by_job[jobId] = PipelineStageRef{state.task.taskId, PipelineStage::DecodeAllReduce};
  }
  SubmitAllReduceForStage(jobId,
                          cfg.pg,
                          activeGpus,
                          DecodeIterationMsgSize(state),
                          false,
                          false,
                          false,
                          PipelineStage::DecodeAllReduce,
                          activeKind);
}

inline void StartDecodeAllToAll(PipelineTaskState& state) {
  uint32_t jobId = JID++;
  auto cfg = GetRuntimeConfig();
  const auto& activeGpus = ActiveDecodeGpus(state);
  uint32_t activePlacementOwner = ActiveDecodePlacementOwner(state);
  auto activeKind = ActiveDecodePlacementKind(state);
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_pipeline_tasks.find(state.task.taskId);
    if (it == g_pipeline_tasks.end())
      return;
    it->second.currentDecodeAllToAllJobId = jobId;
    g_pipeline_stage_by_job[jobId] = PipelineStageRef{state.task.taskId, PipelineStage::DecodeAllToAll};
  }
  SubmitAllToAll(jobId,
                 cfg.pg,
                 activeGpus,
                 DecodeIterationMsgSize(state),
                 cfg.k,
                 cfg.expert_num,
                 cfg.expert_mem_bytes,
                 false,
                 false,
                 false,
                 PipelineStage::DecodeAllToAll,
                 activeKind,
                 activePlacementOwner);
}

inline void FinishPipelineTask(uint32_t taskId) {
  uint64_t nowNs = Simulator::Now().GetNanoSeconds();
  PipelineTaskState state;
  bool found = false;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_pipeline_tasks.find(taskId);
    if (it != g_pipeline_tasks.end()) {
      state = it->second;
      found = true;
      g_pipeline_tasks.erase(it);
    }
  }
  if (!found)
    return;

  if (!state.prefillPlacementReleased) {
    cclScheduler::ReleasePlacement(state.prefillPlacementOwner);
    cclScheduler::ReleaseExpertAlloc(state.prefillPlacementOwner);
  }
  if (!state.decodePlacementReleased) {
    cclScheduler::ReleasePlacement(state.decodePlacementOwner);
    cclScheduler::ReleaseExpertAlloc(state.decodePlacementOwner);
  }

  auto finishCfg = RefreshRuntimeConfig(cclScheduler::NeedEvent::TaskFinish, state.task);
  LogPipelineTaskSummary(state, nowNs, finishCfg, true);
}

inline void LogPipelineTaskSummary(const PipelineTaskState& state,
                                   uint64_t nowNs,
                                   const RuntimeConfig& cfg,
                                   bool completed) {
  uint64_t startNs = state.startTimeNs;
  uint64_t totalFctNs = nowNs >= startNs ? nowNs - startNs : 0;
  uint64_t decodeE2eNs = (state.decodeStartNs > 0 && nowNs >= state.decodeStartNs)
                         ? nowNs - state.decodeStartNs
                         : 0;
  uint64_t ttftNs = (state.firstTokenFinishNs > 0 && state.firstTokenFinishNs >= startNs)
                    ? state.firstTokenFinishNs - startNs
                    : 0;
  uint64_t completedDecodeTokens = state.decodeIteration;
  uint64_t tpotNs = 0;
  if (completedDecodeTokens > 1 && state.firstTokenFinishNs > 0) {
    uint64_t tpotEndNs = completed ? nowNs : state.lastTokenFinishNs;
    if (tpotEndNs >= state.firstTokenFinishNs) {
      uint64_t tpotDenominator = completed ? state.decodeIterations - 1 : completedDecodeTokens - 1;
      tpotNs = (tpotEndNs - state.firstTokenFinishNs) / tpotDenominator;
    }
  }

  uint64_t prefillCollectiveMsgSize =
      cfg.token_msg_size * static_cast<uint64_t>(state.task.prefillLength);
  uint64_t prefillAllReduceMsgSize =
      cfg.token_msg_size * static_cast<uint64_t>(state.task.prefillLength);
  uint64_t decodeCollectiveTotalMsgSize =
      cfg.token_msg_size * static_cast<uint64_t>(state.task.decodeLength);
  uint64_t decodePerIterationMsgSize = cfg.token_msg_size;
  uint64_t kvCacheMsgSize =
      cfg.token_msg_size * static_cast<uint64_t>(state.task.prefillLength);
  std::ostringstream ss;
  ss << "mnCCL: pipeline task " << state.task.taskId
     << (completed ? " finished" : " incomplete at simulation end") << "\n"
     << "  status:\n"
     << "    completed=" << (completed ? 1 : 0) << "\n"
     << "  latency:\n"
     << "    total_fct_ns=" << totalFctNs << "\n"
     << "    TTFT_ns=" << ttftNs << "\n"
     << "    TPOT_ns=" << tpotNs << "\n"
     << "    decode_end_to_end_ns=" << decodeE2eNs << "\n"
     << "  timeline:\n"
     << "    start_ns=" << state.startTimeNs << "\n"
     << "    prefill_finish_ns=" << state.prefillFinishNs << "\n"
     << "    decode_start_ns=" << state.decodeStartNs << "\n"
     << "    kv_cache_finish_ns=" << state.kvCacheFinishNs << "\n"
     << "    first_token_finish_ns=" << state.firstTokenFinishNs << "\n"
     << "    last_token_finish_ns=" << state.lastTokenFinishNs << "\n"
     << "    finish_ns=" << nowNs << "\n"
     << "  task:\n"
     << "    prefill_length=" << state.task.prefillLength << "\n"
     << "    decode_length=" << state.task.decodeLength << "\n"
     << "    decode_iterations=" << state.decodeIterations << "\n"
     << "    decode_iteration_finished=" << state.decodeIteration << "\n"
     << "    prefill_side_decode_iterations=" << state.prefillSideDecodeIterations << "\n"
     << "    need_prefill=" << state.actualNeedPrefill << "\n"
     << "    need_decode=" << state.actualNeedDecode << "\n"
     << "    current_need_prefill=" << cfg.need_prefill << "\n"
     << "    current_need_decode=" << cfg.need_decode << "\n"
     << "    expert_num=" << cfg.expert_num << "\n"
     << "  comm:\n"
     << "    single_token_length=" << cfg.single_token_length << "\n"
     << "    bytes_per_token=" << cfg.token_msg_size << "\n"
     << "    prefill_allreduce_msg_size=" << prefillAllReduceMsgSize << "\n"
     << "    prefill_alltoall_msg_size=" << prefillCollectiveMsgSize << "\n"
     << "    kv_cache_msg_size=" << kvCacheMsgSize << "\n"
     << "    decode_allreduce_msg_size_per_iteration=" << decodePerIterationMsgSize << "\n"
     << "    decode_allreduce_total_msg_size=" << decodeCollectiveTotalMsgSize << "\n"
     << "    decode_alltoall_msg_size_per_iteration=" << decodePerIterationMsgSize << "\n"
     << "    decode_alltoall_total_msg_size=" << decodeCollectiveTotalMsgSize << "\n"
     << "  jobs:\n"
     << "    prefill_ar_JID=" << state.prefillAllReduceJobId << "\n"
     << "    prefill_a2a_JID=" << state.prefillAllToAllJobId << "\n"
     << "    kv_JID=" << state.kvCacheJobId << "\n"
     << "    last_decode_ar_JID=" << state.currentDecodeAllReduceJobId << "\n"
     << "    last_decode_a2a_JID=" << state.currentDecodeAllToAllJobId;
  ccl::CclLog(ss.str());
}

inline void LogPipelineSummariesAtSimulationEnd() {
  std::vector<PipelineTaskState> activeTasks;
  std::vector<std::pair<uint32_t, PipelineStageRef>> outstandingStageRefs;
  RuntimeConfig cfg = GetRuntimeConfig();
  uint64_t nowNs = Simulator::Now().GetNanoSeconds();
  uint32_t outstandingJobs = 0;
  uint64_t queuedFlows = 0;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    for (const auto& kv : g_pipeline_tasks)
      activeTasks.push_back(kv.second);
    outstandingJobs = static_cast<uint32_t>(g_outstanding.size());
    for (const auto& kv : g_local_flow_queues)
      queuedFlows += kv.second.size();
    for (const auto& kv : g_pipeline_stage_by_job)
      outstandingStageRefs.push_back(kv);
  }

  {
    std::ostringstream ss;
    ss << "mnCCL: simulation end summary active_tasks=" << activeTasks.size()
       << " outstanding_collectives=" << outstandingJobs
       << " queued_local_flows=" << queuedFlows;
    ccl::CclLog(ss.str());
  }

  for (const auto& kv : outstandingStageRefs) {
    const auto& ref = kv.second;
    auto it = std::find_if(activeTasks.begin(),
                           activeTasks.end(),
                           [&](const PipelineTaskState& state) {
                             return state.task.taskId == ref.taskId;
                           });
    std::ostringstream ss;
    ss << "mnCCL: outstanding collective JID=" << kv.first
       << " task=" << ref.taskId
       << " stage=" << PipelineStageName(ref.stage);
    if (it != activeTasks.end()) {
      ss << " decode_iteration=" << it->decodeIteration
         << "/" << it->decodeIterations;
    }
    ccl::CclLog(ss.str());
  }

  for (const auto& state : activeTasks)
    LogPipelineTaskSummary(state, nowNs, cfg, false);
}

inline void OnCollectiveFinished(uint32_t jobId) {
  PipelineStageRef ref{};
  PipelineTaskState state;
  bool found = false;
  uint64_t nowNs = Simulator::Now().GetNanoSeconds();
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it_stage = g_pipeline_stage_by_job.find(jobId);
    if (it_stage == g_pipeline_stage_by_job.end())
      return;
    ref = it_stage->second;
    g_pipeline_stage_by_job.erase(it_stage);

    auto it_task = g_pipeline_tasks.find(ref.taskId);
    if (it_task == g_pipeline_tasks.end())
      return;

    if (ref.stage == PipelineStage::PrefillAllToAll) {
      it_task->second.prefillFinishNs = nowNs;
    } else if (ref.stage == PipelineStage::KvCacheTransfer) {
      it_task->second.kvCacheTransferred = true;
      it_task->second.kvCacheFinishNs = nowNs;
      if (!it_task->second.prefillPlacementReleased) {
        cclScheduler::ReleasePlacement(it_task->second.prefillPlacementOwner);
        cclScheduler::ReleaseExpertAlloc(it_task->second.prefillPlacementOwner);
        it_task->second.prefillPlacementReleased = true;
      }
    } else if (ref.stage == PipelineStage::DecodeAllReduce) {
      // The decode token is counted after the following alltoall completes.
    } else if (ref.stage == PipelineStage::DecodeAllToAll) {
      it_task->second.decodeIteration++;
      if (it_task->second.decodeIteration == 1)
        it_task->second.firstTokenFinishNs = nowNs;
      it_task->second.lastTokenFinishNs = nowNs;
    }
    state = it_task->second;
    found = true;
  }

  if (!found)
    return;

  switch (ref.stage) {
    case PipelineStage::PrefillAllReduce:
      StartPrefillAllToAll(state);
      break;
    case PipelineStage::PrefillAllToAll:
      TryStartDecode(ref.taskId);
      break;
    case PipelineStage::KvCacheTransfer:
      StartNextDecodeToken(ref.taskId);
      break;
    case PipelineStage::DecodeAllReduce:
      StartDecodeAllToAll(state);
      break;
    case PipelineStage::DecodeAllToAll:
      StartNextDecodeToken(ref.taskId);
      break;
    case PipelineStage::Generic:
      break;
  }
}

// Called from NormalNetwork's message_finish callback to let mnCCL handle
// synchronization and resource release when an mnCCL message completes.
inline bool OnMessageFinish(FILE* /*fout*/, Ptr<RdmaQueuePair> q, uint64_t msgSize) {
  //通过pg,src,dst,port四元组映射到JID
  uint16_t pg = q->m_pg;
  uint32_t src = ip_to_node_id(q->sip);
  uint32_t dst = ip_to_node_id(q->dip);
  uint16_t port = q->sport;
  uint64_t key = MakeKey(pg, src, dst, port);
  FlowFinishRecord record{
      q,
      msgSize,
      static_cast<uint64_t>(Simulator::Now().GetNanoSeconds()),
      static_cast<uint64_t>((Simulator::Now() - q->startTime).GetTimeStep()),
      CalculateStandaloneFct(src, dst, msgSize)};

  uint32_t jobId = 0;
  std::vector<std::pair<uint32_t,uint32_t>> alloc;
  std::vector<FlowFinishRecord> flowRecords;
  bool finished = false;
  bool releaseOnFinish = true;
  bool recordFlowFinish = true;
  bool logCollectiveFinish = true;
  uint64_t localGpuKey = 0;
  bool shouldScheduleLocalGpu = false;

  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it_local = g_flow_key_to_local_gpu.find(key);
    if (it_local != g_flow_key_to_local_gpu.end()) {
      localGpuKey = it_local->second;
      g_flow_key_to_local_gpu.erase(it_local);
      g_local_flow_active[localGpuKey] = false;
      shouldScheduleLocalGpu = true;
    }

    auto it_key = g_key2JID.find(key);
    if (it_key == g_key2JID.end()) {
      if (shouldScheduleLocalGpu)
        Simulator::ScheduleNow([localGpuKey](){ TryScheduleLocalGpu(localGpuKey); });
      return false; // not found, maybe not an mnCCL message
    }
    jobId = it_key->second;

    auto it = g_outstanding.find(jobId);
    if (it == g_outstanding.end()) {
      if (shouldScheduleLocalGpu)
        Simulator::ScheduleNow([localGpuKey](){ TryScheduleLocalGpu(localGpuKey); });
      return false;
    }
    if (it->second == 0) {
      if (shouldScheduleLocalGpu)
        Simulator::ScheduleNow([localGpuKey](){ TryScheduleLocalGpu(localGpuKey); });
      return false;
    }

    bool jobRecordsFlow = g_record_flow_finish.count(jobId) ? g_record_flow_finish[jobId] : true;
    if (jobRecordsFlow)
      g_JID2FlowFinishes[jobId].push_back(record);
    it->second--;
    if (it->second == 0) {
      alloc = g_alloc[jobId];
      g_alloc.erase(jobId);
      g_outstanding.erase(jobId);
      g_key2JID.erase(key);
      flowRecords = g_JID2FlowFinishes[jobId];
      g_JID2FlowFinishes.erase(jobId);
      releaseOnFinish = g_release_on_finish.count(jobId) ? g_release_on_finish[jobId] : true;
      recordFlowFinish = g_record_flow_finish.count(jobId) ? g_record_flow_finish[jobId] : true;
      logCollectiveFinish = g_log_collective_submit.count(jobId) ? g_log_collective_submit[jobId] : true;
      g_release_on_finish.erase(jobId);
      g_record_flow_finish.erase(jobId);
      g_log_collective_submit.erase(jobId);
      finished = true;
    } else {
      g_key2JID.erase(key);
    }
  }

  if (finished) {
    if (releaseOnFinish) {
      cclScheduler::ReleasePlacement(jobId);
      cclScheduler::ReleaseExpertAlloc(jobId);
    }
    if (recordFlowFinish)
      WriteFlowFinishRecords(jobId, flowRecords);
    OnCollectiveFinished(jobId);
    if (logCollectiveFinish) {
      std::ostringstream ss;
      ss << "mnCCL: JID=" << jobId << " finished";
      ccl::CclLog(ss.str());
    }
  }
  if (shouldScheduleLocalGpu)
    TryScheduleLocalGpu(localGpuKey);
  return finished;
}

} // namespace mnccl

#endif // MNCCL_H
