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

struct PipelineLatencyStats {
  uint64_t completedTasks = 0;
  uint64_t sumTtftNs = 0;
  uint64_t sumTpotNs = 0;
};

// Map JID -> completed mnCCL messages. Stored for logging when the whole job
// finishes. Protected by g_mutex.
static std::map<uint32_t, std::vector<FlowFinishRecord>> g_JID2FlowFinishes;
static PipelineLatencyStats g_pipeline_latency_stats;

// Mutex protecting the above maps
static std::mutex g_mutex;
// Path to write flow completion records.
static std::string g_flow_finish_log_path;
static std::string g_cluster_monitor_log_path;
static uint64_t g_cluster_monitor_interval_ns = 1000000;
static bool g_cluster_monitor_enabled = false;

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
  uint32_t expert_per_gpu;
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
  uint32_t expert_per_gpu;
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
  RuntimeConfig dispatchConfig;
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
  bool placementAttempted;
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
inline bool PreparePipelineTaskPlacement(uint32_t taskId);
inline void RetryWaitingPipelineTasks();
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
                                   bool completed,
                                   bool accumulateStats);
inline void LogPipelineSummariesAtSimulationEnd();
inline void SetClusterMonitorLogPath(const std::string& path);
inline void StartClusterTimeSeriesMonitor(uint64_t intervalNs);
inline void SampleClusterTimeSeries();
inline uint64_t DecodeComputeDelayNs(const PipelineTaskState& state);
inline bool ExpertMapReady(
    const std::vector<std::vector<std::pair<uint32_t,uint32_t>>>& expertMap,
    uint32_t expertNum);
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
inline RuntimeConfig TaskRuntimeConfig(const PipelineTaskState& state);
inline RuntimeConfig RefreshRuntimeConfig(cclScheduler::NeedEvent event, const PipelineTask& task);
inline uint64_t PrefillAllReduceMsgSize(const PipelineTask& task);
inline uint64_t PrefillAllReduceMsgSize(const PipelineTask& task, const RuntimeConfig& cfg);
inline uint64_t PrefillAllToAllMsgSize(const PipelineTask& task);
inline uint64_t PrefillAllToAllMsgSize(const PipelineTask& task, const RuntimeConfig& cfg);
inline uint64_t KvCacheMsgSize(const PipelineTask& task);
inline uint64_t KvCacheMsgSize(const PipelineTask& task, const RuntimeConfig& cfg);
inline uint64_t TokenBytes(uint32_t tokens);
inline uint64_t TokenBytes(const RuntimeConfig& cfg, uint32_t tokens);
inline uint32_t DecodeIterationsForTask(const PipelineTask& task);
inline uint64_t DecodeIterationMsgSize(const PipelineTaskState& state);
inline void OnCollectiveFinished(uint32_t jobId);
inline uint64_t MakeGpuKey(uint32_t node, uint32_t gpu);
inline void TryScheduleLocalGpu(uint64_t localGpuKey);
inline size_t StaticLocalFlowScheduleIndex(const std::vector<LocalFlowTask>& queue);
inline void PrepareExpertRoutesForPlacement(uint32_t placementOwner,
                                            PipelineStage stage,
                                            cclScheduler::PlacementKind pdKind,
                                            const RuntimeConfig& cfg);
inline std::pair<uint32_t,uint32_t> ResolveCachedExpertRoute(
    uint32_t placementOwner,
    const std::pair<uint32_t,uint32_t>& sourceGpu,
    uint32_t dstExpert,
    const std::vector<std::pair<uint32_t,uint32_t>>& candidates);
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

inline void SetClusterMonitorLogPath(const std::string& path) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_cluster_monitor_log_path = path;
}

inline void StartClusterTimeSeriesMonitor(uint64_t intervalNs) {
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_cluster_monitor_interval_ns = std::max<uint64_t>(1, intervalNs);
    g_cluster_monitor_enabled = !g_cluster_monitor_log_path.empty();
  }
  if (g_cluster_monitor_enabled)
    Simulator::ScheduleNow(&SampleClusterTimeSeries);
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
static std::map<uint32_t, std::map<uint64_t, std::map<uint32_t, std::pair<uint32_t,uint32_t>>>> g_expert_route_cache;
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

inline size_t StaticLocalFlowScheduleIndex(const std::vector<LocalFlowTask>& queue) {
  if (queue.empty())
    return 0;

  size_t bestIdx = 0;
  auto score = [](const LocalFlowTask& task) {
    uint8_t priority = 1;
    if (task.stage == PipelineStage::PrefillAllReduce || task.stage == PipelineStage::PrefillAllToAll)
      priority = 0;
    else if (task.stage == PipelineStage::DecodeAllReduce || task.stage == PipelineStage::DecodeAllToAll)
      priority = 1;
    else
      priority = 2;
    return std::make_tuple(priority, task.msgSize, task.arrivalTimeNs, task.sequence);
  };
  auto bestScore = score(queue.front());
  for (size_t i = 1; i < queue.size(); ++i) {
    auto curScore = score(queue[i]);
    if (curScore < bestScore) {
      bestScore = curScore;
      bestIdx = i;
    }
  }
  return bestIdx;
}

inline std::pair<uint32_t,uint32_t> ResolveCachedExpertRoute(
    uint32_t placementOwner,
    const std::pair<uint32_t,uint32_t>& sourceGpu,
    uint32_t dstExpert,
    const std::vector<std::pair<uint32_t,uint32_t>>& candidates) {
  if (candidates.empty())
    return sourceGpu;

  uint64_t sourceKey = MakeGpuKey(sourceGpu.first, sourceGpu.second);
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto ownerIt = g_expert_route_cache.find(placementOwner);
    if (ownerIt != g_expert_route_cache.end()) {
      auto srcIt = ownerIt->second.find(sourceKey);
      if (srcIt != ownerIt->second.end()) {
        auto dstIt = srcIt->second.find(dstExpert);
        if (dstIt != srcIt->second.end())
          return dstIt->second;
      }
    }
  }

  for (const auto& candidate : candidates) {
    if (candidate != sourceGpu)
      return candidate;
  }
  return candidates.front();
}

inline void PrepareExpertRoutesForPlacement(uint32_t placementOwner,
                                            PipelineStage stage,
                                            cclScheduler::PlacementKind pdKind,
                                            const RuntimeConfig& cfg) {
  std::map<uint64_t, std::map<uint32_t, std::pair<uint32_t,uint32_t>>> ownerCache;
  ExpertRoutePolicy routePolicy;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    routePolicy = g_expert_route_policy ? g_expert_route_policy : RandomExpertRoutePolicy;
  }

  for (uint32_t sourceExpert = 0; sourceExpert < cfg.expert_num; ++sourceExpert) {
    auto sourceReplicas = cclScheduler::GetExpertReplicas(placementOwner, sourceExpert);
    for (const auto& sourceGpu : sourceReplicas) {
      uint64_t sourceKey = MakeGpuKey(sourceGpu.first, sourceGpu.second);
      auto& perSource = ownerCache[sourceKey];
      for (uint32_t dstExpert = 0; dstExpert < cfg.expert_num; ++dstExpert) {
        auto candidates = cclScheduler::GetExpertReplicas(placementOwner, dstExpert);
        if (candidates.empty())
          continue;
        perSource[dstExpert] = routePolicy(ExpertRouteRequest{
            0,
            placementOwner,
            dstExpert,
            stage,
            pdKind,
            sourceGpu,
            candidates});
      }
    }
  }

  std::lock_guard<std::mutex> lk(g_mutex);
  g_expert_route_cache[placementOwner] = std::move(ownerCache);
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

inline RuntimeConfig TaskRuntimeConfig(const PipelineTaskState& state) {
  return state.dispatchConfig;
}

inline RuntimeConfig RefreshRuntimeConfig(cclScheduler::NeedEvent event, const PipelineTask& task) {
  RuntimeConfig cfg = GetRuntimeConfig();
  if (event != cclScheduler::NeedEvent::TaskDispatch)
    return cfg;
  cclScheduler::NeedState state{
      cfg.need_prefill,
      cfg.need_decode,
      cfg.expert_num,
      cfg.expert_per_gpu};
  cclScheduler::NotifyNeedEvent(event, task.taskId, task.prefillLength, task.decodeLength, state);
  cfg.need_prefill = state.need_prefill;
  cfg.need_decode = state.need_decode;
  cfg.expert_num = state.expert_num;
  cfg.expert_per_gpu = state.expert_per_gpu;
  ConfigureRuntime(cfg);
  return cfg;
}

inline uint64_t TokenBytes(uint32_t tokens) {
  auto cfg = GetRuntimeConfig();
  return cfg.token_msg_size * static_cast<uint64_t>(std::max(1u, tokens));
}

inline uint64_t TokenBytes(const RuntimeConfig& cfg, uint32_t tokens) {
  return cfg.token_msg_size * static_cast<uint64_t>(std::max(1u, tokens));
}

inline uint64_t PrefillAllReduceMsgSize(const PipelineTask& task) {
  return TokenBytes(task.prefillLength);
}

inline uint64_t PrefillAllReduceMsgSize(const PipelineTask& task, const RuntimeConfig& cfg) {
  return TokenBytes(cfg, task.prefillLength);
}

inline uint64_t PrefillAllToAllMsgSize(const PipelineTask& task) {
  return TokenBytes(task.prefillLength);
}

inline uint64_t PrefillAllToAllMsgSize(const PipelineTask& task, const RuntimeConfig& cfg) {
  return TokenBytes(cfg, task.prefillLength);
}

inline uint64_t KvCacheMsgSize(const PipelineTask& task) {
  return TokenBytes(task.prefillLength);
}

inline uint64_t KvCacheMsgSize(const PipelineTask& task, const RuntimeConfig& cfg) {
  return TokenBytes(cfg, task.prefillLength);
}

inline uint32_t DecodeIterationsForTask(const PipelineTask& task) {
  return std::max(1u, task.decodeLength);
}

inline uint64_t DecodeIterationMsgSize(const PipelineTaskState& state) {
  return TokenBytes(TaskRuntimeConfig(state), 1);
}

inline bool ExpertMapReady(
    const std::vector<std::vector<std::pair<uint32_t,uint32_t>>>& expertMap,
    uint32_t expertNum) {
  if (expertMap.size() != expertNum)
    return false;
  for (const auto& replicas : expertMap) {
    if (replicas.empty())
      return false;
  }
  return true;
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
  SubmitColJob(CollectiveJob{
      jobId, CollectiveOp::AllReduce, default_pg, static_cast<uint32_t>(need),
      sim_time, workload, 0, 1, 1, 0, 1});
}

inline void SubmitColJob(const CollectiveJob& job) {
  Simulator::Schedule(Seconds(job.submitTime), [job](){
    cclScheduler::EnqueuePendingTask(job.jobId,
                                     job.op,
                                     job.pg,
                                     job.need,
                                     job.msgSize,
                                     job.root,
                                     job.k,
                                     job.expert_num,
                                     job.expert_mem_bytes,
                                     job.expert_per_gpu);
    cclScheduler::ScheduleTask();
  });
}

inline void SubmitPipelineTask(const PipelineTask& task) {
  RuntimeConfig initialCfg = GetRuntimeConfig();
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    PipelineTaskState state{};
    state.task = task;
    state.prefillPlacementOwner = PID++;
    state.decodePlacementOwner = PID++;
    state.decodeIterations = DecodeIterationsForTask(task);
    state.prefillSideDecodeIterations = 0;
    state.placementAttempted = false;
    state.prefillPlacementReleased = false;
    state.decodePlacementReleased = false;
    state.kvCacheTransferred = false;
    state.dispatchConfig = initialCfg;
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
        it->second.dispatchConfig = cfg;
      }
    }
    {
      std::ostringstream ss;
      ss << "mnCCL: pipeline task " << task.taskId
         << " dispatch need_prefill=" << cfg.need_prefill
         << " need_decode=" << cfg.need_decode
         << " expert_num=" << cfg.expert_num
         << " expert_per_gpu=" << cfg.expert_per_gpu
         << " prefill_length=" << task.prefillLength
         << " decode_length=" << task.decodeLength
         << " single_token_length=" << cfg.single_token_length
         << " prefill_side_decode_iterations=" << prefillSideDecodeIterations;
      ccl::CclLog(ss.str());
    }
    if (!PreparePipelineTaskPlacement(task.taskId)) {
      ccl::CclLog("mnCCL: pipeline task " + std::to_string(task.taskId) +
                  " dispatch placement failed; waiting for a future task-event policy decision");
      return;
    }
    TryStartPipelineTask(task.taskId);
  });
}

inline bool PreparePipelineTaskPlacement(uint32_t taskId) {
  PipelineTaskState state;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_pipeline_tasks.find(taskId);
    if (it == g_pipeline_tasks.end())
      return false;
    state = it->second;
  }

  auto cfg = TaskRuntimeConfig(state);
  auto prefillGpus = cclScheduler::TryPlaceNeed(
      state.prefillPlacementOwner,
      cclScheduler::PlacementKind::Prefill,
      cfg.need_prefill,
      cfg.expert_num,
      cfg.expert_mem_bytes,
      cfg.expert_per_gpu);
  if (prefillGpus.size() != cfg.need_prefill)
    return false;

  auto prefillExpertMap = cclScheduler::AssignExperts(
      state.prefillPlacementOwner,
      cfg.expert_num,
      cfg.expert_mem_bytes,
      cfg.expert_per_gpu,
      prefillGpus);
  if (!ExpertMapReady(prefillExpertMap, cfg.expert_num)) {
    cclScheduler::ReleasePlacement(state.prefillPlacementOwner);
    cclScheduler::ReleaseExpertAlloc(state.prefillPlacementOwner);
    return false;
  }
  PrepareExpertRoutesForPlacement(state.prefillPlacementOwner,
                                  PipelineStage::PrefillAllToAll,
                                  cclScheduler::PlacementKind::Prefill,
                                  cfg);

  auto decodeGpus = cclScheduler::TryPlaceNeed(
      state.decodePlacementOwner,
      cclScheduler::PlacementKind::Decode,
      cfg.need_decode,
      cfg.expert_num,
      cfg.expert_mem_bytes,
      cfg.expert_per_gpu);
  if (decodeGpus.size() != cfg.need_decode) {
    {
      std::lock_guard<std::mutex> lk(g_mutex);
      g_expert_route_cache.erase(state.prefillPlacementOwner);
    }
    cclScheduler::ReleasePlacement(state.prefillPlacementOwner);
    cclScheduler::ReleaseExpertAlloc(state.prefillPlacementOwner);
    return false;
  }

  auto decodeExpertMap = cclScheduler::AssignExperts(
      state.decodePlacementOwner,
      cfg.expert_num,
      cfg.expert_mem_bytes,
      cfg.expert_per_gpu,
      decodeGpus);
  if (!ExpertMapReady(decodeExpertMap, cfg.expert_num)) {
    {
      std::lock_guard<std::mutex> lk(g_mutex);
      g_expert_route_cache.erase(state.prefillPlacementOwner);
    }
    cclScheduler::ReleasePlacement(state.prefillPlacementOwner);
    cclScheduler::ReleaseExpertAlloc(state.prefillPlacementOwner);
    cclScheduler::ReleasePlacement(state.decodePlacementOwner);
    cclScheduler::ReleaseExpertAlloc(state.decodePlacementOwner);
    return false;
  }
  PrepareExpertRoutesForPlacement(state.decodePlacementOwner,
                                  PipelineStage::DecodeAllToAll,
                                  cclScheduler::PlacementKind::Decode,
                                  cfg);

  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_pipeline_tasks.find(taskId);
    if (it == g_pipeline_tasks.end()) {
      g_expert_route_cache.erase(state.prefillPlacementOwner);
      g_expert_route_cache.erase(state.decodePlacementOwner);
      cclScheduler::ReleasePlacement(state.prefillPlacementOwner);
      cclScheduler::ReleaseExpertAlloc(state.prefillPlacementOwner);
      cclScheduler::ReleasePlacement(state.decodePlacementOwner);
      cclScheduler::ReleaseExpertAlloc(state.decodePlacementOwner);
      return false;
    }
    it->second.prefillGpus = prefillGpus;
    it->second.decodeGpus = decodeGpus;
    it->second.actualNeedPrefill = cfg.need_prefill;
    it->second.actualNeedDecode = cfg.need_decode;
    it->second.placementAttempted = true;
  }
  return true;
}

inline void RetryWaitingPipelineTasks() {
  std::vector<uint32_t> taskIds;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    for (const auto& kv : g_pipeline_tasks) {
      const auto& state = kv.second;
      if (!state.placementAttempted && state.prefillGpus.empty() && state.decodeGpus.empty())
        taskIds.push_back(kv.first);
    }
  }

  for (uint32_t taskId : taskIds) {
    if (PreparePipelineTaskPlacement(taskId)) {
      ccl::CclLog("mnCCL: pipeline task " + std::to_string(taskId) +
                  " placement retry succeeded");
      TryStartPipelineTask(taskId);
    } else {
      ccl::CclLog("mnCCL: pipeline task " + std::to_string(taskId) +
                  " placement retry still waiting for resources");
    }
  }
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
        auto dstGpu = ResolveCachedExpertRoute(
            placementOwner,
            sourceGpu,
            static_cast<uint32_t>(dstExpert),
            candidates);
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

  if (state.prefillGpus.empty()) {
    return;
  }

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
  auto cfg = TaskRuntimeConfig(state);
  SubmitAllReduceForStage(jobId,
                          cfg.pg,
                          state.prefillGpus,
                          PrefillAllReduceMsgSize(state.task, cfg),
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
  auto cfg = TaskRuntimeConfig(state);
  SubmitAllToAll(jobId,
                 cfg.pg,
                 state.prefillGpus,
                 PrefillAllToAllMsgSize(state.task, cfg),
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

  if (state.decodeGpus.empty()) {
    return;
  }

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
  auto cfg = TaskRuntimeConfig(state);
  SubmitKvCacheTransfer(jobId, cfg.pg, state.prefillGpus, state.decodeGpus, KvCacheMsgSize(state.task, cfg));
}

inline uint64_t DecodeComputeDelayNs(const PipelineTaskState& state) {
  const auto& activeGpus = ActiveDecodeGpus(state);
  uint64_t avgRemaining = cclScheduler::GetAverageRemainingMemory(activeGpus);
  uint64_t avgCapacity = cclScheduler::GetAverageCapacity(activeGpus);
  auto cfg = TaskRuntimeConfig(state);
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
  auto cfg = TaskRuntimeConfig(state);
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
  auto cfg = TaskRuntimeConfig(state);
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
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_expert_route_cache.erase(state.prefillPlacementOwner);
    g_expert_route_cache.erase(state.decodePlacementOwner);
  }

  auto taskCfg = TaskRuntimeConfig(state);
  LogPipelineTaskSummary(state, nowNs, taskCfg, true, true);
  RetryWaitingPipelineTasks();
}

inline void LogPipelineTaskSummary(const PipelineTaskState& state,
                                   uint64_t nowNs,
                                   const RuntimeConfig& cfg,
                                   bool completed,
                                   bool accumulateStats) {
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

  if (completed && accumulateStats) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_pipeline_latency_stats.completedTasks++;
    g_pipeline_latency_stats.sumTtftNs += ttftNs;
    g_pipeline_latency_stats.sumTpotNs += tpotNs;
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
     << "    dispatch_need_prefill=" << cfg.need_prefill << "\n"
     << "    dispatch_need_decode=" << cfg.need_decode << "\n"
     << "    expert_num=" << cfg.expert_num << "\n"
     << "    expert_per_gpu=" << cfg.expert_per_gpu << "\n"
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
  std::vector<std::pair<uint64_t, uint64_t>> localQueueSizes;
  std::vector<std::pair<uint64_t, bool>> localActiveStates;
  uint64_t nowNs = Simulator::Now().GetNanoSeconds();
  uint32_t outstandingJobs = 0;
  uint64_t queuedFlows = 0;
  PipelineLatencyStats latencyStats{};
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    for (const auto& kv : g_pipeline_tasks)
      activeTasks.push_back(kv.second);
    outstandingJobs = static_cast<uint32_t>(g_outstanding.size());
    for (const auto& kv : g_local_flow_queues) {
      queuedFlows += kv.second.size();
      if (!kv.second.empty())
        localQueueSizes.emplace_back(kv.first, static_cast<uint64_t>(kv.second.size()));
    }
    for (const auto& kv : g_local_flow_active)
      if (kv.second)
        localActiveStates.emplace_back(kv.first, kv.second);
    for (const auto& kv : g_pipeline_stage_by_job)
      outstandingStageRefs.push_back(kv);
    latencyStats = g_pipeline_latency_stats;
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

  for (const auto& kv : localQueueSizes) {
    std::ostringstream ss;
    ss << "mnCCL: queued local GPU key=" << kv.first
       << " queued_flows=" << kv.second;
    ccl::CclLog(ss.str());
  }
  for (const auto& kv : localActiveStates) {
    std::ostringstream ss;
    ss << "mnCCL: active local GPU key=" << kv.first;
    ccl::CclLog(ss.str());
  }

  for (const auto& state : activeTasks)
    LogPipelineTaskSummary(state, nowNs, TaskRuntimeConfig(state), false, false);

  {
    uint64_t avgTtftNs = latencyStats.completedTasks == 0
                         ? 0
                         : latencyStats.sumTtftNs / latencyStats.completedTasks;
    uint64_t avgTpotNs = latencyStats.completedTasks == 0
                         ? 0
                         : latencyStats.sumTpotNs / latencyStats.completedTasks;
    std::ostringstream ss;
    ss << "mnCCL: completed task latency average\n"
       << "  completed_tasks=" << latencyStats.completedTasks << "\n"
       << "  average_TTFT_ns=" << avgTtftNs << "\n"
       << "  average_TPOT_ns=" << avgTpotNs;
    ccl::CclLog(ss.str());
  }
}

inline void SampleClusterTimeSeries() {
  std::string path;
  uint64_t intervalNs = 0;
  uint64_t activeTasks = 0;
  uint64_t outstandingJobs = 0;
  uint64_t queuedFlows = 0;
  uint64_t activeLocalGpus = 0;
  uint64_t expertMemBytes = 0;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (!g_cluster_monitor_enabled || g_cluster_monitor_log_path.empty())
      return;
    path = g_cluster_monitor_log_path;
    intervalNs = g_cluster_monitor_interval_ns;
    activeTasks = g_pipeline_tasks.size();
    outstandingJobs = g_outstanding.size();
    expertMemBytes = g_runtime_config.expert_mem_bytes;
    for (const auto& kv : g_local_flow_queues)
      queuedFlows += kv.second.size();
    for (const auto& kv : g_local_flow_active)
      if (kv.second)
        activeLocalGpus++;
  }

  std::vector<std::vector<uint32_t>> expertSlots;
  uint32_t numNodes = 0;
  uint32_t gpusPerServer = 0;
  uint32_t maxExpertsPerGpu = 0;
  cclScheduler::GetActiveExpertSlotSnapshot(
      expertSlots, numNodes, gpusPerServer, maxExpertsPerGpu);

  uint64_t totalGpus = 0;
  uint64_t usedGpus = 0;
  uint64_t totalExpertSlots = 0;
  uint64_t usedExpertSlots = 0;
  uint64_t freeExpertSlots = 0;
  uint64_t maxFreeExpertSlots = 0;
  uint64_t sumFreeSq = 0;
  uint64_t totalGpuMemBytes = 0;
  uint64_t usedGpuMemBytes = 0;
  for (uint32_t node = 0; node < numNodes; ++node) {
    for (uint32_t gpu = 0; gpu < gpusPerServer; ++gpu) {
      uint64_t used = 0;
      if (node < expertSlots.size() && gpu < expertSlots[node].size())
        used = expertSlots[node][gpu];
      uint64_t capacity = maxExpertsPerGpu;
      uint64_t cappedUsed = std::min(used, capacity);
      uint64_t freeSlots = capacity > cappedUsed ? capacity - cappedUsed : 0;
      uint64_t gpuMemBytes = capacity * expertMemBytes;
      totalGpus++;
      totalExpertSlots += capacity;
      usedExpertSlots += cappedUsed;
      freeExpertSlots += freeSlots;
      maxFreeExpertSlots = std::max(maxFreeExpertSlots, freeSlots);
      sumFreeSq += freeSlots * freeSlots;
      totalGpuMemBytes += gpuMemBytes;
      usedGpuMemBytes += cappedUsed * expertMemBytes;
      if (used > 0)
        usedGpus++;
    }
  }

  double placementGpuUtilization = totalGpus == 0 ? 0.0 : static_cast<double>(usedGpus) / totalGpus;
  double gpuUtilization = totalGpuMemBytes == 0
                          ? 0.0
                          : static_cast<double>(usedGpuMemBytes) / totalGpuMemBytes;
  double expertSlotUtilization = totalExpertSlots == 0
                                 ? 0.0
                                 : static_cast<double>(usedExpertSlots) / totalExpertSlots;
  double fragmentation = (freeExpertSlots == 0 || usedExpertSlots == 0)
                         ? 0.0
                         : 1.0 - static_cast<double>(maxFreeExpertSlots) / freeExpertSlots;
  double freeSlotVariance = 0.0;
  if (totalGpus > 0) {
    double meanFree = static_cast<double>(freeExpertSlots) / totalGpus;
    freeSlotVariance = static_cast<double>(sumFreeSq) / totalGpus - meanFree * meanFree;
  }

  std::ofstream ofs(path, std::ofstream::app);
  if (ofs) {
    ofs << Simulator::Now().GetNanoSeconds() << ","
        << activeTasks << ","
        << outstandingJobs << ","
        << queuedFlows << ","
        << activeLocalGpus << ","
        << totalGpus << ","
        << usedGpus << ","
        << totalExpertSlots << ","
        << usedExpertSlots << ","
        << freeExpertSlots << ","
        << maxFreeExpertSlots << ","
        << gpuUtilization << ","
        << expertSlotUtilization << ","
        << fragmentation << ","
        << freeSlotVariance << ","
        << placementGpuUtilization << ","
        << usedGpuMemBytes << ","
        << totalGpuMemBytes << "\n";
    ofs.flush();
  }

  Simulator::Schedule(NanoSeconds(intervalNs), &SampleClusterTimeSeries);
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
