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
#include <set>
#include <limits>
#include <numeric>
#include <tuple>
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
  uint64_t actualFctNs;
  uint64_t standaloneFctNs;
};

struct PipelineLatencyStats {
  uint64_t completedTasks = 0;
  uint64_t sumTtftNs = 0;
  uint64_t sumTpotNs = 0;
};

struct DispatchLatencyStats {
  uint64_t completedTasks = 0;
  uint64_t sumLatencyNs = 0;
  uint64_t sumTpotNs = 0;
};

struct TrafficStats {
  uint64_t dataSubmittedFlows = 0;
  uint64_t dataSubmittedBytes = 0;
  uint64_t routeProbeSubmittedFlows = 0;
  uint64_t routeProbeSubmittedBytes = 0;
  uint64_t routeProbeFinishedFlows = 0;
  uint64_t routeProbeFinishedBytes = 0;
};

// Map JID -> completed mnCCL messages. Stored for logging when the whole job
// finishes. Protected by g_mutex.
static std::map<uint32_t, std::vector<FlowFinishRecord>> g_JID2FlowFinishes;
static PipelineLatencyStats g_pipeline_latency_stats;
static DispatchLatencyStats g_dispatch_latency_stats;
static TrafficStats g_traffic_stats;

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

struct DispatchTask {
  uint32_t taskId;
  double submitTime;
  uint32_t dispatchN;
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
  uint32_t dispatch_n = 1;
  uint64_t dispatch_expert_ffn_params = 1000000000ULL;
  uint32_t dispatch_precision_bytes = 1;
  uint32_t dispatch_batch_size = 16;
};

enum class PipelineStage : uint8_t {
  Dispatch = 0,
  PrefillAllReduce,
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

struct DispatchTaskState {
  DispatchTask task;
  uint32_t placementOwner = 0;
  uint32_t dispatchJobId = 0;
  uint32_t actualNeed = 0;
  uint32_t traceLayerId = 0;
  RuntimeConfig dispatchConfig{};
  uint64_t startTimeNs = 0;
  uint64_t finishTimeNs = 0;
  std::vector<std::pair<uint32_t,uint32_t>> gpus;
  bool placementAttempted = false;
};

struct PipelineTaskPolicySnapshot {
  bool found = false;
  uint32_t taskId = 0;
  uint32_t prefillLength = 0;
  uint32_t decodeLength = 0;
  uint32_t decodeIterations = 0;
  uint32_t decodeIteration = 0;
  uint32_t actualNeedPrefill = 0;
  uint32_t actualNeedDecode = 0;
  uint64_t startTimeNs = 0;
  bool prefillPlacementReleased = true;
  bool decodePlacementReleased = true;
  std::vector<std::pair<uint32_t,uint32_t>> prefillGpus;
  std::vector<std::pair<uint32_t,uint32_t>> decodeGpus;
};

struct CollectivePolicySnapshot {
  bool found = false;
  uint32_t jobId = 0;
  CollectiveOp op = CollectiveOp::AllReduce;
  uint32_t need = 0;
  uint32_t outstanding = 0;
  uint32_t totalMessages = 0;
  uint64_t msgSize = 0;
  std::vector<std::pair<uint32_t,uint32_t>> gpus;
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
  uint64_t flowGroupKey = 0;
  uint64_t flowGroupBytes = 0;
  uint64_t flowGroupArrivalTimeNs = 0;
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

struct RouteProbeSample {
  bool found = false;
  uint64_t fctNs = 0;
  uint64_t finishNs = 0;
};

struct PdSplitRequest {
  PipelineTask task;
  RuntimeConfig config;
  uint32_t decodeIterations;
};

using LocalFlowSchedulePolicy =
  std::function<size_t(const std::vector<LocalFlowTask>& queue, uint64_t nowNs)>;

using LocalFlowPgPolicy =
  std::function<uint16_t(const LocalFlowTask& task, uint64_t nowNs)>;

using ExpertRoutePolicy =
  std::function<std::pair<uint32_t,uint32_t>(const ExpertRouteRequest& req)>;

using PdSplitPolicy =
  std::function<uint32_t(const PdSplitRequest& req)>;

//forward declaration
inline void SubmitJob(uint32_t jobId, int need, double sim_time, uint64_t workload);
inline void SubmitColJob(const CollectiveJob& job);
inline void SubmitTrainingAllReduce(uint32_t jobId,
                                    double submitTime,
                                    uint16_t pg,
                                    uint32_t need,
                                    uint64_t msgSize,
                                    uint64_t memBytesPerGpu);
inline void SubmitDispatchTask(const DispatchTask& task);
inline void StartTraceDispatchWorkload(double firstSubmitTime,
                                       double submitInterval,
                                       uint32_t dispatchN,
                                       bool printSubmissions,
                                       std::ostream* output);
inline void SetDispatchTraceTargets(const std::vector<uint64_t>& targetCounts,
                                    bool stopOnComplete);
inline void SetDispatchTraceTargets(const std::vector<std::vector<uint64_t>>& layerTargetCounts,
                                    bool stopOnComplete);
inline void ClearDispatchTraceTargets();
inline bool TraceDispatchEnabled();
inline bool TraceDispatchTargetsReached();
inline bool GetTraceDispatchCurrentLayer(uint32_t* layerOut);
inline void MaybeScheduleNextTraceDispatchTask(uint64_t nowNs);
inline RuntimeConfig RefreshRuntimeConfigForDispatch(cclScheduler::NeedEvent event,
                                                     const DispatchTask& task);
inline bool PrepareDispatchTaskPlacement(uint32_t taskId);
inline bool HotUpdateExpertPlacement(uint32_t placementOwner,
                                     PipelineStage stage,
                                     cclScheduler::PlacementKind pdKind,
                                     const RuntimeConfig& cfg,
                                     const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                                     uint32_t traceLayerId = 0);
inline void RetryWaitingDispatchTasks();
inline void TryStartDispatchTask(uint32_t taskId);
inline void FinishDispatchTask(uint32_t taskId, uint32_t jobId);
inline uint64_t DispatchMsgSize(const DispatchTask& task, const RuntimeConfig& cfg);
inline uint64_t DispatchTotalBytesPerSource(const DispatchTask& task, const RuntimeConfig& cfg);
inline void LogDispatchTaskSummary(const DispatchTaskState& state,
                                   uint64_t nowNs,
                                   const RuntimeConfig& cfg,
                                   bool completed,
                                   bool accumulateStats);
inline void SubmitPipelineTask(const PipelineTask& task);
inline uint16_t SubmitExpertDispatch(uint32_t jobId,
                                     uint16_t pg,
                                     const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                                     uint64_t msgSize,
                                     uint32_t topk,
                                     uint32_t expertNum,
                                     uint32_t expertPlacementOwner,
                                     bool releaseOnFinish,
                                     bool recordFlowFinish,
                                     bool logSubmit);
inline uint16_t SubmitTraceExpertDispatch(uint32_t jobId,
                                          uint16_t pg,
                                          const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                                          uint64_t msgSize,
                                          uint32_t topk,
                                          uint32_t expertNum,
                                          uint32_t expertPlacementOwner,
                                          uint32_t traceLayerHint,
                                          bool releaseOnFinish,
                                          bool recordFlowFinish,
                                          bool logSubmit);
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
inline uint32_t L1GroupOfNetworkNode(uint32_t nodeId);
inline uint64_t PairRttOrMax(uint32_t a, uint32_t b);
inline uint64_t CalculateStandaloneFct(uint32_t src, uint32_t dst, uint64_t size);
inline PipelineTaskPolicySnapshot GetPipelineTaskPolicySnapshotForJob(uint32_t jobId);
inline CollectivePolicySnapshot GetCollectivePolicySnapshot(uint32_t jobId);
inline uint64_t DecodeComputeDelayNs(const PipelineTaskState& state);
inline bool ExpertMapReady(
    const std::vector<std::vector<std::pair<uint32_t,uint32_t>>>& expertMap,
    uint32_t expertNum);
inline void ConfigureRuntime(const RuntimeConfig& cfg);
inline void SetGlobalPipelineNeeds(uint32_t needPrefill, uint32_t needDecode);
inline void SetModelParallelConfig(uint32_t expertNum);
inline size_t FifoLocalFlowSchedulePolicy(const std::vector<LocalFlowTask>& queue, uint64_t nowNs);
inline void SetLocalFlowSchedulePolicy(LocalFlowSchedulePolicy policy);
inline uint16_t KeepLocalFlowPgPolicy(const LocalFlowTask& task, uint64_t nowNs);
inline void SetLocalFlowPgPolicy(LocalFlowPgPolicy policy);
inline std::pair<uint32_t,uint32_t> RandomExpertRoutePolicy(const ExpertRouteRequest& req);
inline void SetExpertRoutePolicy(ExpertRoutePolicy policy);
inline void SubmitRouteProbe(uint16_t pg,
                             const std::pair<uint32_t,uint32_t>& srcGpu,
                             const std::pair<uint32_t,uint32_t>& dstGpu,
                             uint64_t bytes,
                             bool refreshStale = false);
inline void SetRouteProbeMaxInflight(size_t maxInflight);
inline void SetAsyncRouteProbeConfig(bool enabled,
                                     bool liveRoute,
                                     uint16_t highPg,
                                     uint16_t lowPg,
                                     uint64_t bytes,
                                     uint64_t intervalNs,
                                     uint32_t budgetPerTick,
                                     uint64_t refreshNs,
                                     uint32_t topK);
inline bool AsyncRouteProbeEnabled();
inline bool LiveRouteResolutionEnabled();
inline void RegisterAsyncRouteProbePairs(
    const std::vector<std::pair<std::pair<uint32_t,uint32_t>,
                                std::pair<uint32_t,uint32_t>>>& pairs);
inline void ScheduleAsyncRouteProbePump(uint64_t delayNs = 0);
inline void ResetExpertAccessStats(uint32_t expertNum, double prior);
inline void RecordExpertAccess(uint32_t expertId);
inline double GetExpertAccessEstimate(uint32_t expertId, uint32_t expertNum);
inline std::vector<double> GetExpertAccessEstimates(uint32_t expertNum);
inline uint64_t GetObservedExpertAccessSamples();
inline RouteProbeSample GetRouteProbeSample(uint16_t pg,
                                            const std::pair<uint32_t,uint32_t>& srcGpu,
                                            const std::pair<uint32_t,uint32_t>& dstGpu);
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
inline uint32_t NetworkNodeId(const std::pair<uint32_t,uint32_t>& gpu);
inline double GetLocalGpuQueuePressure(const std::pair<uint32_t,uint32_t>& gpu);
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
inline uint16_t FinishSubmit(uint32_t jobId,
                             CollectiveOp op,
                             const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                             const std::vector<uint64_t>& keys,
                             bool releaseOnFinish,
                             bool recordFlowFinish,
                             bool logSubmit);
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

inline uint32_t L1GroupOfNetworkNode(uint32_t nodeId) {
  if (nodeId < n.GetN()) {
    Ptr<Node> gpuNode = n.Get(nodeId);
    for (const auto& kv : nbr2if[gpuNode]) {
      if (kv.first->GetNodeType() == 1)
        return kv.first->GetId();
    }
  }
  return nodeId;
}

inline uint64_t PairRttOrMax(uint32_t a, uint32_t b) {
  auto outer = pairRtt.find(a);
  if (outer != pairRtt.end()) {
    auto inner = outer->second.find(b);
    if (inner != outer->second.end() && inner->second > 0)
      return inner->second;
  }
  return maxRtt > 0 ? maxRtt : 1;
}

inline uint64_t PairBwOrZero(uint32_t a, uint32_t b) {
  auto outer = pairBw.find(a);
  if (outer != pairBw.end()) {
    auto inner = outer->second.find(b);
    if (inner != outer->second.end())
      return inner->second;
  }
  return 0;
}

inline double NetworkLevelPenalty(uint32_t a, uint32_t b) {
  if (a == b)
    return 1.0;
  uint32_t gpusPerHost = std::max(1u, gpus_per_server);
  if (a / gpusPerHost == b / gpusPerHost)
    return 1.0;
  if (L1GroupOfNetworkNode(a) == L1GroupOfNetworkNode(b))
    return 0.82;
  return 0.70;
}

inline double PairAffinity(uint32_t a, uint32_t b) {
  if (a == b)
    return 1.0;
  uint64_t bw = PairBwOrZero(a, b);
  static uint64_t cachedMaxBw = 0;
  static uint64_t cachedMinRtt = 0;
  if (cachedMaxBw == 0) {
    for (const auto& outer : pairBw) {
      for (const auto& inner : outer.second)
        cachedMaxBw = std::max(cachedMaxBw, inner.second);
    }
  }
  uint64_t maxBw = cachedMaxBw;
  double bwTerm = (maxBw == 0 || bw == 0)
                  ? 1.0
                  : static_cast<double>(bw) / static_cast<double>(maxBw);
  if (cachedMinRtt == 0) {
    uint64_t minRtt = UINT64_MAX;
    for (const auto& outer : pairRtt) {
      for (const auto& inner : outer.second) {
        if (inner.first != outer.first && inner.second > 0)
          minRtt = std::min(minRtt, inner.second);
      }
    }
    cachedMinRtt = minRtt == UINT64_MAX ? 1 : minRtt;
  }
  uint64_t rtt = PairRttOrMax(a, b);
  double delayTerm = rtt == 0 ? 1.0 : static_cast<double>(cachedMinRtt) / static_cast<double>(rtt);
  if (delayTerm > 1.0)
    delayTerm = 1.0;
  return bwTerm * delayTerm * NetworkLevelPenalty(a, b);
}

inline double ResourceBlockAffinity(const std::vector<uint32_t>& nodes) {
  if (nodes.size() < 2)
    return 1.0;
  double sum = 0.0;
  uint64_t pairs = 0;
  for (size_t i = 0; i < nodes.size(); ++i) {
    for (size_t j = i + 1; j < nodes.size(); ++j) {
      sum += PairAffinity(nodes[i], nodes[j]);
      pairs++;
    }
  }
  return pairs == 0 ? 1.0 : sum / static_cast<double>(pairs);
}

inline bool AffinityAccepts(const std::vector<uint32_t>& nodes, double theta) {
  if (nodes.size() < 2)
    return true;
  if (theta <= 0.000001)
    return true;
  double affinity = ResourceBlockAffinity(nodes);
  return affinity >= theta || affinity <= 0.0;
}

inline std::vector<uint32_t> BuildFreeNetworkNodes(
    const std::vector<std::vector<uint64_t>>& memUsed,
    const std::vector<std::vector<uint32_t>>& expertUsed,
    uint32_t numNodes,
    uint32_t gpusPerServer,
    const std::set<uint32_t>& releaseNodes = {}) {
  std::vector<uint32_t> freeNodes;
  for (uint32_t node = 0; node < numNodes; ++node) {
    for (uint32_t gpu = 0; gpu < gpusPerServer; ++gpu) {
      uint32_t networkNode = node * std::max(1u, gpusPerServer) + gpu;
      bool released = releaseNodes.count(networkNode) != 0;
      uint64_t memBytes = (node < memUsed.size() && gpu < memUsed[node].size())
                          ? memUsed[node][gpu]
                          : 0;
      uint32_t experts = (node < expertUsed.size() && gpu < expertUsed[node].size())
                         ? expertUsed[node][gpu]
                         : 0;
      if (released || (memBytes == 0 && experts == 0))
        freeNodes.push_back(networkNode);
    }
  }
  return freeNodes;
}

inline bool DocumentTaskSchedulable(
    const std::vector<uint32_t>& freeNodes,
    uint32_t need,
    double theta) {
  if (freeNodes.size() < need)
    return false;
  std::map<uint32_t, std::vector<uint32_t>> byHost;
  std::map<uint32_t, std::vector<uint32_t>> byL1;
  uint32_t stride = std::max(1u, gpus_per_server);
  for (uint32_t node : freeNodes) {
    byHost[node / stride].push_back(node);
    uint32_t l1 = L1GroupOfNetworkNode(node);
    if (l1 == node)
      l1 = node / 8;
    byL1[l1].push_back(node);
  }
  auto checkGroups = [&](const std::map<uint32_t, std::vector<uint32_t>>& groups) {
    for (const auto& kv : groups) {
      if (kv.second.size() >= need) {
        std::vector<uint32_t> chosen(kv.second.begin(), kv.second.begin() + need);
        if (AffinityAccepts(chosen, theta))
          return true;
      }
    }
    return false;
  };
  if (checkGroups(byHost) || checkGroups(byL1))
    return true;

  std::vector<uint32_t> chosen;
  std::vector<std::pair<size_t, uint32_t>> groupsBySize;
  for (const auto& kv : byL1)
    groupsBySize.emplace_back(kv.second.size(), kv.first);
  std::sort(groupsBySize.begin(), groupsBySize.end(), std::greater<std::pair<size_t, uint32_t>>());
  for (const auto& groupRef : groupsBySize) {
    const auto& group = byL1[groupRef.second];
    for (uint32_t node : group) {
      chosen.push_back(node);
      if (chosen.size() == need)
        break;
    }
    if (chosen.size() == need)
      break;
  }
  return chosen.size() == need && AffinityAccepts(chosen, theta);
}

inline double DocumentCnFragmentation(
    const std::vector<std::vector<uint64_t>>& memUsed,
    const std::vector<std::vector<uint32_t>>& expertUsed,
    uint32_t numNodes,
    uint32_t gpusPerServer,
    const std::set<uint32_t>& releaseNodes = {}) {
  auto freeNodes = BuildFreeNetworkNodes(memUsed, expertUsed, numNodes, gpusPerServer, releaseNodes);
  std::vector<std::tuple<uint32_t, double, double>> futureTasks{
      {2, 0.05, 0.000001},
      {4, 0.07, 0.000001},
      {8, 0.10, 0.000001},
      {16, 0.14, 0.000001},
      {32, 0.16, 0.000001},
      {64, 0.18, 0.000001},
      {128, 0.16, 0.000001},
      {192, 0.09, 0.000001},
      {256, 0.05, 0.000001}};
  double schedMass = 0.0;
  for (const auto& task : futureTasks) {
    uint32_t need = std::get<0>(task);
    double probability = std::get<1>(task);
    double theta = std::get<2>(task);
    if (DocumentTaskSchedulable(freeNodes, need, theta))
      schedMass += probability;
  }
  return std::max(0.0, std::min(1.0, 1.0 - schedMass));
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
static std::map<uint32_t, DispatchTaskState> g_dispatch_tasks;
static std::map<uint32_t, PipelineStageRef> g_pipeline_stage_by_job;
static std::map<uint32_t, uint32_t> g_dispatch_task_by_job;
static std::map<uint32_t, bool> g_release_on_finish;
static std::map<uint32_t, bool> g_record_flow_finish;
static std::map<uint32_t, bool> g_log_collective_submit;
static std::map<uint32_t, CollectivePolicySnapshot> g_collective_policy_snapshots;
static std::map<uint64_t, std::vector<LocalFlowTask>> g_local_flow_queues;
static std::map<uint64_t, bool> g_local_flow_active;
static std::map<uint64_t, uint64_t> g_flow_key_to_local_gpu;
static std::map<uint64_t, uint64_t> g_flow_key_alias;
static std::map<uint32_t, std::map<uint64_t, std::map<uint32_t, std::pair<uint32_t,uint32_t>>>> g_expert_route_cache;
static uint32_t g_route_probe_job_id = 0x70000000u;
static std::map<uint32_t, std::tuple<uint16_t, std::pair<uint32_t,uint32_t>, std::pair<uint32_t,uint32_t>>> g_route_probe_jobs;
static std::map<std::tuple<uint16_t,uint64_t,uint64_t>, RouteProbeSample> g_route_probe_samples;
static std::set<std::tuple<uint16_t,uint64_t,uint64_t>> g_route_probe_inflight;
static size_t g_route_probe_max_inflight = 64;
static bool g_async_route_probe_enabled = false;
static bool g_live_route_resolution_enabled = false;
static uint16_t g_async_route_probe_high_pg = 0;
static uint16_t g_async_route_probe_low_pg = 7;
static uint64_t g_async_route_probe_bytes = 1;
static uint64_t g_async_route_probe_interval_ns = 50000;
static uint32_t g_async_route_probe_budget_per_tick = 64;
static uint64_t g_async_route_probe_refresh_ns = 1000000;
static uint32_t g_async_route_probe_topk = 2;
static bool g_async_route_probe_scheduled = false;
static size_t g_async_route_probe_cursor = 0;
static std::vector<std::pair<std::pair<uint32_t,uint32_t>, std::pair<uint32_t,uint32_t>>>
    g_async_route_probe_pairs;
static std::set<std::pair<uint64_t,uint64_t>> g_async_route_probe_pair_keys;
static uint64_t g_local_flow_sequence = 0;
static LocalFlowSchedulePolicy g_local_flow_schedule_policy = FifoLocalFlowSchedulePolicy;
static LocalFlowPgPolicy g_local_flow_pg_policy = KeepLocalFlowPgPolicy;
static ExpertRoutePolicy g_expert_route_policy = RandomExpertRoutePolicy;
static PdSplitPolicy g_pd_split_policy = NoPrefillSideDecodeSplitPolicy;
static uint64_t g_local_flow_preemptive_chunk_bytes = 0;
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
    1,
    1,
    1000000000ULL,
    1,
    16};

// Global, unique probability table used by all alltoall jobs. If empty,
// a uniform distribution will be used for the group size when needed.
static std::vector<double> g_prob_table;
static std::vector<double> g_observed_expert_access;
static double g_expert_access_prior = 1.0;
static uint64_t g_observed_expert_access_samples = 0;
static bool g_trace_dispatch_enabled = false;
static bool g_trace_dispatch_stop_on_complete = true;
static bool g_trace_dispatch_started = false;
static double g_trace_dispatch_submit_interval = 0.00001;
static uint32_t g_trace_dispatch_dispatch_n = 1;
static uint64_t g_trace_dispatch_submitted_tasks = 0;
static bool g_trace_dispatch_print_submissions = false;
static std::ostream* g_trace_dispatch_output = &std::cout;
static std::vector<uint64_t> g_trace_dispatch_target_counts;
static std::vector<std::vector<uint64_t>> g_trace_dispatch_layer_target_counts;
static std::vector<std::vector<uint64_t>> g_trace_dispatch_layer_remaining_counts;
static uint32_t g_trace_dispatch_current_layer = 0;
static std::mt19937 g_trace_dispatch_rng(20260727u);

inline void SetGlobalProbTable(const std::vector<double>& table) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_prob_table = table;
}

inline void ResetExpertAccessStats(uint32_t expertNum, double prior) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_expert_access_prior = std::max(0.0, prior);
  g_observed_expert_access.assign(expertNum, 0.0);
  g_observed_expert_access_samples = 0;
}

inline void RecordExpertAccess(uint32_t expertId) {
  std::lock_guard<std::mutex> lk(g_mutex);
  if (expertId >= g_observed_expert_access.size())
    g_observed_expert_access.resize(expertId + 1, 0.0);
  g_observed_expert_access[expertId] += 1.0;
  g_observed_expert_access_samples += 1;
}

inline uint64_t ObservedExpertAccessCountLocked(uint32_t expertId) {
  if (expertId >= g_observed_expert_access.size())
    return 0;
  return static_cast<uint64_t>(g_observed_expert_access[expertId]);
}

inline uint64_t TraceDispatchLayerRemainingLocked(uint32_t layerId) {
  if (layerId >= g_trace_dispatch_layer_remaining_counts.size())
    return 0;
  return std::accumulate(g_trace_dispatch_layer_remaining_counts[layerId].begin(),
                         g_trace_dispatch_layer_remaining_counts[layerId].end(),
                         uint64_t{0});
}

inline bool TraceDispatchAdvanceLayerLocked(bool forceAdvance) {
  uint32_t layerCount =
      static_cast<uint32_t>(g_trace_dispatch_layer_remaining_counts.size());
  if (layerCount == 0)
    return false;
  uint32_t start = g_trace_dispatch_current_layer % layerCount;
  if (forceAdvance)
    start = (start + 1) % layerCount;
  for (uint32_t offset = 0; offset < layerCount; ++offset) {
    uint32_t layer = (start + offset) % layerCount;
    if (TraceDispatchLayerRemainingLocked(layer) > 0) {
      g_trace_dispatch_current_layer = layer;
      return true;
    }
  }
  return false;
}

inline bool TraceDispatchCurrentLayerIndexLocked(uint32_t* layerOut) {
  if (layerOut == nullptr)
    return false;
  if (!TraceDispatchAdvanceLayerLocked(false))
    return false;
  *layerOut = g_trace_dispatch_current_layer;
  return true;
}

inline bool TraceDispatchCurrentLayerLocked(uint32_t* layerOut) {
  std::lock_guard<std::mutex> lk(g_mutex);
  return TraceDispatchCurrentLayerIndexLocked(layerOut);
}

inline bool TraceDispatchTargetCompleteLocked(uint64_t* remainingOut = nullptr,
                                              uint32_t* unmetExpertsOut = nullptr,
                                              uint64_t* maxRemainingOut = nullptr) {
  uint64_t remaining = 0;
  std::vector<uint64_t> aggregateRemaining(g_trace_dispatch_target_counts.size(), 0);
  for (const auto& layerRemaining : g_trace_dispatch_layer_remaining_counts) {
    if (layerRemaining.size() > aggregateRemaining.size())
      aggregateRemaining.resize(layerRemaining.size(), 0);
    for (size_t expert = 0; expert < layerRemaining.size(); ++expert) {
      aggregateRemaining[expert] += layerRemaining[expert];
      remaining += layerRemaining[expert];
    }
  }
  uint32_t unmetExperts = 0;
  uint64_t maxRemaining = 0;
  for (uint64_t diff : aggregateRemaining) {
    if (diff > 0) {
      ++unmetExperts;
      maxRemaining = std::max(maxRemaining, diff);
    }
  }
  if (remainingOut != nullptr)
    *remainingOut = remaining;
  if (unmetExpertsOut != nullptr)
    *unmetExpertsOut = unmetExperts;
  if (maxRemainingOut != nullptr)
    *maxRemainingOut = maxRemaining;
  return remaining == 0;
}

inline bool TraceDispatchEnabled() {
  std::lock_guard<std::mutex> lk(g_mutex);
  return g_trace_dispatch_enabled;
}

inline bool TraceDispatchTargetsReached() {
  std::lock_guard<std::mutex> lk(g_mutex);
  if (!g_trace_dispatch_enabled)
    return false;
  return TraceDispatchTargetCompleteLocked();
}

inline bool GetTraceDispatchCurrentLayer(uint32_t* layerOut) {
  std::lock_guard<std::mutex> lk(g_mutex);
  return TraceDispatchCurrentLayerIndexLocked(layerOut);
}

inline void SetDispatchTraceTargets(const std::vector<uint64_t>& targetCounts,
                                    bool stopOnComplete) {
  std::vector<std::vector<uint64_t>> layers;
  if (!targetCounts.empty())
    layers.push_back(targetCounts);
  SetDispatchTraceTargets(layers, stopOnComplete);
}

inline void SetDispatchTraceTargets(const std::vector<std::vector<uint64_t>>& layerTargetCounts,
                                    bool stopOnComplete) {
  size_t expertCount = 0;
  uint64_t totalTarget = 0;
  uint32_t nonzeroExperts = 0;
  uint32_t nonzeroLayers = 0;
  uint64_t maxTarget = 0;
  for (const auto& targets : layerTargetCounts)
    expertCount = std::max(expertCount, targets.size());
  std::vector<uint64_t> aggregateTargets(expertCount, 0);
  for (const auto& targets : layerTargetCounts) {
    uint64_t layerTotal = 0;
    for (size_t expert = 0; expert < targets.size(); ++expert) {
      aggregateTargets[expert] += targets[expert];
      totalTarget += targets[expert];
      layerTotal += targets[expert];
      maxTarget = std::max(maxTarget, targets[expert]);
    }
    if (layerTotal > 0)
      ++nonzeroLayers;
  }
  for (uint64_t count : aggregateTargets) {
    if (count > 0)
      ++nonzeroExperts;
  }
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_trace_dispatch_target_counts = aggregateTargets;
    g_trace_dispatch_layer_target_counts = layerTargetCounts;
    g_trace_dispatch_layer_remaining_counts = layerTargetCounts;
    g_trace_dispatch_enabled = !layerTargetCounts.empty() && totalTarget > 0;
    g_trace_dispatch_stop_on_complete = stopOnComplete;
    g_trace_dispatch_started = false;
    g_trace_dispatch_submitted_tasks = 0;
    g_trace_dispatch_current_layer = 0;
    g_trace_dispatch_rng.seed(20260727u);
    TraceDispatchAdvanceLayerLocked(false);
  }
  std::ostringstream ss;
  ss << "mnCCL: dispatch trace targets configured"
     << " enabled=" << (totalTarget > 0 ? 1 : 0)
     << " layers=" << layerTargetCounts.size()
     << " nonzero_layers=" << nonzeroLayers
     << " experts=" << aggregateTargets.size()
     << " nonzero_experts=" << nonzeroExperts
     << " total_target_accesses=" << totalTarget
     << " max_expert_target=" << maxTarget
     << " stop_on_complete=" << (stopOnComplete ? 1 : 0);
  ccl::CclLog(ss.str());
}

inline void ClearDispatchTraceTargets() {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_trace_dispatch_enabled = false;
  g_trace_dispatch_started = false;
  g_trace_dispatch_submitted_tasks = 0;
  g_trace_dispatch_target_counts.clear();
  g_trace_dispatch_layer_target_counts.clear();
  g_trace_dispatch_layer_remaining_counts.clear();
  g_trace_dispatch_current_layer = 0;
}

inline bool BeginTraceDispatchLayer(uint32_t* layerOut,
                                    uint64_t* layerRemainingOut = nullptr) {
  std::lock_guard<std::mutex> lk(g_mutex);
  if (!g_trace_dispatch_enabled || layerOut == nullptr)
    return false;
  if (!TraceDispatchCurrentLayerIndexLocked(layerOut))
    return false;
  if (layerRemainingOut != nullptr)
    *layerRemainingOut = TraceDispatchLayerRemainingLocked(*layerOut);
  return true;
}

inline void FinishTraceDispatchLayerTask(uint32_t layerId) {
  std::lock_guard<std::mutex> lk(g_mutex);
  if (!g_trace_dispatch_enabled || g_trace_dispatch_layer_remaining_counts.empty())
    return;
  if (layerId < g_trace_dispatch_layer_remaining_counts.size())
    g_trace_dispatch_current_layer = layerId;
  TraceDispatchAdvanceLayerLocked(true);
}

inline uint64_t TraceDispatchLayerRemaining(uint32_t layerId) {
  std::lock_guard<std::mutex> lk(g_mutex);
  return TraceDispatchLayerRemainingLocked(layerId);
}

inline bool TakeTraceDispatchExpertFromLayer(uint32_t expertNum,
                                             uint32_t layerId,
                                             const std::set<uint32_t>& excluded,
                                             uint32_t* expertOut,
                                             bool* layerExhaustedOut = nullptr) {
  std::lock_guard<std::mutex> lk(g_mutex);
  if (layerExhaustedOut != nullptr)
    *layerExhaustedOut = true;
  if (!g_trace_dispatch_enabled || expertOut == nullptr ||
      layerId >= g_trace_dispatch_layer_remaining_counts.size())
    return false;

  auto& remainingCounts = g_trace_dispatch_layer_remaining_counts[layerId];
  uint32_t limit = std::min<uint32_t>(
      expertNum,
      static_cast<uint32_t>(remainingCounts.size()));
  uint64_t layerRemaining = std::accumulate(remainingCounts.begin(),
                                            remainingCounts.end(),
                                            uint64_t{0});
  uint64_t eligibleRemaining = 0;
  for (uint32_t expert = 0; expert < limit; ++expert) {
    uint64_t remaining = remainingCounts[expert];
    if (excluded.count(expert) != 0)
      continue;
    eligibleRemaining += remaining;
  }
  if (layerExhaustedOut != nullptr)
    *layerExhaustedOut = layerRemaining == 0;
  if (eligibleRemaining == 0)
    return false;

  std::uniform_int_distribution<uint64_t> dist(1, eligibleRemaining);
  uint64_t pick = dist(g_trace_dispatch_rng);
  uint32_t selected = std::numeric_limits<uint32_t>::max();
  for (uint32_t expert = 0; expert < limit; ++expert) {
    if (excluded.count(expert) != 0 || remainingCounts[expert] == 0)
      continue;
    if (pick <= remainingCounts[expert]) {
      selected = expert;
      break;
    }
    pick -= remainingCounts[expert];
  }
  if (selected == std::numeric_limits<uint32_t>::max())
    return false;

  --remainingCounts[selected];
  if (selected >= g_observed_expert_access.size())
    g_observed_expert_access.resize(selected + 1, 0.0);
  g_observed_expert_access[selected] += 1.0;
  g_observed_expert_access_samples += 1;
  *expertOut = selected;
  return true;
}

inline double GetExpertAccessEstimate(uint32_t expertId, uint32_t expertNum) {
  std::lock_guard<std::mutex> lk(g_mutex);
  if (expertNum == 0)
    expertNum = std::max<uint32_t>(
        1, static_cast<uint32_t>(g_observed_expert_access.size()));
  double estimate = g_expert_access_prior;
  if (expertId < g_observed_expert_access.size())
    estimate += g_observed_expert_access[expertId];
  return estimate > 0.0 ? estimate : 1.0;
}

inline std::vector<double> GetExpertAccessEstimates(uint32_t expertNum) {
  std::lock_guard<std::mutex> lk(g_mutex);
  expertNum = std::max<uint32_t>(
      expertNum, static_cast<uint32_t>(g_observed_expert_access.size()));
  if (expertNum == 0)
    expertNum = 1;
  std::vector<double> estimates(expertNum, std::max(0.0, g_expert_access_prior));
  for (size_t i = 0; i < g_observed_expert_access.size() && i < estimates.size(); ++i)
    estimates[i] += g_observed_expert_access[i];
  for (double& estimate : estimates) {
    if (estimate <= 0.0)
      estimate = 1.0;
  }
  return estimates;
}

inline uint64_t GetObservedExpertAccessSamples() {
  std::lock_guard<std::mutex> lk(g_mutex);
  return g_observed_expert_access_samples;
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

inline uint16_t KeepLocalFlowPgPolicy(const LocalFlowTask& task, uint64_t /*nowNs*/) {
  return task.pg;
}

inline void SetLocalFlowPgPolicy(LocalFlowPgPolicy policy) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_local_flow_pg_policy = policy ? policy : KeepLocalFlowPgPolicy;
}

inline void SetLocalFlowPreemptiveChunkBytes(uint64_t chunkBytes) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_local_flow_preemptive_chunk_bytes = chunkBytes;
}

inline bool LocalFlowPreemptionEnabledFor(const LocalFlowTask& task) {
  if (g_local_flow_preemptive_chunk_bytes == 0)
    return false;
  if (task.msgSize <= g_local_flow_preemptive_chunk_bytes)
    return false;
  return task.stage == PipelineStage::Dispatch ||
         task.stage == PipelineStage::PrefillAllReduce ||
         task.stage == PipelineStage::PrefillAllToAll ||
         task.pdKind == cclScheduler::PlacementKind::Prefill;
}

inline double GetLocalGpuQueuePressure(const std::pair<uint32_t,uint32_t>& gpu) {
  uint64_t localGpuKey = MakeGpuKey(gpu.first, gpu.second);
  std::lock_guard<std::mutex> lk(g_mutex);
  double pressure = 0.0;
  auto queueIt = g_local_flow_queues.find(localGpuKey);
  if (queueIt != g_local_flow_queues.end())
    pressure += static_cast<double>(queueIt->second.size());
  auto activeIt = g_local_flow_active.find(localGpuKey);
  if (activeIt != g_local_flow_active.end() && activeIt->second)
    pressure += 1.0;
  return pressure;
}

inline std::pair<uint32_t,uint32_t> SelectLocalExpertReplica(
    const std::pair<uint32_t,uint32_t>& sourceGpu,
    const std::vector<std::pair<uint32_t,uint32_t>>& candidates,
    bool* found = nullptr) {
  for (const auto& candidate : candidates) {
    if (candidate == sourceGpu) {
      if (found)
        *found = true;
      return candidate;
    }
  }
  for (const auto& candidate : candidates) {
    if (candidate.first == sourceGpu.first) {
      if (found)
        *found = true;
      return candidate;
    }
  }
  if (found)
    *found = false;
  return sourceGpu;
}

inline uint64_t StaticRouteRttNs(const std::pair<uint32_t,uint32_t>& sourceGpu,
                                 const std::pair<uint32_t,uint32_t>& dstGpu) {
  if (sourceGpu == dstGpu)
    return 0;
  return PairRttOrMax(NetworkNodeId(sourceGpu), NetworkNodeId(dstGpu));
}

inline std::vector<std::pair<uint32_t,uint32_t>> StaticRttRankedExpertReplicas(
    const std::pair<uint32_t,uint32_t>& sourceGpu,
    const std::vector<std::pair<uint32_t,uint32_t>>& candidates,
    size_t limit = 0) {
  std::vector<std::pair<uint32_t,uint32_t>> ranked;
  ranked.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    if (candidate != sourceGpu)
      ranked.push_back(candidate);
  }
  if (ranked.empty())
    ranked = candidates;

  std::stable_sort(
      ranked.begin(),
      ranked.end(),
      [&](const auto& a, const auto& b) {
        auto scoreA = std::make_tuple(
            StaticRouteRttNs(sourceGpu, a), NetworkNodeId(a), a.first, a.second);
        auto scoreB = std::make_tuple(
            StaticRouteRttNs(sourceGpu, b), NetworkNodeId(b), b.first, b.second);
        return scoreA < scoreB;
      });

  if (limit > 0 && ranked.size() > limit)
    ranked.resize(limit);
  return ranked;
}

// ===== SCHEDULING POLICY INTERFACE: same-rank expert replica routing =====
// When an expert rank has multiple replicas, this policy chooses the target
// replica. Default implementation prefers a same-host replica, then random.
inline std::pair<uint32_t,uint32_t> RandomExpertRoutePolicy(const ExpertRouteRequest& req) {
  if (req.candidates.empty())
    return req.sourceGpu;

  bool hasLocal = false;
  auto local = SelectLocalExpertReplica(req.sourceGpu, req.candidates, &hasLocal);
  if (hasLocal)
    return local;

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

inline void SetAsyncRouteProbeConfig(bool enabled,
                                     bool liveRoute,
                                     uint16_t highPg,
                                     uint16_t lowPg,
                                     uint64_t bytes,
                                     uint64_t intervalNs,
                                     uint32_t budgetPerTick,
                                     uint64_t refreshNs,
                                     uint32_t topK) {
  bool shouldSchedule = false;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_async_route_probe_enabled = enabled;
    g_live_route_resolution_enabled = liveRoute;
    g_async_route_probe_high_pg = highPg;
    g_async_route_probe_low_pg = lowPg;
    g_async_route_probe_bytes = std::max<uint64_t>(1, bytes);
    g_async_route_probe_interval_ns = std::max<uint64_t>(1, intervalNs);
    g_async_route_probe_budget_per_tick = std::max<uint32_t>(1, budgetPerTick);
    g_async_route_probe_refresh_ns = refreshNs;
    g_async_route_probe_topk = std::max<uint32_t>(1, topK);
    if (!enabled) {
      g_async_route_probe_scheduled = false;
      g_async_route_probe_pairs.clear();
      g_async_route_probe_pair_keys.clear();
      g_async_route_probe_cursor = 0;
    } else {
      shouldSchedule = !g_async_route_probe_pairs.empty();
    }
  }
  if (enabled && shouldSchedule)
    ScheduleAsyncRouteProbePump(0);
}

inline bool AsyncRouteProbeEnabled() {
  std::lock_guard<std::mutex> lk(g_mutex);
  return g_async_route_probe_enabled;
}

inline bool LiveRouteResolutionEnabled() {
  std::lock_guard<std::mutex> lk(g_mutex);
  return g_live_route_resolution_enabled;
}

inline void RegisterAsyncRouteProbePairs(
    const std::vector<std::pair<std::pair<uint32_t,uint32_t>,
                                std::pair<uint32_t,uint32_t>>>& pairs) {
  bool shouldSchedule = false;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (!g_async_route_probe_enabled)
      return;
    for (const auto& pair : pairs) {
      const auto& src = pair.first;
      const auto& dst = pair.second;
      if (src == dst)
        continue;
      uint32_t srcNode = NetworkNodeId(src);
      uint32_t dstNode = NetworkNodeId(dst);
      if (srcNode == dstNode)
        continue;
      auto key = std::make_pair(MakeGpuKey(src.first, src.second),
                                MakeGpuKey(dst.first, dst.second));
      if (g_async_route_probe_pair_keys.insert(key).second) {
        g_async_route_probe_pairs.push_back(pair);
        shouldSchedule = true;
      }
    }
  }
  if (shouldSchedule)
    ScheduleAsyncRouteProbePump(0);
}

inline void RunAsyncRouteProbePump() {
  bool enabled = false;
  uint16_t highPg = 0;
  uint16_t lowPg = 0;
  uint64_t bytes = 1;
  uint64_t intervalNs = 1;
  uint64_t refreshNs = 0;
  uint32_t budget = 1;
  size_t startCursor = 0;
  std::vector<std::pair<std::pair<uint32_t,uint32_t>, std::pair<uint32_t,uint32_t>>> pairs;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_async_route_probe_scheduled = false;
    enabled = g_async_route_probe_enabled;
    if (!enabled || g_async_route_probe_pairs.empty())
      return;
    highPg = g_async_route_probe_high_pg;
    lowPg = g_async_route_probe_low_pg;
    bytes = g_async_route_probe_bytes;
    intervalNs = g_async_route_probe_interval_ns;
    refreshNs = g_async_route_probe_refresh_ns;
    budget = g_async_route_probe_budget_per_tick;
    startCursor = g_async_route_probe_cursor;
    pairs = g_async_route_probe_pairs;
  }

  size_t checked = 0;
  size_t submittedPairs = 0;
  while (!pairs.empty() && checked < pairs.size() && submittedPairs < budget) {
    size_t idx = (startCursor + checked) % pairs.size();
    const auto& pair = pairs[idx];
    SubmitRouteProbe(highPg, pair.first, pair.second, bytes, true);
    SubmitRouteProbe(lowPg, pair.first, pair.second, bytes, true);
    ++checked;
    ++submittedPairs;
  }

  {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (!g_async_route_probe_pairs.empty())
      g_async_route_probe_cursor =
          (startCursor + checked) % g_async_route_probe_pairs.size();
  }
  (void)refreshNs;
  ScheduleAsyncRouteProbePump(intervalNs);
}

inline void ScheduleAsyncRouteProbePump(uint64_t delayNs) {
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (!g_async_route_probe_enabled || g_async_route_probe_scheduled ||
        g_async_route_probe_pairs.empty())
      return;
    g_async_route_probe_scheduled = true;
  }
  Simulator::Schedule(NanoSeconds(delayNs), []() {
    RunAsyncRouteProbePump();
  });
}

inline size_t StaticLocalFlowScheduleIndex(const std::vector<LocalFlowTask>& queue) {
  if (queue.empty())
    return 0;

  size_t bestIdx = 0;
  auto score = [](const LocalFlowTask& task) {
    uint8_t priority = 1;
    if (task.stage == PipelineStage::PrefillAllReduce || task.stage == PipelineStage::PrefillAllToAll)
      priority = 0;
    else if (task.stage == PipelineStage::Dispatch ||
             task.stage == PipelineStage::DecodeAllReduce ||
             task.stage == PipelineStage::DecodeAllToAll)
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

  bool hasLocal = false;
  auto local = SelectLocalExpertReplica(sourceGpu, candidates, &hasLocal);
  if (hasLocal)
    return local;

  bool liveRoute = false;
  ExpertRoutePolicy routePolicy;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    liveRoute = g_live_route_resolution_enabled;
    routePolicy = g_expert_route_policy ? g_expert_route_policy : RandomExpertRoutePolicy;
  }
  if (liveRoute) {
    return routePolicy(ExpertRouteRequest{
        0,
        placementOwner,
        dstExpert,
        PipelineStage::Dispatch,
        cclScheduler::PlacementKind::Generic,
        sourceGpu,
        candidates});
  }

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
  bool asyncProbe = false;
  uint32_t asyncTopK = 1;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    routePolicy = g_expert_route_policy ? g_expert_route_policy : RandomExpertRoutePolicy;
    asyncProbe = g_async_route_probe_enabled;
    asyncTopK = std::max<uint32_t>(1, g_async_route_probe_topk);
  }

  uint64_t localRoutes = 0;
  uint64_t totalRoutes = 0;
  std::vector<std::pair<std::pair<uint32_t,uint32_t>, std::pair<uint32_t,uint32_t>>>
      asyncPairs;
  std::set<std::pair<uint64_t,uint64_t>> asyncPairKeys;
  for (uint32_t sourceExpert = 0; sourceExpert < cfg.expert_num; ++sourceExpert) {
    auto sourceReplicas = cclScheduler::GetExpertReplicas(placementOwner, sourceExpert);
    for (const auto& sourceGpu : sourceReplicas) {
      uint64_t sourceKey = MakeGpuKey(sourceGpu.first, sourceGpu.second);
      auto& perSource = ownerCache[sourceKey];
      for (uint32_t dstExpert = 0; dstExpert < cfg.expert_num; ++dstExpert) {
        auto candidates = cclScheduler::GetExpertReplicas(placementOwner, dstExpert);
        if (candidates.empty())
          continue;
        bool hasLocal = false;
        auto local = SelectLocalExpertReplica(sourceGpu, candidates, &hasLocal);
        if (hasLocal) {
          perSource[dstExpert] = local;
          ++localRoutes;
        } else {
          if (asyncProbe) {
            auto ranked = StaticRttRankedExpertReplicas(
                sourceGpu, candidates, static_cast<size_t>(asyncTopK));
            for (const auto& candidate : ranked) {
              if (candidate == sourceGpu)
                continue;
              auto key = std::make_pair(MakeGpuKey(sourceGpu.first, sourceGpu.second),
                                        MakeGpuKey(candidate.first, candidate.second));
              if (asyncPairKeys.insert(key).second)
                asyncPairs.emplace_back(sourceGpu, candidate);
            }
            perSource[dstExpert] = ranked.empty() ? candidates.front() : ranked.front();
          } else {
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
        ++totalRoutes;
      }
    }
  }
  if (asyncProbe)
    RegisterAsyncRouteProbePairs(asyncPairs);

  {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_expert_route_cache[placementOwner] = std::move(ownerCache);
  }
  {
    std::ostringstream ss;
    ss << "mnCCL: expert route cache hot update"
       << " owner=" << placementOwner
       << " stage=" << PipelineStageName(stage)
       << " kind=" << cclScheduler::PlacementKindName(pdKind)
       << " routes=" << totalRoutes
       << " local_routes=" << localRoutes
       << " async_probe=" << (asyncProbe ? 1 : 0)
       << " async_probe_topk=" << (asyncProbe ? asyncTopK : 0)
       << " async_probe_candidates=" << asyncPairs.size();
    ccl::CclLog(ss.str());
  }
}

inline bool HotUpdateExpertPlacement(
    uint32_t placementOwner,
    PipelineStage stage,
    cclScheduler::PlacementKind pdKind,
    const RuntimeConfig& cfg,
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
    uint32_t traceLayerId) {
  auto expertMap = cclScheduler::AssignExperts(
      placementOwner,
      cfg.expert_num,
      cfg.expert_mem_bytes,
      cfg.expert_per_gpu,
      gpus,
      traceLayerId);
  if (!ExpertMapReady(expertMap, cfg.expert_num))
    return false;

  PrepareExpertRoutesForPlacement(placementOwner, stage, pdKind, cfg);
  {
    std::ostringstream ss;
    ss << "mnCCL: expert placement hot update dispatch"
       << " owner=" << placementOwner
       << " stage=" << PipelineStageName(stage)
       << " kind=" << cclScheduler::PlacementKindName(pdKind)
       << " trace_layer=" << traceLayerId
       << " expert_num=" << cfg.expert_num
       << " expert_per_gpu=" << cfg.expert_per_gpu
       << " selected_gpus=" << gpus.size();
    ccl::CclLog(ss.str());
  }
  return true;
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

inline PipelineTaskPolicySnapshot GetPipelineTaskPolicySnapshotForJob(uint32_t jobId) {
  PipelineTaskPolicySnapshot snapshot;
  std::lock_guard<std::mutex> lk(g_mutex);
  auto it_stage = g_pipeline_stage_by_job.find(jobId);
  if (it_stage == g_pipeline_stage_by_job.end())
    return snapshot;
  auto it_task = g_pipeline_tasks.find(it_stage->second.taskId);
  if (it_task == g_pipeline_tasks.end())
    return snapshot;

  const auto& state = it_task->second;
  snapshot.found = true;
  snapshot.taskId = state.task.taskId;
  snapshot.prefillLength = state.task.prefillLength;
  snapshot.decodeLength = state.task.decodeLength;
  snapshot.decodeIterations = state.decodeIterations;
  snapshot.decodeIteration = state.decodeIteration;
  snapshot.actualNeedPrefill = state.actualNeedPrefill;
  snapshot.actualNeedDecode = state.actualNeedDecode;
  snapshot.startTimeNs = state.startTimeNs;
  snapshot.prefillPlacementReleased = state.prefillPlacementReleased;
  snapshot.decodePlacementReleased = state.decodePlacementReleased;
  snapshot.prefillGpus = state.prefillGpus;
  snapshot.decodeGpus = state.decodeGpus;
  return snapshot;
}

inline CollectivePolicySnapshot GetCollectivePolicySnapshot(uint32_t jobId) {
  std::lock_guard<std::mutex> lk(g_mutex);
  auto it = g_collective_policy_snapshots.find(jobId);
  if (it == g_collective_policy_snapshots.end())
    return CollectivePolicySnapshot{};
  auto snapshot = it->second;
  auto out = g_outstanding.find(jobId);
  snapshot.outstanding = out == g_outstanding.end() ? 0 : out->second;
  return snapshot;
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

inline RuntimeConfig RefreshRuntimeConfigForDispatch(cclScheduler::NeedEvent event,
                                                     const DispatchTask& task) {
  RuntimeConfig cfg = GetRuntimeConfig();
  if (event != cclScheduler::NeedEvent::TaskDispatch)
    return cfg;
  cclScheduler::NeedState state{
      cfg.need_prefill,
      cfg.need_decode,
      cfg.expert_num,
      cfg.expert_per_gpu};
  cclScheduler::NotifyNeedEvent(event, task.taskId, task.dispatchN, 0, state);
  cfg.need_prefill = state.need_prefill;
  cfg.need_decode = state.need_decode;
  cfg.expert_num = state.expert_num;
  cfg.expert_per_gpu = state.expert_per_gpu;
  cfg.dispatch_n = std::max(1u, task.dispatchN);
  ConfigureRuntime(cfg);
  return cfg;
}

inline uint64_t SaturatingMul(uint64_t a, uint64_t b) {
  if (a == 0 || b == 0)
    return 0;
  if (a > std::numeric_limits<uint64_t>::max() / b)
    return std::numeric_limits<uint64_t>::max();
  return a * b;
}

inline uint64_t DispatchMsgSize(const DispatchTask& task, const RuntimeConfig& cfg) {
  uint64_t bytes = std::max<uint64_t>(1, cfg.dispatch_expert_ffn_params);
  bytes = SaturatingMul(bytes, std::max<uint32_t>(1, cfg.dispatch_precision_bytes));
  bytes = SaturatingMul(bytes, std::max<uint32_t>(1, cfg.dispatch_batch_size));
  bytes = SaturatingMul(bytes, std::max<uint32_t>(1, task.dispatchN));
  return std::max<uint64_t>(1, bytes);
}

inline uint64_t DispatchTotalBytesPerSource(const DispatchTask& task,
                                            const RuntimeConfig& cfg) {
  return SaturatingMul(DispatchMsgSize(task, cfg), std::max<uint32_t>(1, cfg.k));
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

inline void SubmitTrainingAllReduce(uint32_t jobId,
                                    double submitTime,
                                    uint16_t pg,
                                    uint32_t need,
                                    uint64_t msgSize,
                                    uint64_t memBytesPerGpu) {
  Simulator::Schedule(Seconds(submitTime), [jobId, pg, need, msgSize, memBytesPerGpu](){
    auto allocated = cclScheduler::TryPlaceNeed(
        jobId,
        cclScheduler::PlacementKind::Generic,
        need,
        1,
        memBytesPerGpu,
        1);
    if (allocated.size() != need) {
      ccl::CclLog("mnCCL: training allreduce job " + std::to_string(jobId) +
                  " placement failed need=" + std::to_string(need));
      cclScheduler::ReleasePlacement(jobId);
      return;
    }
    if (!cclScheduler::ReservePlacementMemory(jobId, allocated, memBytesPerGpu)) {
      ccl::CclLog("mnCCL: training allreduce job " + std::to_string(jobId) +
                  " reservation failed");
      cclScheduler::ReleasePlacement(jobId);
      return;
    }
    ccl::CclLog("mnCCL: training allreduce submit job=" + std::to_string(jobId) +
                " need=" + std::to_string(need) +
                " msg_size=" + std::to_string(msgSize) +
                " pg=" + std::to_string(pg));
    SubmitCollective(jobId,
                     CollectiveOp::AllReduce,
                     pg,
                     allocated,
                     msgSize,
                     0,
                     1,
                     1,
                     memBytesPerGpu,
                     true,
                     true,
                     true);
  });
}

inline void SubmitDispatchTask(const DispatchTask& task) {
  RuntimeConfig initialCfg = GetRuntimeConfig();
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    DispatchTaskState state{};
    state.task = task;
    state.placementOwner = PID++;
    state.dispatchConfig = initialCfg;
    g_dispatch_tasks[task.taskId] = state;
  }

  Simulator::Schedule(Seconds(task.submitTime), [task]() {
    auto cfg = RefreshRuntimeConfigForDispatch(
        cclScheduler::NeedEvent::TaskDispatch, task);
    {
      std::lock_guard<std::mutex> lk(g_mutex);
      auto it = g_dispatch_tasks.find(task.taskId);
      if (it != g_dispatch_tasks.end()) {
        it->second.startTimeNs = Simulator::Now().GetNanoSeconds();
        it->second.dispatchConfig = cfg;
      }
    }
    {
      std::ostringstream ss;
      ss << "mnCCL: dispatch task " << task.taskId
         << " submit"
         << " need=" << cfg.need_prefill
         << " expert_num=" << cfg.expert_num
         << " expert_per_gpu=" << cfg.expert_per_gpu
         << " topk=" << cfg.k
         << " dispatch_N=" << task.dispatchN
         << " msg_size_per_expert=" << DispatchMsgSize(task, cfg)
         << " total_bytes_per_source=" << DispatchTotalBytesPerSource(task, cfg);
      ccl::CclLog(ss.str());
    }
    if (!PrepareDispatchTaskPlacement(task.taskId)) {
      ccl::CclLog("mnCCL: dispatch task " + std::to_string(task.taskId) +
                  " placement failed; waiting for task completion retry");
      return;
    }
    TryStartDispatchTask(task.taskId);
  });
}

inline bool PrepareDispatchTaskPlacement(uint32_t taskId) {
  DispatchTaskState state;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_dispatch_tasks.find(taskId);
    if (it == g_dispatch_tasks.end())
      return false;
    state = it->second;
  }

  auto cfg = state.dispatchConfig;
  uint32_t traceLayerId = 0;
  if (TraceDispatchEnabled()) {
    GetTraceDispatchCurrentLayer(&traceLayerId);
    {
      std::lock_guard<std::mutex> lk(g_mutex);
      auto it = g_dispatch_tasks.find(taskId);
      if (it != g_dispatch_tasks.end())
        it->second.traceLayerId = traceLayerId;
    }
  }
  uint32_t dispatchNeed = std::max(1u, cfg.need_prefill);
  auto gpus = cclScheduler::TryPlaceNeed(
      state.placementOwner,
      cclScheduler::PlacementKind::Generic,
      dispatchNeed,
      cfg.expert_num,
      cfg.expert_mem_bytes,
      cfg.expert_per_gpu);
  if (gpus.size() != dispatchNeed)
    return false;

  if (!HotUpdateExpertPlacement(state.placementOwner,
                                PipelineStage::Dispatch,
                                cclScheduler::PlacementKind::Generic,
                                cfg,
                                gpus,
                                traceLayerId)) {
    cclScheduler::ReleasePlacement(state.placementOwner);
    cclScheduler::ReleaseExpertAlloc(state.placementOwner);
    return false;
  }

  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_dispatch_tasks.find(taskId);
    if (it == g_dispatch_tasks.end()) {
      g_expert_route_cache.erase(state.placementOwner);
      cclScheduler::ReleasePlacement(state.placementOwner);
      cclScheduler::ReleaseExpertAlloc(state.placementOwner);
      return false;
    }
    it->second.gpus = gpus;
    it->second.actualNeed = dispatchNeed;
    it->second.placementAttempted = true;
  }
  return true;
}

inline void RetryWaitingDispatchTasks() {
  std::vector<uint32_t> taskIds;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    for (const auto& kv : g_dispatch_tasks) {
      const auto& state = kv.second;
      if (state.startTimeNs > 0 && !state.placementAttempted && state.gpus.empty())
        taskIds.push_back(kv.first);
    }
  }

  for (uint32_t taskId : taskIds) {
    if (PrepareDispatchTaskPlacement(taskId)) {
      ccl::CclLog("mnCCL: dispatch task " + std::to_string(taskId) +
                  " placement retry succeeded");
      TryStartDispatchTask(taskId);
    } else {
      ccl::CclLog("mnCCL: dispatch task " + std::to_string(taskId) +
                  " placement retry still waiting for resources");
    }
  }
}

inline void TryStartDispatchTask(uint32_t taskId) {
  DispatchTaskState state;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_dispatch_tasks.find(taskId);
    if (it == g_dispatch_tasks.end())
      return;
    state = it->second;
  }
  if (state.gpus.empty())
    return;

  uint32_t jobId = JID++;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_dispatch_tasks.find(taskId);
    if (it == g_dispatch_tasks.end())
      return;
    it->second.dispatchJobId = jobId;
    g_dispatch_task_by_job[jobId] = taskId;
  }

  auto cfg = state.dispatchConfig;
  if (TraceDispatchEnabled()) {
    SubmitTraceExpertDispatch(jobId,
                              cfg.pg,
                              state.gpus,
                              DispatchMsgSize(state.task, cfg),
                              std::max(1u, cfg.k),
                              cfg.expert_num,
                              state.placementOwner,
                              state.traceLayerId,
                              false,
                              true,
                              true);
  } else {
    SubmitExpertDispatch(jobId,
                         cfg.pg,
                         state.gpus,
                         DispatchMsgSize(state.task, cfg),
                         std::max(1u, cfg.k),
                         cfg.expert_num,
                         state.placementOwner,
                         false,
                         true,
                         true);
  }
}

inline void FinishDispatchTask(uint32_t taskId, uint32_t jobId) {
  uint64_t nowNs = Simulator::Now().GetNanoSeconds();
  DispatchTaskState state;
  bool found = false;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_dispatch_tasks.find(taskId);
    if (it != g_dispatch_tasks.end()) {
      state = it->second;
      state.finishTimeNs = nowNs;
      found = true;
      g_dispatch_tasks.erase(it);
    }
    g_dispatch_task_by_job.erase(jobId);
    g_expert_route_cache.erase(state.placementOwner);
  }
  if (!found)
    return;

  cclScheduler::ReleasePlacement(state.placementOwner);
  cclScheduler::ReleaseExpertAlloc(state.placementOwner);
  LogDispatchTaskSummary(state, nowNs, state.dispatchConfig, true, true);
  RetryWaitingDispatchTasks();
  MaybeScheduleNextTraceDispatchTask(nowNs);
}

inline void SubmitNextTraceDispatchTaskAt(double submitTime) {
  uint32_t taskId = TID++;
  uint32_t dispatchN = 1;
  bool printSubmissions = false;
  std::ostream* output = nullptr;
  uint64_t submittedIndex = 0;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    dispatchN = std::max(1u, g_trace_dispatch_dispatch_n);
    printSubmissions = g_trace_dispatch_print_submissions;
    output = g_trace_dispatch_output;
    submittedIndex = ++g_trace_dispatch_submitted_tasks;
  }
  DispatchTask task{taskId, submitTime, dispatchN};
  SubmitDispatchTask(task);
  if (printSubmissions && output != nullptr) {
    *output << "NormalNetwork: scheduled trace dispatch task " << task.taskId
            << " trace_index=" << submittedIndex
            << " dispatch_N=" << task.dispatchN
            << " at " << task.submitTime << "s" << std::endl;
  }
}

inline void StartTraceDispatchWorkload(double firstSubmitTime,
                                       double submitInterval,
                                       uint32_t dispatchN,
                                       bool printSubmissions,
                                       std::ostream* output) {
  uint64_t remaining = 0;
  uint32_t unmetExperts = 0;
  uint64_t maxRemaining = 0;
  uint32_t currentLayer = 0;
  uint32_t layerCount = 0;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (!g_trace_dispatch_enabled) {
      ccl::CclLog("mnCCL: trace dispatch workload requested but no targets are enabled");
      return;
    }
    if (g_trace_dispatch_started) {
      ccl::CclLog("mnCCL: trace dispatch workload already started");
      return;
    }
    g_trace_dispatch_started = true;
    g_trace_dispatch_submit_interval = std::max(0.0, submitInterval);
    g_trace_dispatch_dispatch_n = std::max(1u, dispatchN);
    g_trace_dispatch_print_submissions = printSubmissions;
    g_trace_dispatch_output = output;
    TraceDispatchTargetCompleteLocked(&remaining, &unmetExperts, &maxRemaining);
    TraceDispatchCurrentLayerIndexLocked(&currentLayer);
    layerCount = static_cast<uint32_t>(g_trace_dispatch_layer_remaining_counts.size());
  }

  {
    std::ostringstream ss;
    ss << "mnCCL: trace dispatch workload start"
       << " first_submit_time=" << firstSubmitTime
       << " submit_interval=" << submitInterval
       << " dispatch_N=" << dispatchN
       << " current_layer=" << currentLayer
       << " layer_count=" << layerCount
       << " remaining_accesses=" << remaining
       << " unmet_experts=" << unmetExperts
       << " max_expert_remaining=" << maxRemaining;
    ccl::CclLog(ss.str());
  }

  if (remaining == 0) {
    ccl::CclLog("mnCCL: trace dispatch targets already satisfied; stopping simulation");
    if (g_trace_dispatch_stop_on_complete)
      Simulator::Stop();
    return;
  }
  SubmitNextTraceDispatchTaskAt(firstSubmitTime);
}

inline void MaybeScheduleNextTraceDispatchTask(uint64_t nowNs) {
  bool enabled = false;
  bool stopOnComplete = true;
  bool complete = false;
  double interval = 0.0;
  uint64_t remaining = 0;
  uint32_t unmetExperts = 0;
  uint64_t maxRemaining = 0;
  uint64_t submittedTasks = 0;
  uint32_t currentLayer = 0;
  uint32_t layerCount = 0;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    enabled = g_trace_dispatch_enabled && g_trace_dispatch_started;
    if (!enabled)
      return;
    complete = TraceDispatchTargetCompleteLocked(
        &remaining, &unmetExperts, &maxRemaining);
    stopOnComplete = g_trace_dispatch_stop_on_complete;
    interval = g_trace_dispatch_submit_interval;
    submittedTasks = g_trace_dispatch_submitted_tasks;
    TraceDispatchCurrentLayerIndexLocked(&currentLayer);
    layerCount = static_cast<uint32_t>(g_trace_dispatch_layer_remaining_counts.size());
  }

  if (complete) {
    std::ostringstream ss;
    ss << "mnCCL: trace dispatch targets reached"
       << " submitted_trace_tasks=" << submittedTasks
       << " observed_accesses=" << GetObservedExpertAccessSamples()
       << " now_ns=" << nowNs;
    ccl::CclLog(ss.str());
    if (stopOnComplete)
      Simulator::Stop();
    return;
  }

  {
    std::ostringstream ss;
    ss << "mnCCL: trace dispatch scheduling next task"
       << " submitted_trace_tasks=" << submittedTasks
       << " current_layer=" << currentLayer
       << " layer_count=" << layerCount
       << " remaining_accesses=" << remaining
       << " unmet_experts=" << unmetExperts
       << " max_expert_remaining=" << maxRemaining;
    ccl::CclLog(ss.str());
  }
  (void)nowNs;
  SubmitNextTraceDispatchTaskAt(interval);
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

  if (!HotUpdateExpertPlacement(state.prefillPlacementOwner,
                                PipelineStage::PrefillAllToAll,
                                cclScheduler::PlacementKind::Prefill,
                                cfg,
                                prefillGpus)) {
    cclScheduler::ReleasePlacement(state.prefillPlacementOwner);
    cclScheduler::ReleaseExpertAlloc(state.prefillPlacementOwner);
    return false;
  }

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

  if (!HotUpdateExpertPlacement(state.decodePlacementOwner,
                                PipelineStage::DecodeAllToAll,
                                cclScheduler::PlacementKind::Decode,
                                cfg,
                                decodeGpus)) {
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
    case PipelineStage::Dispatch: return "dispatch";
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
  std::vector<LocalFlowTask> queueSnapshot;
  uint64_t nowNs = static_cast<uint64_t>(Simulator::Now().GetNanoSeconds());
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_local_flow_active[localGpuKey])
      return;

    auto it = g_local_flow_queues.find(localGpuKey);
    if (it == g_local_flow_queues.end() || it->second.empty())
      return;

    queueSnapshot = it->second;
  }

  auto policy = g_local_flow_schedule_policy ? g_local_flow_schedule_policy : FifoLocalFlowSchedulePolicy;
  size_t idx = policy(queueSnapshot, nowNs);
  if (idx >= queueSnapshot.size())
    idx = 0;
  uint64_t selectedKey = queueSnapshot[idx].key;
  LocalFlowPgPolicy pgPolicy;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    pgPolicy = g_local_flow_pg_policy ? g_local_flow_pg_policy : KeepLocalFlowPgPolicy;
  }

  {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_local_flow_active[localGpuKey])
      return;

    auto it = g_local_flow_queues.find(localGpuKey);
    if (it == g_local_flow_queues.end() || it->second.empty())
      return;

    auto selected = std::find_if(it->second.begin(), it->second.end(),
                                [selectedKey](const LocalFlowTask& candidate) {
                                  return candidate.key == selectedKey;
                                });
    if (selected == it->second.end())
      selected = it->second.begin();
    task = *selected;
  }

  uint16_t mappedPg = pgPolicy(task, nowNs);

  {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (g_local_flow_active[localGpuKey])
      return;

    auto it = g_local_flow_queues.find(localGpuKey);
    if (it == g_local_flow_queues.end() || it->second.empty())
      return;

    auto selected = std::find_if(it->second.begin(), it->second.end(),
                                [selectedKey](const LocalFlowTask& candidate) {
                                  return candidate.key == selectedKey;
                                });
    if (selected == it->second.end())
      selected = it->second.begin();
    task = *selected;
    it->second.erase(selected);
    if (mappedPg != task.pg) {
      uint64_t oldKey = task.key;
      task.pg = mappedPg;
      task.srcPort = portNumber[task.srcNode][task.dstNode]++;
      task.key = MakeKey(task.pg, task.srcNode, task.dstNode, task.srcPort);
      g_flow_key_alias[oldKey] = task.key;
      g_flow_key_to_local_gpu.erase(oldKey);
      g_flow_key_to_local_gpu[task.key] = localGpuKey;
      auto itJob = g_key2JID.find(oldKey);
      if (itJob != g_key2JID.end()) {
        uint32_t jobId = itJob->second;
        g_key2JID.erase(itJob);
        g_key2JID[task.key] = jobId;
      }
    }
    g_local_flow_active[localGpuKey] = true;
    hasTask = true;
  }

  if (hasTask)
    DispatchLocalFlow(task);
}

inline std::vector<LocalFlowTask> ExpandLocalFlowTaskIntoChunks(const LocalFlowTask& baseTask) {
  uint64_t chunkBytes = 0;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    chunkBytes = g_local_flow_preemptive_chunk_bytes;
  }

  auto normalized = baseTask;
  normalized.flowGroupKey = baseTask.flowGroupKey == 0 ? baseTask.key : baseTask.flowGroupKey;
  normalized.flowGroupBytes = baseTask.flowGroupBytes == 0 ? baseTask.msgSize : baseTask.flowGroupBytes;
  normalized.flowGroupArrivalTimeNs = baseTask.flowGroupArrivalTimeNs == 0
                                      ? baseTask.arrivalTimeNs
                                      : baseTask.flowGroupArrivalTimeNs;

  bool chunkable = chunkBytes > 0 &&
                   baseTask.msgSize > chunkBytes &&
                   (baseTask.stage == PipelineStage::Dispatch ||
                    baseTask.stage == PipelineStage::PrefillAllReduce ||
                    baseTask.stage == PipelineStage::PrefillAllToAll ||
                    baseTask.pdKind == cclScheduler::PlacementKind::Prefill);
  if (!chunkable)
    return std::vector<LocalFlowTask>{normalized};

  std::vector<LocalFlowTask> chunks;
  uint64_t remaining = baseTask.msgSize;
  bool first = true;
  while (remaining > 0) {
    LocalFlowTask chunk = normalized;
    uint64_t bytes = std::min(chunkBytes, remaining);
    chunk.msgSize = bytes;
    chunk.flowGroupKey = baseTask.key;
    chunk.flowGroupBytes = baseTask.msgSize;
    chunk.flowGroupArrivalTimeNs = baseTask.arrivalTimeNs;
    if (!first) {
      chunk.srcPort = portNumber[chunk.srcNode][chunk.dstNode]++;
      chunk.key = MakeKey(chunk.pg, chunk.srcNode, chunk.dstNode, chunk.srcPort);
      chunk.sequence = g_local_flow_sequence++;
    }
    chunks.push_back(chunk);
    remaining -= bytes;
    first = false;
  }
  return chunks;
}

inline void SubmitLocalFlowTasks(std::vector<LocalFlowTask>& tasks,
                                 uint64_t localGpuKey,
                                 std::vector<uint64_t>& keys,
                                 bool useQueue) {
  if (tasks.empty())
    return;

  if (!useQueue) {
    for (const auto& task : tasks) {
      keys.push_back(task.key);
      {
        std::lock_guard<std::mutex> lk(g_mutex);
        g_traffic_stats.dataSubmittedFlows++;
        g_traffic_stats.dataSubmittedBytes += task.msgSize;
      }
      DispatchLocalFlow(task);
    }
    return;
  }

  {
    std::lock_guard<std::mutex> lk(g_mutex);
    for (const auto& task : tasks) {
      keys.push_back(task.key);
      g_traffic_stats.dataSubmittedFlows++;
      g_traffic_stats.dataSubmittedBytes += task.msgSize;
      g_local_flow_queues[localGpuKey].push_back(task);
      g_flow_key_to_local_gpu[task.key] = localGpuKey;
    }
  }
  TryScheduleLocalGpu(localGpuKey);
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

  auto tasks = ExpandLocalFlowTaskIntoChunks(task);
  SubmitLocalFlowTasks(tasks, localGpuKey, keys, true);
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

  bool useQueue = UseLocalFlowQueue(stage);
  auto tasks = ExpandLocalFlowTaskIntoChunks(task);
  SubmitLocalFlowTasks(tasks, localGpuKey, keys, useQueue);
}

inline void SubmitRouteProbe(uint16_t pg,
                             const std::pair<uint32_t,uint32_t>& srcGpu,
                             const std::pair<uint32_t,uint32_t>& dstGpu,
                             uint64_t bytes,
                             bool refreshStale) {
  if (srcGpu == dstGpu || bytes == 0)
    return;
  uint32_t jobId = 0;
  std::vector<uint64_t> keys;
  uint32_t srcNode = NetworkNodeId(srcGpu);
  uint32_t dstNode = NetworkNodeId(dstGpu);
  if (srcNode == dstNode)
    return;
  uint16_t port = portNumber[srcNode][dstNode]++;
  uint64_t key = MakeKey(pg, srcNode, dstNode, port);
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto sampleKey = std::make_tuple(pg,
                                     MakeGpuKey(srcGpu.first, srcGpu.second),
                                     MakeGpuKey(dstGpu.first, dstGpu.second));
    if (g_route_probe_inflight.count(sampleKey) != 0)
      return;
    auto sampleIt = g_route_probe_samples.find(sampleKey);
    if (sampleIt != g_route_probe_samples.end()) {
      uint64_t nowNs = static_cast<uint64_t>(Simulator::Now().GetNanoSeconds());
      bool stale = refreshStale &&
                   g_async_route_probe_refresh_ns > 0 &&
                   nowNs >= sampleIt->second.finishNs &&
                   nowNs - sampleIt->second.finishNs >= g_async_route_probe_refresh_ns;
      if (!stale)
        return;
    }
    if (g_route_probe_inflight.size() >= g_route_probe_max_inflight)
      return;
    jobId = g_route_probe_job_id++;
    g_route_probe_jobs[jobId] = std::make_tuple(pg, srcGpu, dstGpu);
    g_route_probe_inflight.insert(sampleKey);
    g_traffic_stats.routeProbeSubmittedFlows++;
    g_traffic_stats.routeProbeSubmittedBytes += bytes;
  }

  LocalFlowTask task{
      key,
      g_local_flow_sequence++,
      jobId,
      CollectiveOp::AllToAll,
      PipelineStage::Generic,
      cclScheduler::PlacementKind::Generic,
      pg,
      srcNode,
      srcGpu.second,
      dstNode,
      dstGpu.second,
      port,
      bytes,
      static_cast<uint64_t>(Simulator::Now().GetNanoSeconds())};
  keys.push_back(key);
  FinishSubmit(jobId,
               CollectiveOp::AllToAll,
               std::vector<std::pair<uint32_t,uint32_t>>{srcGpu, dstGpu},
               keys,
               false,
               false,
               false);
  DispatchLocalFlow(task);
}

inline void SetRouteProbeMaxInflight(size_t maxInflight) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_route_probe_max_inflight = maxInflight;
}

inline RouteProbeSample GetRouteProbeSample(uint16_t pg,
                                            const std::pair<uint32_t,uint32_t>& srcGpu,
                                            const std::pair<uint32_t,uint32_t>& dstGpu) {
  std::lock_guard<std::mutex> lk(g_mutex);
  auto key = std::make_tuple(pg,
                             MakeGpuKey(srcGpu.first, srcGpu.second),
                             MakeGpuKey(dstGpu.first, dstGpu.second));
  auto it = g_route_probe_samples.find(key);
  return it == g_route_probe_samples.end() ? RouteProbeSample{} : it->second;
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
    g_collective_policy_snapshots[jobId] = CollectivePolicySnapshot{
        true,
        jobId,
        op,
        static_cast<uint32_t>(gpus.size()),
        static_cast<uint32_t>(keys.size()),
        static_cast<uint32_t>(keys.size()),
        0,
        gpus};
    for (auto key : keys) {
      auto alias = g_flow_key_alias.find(key);
      uint64_t effectiveKey = alias == g_flow_key_alias.end() ? key : alias->second;
      g_key2JID[effectiveKey] = jobId;
      if (alias != g_flow_key_alias.end())
        g_flow_key_alias.erase(alias);
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
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_collective_policy_snapshots[jobId] = CollectivePolicySnapshot{
        true,
        jobId,
        CollectiveOp::AllReduce,
        static_cast<uint32_t>(gpus.size()),
        static_cast<uint32_t>(gpus.size()),
        static_cast<uint32_t>(gpus.size()),
        0,
        gpus};
  }
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

inline uint16_t SubmitExpertDispatch(uint32_t jobId,
                                     uint16_t pg,
                                     const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                                     uint64_t msgSize,
                                     uint32_t topk,
                                     uint32_t expertNum,
                                     uint32_t expertPlacementOwner,
                                     bool releaseOnFinish,
                                     bool recordFlowFinish,
                                     bool logSubmit) {
  if (gpus.empty() || topk == 0 || expertNum == 0)
    return FinishSubmit(jobId,
                        CollectiveOp::AllToAll,
                        gpus,
                        {},
                        releaseOnFinish,
                        recordFlowFinish,
                        logSubmit);

  std::vector<double> base;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (!g_prob_table.empty())
      base = g_prob_table;
  }
  if (base.empty())
    base.assign(expertNum, 1.0);

  std::vector<double> expertWeights(expertNum, 1.0);
  for (uint32_t e = 0; e < expertNum; ++e) {
    double v = base[e % base.size()];
    expertWeights[e] = v > 0.0 ? v : 1.0;
  }

  std::vector<uint64_t> keys;
  uint32_t draws = std::min(topk, expertNum);
  uint32_t placementOwner =
      expertPlacementOwner == 0 ? jobId : expertPlacementOwner;
  for (size_t srcIdx = 0; srcIdx < gpus.size(); ++srcIdx) {
    const auto& sourceGpu = gpus[srcIdx];
    auto weights = expertWeights;
    std::mt19937 rng(static_cast<uint32_t>(
        jobId ^ (srcIdx * 2654435761u) ^
        (sourceGpu.first * 16777619u) ^
        (sourceGpu.second * 2166136261u)));

    for (uint32_t draw = 0; draw < draws; ++draw) {
      double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
      if (sum <= 0.0)
        break;
      std::discrete_distribution<uint32_t> dist(weights.begin(), weights.end());
      uint32_t dstExpert = dist(rng);
      weights[dstExpert] = 0.0;

      auto candidates = cclScheduler::GetExpertReplicas(placementOwner, dstExpert);
      if (candidates.empty())
        continue;
      RecordExpertAccess(dstExpert);
      auto dstGpu = ResolveCachedExpertRoute(
          placementOwner, sourceGpu, dstExpert, candidates);
      SubmitFlow(pg,
                 sourceGpu,
                 dstGpu,
                 msgSize,
                 keys,
                 jobId,
                 CollectiveOp::AllToAll,
                 PipelineStage::Dispatch,
                 cclScheduler::PlacementKind::Generic);
    }
  }

  return FinishSubmit(jobId,
                      CollectiveOp::AllToAll,
                      gpus,
                      keys,
                      releaseOnFinish,
                      recordFlowFinish,
                      logSubmit);
}

inline uint16_t SubmitTraceExpertDispatch(uint32_t jobId,
                                          uint16_t pg,
                                          const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                                          uint64_t msgSize,
                                          uint32_t topk,
                                          uint32_t expertNum,
                                          uint32_t expertPlacementOwner,
                                          uint32_t traceLayerHint,
                                          bool releaseOnFinish,
                                          bool recordFlowFinish,
                                          bool logSubmit) {
  if (gpus.empty() || topk == 0 || expertNum == 0)
    return FinishSubmit(jobId,
                        CollectiveOp::AllToAll,
                        gpus,
                        {},
                        releaseOnFinish,
                        recordFlowFinish,
                        logSubmit);

  std::vector<uint64_t> keys;
  uint32_t draws = std::min(topk, expertNum);
  uint32_t placementOwner =
      expertPlacementOwner == 0 ? jobId : expertPlacementOwner;
  uint64_t submittedAccesses = 0;
  bool exhausted = false;
  uint32_t traceLayer = 0;
  uint64_t layerRemainingAtStart = 0;
  bool foundLayer = false;
  if (traceLayerHint != std::numeric_limits<uint32_t>::max()) {
    uint64_t hintedRemaining = TraceDispatchLayerRemaining(traceLayerHint);
    if (hintedRemaining > 0) {
      traceLayer = traceLayerHint;
      layerRemainingAtStart = hintedRemaining;
      foundLayer = true;
    }
  }
  if (!foundLayer && !BeginTraceDispatchLayer(&traceLayer, &layerRemainingAtStart)) {
    return FinishSubmit(jobId,
                        CollectiveOp::AllToAll,
                        gpus,
                        keys,
                        releaseOnFinish,
                        recordFlowFinish,
                        logSubmit);
  }

  for (size_t srcIdx = 0; srcIdx < gpus.size() && !exhausted; ++srcIdx) {
    const auto& sourceGpu = gpus[srcIdx];
    std::set<uint32_t> pickedForSource;
    for (uint32_t draw = 0; draw < draws; ++draw) {
      uint32_t dstExpert = 0;
      bool layerExhausted = false;
      if (!TakeTraceDispatchExpertFromLayer(expertNum,
                                            traceLayer,
                                            pickedForSource,
                                            &dstExpert,
                                            &layerExhausted)) {
        if (layerExhausted)
          exhausted = true;
        break;
      }
      pickedForSource.insert(dstExpert);
      auto candidates = cclScheduler::GetExpertReplicas(placementOwner, dstExpert);
      if (candidates.empty()) {
        ccl::CclLog("mnCCL: trace dispatch target expert has no replicas expert=" +
                    std::to_string(dstExpert));
        continue;
      }
      auto dstGpu = ResolveCachedExpertRoute(
          placementOwner, sourceGpu, dstExpert, candidates);
      SubmitFlow(pg,
                 sourceGpu,
                 dstGpu,
                 msgSize,
                 keys,
                 jobId,
                 CollectiveOp::AllToAll,
                 PipelineStage::Dispatch,
                 cclScheduler::PlacementKind::Generic);
      ++submittedAccesses;
    }
  }
  uint64_t layerRemainingAfterSubmit = TraceDispatchLayerRemaining(traceLayer);
  FinishTraceDispatchLayerTask(traceLayer);

  if (logSubmit) {
    uint64_t remaining = 0;
    uint32_t unmetExperts = 0;
    uint64_t maxRemaining = 0;
    {
      std::lock_guard<std::mutex> lk(g_mutex);
      TraceDispatchTargetCompleteLocked(&remaining, &unmetExperts, &maxRemaining);
    }
    std::ostringstream ss;
    ss << "mnCCL: trace dispatch submitted"
       << " JID=" << jobId
       << " trace_layer=" << traceLayer
       << " participants=" << gpus.size()
       << " sends=" << submittedAccesses
       << " layer_remaining_before=" << layerRemainingAtStart
       << " layer_remaining_after=" << layerRemainingAfterSubmit
       << " remaining_accesses=" << remaining
       << " unmet_experts=" << unmetExperts
       << " max_expert_remaining=" << maxRemaining;
    ccl::CclLog(ss.str());
  }

  return FinishSubmit(jobId,
                      CollectiveOp::AllToAll,
                      gpus,
                      keys,
                      releaseOnFinish,
                      recordFlowFinish,
                      logSubmit);
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
  if (base.empty()) {
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
        RecordExpertAccess(static_cast<uint32_t>(dstExpert));
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
        << msg.qp->startTime.GetTimeStep() << "," << msg.actualFctNs << ","
        << msg.standaloneFctNs << "\n";
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

inline void LogDispatchTaskSummary(const DispatchTaskState& state,
                                   uint64_t nowNs,
                                   const RuntimeConfig& cfg,
                                   bool completed,
                                   bool accumulateStats) {
  uint64_t startNs = state.startTimeNs;
  uint64_t totalFctNs = nowNs >= startNs ? nowNs - startNs : 0;
  uint32_t n = std::max(1u, state.task.dispatchN);
  uint64_t tpotNs = totalFctNs / n;

  if (completed && accumulateStats) {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_dispatch_latency_stats.completedTasks++;
    g_dispatch_latency_stats.sumLatencyNs += totalFctNs;
    g_dispatch_latency_stats.sumTpotNs += tpotNs;
  }

  std::ostringstream ss;
  ss << "mnCCL: dispatch task " << state.task.taskId
     << (completed ? " finished" : " incomplete at simulation end") << "\n"
     << "  status:\n"
     << "    completed=" << (completed ? 1 : 0) << "\n"
     << "  latency:\n"
     << "    total_fct_ns=" << totalFctNs << "\n"
     << "    TPOT_ns=" << tpotNs << "\n"
     << "  timeline:\n"
     << "    start_ns=" << state.startTimeNs << "\n"
     << "    finish_ns=" << nowNs << "\n"
     << "  task:\n"
     << "    dispatch_N=" << state.task.dispatchN << "\n"
     << "    need=" << state.actualNeed << "\n"
     << "    dispatch_need=" << cfg.need_prefill << "\n"
     << "    expert_num=" << cfg.expert_num << "\n"
     << "    expert_per_gpu=" << cfg.expert_per_gpu << "\n"
     << "    topk=" << cfg.k << "\n"
     << "  comm:\n"
     << "    expert_ffn_params=" << cfg.dispatch_expert_ffn_params << "\n"
     << "    precision_bytes=" << cfg.dispatch_precision_bytes << "\n"
     << "    batch_size=" << cfg.dispatch_batch_size << "\n"
     << "    msg_size_per_expert=" << DispatchMsgSize(state.task, cfg) << "\n"
     << "    total_bytes_per_source=" << DispatchTotalBytesPerSource(state.task, cfg)
     << "\n"
     << "  jobs:\n"
     << "    dispatch_JID=" << state.dispatchJobId;
  ccl::CclLog(ss.str());
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
  std::vector<DispatchTaskState> activeDispatchTasks;
  std::vector<std::pair<uint32_t, PipelineStageRef>> outstandingStageRefs;
  std::vector<std::pair<uint64_t, uint64_t>> localQueueSizes;
  std::vector<std::pair<uint64_t, bool>> localActiveStates;
  uint64_t nowNs = Simulator::Now().GetNanoSeconds();
  uint32_t outstandingJobs = 0;
  uint64_t queuedFlows = 0;
  PipelineLatencyStats latencyStats{};
  DispatchLatencyStats dispatchLatencyStats{};
  TrafficStats trafficStats{};
  uint64_t routeProbeInflight = 0;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    for (const auto& kv : g_pipeline_tasks)
      activeTasks.push_back(kv.second);
    for (const auto& kv : g_dispatch_tasks)
      activeDispatchTasks.push_back(kv.second);
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
    dispatchLatencyStats = g_dispatch_latency_stats;
    trafficStats = g_traffic_stats;
    routeProbeInflight = g_route_probe_inflight.size();
  }

  {
    std::ostringstream ss;
    ss << "mnCCL: simulation end summary active_tasks="
       << (activeTasks.size() + activeDispatchTasks.size())
       << " outstanding_collectives=" << outstandingJobs
       << " queued_local_flows=" << queuedFlows;
    ccl::CclLog(ss.str());
  }

  {
    uint64_t totalSubmittedBytes =
        trafficStats.dataSubmittedBytes +
        trafficStats.routeProbeSubmittedBytes;
    uint64_t totalSubmittedFlows =
        trafficStats.dataSubmittedFlows +
        trafficStats.routeProbeSubmittedFlows;
    double probeByteRatio = totalSubmittedBytes == 0
                            ? 0.0
                            : static_cast<double>(trafficStats.routeProbeSubmittedBytes) /
                                  static_cast<double>(totalSubmittedBytes);
    double probeFlowRatio = totalSubmittedFlows == 0
                            ? 0.0
                            : static_cast<double>(trafficStats.routeProbeSubmittedFlows) /
                                  static_cast<double>(totalSubmittedFlows);
    std::ostringstream ss;
    ss << "mnCCL: traffic summary\n"
       << "  data_submitted_flows=" << trafficStats.dataSubmittedFlows << "\n"
       << "  data_submitted_bytes=" << trafficStats.dataSubmittedBytes << "\n"
       << "  probe_submitted_flows=" << trafficStats.routeProbeSubmittedFlows << "\n"
       << "  probe_submitted_bytes=" << trafficStats.routeProbeSubmittedBytes << "\n"
       << "  total_submitted_flows=" << totalSubmittedFlows << "\n"
       << "  total_submitted_bytes=" << totalSubmittedBytes << "\n"
       << "  probe_submitted_flow_ratio=" << probeFlowRatio << "\n"
       << "  probe_submitted_byte_ratio=" << probeByteRatio << "\n"
       << "  probe_submitted_percent=" << (probeByteRatio * 100.0) << "\n"
       << "  probe_finished_flows=" << trafficStats.routeProbeFinishedFlows << "\n"
       << "  probe_finished_bytes=" << trafficStats.routeProbeFinishedBytes << "\n"
       << "  probe_inflight=" << routeProbeInflight;
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
  for (const auto& state : activeDispatchTasks)
    LogDispatchTaskSummary(state, nowNs, state.dispatchConfig, false, false);

  if (latencyStats.completedTasks > 0 || !activeTasks.empty()) {
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
  {
    uint64_t avgLatencyNs = dispatchLatencyStats.completedTasks == 0
                            ? 0
                            : dispatchLatencyStats.sumLatencyNs /
                                  dispatchLatencyStats.completedTasks;
    uint64_t avgTpotNs = dispatchLatencyStats.completedTasks == 0
                         ? 0
                         : dispatchLatencyStats.sumTpotNs /
                               dispatchLatencyStats.completedTasks;
    std::ostringstream ss;
    ss << "mnCCL: completed dispatch task latency average\n"
       << "  completed_tasks=" << dispatchLatencyStats.completedTasks << "\n"
       << "  average_latency_ns=" << avgLatencyNs << "\n"
       << "  average_TPOT_ns=" << avgTpotNs;
    ccl::CclLog(ss.str());
  }
  {
    bool traceEnabled = false;
    bool complete = false;
    uint64_t remaining = 0;
    uint32_t unmetExperts = 0;
    uint64_t maxRemaining = 0;
    uint64_t totalTarget = 0;
    uint64_t submittedTasks = 0;
    uint64_t observed = 0;
    uint32_t layerCount = 0;
    uint32_t currentLayer = 0;
    {
      std::lock_guard<std::mutex> lk(g_mutex);
      traceEnabled = g_trace_dispatch_enabled;
      if (traceEnabled) {
        complete = TraceDispatchTargetCompleteLocked(
            &remaining, &unmetExperts, &maxRemaining);
        for (uint64_t target : g_trace_dispatch_target_counts)
          totalTarget += target;
        submittedTasks = g_trace_dispatch_submitted_tasks;
        observed = g_observed_expert_access_samples;
        layerCount = static_cast<uint32_t>(g_trace_dispatch_layer_remaining_counts.size());
        TraceDispatchCurrentLayerIndexLocked(&currentLayer);
      }
    }
    if (traceEnabled) {
      std::ostringstream ss;
      ss << "mnCCL: trace dispatch target summary\n"
         << "  completed=" << (complete ? 1 : 0) << "\n"
         << "  submitted_trace_tasks=" << submittedTasks << "\n"
         << "  moe_layers=" << layerCount << "\n"
         << "  current_layer=" << currentLayer << "\n"
         << "  target_accesses=" << totalTarget << "\n"
         << "  observed_accesses=" << observed << "\n"
         << "  remaining_accesses=" << remaining << "\n"
         << "  unmet_experts=" << unmetExperts << "\n"
         << "  max_expert_remaining=" << maxRemaining;
      ccl::CclLog(ss.str());
    }
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
    activeTasks = g_pipeline_tasks.size() + g_dispatch_tasks.size();
    outstandingJobs = g_outstanding.size();
    expertMemBytes = g_runtime_config.expert_mem_bytes;
    for (const auto& kv : g_local_flow_queues)
      queuedFlows += kv.second.size();
    for (const auto& kv : g_local_flow_active)
      if (kv.second)
        activeLocalGpus++;
  }

  std::vector<std::vector<uint32_t>> expertSlots;
  std::vector<std::vector<uint64_t>> memUsed;
  std::vector<std::vector<uint64_t>> gpuMem;
  std::vector<std::vector<uint32_t>> expertUsed;
  uint32_t numNodes = 0;
  uint32_t gpusPerServer = 0;
  uint32_t maxExpertsPerGpu = 0;
  cclScheduler::GetActiveExpertSlotSnapshot(
      expertSlots, numNodes, gpusPerServer, maxExpertsPerGpu);
  cclScheduler::GetResourceSnapshot(
      memUsed, gpuMem, expertUsed, numNodes, gpusPerServer, maxExpertsPerGpu);

  uint64_t totalGpus = 0;
  uint64_t usedGpus = 0;
  uint64_t freeGpus = 0;
  uint64_t totalExpertSlots = 0;
  uint64_t usedExpertSlots = 0;
  uint64_t freeExpertSlots = 0;
  uint64_t maxFreeExpertSlots = 0;
  uint64_t sumFreeSq = 0;
  uint64_t totalGpuMemBytes = 0;
  uint64_t usedGpuMemBytes = 0;
  uint32_t complete8GpuHosts = 0;
  std::map<uint32_t, uint32_t> freeGpusByL1;
  std::vector<uint32_t> freeNetworkNodes;
  for (uint32_t node = 0; node < numNodes; ++node) {
    uint32_t freeOnHost = 0;
    for (uint32_t gpu = 0; gpu < gpusPerServer; ++gpu) {
      uint64_t used = 0;
      if (node < expertSlots.size() && gpu < expertSlots[node].size())
        used = expertSlots[node][gpu];
      uint64_t capacity = maxExpertsPerGpu;
      uint64_t cappedUsed = std::min(used, capacity);
      uint64_t freeSlots = capacity > cappedUsed ? capacity - cappedUsed : 0;
      uint64_t gpuMemBytes = (node < gpuMem.size() && gpu < gpuMem[node].size())
                             ? gpuMem[node][gpu]
                             : capacity * expertMemBytes;
      uint64_t memBytes = (node < memUsed.size() && gpu < memUsed[node].size())
                          ? memUsed[node][gpu]
                          : cappedUsed * expertMemBytes;
      bool gpuBusy = memBytes > 0 || cappedUsed > 0;
      totalGpus++;
      totalExpertSlots += capacity;
      usedExpertSlots += cappedUsed;
      freeExpertSlots += freeSlots;
      maxFreeExpertSlots = std::max(maxFreeExpertSlots, freeSlots);
      sumFreeSq += freeSlots * freeSlots;
      totalGpuMemBytes += gpuMemBytes;
      usedGpuMemBytes += memBytes;
      if (gpuBusy)
        usedGpus++;
      else {
        freeGpus++;
        freeOnHost++;
        uint32_t networkNode = node * std::max(1u, gpusPerServer) + gpu;
        freeNetworkNodes.push_back(networkNode);
        freeGpusByL1[L1GroupOfNetworkNode(networkNode)]++;
      }
    }
    if (gpusPerServer >= 8 && freeOnHost >= 8)
      complete8GpuHosts += freeOnHost / 8;
  }

  uint32_t sameTor8GpuBlocks = 0;
  for (const auto& kv : freeGpusByL1)
    sameTor8GpuBlocks += kv.second / 8;

  double cnFragmentation = DocumentCnFragmentation(
      memUsed, expertUsed, numNodes, gpusPerServer);
  double schedSuccess = 1.0 - cnFragmentation;
  double fragBlocked = 0.0;
  std::vector<std::tuple<uint32_t, double, double>> futureTasks{
      {2, 0.05, 0.000001},
      {4, 0.07, 0.000001},
      {8, 0.10, 0.000001},
      {16, 0.14, 0.000001},
      {32, 0.16, 0.000001},
      {64, 0.18, 0.000001},
      {128, 0.16, 0.000001},
      {192, 0.09, 0.000001},
      {256, 0.05, 0.000001}};
  for (const auto& task : futureTasks) {
    uint32_t need = std::get<0>(task);
    double probability = std::get<1>(task);
    double theta = std::get<2>(task);
    bool enoughGpu = freeGpus >= need;
    bool ok = DocumentTaskSchedulable(freeNetworkNodes, need, theta);
    if (enoughGpu && !ok)
      fragBlocked += probability;
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
        << totalGpuMemBytes << ","
        << cnFragmentation << ","
        << schedSuccess << ","
        << fragBlocked << ","
        << complete8GpuHosts << ","
        << sameTor8GpuBlocks << ","
        << freeGpus << "\n";
    ofs.flush();
  }

  Simulator::Schedule(NanoSeconds(intervalNs), &SampleClusterTimeSeries);
}

inline void OnCollectiveFinished(uint32_t jobId) {
  uint32_t dispatchTaskId = 0;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it_dispatch = g_dispatch_task_by_job.find(jobId);
    if (it_dispatch != g_dispatch_task_by_job.end())
      dispatchTaskId = it_dispatch->second;
  }
  if (dispatchTaskId != 0) {
    FinishDispatchTask(dispatchTaskId, jobId);
    return;
  }

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
    case PipelineStage::Dispatch:
      break;
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
      static_cast<uint64_t>((Simulator::Now() - q->startTime).GetNanoSeconds()),
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
  bool routeProbeFinished = false;
  std::tuple<uint16_t, std::pair<uint32_t,uint32_t>, std::pair<uint32_t,uint32_t>> routeProbeMeta;

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
      auto probeIt = g_route_probe_jobs.find(jobId);
      if (probeIt != g_route_probe_jobs.end()) {
        routeProbeFinished = true;
        routeProbeMeta = probeIt->second;
        g_route_probe_jobs.erase(probeIt);
      }
      g_release_on_finish.erase(jobId);
      g_record_flow_finish.erase(jobId);
      g_log_collective_submit.erase(jobId);
      g_collective_policy_snapshots.erase(jobId);
      finished = true;
    } else {
      g_key2JID.erase(key);
    }
  }

  if (finished) {
    if (routeProbeFinished) {
      auto pgAndGpus = routeProbeMeta;
      uint16_t probePg = std::get<0>(pgAndGpus);
      auto srcGpu = std::get<1>(pgAndGpus);
      auto dstGpu = std::get<2>(pgAndGpus);
      std::lock_guard<std::mutex> lk(g_mutex);
      auto probeKey = std::make_tuple(probePg,
                                      MakeGpuKey(srcGpu.first, srcGpu.second),
                                      MakeGpuKey(dstGpu.first, dstGpu.second));
      g_route_probe_inflight.erase(probeKey);
      g_traffic_stats.routeProbeFinishedFlows++;
      g_traffic_stats.routeProbeFinishedBytes += msgSize;
      g_route_probe_samples[probeKey] =
          RouteProbeSample{
              true,
              static_cast<uint64_t>((Simulator::Now() - q->startTime).GetNanoSeconds()),
              static_cast<uint64_t>(Simulator::Now().GetNanoSeconds())};
    }
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
