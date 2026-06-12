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
  uint64_t prefill_unit_msg_size;
  uint64_t prefill_alltoall_msg_size;
  uint64_t token_msg_size;
  uint64_t base_decode_compute_delay_ns;
};

enum class PipelineStage : uint8_t {
  PrefillAllReduce = 0,
  PrefillAllToAll,
  KvCacheTransfer,
  DecodeAllReduce,
  DecodeAllToAll
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
  uint64_t startTimeNs;
  uint64_t prefillFinishNs;
  uint64_t decodeStartNs;
  uint64_t firstTokenFinishNs;
  std::vector<std::pair<uint32_t,uint32_t>> prefillGpus;
  std::vector<std::pair<uint32_t,uint32_t>> decodeGpus;
  bool prefillPlacementReleased;
  bool decodePlacementReleased;
};

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
inline uint64_t DecodeComputeDelayNs(const PipelineTaskState& state);
inline void ConfigureRuntime(const RuntimeConfig& cfg);
inline void SetGlobalPipelineNeeds(uint32_t needPrefill, uint32_t needDecode);
inline void SetModelParallelConfig(uint32_t expertNum);
inline RuntimeConfig GetRuntimeConfig();
inline RuntimeConfig RefreshRuntimeConfig(cclScheduler::NeedEvent event, const PipelineTask& task);
inline uint64_t PrefillAllReduceMsgSize(const PipelineTask& task);
inline uint64_t PrefillAllToAllMsgSize();
inline uint64_t KvCacheMsgSize(const PipelineTask& task);
inline void OnCollectiveFinished(uint32_t jobId);
inline uint16_t SubmitAllReduce(uint32_t jobId,
                                uint16_t pg,
                                const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                                uint64_t msgSize,
                                bool releaseOnFinish,
                                bool recordFlowFinish,
                                bool logSubmit);
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
                               bool logSubmit = true);
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
inline bool OnMessageFinish(FILE* fout, Ptr<RdmaQueuePair> q, uint64_t msgSize);

inline void SetFlowFinishLogPath(const std::string &path) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_flow_finish_log_path = path;
}

std::vector<std::pair<uint32_t,uint32_t>> participants;
uint64_t msgSize = 1024 * 1024 * 100; // 1 MB
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
static RuntimeConfig g_runtime_config{
    default_pg,
    1,
    1,
    1,
    1,
    64,
    0,
    1ULL * 1024ULL * 1024ULL,
    512ULL * 1024ULL,
    512ULL * 1024ULL,
    50000};

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

inline uint64_t PrefillAllReduceMsgSize(const PipelineTask& task) {
  auto cfg = GetRuntimeConfig();
  uint32_t scale = std::max(1u, (task.prefillLength + cfg.single_token_length - 1) / cfg.single_token_length);
  return cfg.prefill_unit_msg_size * scale;
}

inline uint64_t PrefillAllToAllMsgSize() {
  return GetRuntimeConfig().prefill_alltoall_msg_size;
}

inline uint64_t KvCacheMsgSize(const PipelineTask& task) {
  auto cfg = GetRuntimeConfig();
  return cfg.token_msg_size * std::max(1u, task.prefillLength / cfg.single_token_length);
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
    auto cfg = g_runtime_config;
    state.decodeIterations = std::max(1u, (task.decodeLength + cfg.single_token_length - 1) / cfg.single_token_length);
    state.prefillPlacementReleased = false;
    state.decodePlacementReleased = false;
    g_pipeline_tasks[task.taskId] = state;
  }

  Simulator::Schedule(Seconds(task.submitTime), [task](){
    auto cfg = RefreshRuntimeConfig(cclScheduler::NeedEvent::TaskDispatch, task);
    {
      std::lock_guard<std::mutex> lk(g_mutex);
      auto it = g_pipeline_tasks.find(task.taskId);
      if (it != g_pipeline_tasks.end()) {
        it->second.startTimeNs = Simulator::Now().GetNanoSeconds();
        it->second.decodeIterations = std::max(1u, (task.decodeLength + cfg.single_token_length - 1) / cfg.single_token_length);
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
         << " single_token_length=" << cfg.single_token_length;
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

inline uint64_t MakeKey(uint16_t pg, uint32_t src, uint32_t dst, uint16_t port) {
  return ((uint64_t)pg << 48) | ((uint64_t)src << 32) | ((uint64_t)dst << 16) | port;
}

inline void SubmitFlow(uint16_t pg,
                       uint32_t src,
                       uint32_t dst,
                       uint64_t bytes,
                       std::vector<uint64_t>& keys) {
  if (src == dst || bytes == 0)
    return;

  uint16_t port = portNumber[src][dst]++;
  RdmaClientHelper clientHelper(
      pg,
      serverAddress[src],
      serverAddress[dst],
      port,
      0,
      bytes,
      has_win ? (global_t == 1 ? maxBdp : pairBdp[n.Get(src)][n.Get(dst)]) : 0,
      global_t == 1 ? maxRtt : pairRtt[src][dst],
      nullptr,
      nullptr,
      1,
      src,
      dst);
  ApplicationContainer apps = clientHelper.Install(n.Get(src));
  apps.Start(Simulator::Now());
  keys.push_back(MakeKey(pg, src, dst, port));
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
  if (gpus.size() < 2)
    return FinishSubmit(jobId, CollectiveOp::AllReduce, gpus, {}, releaseOnFinish, recordFlowFinish, logSubmit);

  std::vector<uint64_t> keys;
  for (size_t i = 0; i < gpus.size(); ++i) {
    uint32_t src = gpus[i].first;
    uint32_t dst = gpus[(i + 1) % gpus.size()].first;
    SubmitFlow(pg, src, dst, msgSize, keys);
  }

  return FinishSubmit(jobId, CollectiveOp::AllReduce, gpus, keys, releaseOnFinish, recordFlowFinish, logSubmit);
}

inline uint16_t SubmitBroadcast(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root) {
  if (gpus.size() < 2)
    return jobId;
  root %= gpus.size();

  uint32_t src = gpus[root].first;
  std::vector<uint64_t> keys;
  for (size_t i = 0; i < gpus.size(); ++i) {
    if (i == root)
      continue;
    SubmitFlow(pg, src, gpus[i].first, msgSize, keys);
  }
  return FinishSubmit(jobId, CollectiveOp::Broadcast, gpus, keys);
}

inline uint16_t SubmitReduce(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root) {
  if (gpus.size() < 2)
    return jobId;
  root %= gpus.size();

  uint32_t dst = gpus[root].first;
  std::vector<uint64_t> keys;
  for (size_t i = 0; i < gpus.size(); ++i) {
    if (i == root)
      continue;
    SubmitFlow(pg, gpus[i].first, dst, msgSize, keys);
  }
  return FinishSubmit(jobId, CollectiveOp::Reduce, gpus, keys);
}

inline uint16_t SubmitGather(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root) {
  if (gpus.size() < 2)
    return jobId;
  root %= gpus.size();

  uint32_t dst = gpus[root].first;
  uint64_t chunkSize = (msgSize + gpus.size() - 1) / gpus.size();
  std::vector<uint64_t> keys;
  for (size_t i = 0; i < gpus.size(); ++i) {
    if (i == root)
      continue;
    SubmitFlow(pg, gpus[i].first, dst, chunkSize, keys);
  }
  return FinishSubmit(jobId, CollectiveOp::Gather, gpus, keys);
}

inline uint16_t SubmitScatter(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root) {
  if (gpus.size() < 2)
    return jobId;
  root %= gpus.size();

  uint32_t src = gpus[root].first;
  uint64_t chunkSize = (msgSize + gpus.size() - 1) / gpus.size();
  std::vector<uint64_t> keys;
  for (size_t i = 0; i < gpus.size(); ++i) {
    if (i == root)
      continue;
    SubmitFlow(pg, src, gpus[i].first, chunkSize, keys);
  }
  return FinishSubmit(jobId, CollectiveOp::Scatter, gpus, keys);
}

inline uint16_t SubmitAllGather(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize) {
  if (gpus.size() < 2)
    return jobId;

  uint64_t chunkSize = (msgSize + gpus.size() - 1) / gpus.size();
  std::vector<uint64_t> keys;
  for (size_t i = 0; i < gpus.size(); ++i) {
    uint32_t src = gpus[i].first;
    uint32_t dst = gpus[(i + 1) % gpus.size()].first;
    SubmitFlow(pg, src, dst, chunkSize, keys);
  }
  return FinishSubmit(jobId, CollectiveOp::AllGather, gpus, keys);
}

inline uint16_t SubmitReduceScatter(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize) {
  if (gpus.size() < 2)
    return jobId;

  uint64_t chunkSize = (msgSize + gpus.size() - 1) / gpus.size();
  std::vector<uint64_t> keys;
  for (size_t i = 0; i < gpus.size(); ++i) {
    uint32_t src = gpus[i].first;
    uint32_t dst = gpus[(i + 1) % gpus.size()].first;
    SubmitFlow(pg, src, dst, chunkSize, keys);
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
                               bool logSubmit) {
  if (gpus.size() < 2 || (k == 0 && expert_num == 0))
    return FinishSubmit(jobId, CollectiveOp::AllToAll, gpus, {}, releaseOnFinish, recordFlowFinish, logSubmit);

  size_t n = gpus.size();
  std::vector<uint64_t> keys;

  // store local ranks 0..n-1 for this job
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_job_ranks[jobId].clear();
    for (uint32_t r = 0; r < n; ++r)
      g_job_ranks[jobId].push_back(r);
  }

  // base probability table: use global table if available, otherwise uniform
  std::vector<double> base;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    if (!g_prob_table.empty())
      base = g_prob_table;
  }
  if (base.size() < n) {
    base.assign(n, 1.0);
  }

  // For expert parallelism: there are `expert_num` expert types.
  // Assign each expert type e to node (e % n). For each expert hosted on
  // a sender node, generate `k` flows whose destinations are sampled from
  // a probability table shifted by the expert id.
  for (size_t s = 0; s < n; ++s) {
    uint32_t src = gpus[s].first;
    // collect expert ids hosted on this sender
    std::vector<uint32_t> experts;
    for (uint32_t e = 0; e < expert_num; ++e) {
      if ((e % n) == s)
        experts.push_back(e);
    }
    if (experts.empty() && k == 0)
      continue;

    for (uint32_t e : experts) {
      // build per-expert probability vector by shifting the base table by expert id
      std::vector<double> probs(n);
      double sum = 0.0;
      for (size_t j = 0; j < n; ++j) {
        probs[j] = base[(j + (e % base.size())) % base.size()];
        sum += probs[j];
      }
      if (sum <= 0.0) {
        for (size_t j = 0; j < n; ++j) probs[j] = 1.0;
        sum = (double)n;
      }
      for (size_t j = 0; j < n; ++j) probs[j] /= sum;

      std::mt19937 rng(static_cast<uint32_t>(jobId ^ (s * 16777619u) ^ (e * 92717u)));
      std::discrete_distribution<int> dist(probs.begin(), probs.end());

      for (uint32_t t = 0; t < k; ++t) {
        int dst_idx = dist(rng);
        int tries = 0;
        while (dst_idx == (int)s && tries < 3) {
          dst_idx = dist(rng);
          ++tries;
        }
        if (dst_idx == (int)s)
          continue;
        uint32_t dst = gpus[dst_idx].first;
        SubmitFlow(pg, src, dst, msgSize, keys);
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
      return SubmitAllToAll(jobId, pg, gpus, msgSize, k, expert_num, expert_mem_bytes, releaseOnFinish, recordFlowFinish, logSubmit);
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
    uint32_t src = prefillGpus[i % prefillGpus.size()].first;
    uint32_t dst = decodeGpus[i % decodeGpus.size()].first;
    SubmitFlow(pg, src, dst, bytesPerFlow, keys);
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
  SubmitAllReduce(jobId, cfg.pg, state.prefillGpus, PrefillAllReduceMsgSize(state.task), false, true, true);
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
                 PrefillAllToAllMsgSize(),
                 cfg.k,
                 cfg.expert_num,
                 cfg.expert_mem_bytes,
                 false,
                 true,
                 true);
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
  uint64_t avgRemaining = cclScheduler::GetAverageRemainingMemory(state.decodeGpus);
  uint64_t avgCapacity = cclScheduler::GetAverageCapacity(state.decodeGpus);
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

inline void StartNextDecodeToken(uint32_t taskId) {
  PipelineTaskState state;
  bool shouldFinish = false;
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_pipeline_tasks.find(taskId);
    if (it == g_pipeline_tasks.end())
      return;
    if (it->second.decodeIteration >= it->second.decodeIterations) {
      shouldFinish = true;
    } else {
      state = it->second;
    }
  }

  if (shouldFinish) {
    FinishPipelineTask(taskId);
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
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_pipeline_tasks.find(state.task.taskId);
    if (it == g_pipeline_tasks.end())
      return;
    it->second.currentDecodeAllReduceJobId = jobId;
    g_pipeline_stage_by_job[jobId] = PipelineStageRef{state.task.taskId, PipelineStage::DecodeAllReduce};
  }
  SubmitAllReduce(jobId, cfg.pg, state.decodeGpus, cfg.token_msg_size, false, false, false);
}

inline void StartDecodeAllToAll(PipelineTaskState& state) {
  uint32_t jobId = JID++;
  auto cfg = GetRuntimeConfig();
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
                 state.decodeGpus,
                 cfg.token_msg_size,
                 cfg.k,
                 cfg.expert_num,
                 cfg.expert_mem_bytes,
                 false,
                 false,
                 false);
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

  uint64_t totalFctNs = nowNs - state.startTimeNs;
  uint64_t decodeE2eNs = nowNs - state.decodeStartNs;
  uint64_t ttftNs = state.firstTokenFinishNs > 0
                    ? state.firstTokenFinishNs - state.startTimeNs
                    : 0;
  uint64_t tpotNs = 0;
  if (state.decodeIterations > 1 && state.firstTokenFinishNs > 0 && nowNs >= state.firstTokenFinishNs) {
    tpotNs = (nowNs - state.firstTokenFinishNs) / (state.decodeIterations - 1);
  }
  auto finishCfg = RefreshRuntimeConfig(cclScheduler::NeedEvent::TaskFinish, state.task);

  std::ostringstream ss;
  ss << "mnCCL: pipeline task " << taskId << " finished\n"
     << "  latency:\n"
     << "    total_fct_ns=" << totalFctNs << "\n"
     << "    TTFT_ns=" << ttftNs << "\n"
     << "    TPOT_ns=" << tpotNs << "\n"
     << "    decode_end_to_end_ns=" << decodeE2eNs << "\n"
     << "  timeline:\n"
     << "    start_ns=" << state.startTimeNs << "\n"
     << "    prefill_finish_ns=" << state.prefillFinishNs << "\n"
     << "    decode_start_ns=" << state.decodeStartNs << "\n"
     << "    first_token_finish_ns=" << state.firstTokenFinishNs << "\n"
     << "    finish_ns=" << nowNs << "\n"
     << "  task:\n"
     << "    prefill_length=" << state.task.prefillLength << "\n"
     << "    decode_length=" << state.task.decodeLength << "\n"
     << "    decode_iterations=" << state.decodeIterations << "\n"
     << "    need_prefill=" << state.actualNeedPrefill << "\n"
     << "    need_decode=" << state.actualNeedDecode << "\n"
     << "    next_need_prefill=" << finishCfg.need_prefill << "\n"
     << "    next_need_decode=" << finishCfg.need_decode << "\n"
     << "    expert_num=" << finishCfg.expert_num << "\n"
     << "  jobs:\n"
     << "    prefill_ar_JID=" << state.prefillAllReduceJobId << "\n"
     << "    prefill_a2a_JID=" << state.prefillAllToAllJobId << "\n"
     << "    kv_JID=" << state.kvCacheJobId << "\n"
     << "    last_decode_ar_JID=" << state.currentDecodeAllReduceJobId << "\n"
     << "    last_decode_a2a_JID=" << state.currentDecodeAllToAllJobId;
  ccl::CclLog(ss.str());
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
      if (!it_task->second.prefillPlacementReleased) {
        cclScheduler::ReleasePlacement(it_task->second.prefillPlacementOwner);
        cclScheduler::ReleaseExpertAlloc(it_task->second.prefillPlacementOwner);
        it_task->second.prefillPlacementReleased = true;
      }
    } else if (ref.stage == PipelineStage::DecodeAllToAll) {
      it_task->second.decodeIteration++;
      if (it_task->second.decodeIteration == 1)
        it_task->second.firstTokenFinishNs = nowNs;
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

  {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it_key = g_key2JID.find(key);
    if (it_key == g_key2JID.end())
      return false; // not found, maybe not an mnCCL message
    jobId = it_key->second;

    auto it = g_outstanding.find(jobId);
    if (it == g_outstanding.end())
      return false;
    if (it->second == 0)
      return false;

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
  return finished;
}

} // namespace mnccl

#endif // MNCCL_H
