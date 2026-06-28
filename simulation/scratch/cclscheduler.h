// Lightweight scheduler for allocating/freeing GPUs based on send completions.
// Header-only implementation so it can be included into a single scratch
// translation unit without causing multiple-definition linker errors.
#ifndef CCLSCHEDULER_H
#define CCLSCHEDULER_H

#include <vector>
#include <utility>
#include <deque>
#include <functional>
#include <algorithm>
#include <map>
#include <set>
#include <mutex>
#include <iostream>
#include <sstream>
#include <cstdint>
#include <string>
#include "common.h"
#include "ccl_log.h"

// Forward-declare mnCCL SubmitAllReduce to avoid circular include with
// `mnCCL.h` (which itself includes this header). Only the symbol is needed
// here, so a forward declaration prevents the 'mnccl' not declared error.
namespace mnccl {
enum class CollectiveOp : uint8_t {
  AllReduce = 0,
  AllToAll,
  Broadcast,
  Reduce,
  Gather,
  Scatter,
  AllGather,
  ReduceScatter
};

inline uint16_t SubmitCollective(uint32_t jobId,
                                 CollectiveOp op,
                                 uint16_t pg,
                                 const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                                 uint64_t msgSize,
                                 uint32_t root,
                                 uint32_t k = 1,
                                 uint32_t expert_num = 1,
                                 uint64_t expert_mem_bytes = 0,
                                 bool releaseOnFinish = true,
                                 bool recordFlowFinish = true,
                                 bool logSubmit = true);
}

namespace cclScheduler {


static uint32_t s_num_nodes = 0;
static uint32_t s_gpus_per_server = 0;
static std::mutex s_mutex;
static std::mutex s_schedule_mutex;
// Mutex protecting the pending queue
static std::mutex s_pending_mutex;

enum class PlacementKind : uint8_t {
  Generic = 0,
  Prefill,
  Decode
};

enum class NeedEvent : uint8_t {
  TaskDispatch = 0,
  TaskFinish
};

struct NeedState {
  uint32_t need_prefill;
  uint32_t need_decode;
  uint32_t expert_num;
  uint32_t expert_per_gpu;
};

using NeedUpdateCallback =
  std::function<void(NeedEvent event,
                     uint32_t taskId,
                     uint32_t prefillLength,
                     uint32_t decodeLength,
                     NeedState& state)>;

// Pending tasks queue (FIFO) for jobs that couldn't be scheduled due to
// insufficient free GPUs. External code may push tasks into this queue via
// `EnqueuePendingTask`. `SchedulePendingTasks` will try to schedule them in
// FIFO order.
struct PendingTask {
  uint32_t jobId;
  mnccl::CollectiveOp op;
  uint16_t pg;
  uint32_t need;    // number of GPUs required
  uint64_t msgSize; // optional payload size
  uint32_t root;
  uint32_t k;       // number of flows per node (for alltoall)
  uint32_t expert_num; // total number of experts
  uint64_t expert_mem_bytes; // memory per expert in bytes
  uint32_t expert_per_gpu; // target experts hosted by each selected GPU
  PlacementKind kind;
};

static std::deque<PendingTask> s_pending_queue;
static std::vector<std::vector<bool>> s_busy; // [node][gpu]
static std::vector<std::vector<uint64_t>> s_gpu_mem; // capacity per GPU
static std::vector<std::vector<uint64_t>> s_mem_used; // used per GPU
static std::vector<std::vector<uint32_t>> s_expert_used; // experts hosted per GPU
static std::map<uint32_t, std::map<uint64_t, uint64_t>> s_alloc_mem_by_job;
static std::map<uint32_t, std::map<uint64_t, uint32_t>> s_alloc_experts_by_job;
static std::map<uint32_t, PlacementKind> s_placement_kind_by_owner;
static const uint32_t kMaxExpertsPerGpu = 9;
// jobId -> expert_id -> list of (node,gpu). We store a vector per expert
// because an expert may be assigned to multiple GPUs (or a GPU may host
// multiple experts when expert_num > need).
static std::map<uint32_t, std::vector<std::vector<std::pair<uint32_t,uint32_t>>>> s_expert_alloc;

struct PlacementRequest {
  uint32_t ownerId;
  PlacementKind kind;
  uint32_t need;
  uint32_t expert_num;
  uint64_t expert_mem_bytes;
  uint32_t expert_per_gpu;
};

struct ExpertPlacementRequest {
  uint32_t ownerId;
  PlacementKind kind;
  uint32_t expert_num;
  uint64_t expert_mem_bytes;
  uint32_t expert_per_gpu;
  std::vector<std::pair<uint32_t,uint32_t>> gpus;
};

// ===== SCHEDULING POLICY INTERFACE: PD placement / need placement =====
using NeedPlacementPolicy =
  std::function<std::vector<std::pair<uint32_t, uint32_t>>(
    const PlacementRequest&,
    const std::vector<std::vector<uint64_t>>&,
    const std::vector<std::vector<uint64_t>>&,
    const std::vector<std::vector<uint32_t>>&,
    uint32_t,
    uint32_t,
    uint32_t)>;

// ===== SCHEDULING POLICY INTERFACE: expert placement on selected PD GPUs =====
using ExpertPlacementPolicy =
  std::function<std::vector<std::vector<std::pair<uint32_t,uint32_t>>>(
    const ExpertPlacementRequest&)>;

// Forward declarations for functions used before their full definitions.
inline void ScheduleTask();
inline PendingTask SchedulePendingTasks();
inline std::vector<std::vector<std::pair<uint32_t,uint32_t>>> AssignExperts(uint32_t jobId, uint32_t expert_num, uint64_t expert_mem_bytes, uint32_t expert_per_gpu, const std::vector<std::pair<uint32_t,uint32_t>>& gpus);
inline void ReleaseExpertAlloc(uint32_t jobId);
// Return flattened list of all <node,gpu> assigned to the job (across all experts)
inline std::vector<std::pair<uint32_t,uint32_t>> GetExpertAlloc(uint32_t jobId);
inline std::vector<std::pair<uint32_t,uint32_t>> GetExpertReplicas(uint32_t jobId, uint32_t expertId);
inline const char* PlacementKindName(PlacementKind kind);
inline uint64_t MakeGpuKey(uint32_t node, uint32_t gpu);
inline uint32_t ExpertsPerGpuForNeed(uint32_t expert_num, uint32_t ngpus, uint32_t expert_per_gpu, uint64_t expert_mem_bytes);
inline uint32_t ExpertsOnGpuIndex(uint32_t expert_num, uint32_t ngpus, uint32_t idx, uint64_t expert_mem_bytes, uint32_t expert_per_gpu);
inline std::vector<std::vector<std::pair<uint32_t,uint32_t>>> RoundRobinExpertPlacementPolicy(const ExpertPlacementRequest& req);
inline std::vector<std::pair<uint32_t, uint32_t>> FillNeedPlacementPolicy(
  const PlacementRequest& req,
  const std::vector<std::vector<uint64_t>>& mem_used,
  const std::vector<std::vector<uint64_t>>& gpu_mem,
  const std::vector<std::vector<uint32_t>>& expert_used,
  uint32_t num_nodes,
  uint32_t gpus_per_server,
  uint32_t max_experts_per_gpu);
inline std::vector<std::pair<uint32_t, uint32_t>> TryPlaceNeed(const PlacementRequest& req);
inline std::vector<std::pair<uint32_t, uint32_t>> TryPlaceNeed(uint32_t ownerId,
                                                               PlacementKind kind,
                                                               uint32_t need,
                                                               uint32_t expert_num,
                                                               uint64_t expert_mem_bytes,
                                                               uint32_t expert_per_gpu = 1);
inline void SetNeedUpdateCallback(NeedUpdateCallback cb);
inline void NotifyNeedEvent(NeedEvent event,
                            uint32_t taskId,
                            uint32_t prefillLength,
                            uint32_t decodeLength,
                            NeedState& state);
inline void ReleasePlacement(uint32_t ownerId);
inline uint64_t GetMinRemainingMemory(const std::vector<std::pair<uint32_t,uint32_t>>& gpus);
inline uint64_t GetAverageRemainingMemory(const std::vector<std::pair<uint32_t,uint32_t>>& gpus);
inline uint64_t GetAverageCapacity(const std::vector<std::pair<uint32_t,uint32_t>>& gpus);
inline void GetResourceSnapshot(std::vector<std::vector<uint64_t>>& mem_used,
                                std::vector<std::vector<uint64_t>>& gpu_mem,
                                std::vector<std::vector<uint32_t>>& expert_used,
                                uint32_t& num_nodes,
                                uint32_t& gpus_per_server,
                                uint32_t& max_experts_per_gpu);
inline void SetNeedPlacementPolicy(NeedPlacementPolicy policy);
inline void SetExpertPlacementPolicy(ExpertPlacementPolicy policy);
using AllocationPolicy =
  std::function<std::vector<std::pair<uint32_t, uint32_t>>(
    uint32_t,
    std::vector<std::vector<bool>>&,
    uint32_t,
    uint32_t,
    uint64_t)>; // last arg: needed_mem_per_gpu
inline std::vector<std::pair<uint32_t, uint32_t>> FillAllocateGPUs(
  uint32_t ngpus,
  std::vector<std::vector<bool>>& busy,
  uint32_t num_nodes,
  uint32_t gpus_per_server,
  uint64_t needed_mem_per_gpu);
inline std::vector<std::pair<uint32_t, uint32_t>> AllocateGPUs(uint32_t ngpus, uint32_t expert_num, uint64_t expert_mem_bytes);
inline void SetAllocationPolicy(AllocationPolicy policy);
inline void ReleaseGPUs(const std::vector<std::pair<uint32_t, uint32_t>>& gpus);
inline int HasFreeGPUs(uint32_t need);
inline void DequeuePendingTask();

static AllocationPolicy s_allocate_policy = FillAllocateGPUs;
static NeedPlacementPolicy s_need_placement_policy = FillNeedPlacementPolicy;
static ExpertPlacementPolicy s_expert_placement_policy = RoundRobinExpertPlacementPolicy;
static NeedUpdateCallback s_need_update_callback;

// Enqueue a pending task (external code should call this when allocation
// fails and it wants the scheduler to retry later).
inline void EnqueuePendingTask(uint32_t jobId,
                               mnccl::CollectiveOp op,
                               uint16_t pg,
                               uint32_t need,
                               uint64_t msgSize,
                               uint32_t root = 0,
                               uint32_t k = 1,
                               uint32_t expert_num = 1,
                               uint64_t expert_mem_bytes = 0,
                               uint32_t expert_per_gpu = 1,
                               PlacementKind kind = PlacementKind::Generic) {
  {
    std::lock_guard<std::mutex> lk(s_pending_mutex);
    s_pending_queue.push_back(PendingTask{
        jobId, op, pg, need, msgSize, root, k, expert_num, expert_mem_bytes, expert_per_gpu, kind});
  }
  {
    std::ostringstream ss;
    ss << "Scheduler: enqueued pending job " << jobId << " needing " << need
       << " GPUs (expert_num=" << expert_num
       << ", expert_per_gpu=" << expert_per_gpu
       << ", expert_mem_bytes=" << expert_mem_bytes << ")";
    ccl::CclLog(ss.str());
  }
}

inline void DequeuePendingTask() {
  std::lock_guard<std::mutex> lk(s_pending_mutex);
  if (s_pending_queue.empty())
    return;
  s_pending_queue.pop_front();

}

inline void ScheduleTask(){
  std::lock_guard<std::mutex> schedule_lk(s_schedule_mutex);
  auto task = SchedulePendingTasks();
  // std::cout << "Scheduler: ScheduleTask called at " << Simulator::Now().GetNanoSeconds() << "ns\n";
  if (task.jobId != 0) {
    auto allocated = TryPlaceNeed(task.jobId,
                                  task.kind,
                                  task.need,
                                  task.expert_num,
                                  task.expert_mem_bytes,
                                  task.expert_per_gpu);
    if (allocated.size() != task.need) {
      if (!allocated.empty())
        ReleasePlacement(task.jobId);
      {
        std::ostringstream ss;
        ss << "Scheduler: insufficient placement for pending job " << task.jobId
           << " allocated=" << allocated.size() << " need=" << task.need;
        ccl::CclLog(ss.str());
      }
      return;
    }
    DequeuePendingTask();
    auto expert_map = AssignExperts(
        task.jobId, task.expert_num, task.expert_mem_bytes, task.expert_per_gpu, allocated);
    (void)expert_map;
    mnccl::SubmitCollective(task.jobId, task.op, task.pg, allocated, task.msgSize, task.root, task.k, task.expert_num, task.expert_mem_bytes);
    {
      std::ostringstream ss;
      ss << "Scheduler: scheduled pending job " << task.jobId
         << " with allocated GPUs: "<< allocated.size();
      ccl::CclLog(ss.str());
    }
  }
  else {
    // no pending task
    // cout<< "Scheduler: no pending tasks to schedule at" << Simulator::Now().GetNanoSeconds() << "ns\n";
    return;
  }
}


// Try to schedule pending tasks in FIFO order. For each task at queue head,
// attempt to allocate `need` GPUs; if successful, pop the task and invoke the
// registered submit callback with the allocated participants. If the head
// task cannot be allocated, stop (strict FIFO).
inline PendingTask SchedulePendingTasks() {
  PendingTask task;
  {
    std::lock_guard<std::mutex> lk(s_pending_mutex);
    if (s_pending_queue.empty())
      return PendingTask{0, mnccl::CollectiveOp{}, 0, 0, 0, 0, 1, 1, 0, 1, PlacementKind::Generic};
    task = s_pending_queue.front();
  }
  return task;
}



inline void Init(uint32_t num_nodes, uint32_t gpus_per_server, uint64_t default_mem_per_gpu = 32ULL*1024*1024*1024ULL) {

  std::lock_guard<std::mutex> lk(s_mutex);
  s_num_nodes = num_nodes;
  s_gpus_per_server = gpus_per_server;
  s_busy.assign(num_nodes, std::vector<bool>(gpus_per_server, false));
  s_gpu_mem.assign(num_nodes, std::vector<uint64_t>(gpus_per_server, default_mem_per_gpu));
  s_mem_used.assign(num_nodes, std::vector<uint64_t>(gpus_per_server, 0));
  s_expert_used.assign(num_nodes, std::vector<uint32_t>(gpus_per_server, 0));
  s_alloc_mem_by_job.clear();
  s_alloc_experts_by_job.clear();
  s_placement_kind_by_owner.clear();
  s_expert_alloc.clear();
}

inline std::vector<std::vector<std::pair<uint32_t,uint32_t>>> RoundRobinExpertPlacementPolicy(const ExpertPlacementRequest& req) {
  std::vector<std::vector<std::pair<uint32_t,uint32_t>>> res;
  if (req.gpus.empty() || req.expert_num == 0)
    return res;
  res.assign(req.expert_num, {});
  uint32_t expertsPerGpu = ExpertsPerGpuForNeed(
      req.expert_num, static_cast<uint32_t>(req.gpus.size()), req.expert_per_gpu, 1);
  if (expertsPerGpu == 0)
    return res;
  uint32_t expert_id = 0;
  for (const auto& gpu : req.gpus) {
    for (uint32_t slot = 0; slot < expertsPerGpu; ++slot) {
      res[expert_id % req.expert_num].push_back(gpu);
      ++expert_id;
    }
  }
  return res;
}

// Assign experts (0..expert_num-1) to the provided GPUs through the configured
// expert placement policy. Stores mapping in `s_expert_alloc` under `jobId`.
inline std::vector<std::vector<std::pair<uint32_t,uint32_t>>> AssignExperts(uint32_t jobId,
                                                                            uint32_t expert_num,
                                                                            uint64_t expert_mem_bytes,
                                                                            uint32_t expert_per_gpu,
                                                                            const std::vector<std::pair<uint32_t,uint32_t>>& gpus) {
  std::vector<std::vector<std::pair<uint32_t,uint32_t>>> res;
  if (gpus.empty() || expert_num == 0)
    return res;
  PlacementKind kind = PlacementKind::Generic;
  ExpertPlacementPolicy policy;
  {
    std::lock_guard<std::mutex> lk(s_mutex);
    auto kind_it = s_placement_kind_by_owner.find(jobId);
    if (kind_it != s_placement_kind_by_owner.end())
      kind = kind_it->second;
    policy = s_expert_placement_policy ? s_expert_placement_policy : RoundRobinExpertPlacementPolicy;
  }

  res = policy(ExpertPlacementRequest{jobId, kind, expert_num, expert_mem_bytes, expert_per_gpu, gpus});
  if (res.size() != expert_num)
    return {};
  for (const auto& replicas : res) {
    if (replicas.empty())
      return {};
  }

  {
    std::lock_guard<std::mutex> lk(s_mutex);
    std::map<uint64_t, uint64_t> mem_delta;
    std::map<uint64_t, uint32_t> expert_delta;
    std::map<uint64_t, std::set<uint32_t>> experts_by_gpu;
    for (uint32_t expert_id = 0; expert_id < res.size(); ++expert_id) {
      const auto& replicas = res[expert_id];
      for (const auto& gpu : replicas) {
        if (gpu.first >= s_num_nodes || gpu.second >= s_gpus_per_server)
          return {};
        uint64_t key = MakeGpuKey(gpu.first, gpu.second);
        if (!experts_by_gpu[key].insert(expert_id).second)
          return {};
        expert_delta[key] += 1;
        mem_delta[key] += expert_mem_bytes;
      }
    }
    for (const auto& kv : expert_delta) {
      uint32_t node = static_cast<uint32_t>(kv.first >> 32);
      uint32_t gpu = static_cast<uint32_t>(kv.first & 0xffffffffULL);
      if (kv.second > kMaxExpertsPerGpu)
        return {};
      if (mem_delta[kv.first] > s_gpu_mem[node][gpu])
        return {};
    }
    s_expert_alloc[jobId] = res;
  }

  {
    std::ostringstream ss;
    ss << "Scheduler: expert placement placement_owner=" << jobId
       << " kind=" << PlacementKindName(kind)
       << " expert_num=" << expert_num
       << " expert_mem_bytes=" << expert_mem_bytes
       << " expert_per_gpu=" << expert_per_gpu
       << " experts=[";
    for (size_t e = 0; e < res.size(); ++e) {
      if (e) ss << ";";
      ss << e << "->{";
      for (size_t r = 0; r < res[e].size(); ++r) {
        if (r) ss << ",";
        ss << res[e][r].first << ":" << res[e][r].second;
      }
      ss << "}";
    }
    ss << "]";
    ccl::CclLog(ss.str());
  }
  {
    std::map<uint64_t, std::vector<uint32_t>> experts_by_gpu;
    for (uint32_t e = 0; e < res.size(); ++e) {
      for (const auto& gpu : res[e]) {
        experts_by_gpu[MakeGpuKey(gpu.first, gpu.second)].push_back(e);
      }
    }
    for (auto& kv : experts_by_gpu) {
      std::sort(kv.second.begin(), kv.second.end());
      uint32_t node = static_cast<uint32_t>(kv.first >> 32);
      uint32_t gpu = static_cast<uint32_t>(kv.first & 0xffffffffULL);
      std::ostringstream ss;
      ss << "Scheduler: expert placement gpu placement_owner=" << jobId
         << " kind=" << PlacementKindName(kind)
         << " gpu=" << node << ":" << gpu
         << " experts=[";
      for (size_t i = 0; i < kv.second.size(); ++i) {
        if (i) ss << ",";
        ss << kv.second[i];
      }
      ss << "]";
      ccl::CclLog(ss.str());
    }
  }
  return res;
}

inline void ReleaseExpertAlloc(uint32_t jobId) {
  std::lock_guard<std::mutex> lk(s_mutex);
  s_expert_alloc.erase(jobId);
}

inline std::vector<std::pair<uint32_t,uint32_t>> GetExpertAlloc(uint32_t jobId) {
  std::lock_guard<std::mutex> lk(s_mutex);
  auto it = s_expert_alloc.find(jobId);
  if (it == s_expert_alloc.end())
    return {};
  // flatten per-expert lists into a single vector
  std::vector<std::pair<uint32_t,uint32_t>> flat;
  for (const auto &vec : it->second) {
    for (const auto &p : vec)
      flat.push_back(p);
  }
  return flat;
}

inline std::vector<std::pair<uint32_t,uint32_t>> GetExpertReplicas(uint32_t jobId, uint32_t expertId) {
  std::lock_guard<std::mutex> lk(s_mutex);
  auto it = s_expert_alloc.find(jobId);
  if (it == s_expert_alloc.end() || expertId >= it->second.size())
    return {};
  return it->second[expertId];
}

inline const char* PlacementKindName(PlacementKind kind) {
  switch (kind) {
    case PlacementKind::Generic: return "generic";
    case PlacementKind::Prefill: return "prefill";
    case PlacementKind::Decode: return "decode";
  }
  return "unknown";
}

inline uint64_t MakeGpuKey(uint32_t node, uint32_t gpu) {
  return (static_cast<uint64_t>(node) << 32) | gpu;
}

inline uint32_t ExpertsPerGpuForNeed(uint32_t expert_num,
                                     uint32_t ngpus,
                                     uint32_t expert_per_gpu,
                                     uint64_t expert_mem_bytes) {
  if (ngpus == 0 || expert_mem_bytes == 0 || expert_num == 0)
    return 0;
  expert_per_gpu = std::max(1u, expert_per_gpu);
  if (ngpus < expert_num) {
    uint32_t minExpertsPerGpu = (expert_num + ngpus - 1) / ngpus;
    return std::max(expert_per_gpu, minExpertsPerGpu);
  }
  return expert_per_gpu;
}

inline uint32_t ExpertsOnGpuIndex(uint32_t expert_num,
                                  uint32_t ngpus,
                                  uint32_t /*idx*/,
                                  uint64_t expert_mem_bytes,
                                  uint32_t expert_per_gpu) {
  return ExpertsPerGpuForNeed(expert_num, ngpus, expert_per_gpu, expert_mem_bytes);
}

inline std::vector<std::pair<uint32_t, uint32_t>> FillNeedPlacementPolicy(
    const PlacementRequest& req,
    const std::vector<std::vector<uint64_t>>& mem_used,
    const std::vector<std::vector<uint64_t>>& gpu_mem,
    const std::vector<std::vector<uint32_t>>& expert_used,
    uint32_t num_nodes,
    uint32_t gpus_per_server,
    uint32_t max_experts_per_gpu) {
  std::vector<std::pair<uint32_t, uint32_t>> res;
  if (req.need == 0)
    return res;

  std::map<uint64_t, uint64_t> mem_delta;
  std::map<uint64_t, uint32_t> expert_delta;
  while (res.size() < req.need) {
    bool placed = false;
    for (uint32_t node = 0; node < num_nodes && !placed; node++) {
      for (uint32_t g = 0; g < gpus_per_server && !placed; g++) {
        uint64_t linear = static_cast<uint64_t>(node) * gpus_per_server + g;
        uint64_t total = static_cast<uint64_t>(num_nodes) * gpus_per_server;
        uint64_t split = (total * 3) / 4;
        if (split == 0)
          split = 1;
        if (split >= total && total > 1)
          split = total - 1;
        if (req.kind == PlacementKind::Prefill && linear >= split)
          continue;
        if (req.kind == PlacementKind::Decode && linear < split)
          continue;
        uint32_t idx = static_cast<uint32_t>(res.size());
        uint32_t experts = ExpertsOnGpuIndex(
            req.expert_num, req.need, idx, req.expert_mem_bytes, req.expert_per_gpu);
        uint64_t mem_need = static_cast<uint64_t>(experts) * req.expert_mem_bytes;
        if (node >= mem_used.size() || g >= mem_used[node].size())
          continue;
        uint64_t key = MakeGpuKey(node, g);
        if (expert_used[node][g] + expert_delta[key] + experts > max_experts_per_gpu)
          continue;
        if (mem_used[node][g] + mem_delta[key] + mem_need > gpu_mem[node][g])
          continue;
        res.emplace_back(node, g);
        expert_delta[key] += experts;
        mem_delta[key] += mem_need;
        placed = true;
      }
    }
    if (!placed)
      break;
  }
  return res;
}

inline void SetNeedPlacementPolicy(NeedPlacementPolicy policy) {
  std::lock_guard<std::mutex> lk(s_mutex);
  s_need_placement_policy = policy ? policy : FillNeedPlacementPolicy;
}

inline void SetExpertPlacementPolicy(ExpertPlacementPolicy policy) {
  std::lock_guard<std::mutex> lk(s_mutex);
  s_expert_placement_policy = policy ? policy : RoundRobinExpertPlacementPolicy;
}

inline void SetNeedUpdateCallback(NeedUpdateCallback cb) {
  std::lock_guard<std::mutex> lk(s_mutex);
  s_need_update_callback = cb;
}

inline void NotifyNeedEvent(NeedEvent event,
                            uint32_t taskId,
                            uint32_t prefillLength,
                            uint32_t decodeLength,
                            NeedState& state) {
  NeedUpdateCallback cb;
  {
    std::lock_guard<std::mutex> lk(s_mutex);
    cb = s_need_update_callback;
  }
  if (cb)
    cb(event, taskId, prefillLength, decodeLength, state);
}

inline std::vector<std::pair<uint32_t, uint32_t>> TryPlaceNeed(const PlacementRequest& req) {
  std::lock_guard<std::mutex> lk(s_mutex);
  std::vector<std::pair<uint32_t, uint32_t>> allocated;
  if (req.need == 0)
    return allocated;

  auto policy = s_need_placement_policy ? s_need_placement_policy : FillNeedPlacementPolicy;
  std::vector<std::vector<uint64_t>> empty_mem_used(
      s_num_nodes, std::vector<uint64_t>(s_gpus_per_server, 0));
  std::vector<std::vector<uint32_t>> empty_expert_used(
      s_num_nodes, std::vector<uint32_t>(s_gpus_per_server, 0));
  allocated = policy(req,
                     empty_mem_used,
                     s_gpu_mem,
                     empty_expert_used,
                     s_num_nodes,
                     s_gpus_per_server,
                     kMaxExpertsPerGpu);
  if (allocated.size() != req.need)
    return {};
  s_placement_kind_by_owner[req.ownerId] = req.kind;

  {
    std::ostringstream ss;
    ss << "Scheduler: placed placement_owner=" << req.ownerId
       << " kind=" << PlacementKindName(req.kind)
       << " need=" << req.need
       << " expert_num=" << req.expert_num
       << " expert_per_gpu=" << req.expert_per_gpu
       << " selected_gpus=" << allocated.size()
       << " detail=awaiting expert_placement_policy";
    ccl::CclLog(ss.str());
  }
  return allocated;
}

inline std::vector<std::pair<uint32_t, uint32_t>> TryPlaceNeed(uint32_t ownerId,
                                                               PlacementKind kind,
                                                               uint32_t need,
                                                               uint32_t expert_num,
                                                               uint64_t expert_mem_bytes,
                                                               uint32_t expert_per_gpu) {
  return TryPlaceNeed(PlacementRequest{
      ownerId, kind, need, expert_num, expert_mem_bytes, expert_per_gpu});
}

inline void ReleasePlacement(uint32_t ownerId) {
  std::lock_guard<std::mutex> lk(s_mutex);
  auto mem_it = s_alloc_mem_by_job.find(ownerId);
  auto exp_it = s_alloc_experts_by_job.find(ownerId);
  if (mem_it == s_alloc_mem_by_job.end() && exp_it == s_alloc_experts_by_job.end()) {
    s_placement_kind_by_owner.erase(ownerId);
    return;
  }

  if (mem_it != s_alloc_mem_by_job.end()) {
    for (auto &kv : mem_it->second) {
      uint32_t node = static_cast<uint32_t>(kv.first >> 32);
      uint32_t gpu = static_cast<uint32_t>(kv.first & 0xffffffffULL);
      if (node >= s_num_nodes || gpu >= s_gpus_per_server)
        continue;
      s_mem_used[node][gpu] = (s_mem_used[node][gpu] > kv.second) ? (s_mem_used[node][gpu] - kv.second) : 0;
      s_busy[node][gpu] = s_mem_used[node][gpu] > 0;
    }
    s_alloc_mem_by_job.erase(mem_it);
  }

  if (exp_it != s_alloc_experts_by_job.end()) {
    for (auto &kv : exp_it->second) {
      uint32_t node = static_cast<uint32_t>(kv.first >> 32);
      uint32_t gpu = static_cast<uint32_t>(kv.first & 0xffffffffULL);
      if (node >= s_num_nodes || gpu >= s_gpus_per_server)
        continue;
      s_expert_used[node][gpu] = (s_expert_used[node][gpu] > kv.second) ? (s_expert_used[node][gpu] - kv.second) : 0;
    }
    s_alloc_experts_by_job.erase(exp_it);
  }
  s_placement_kind_by_owner.erase(ownerId);

  {
    std::ostringstream ss;
    ss << "Scheduler: released placement placement_owner=" << ownerId;
    ccl::CclLog(ss.str());
  }
}

inline uint64_t GetMinRemainingMemory(const std::vector<std::pair<uint32_t,uint32_t>>& gpus) {
  std::lock_guard<std::mutex> lk(s_mutex);
  uint64_t min_remaining = UINT64_MAX;
  for (auto &p : gpus) {
    if (p.first >= s_num_nodes || p.second >= s_gpus_per_server)
      continue;
    uint64_t remaining = (s_gpu_mem[p.first][p.second] > s_mem_used[p.first][p.second])
                         ? (s_gpu_mem[p.first][p.second] - s_mem_used[p.first][p.second])
                         : 0;
    min_remaining = std::min(min_remaining, remaining);
  }
  return min_remaining == UINT64_MAX ? 0 : min_remaining;
}

inline uint64_t GetAverageRemainingMemory(const std::vector<std::pair<uint32_t,uint32_t>>& gpus) {
  std::lock_guard<std::mutex> lk(s_mutex);
  if (gpus.empty())
    return 0;
  uint64_t total = 0;
  uint32_t count = 0;
  for (auto &p : gpus) {
    if (p.first >= s_num_nodes || p.second >= s_gpus_per_server)
      continue;
    total += (s_gpu_mem[p.first][p.second] > s_mem_used[p.first][p.second])
             ? (s_gpu_mem[p.first][p.second] - s_mem_used[p.first][p.second])
             : 0;
    ++count;
  }
  return count == 0 ? 0 : total / count;
}

inline uint64_t GetAverageCapacity(const std::vector<std::pair<uint32_t,uint32_t>>& gpus) {
  std::lock_guard<std::mutex> lk(s_mutex);
  if (gpus.empty())
    return 0;
  uint64_t total = 0;
  uint32_t count = 0;
  for (auto &p : gpus) {
    if (p.first >= s_num_nodes || p.second >= s_gpus_per_server)
      continue;
    total += s_gpu_mem[p.first][p.second];
    ++count;
  }
  return count == 0 ? 0 : total / count;
}

inline void GetResourceSnapshot(std::vector<std::vector<uint64_t>>& mem_used,
                                std::vector<std::vector<uint64_t>>& gpu_mem,
                                std::vector<std::vector<uint32_t>>& expert_used,
                                uint32_t& num_nodes,
                                uint32_t& gpus_per_server,
                                uint32_t& max_experts_per_gpu) {
  std::lock_guard<std::mutex> lk(s_mutex);
  mem_used = s_mem_used;
  gpu_mem = s_gpu_mem;
  expert_used = s_expert_used;
  num_nodes = s_num_nodes;
  gpus_per_server = s_gpus_per_server;
  max_experts_per_gpu = kMaxExpertsPerGpu;
}

inline void GetActiveExpertSlotSnapshot(std::vector<std::vector<uint32_t>>& expert_slots,
                                        uint32_t& num_nodes,
                                        uint32_t& gpus_per_server,
                                        uint32_t& max_experts_per_gpu) {
  std::lock_guard<std::mutex> lk(s_mutex);
  num_nodes = s_num_nodes;
  gpus_per_server = s_gpus_per_server;
  max_experts_per_gpu = kMaxExpertsPerGpu;
  expert_slots.assign(s_num_nodes, std::vector<uint32_t>(s_gpus_per_server, 0));
  for (const auto& owner : s_expert_alloc) {
    for (const auto& replicas : owner.second) {
      for (const auto& gpu : replicas) {
        if (gpu.first < s_num_nodes && gpu.second < s_gpus_per_server)
          expert_slots[gpu.first][gpu.second] += 1;
      }
    }
  }
}

//check if there are enough free GPUs for a new job
inline int HasFreeGPUs(uint32_t need) {
  std::lock_guard<std::mutex> lk(s_mutex);
  uint32_t free_count = 0;
  for (uint32_t node = 0; node < s_busy.size(); node++) {
    for (uint32_t g = 0; g < s_busy[node].size(); g++) {
      if (s_mem_used[node][g] < s_gpu_mem[node][g] && s_expert_used[node][g] < kMaxExpertsPerGpu) {
        free_count += 1;
        if (free_count >= need)
          return 0;
      }
    }
  }
  return need - free_count;
}

inline std::vector<std::pair<uint32_t, uint32_t>> FillAllocateGPUs(
    uint32_t ngpus,
    std::vector<std::vector<bool>>& busy,
    uint32_t num_nodes,
    uint32_t gpus_per_server,
    uint64_t needed_mem_per_gpu) {
  std::vector<std::pair<uint32_t, uint32_t>> res;
  for (uint32_t node = 0; node < num_nodes && res.size() < ngpus; node++) {
    for (uint32_t g = 0; g < gpus_per_server && res.size() < ngpus; g++) {
      if (!busy[node][g] && (s_mem_used[node][g] + needed_mem_per_gpu <= s_gpu_mem[node][g])) {
        busy[node][g] = true;
        s_mem_used[node][g] += needed_mem_per_gpu;
        res.emplace_back(node, g);
      }
    }
  }
  return res;
}

inline void SetAllocationPolicy(AllocationPolicy policy) {
  std::lock_guard<std::mutex> lk(s_mutex);
  s_allocate_policy = policy ? policy : FillAllocateGPUs;
}

// Example custom allocation policies -------------------------------------------------
// Node-balanced: try to spread allocations evenly across nodes (one GPU per
// node in round-robin) to maximize node-level parallelism.
inline std::vector<std::pair<uint32_t, uint32_t>> NodeBalancedPolicy(
  uint32_t ngpus,
  std::vector<std::vector<bool>>& busy,
  uint32_t num_nodes,
  uint32_t gpus_per_server,
  uint64_t needed_mem_per_gpu) {
  std::vector<std::pair<uint32_t, uint32_t>> res;
  if (ngpus == 0) return res;
  // First pass: try to take at most one GPU from each node in round-robin
  // order to spread load.
  for (uint32_t pass = 0; res.size() < ngpus && pass < gpus_per_server; ++pass) {
    for (uint32_t node = 0; node < num_nodes && res.size() < ngpus; ++node) {
      uint32_t g = pass; // take the pass-th GPU index on this node
      if (g < gpus_per_server && !busy[node][g] && (s_mem_used[node][g] + needed_mem_per_gpu <= s_gpu_mem[node][g])) {
        busy[node][g] = true;
        s_mem_used[node][g] += needed_mem_per_gpu;
        res.emplace_back(node, g);
      }
      // if that slot was busy, try to find any free GPU on this node
      if (res.size() < ngpus && g >= gpus_per_server) {
        for (uint32_t gg = 0; gg < gpus_per_server && res.size() < ngpus; ++gg) {
          if (!busy[node][gg] && (s_mem_used[node][gg] + needed_mem_per_gpu <= s_gpu_mem[node][gg])) {
            busy[node][gg] = true;
            s_mem_used[node][gg] += needed_mem_per_gpu;
            res.emplace_back(node, gg);
          }
        }
      }
    }
  }
  return res;
}

// Memory-priority: pick GPUs with the most available memory first.
inline std::vector<std::pair<uint32_t, uint32_t>> MemoryPriorityPolicy(
  uint32_t ngpus,
  std::vector<std::vector<bool>>& busy,
  uint32_t num_nodes,
  uint32_t gpus_per_server,
  uint64_t needed_mem_per_gpu) {
  std::vector<std::pair<uint32_t, uint32_t>> res;
  if (ngpus == 0) return res;
  struct Candidate { uint32_t node; uint32_t gpu; uint64_t free; };
  std::vector<Candidate> cand;
  for (uint32_t n = 0; n < num_nodes; ++n) {
    for (uint32_t g = 0; g < gpus_per_server; ++g) {
      if (!busy[n][g]) {
        uint64_t free_mem = (s_gpu_mem[n][g] > s_mem_used[n][g]) ? (s_gpu_mem[n][g] - s_mem_used[n][g]) : 0;
        if (free_mem >= needed_mem_per_gpu)
          cand.push_back(Candidate{n, g, free_mem});
      }
    }
  }
  // sort descending by free memory
  std::sort(cand.begin(), cand.end(), [](const Candidate&a, const Candidate&b){ return a.free > b.free; });
  for (size_t i = 0; i < cand.size() && res.size() < ngpus; ++i) {
    busy[cand[i].node][cand[i].gpu] = true;
    s_mem_used[cand[i].node][cand[i].gpu] += needed_mem_per_gpu;
    res.emplace_back(cand[i].node, cand[i].gpu);
  }
  return res;
}

// Helper functions to register example policies
inline void UseNodeBalancedPolicy() {
  SetAllocationPolicy(NodeBalancedPolicy);
  ccl::CclLog("Scheduler: Set allocation policy = NodeBalanced");
}

inline void UseMemoryPriorityPolicy() {
  SetAllocationPolicy(MemoryPriorityPolicy);
  ccl::CclLog("Scheduler: Set allocation policy = MemoryPriority");
}

// Example registration (call one of these early in program startup):
//   cclScheduler::UseNodeBalancedPolicy();
// or
//   cclScheduler::UseMemoryPriorityPolicy();

// Allocate GPUs through the configured allocation policy. Returns vector of <node, gpu_idx>.
inline std::vector<std::pair<uint32_t, uint32_t>> AllocateGPUs(uint32_t ngpus, uint32_t expert_num, uint64_t expert_mem_bytes) {
  std::lock_guard<std::mutex> lk(s_mutex);
  if (ngpus == 0)
    return {};
  // estimate experts assigned per GPU (ceiling)
  uint32_t experts_per_gpu = (expert_num + ngpus - 1) / ngpus;
  uint64_t needed_mem_per_gpu = experts_per_gpu * expert_mem_bytes;
  {
    std::ostringstream ss;
    ss << "Scheduler: AllocateGPUs request ngpus=" << ngpus
       << " expert_num=" << expert_num
       << " experts_per_gpu=" << experts_per_gpu
       << " needed_mem_per_gpu=" << needed_mem_per_gpu;
    ccl::CclLog(ss.str());
  }
  auto allocated = s_allocate_policy(ngpus, s_busy, s_num_nodes, s_gpus_per_server, needed_mem_per_gpu);
  {
    std::ostringstream ss;
    ss << "Scheduler: AllocateGPUs returned " << allocated.size() << " GPUs";
    if (!allocated.empty()) {
      ss << " [";
      for (size_t i = 0; i < allocated.size(); ++i) {
        if (i) ss << ";";
        ss << allocated[i].first << ":" << allocated[i].second;
      }
      ss << "]";
    }
    ccl::CclLog(ss.str());
  }
  return allocated;
}

// Release GPUs (mark them free again).
inline void ReleaseGPUs(const std::vector<std::pair<uint32_t, uint32_t>>& gpus) {

  std::lock_guard<std::mutex> lk(s_mutex);
  for (auto &p : gpus) {
    uint32_t node = p.first;
    uint32_t g = p.second;
    if (node < s_busy.size() && g < s_busy[node].size()) {
      s_busy[node][g] = false;
      s_mem_used[node][g] = 0;
      {
        std::ostringstream ss;
        ss << "Scheduler: ReleaseGPUs freed node=" << node << " gpu=" << g;
        ccl::CclLog(ss.str());
      }
    }
  }
}

// Called from the RDMA send completion callback. We try to free one busy
// GPU on the sending host (if any). This gives the simulation a simple
// way to learn that a node became free.
inline void OnSendFinish(FILE* /*fout*/, Ptr<RdmaQueuePair> q) {
  uint32_t sid = ip_to_node_id(q->sip);
  bool freed = false;
  {
    std::lock_guard<std::mutex> lk(s_mutex);
    if (sid >= s_busy.size())
      return;
    for (uint32_t g = 0; g < s_gpus_per_server; ++g) {
      if (s_busy[sid][g]) {
        s_busy[sid][g] = false;
        s_mem_used[sid][g] = 0;
        // debug
        // std::cout << "Scheduler: freed GPU " << g << " on node " << sid
        //           << " at " << Simulator::Now().GetNanoSeconds() << "ns\n";
        freed = true;
        break;
      }
    }
  }
}

} // namespace cclScheduler

#endif // CCLSCHEDULER_H
