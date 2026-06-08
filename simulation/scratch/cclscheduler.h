// Lightweight scheduler for allocating/freeing GPUs based on send completions.
// Header-only implementation so it can be included into a single scratch
// translation unit without causing multiple-definition linker errors.
#ifndef CCLSCHEDULER_H
#define CCLSCHEDULER_H

#include <vector>
#include <utility>
#include <deque>
#include <functional>
#include <mutex>
#include <iostream>
#include <cstdint>
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
inline uint16_t SubmitAllReduce(uint32_t jobId,
                                uint16_t pg,
                                const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                                uint64_t msgSize);
inline uint16_t SubmitCollective(uint32_t jobId,
                                 CollectiveOp op,
                                 uint16_t pg,
                                 const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
                                 uint64_t msgSize,
                                 uint32_t root,
                                 uint32_t k = 1,
                                 uint32_t ep = 1,
                                 uint64_t expert_mem_bytes = 0);
}

namespace cclScheduler {


static uint32_t s_num_nodes = 0;
static uint32_t s_gpus_per_server = 0;
static std::mutex s_mutex;
// Mutex protecting the pending queue
static std::mutex s_pending_mutex;

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
  uint32_t ep;      // number of experts (expert parallelism)
  uint64_t expert_mem_bytes; // memory per expert in bytes
};

static std::deque<PendingTask> s_pending_queue;
static std::vector<std::vector<bool>> s_busy; // [node][gpu]
static std::vector<std::vector<uint64_t>> s_gpu_mem; // capacity per GPU
static std::vector<std::vector<uint64_t>> s_mem_used; // used per GPU
// jobId -> expert_id -> list of (node,gpu). We store a vector per expert
// because an expert may be assigned to multiple GPUs (or a GPU may host
// multiple experts when ep > need).
static std::map<uint32_t, std::vector<std::vector<std::pair<uint32_t,uint32_t>>>> s_expert_alloc;

// Forward declarations for functions used before their full definitions.
inline void ScheduleTask();
inline PendingTask SchedulePendingTasks();
inline std::vector<std::vector<std::pair<uint32_t,uint32_t>>> AssignExperts(uint32_t jobId, uint32_t ep, const std::vector<std::pair<uint32_t,uint32_t>>& gpus);
inline void ReleaseExpertAlloc(uint32_t jobId);
// Return flattened list of all <node,gpu> assigned to the job (across all experts)
inline std::vector<std::pair<uint32_t,uint32_t>> GetExpertAlloc(uint32_t jobId);
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
inline std::vector<std::pair<uint32_t, uint32_t>> AllocateGPUs(uint32_t ngpus, uint32_t ep, uint64_t expert_mem_bytes);
inline void SetAllocationPolicy(AllocationPolicy policy);
inline void ReleaseGPUs(const std::vector<std::pair<uint32_t, uint32_t>>& gpus);
inline int HasFreeGPUs(uint32_t need);
inline void DequeuePendingTask();

static AllocationPolicy s_allocate_policy = FillAllocateGPUs;

// Enqueue a pending task (external code should call this when allocation
// fails and it wants the scheduler to retry later).
inline void EnqueuePendingTask(uint32_t jobId,
                               mnccl::CollectiveOp op,
                               uint16_t pg,
                               uint32_t need,
                               uint64_t msgSize,
                               uint32_t root = 0,
                               uint32_t k = 1,
                               uint32_t ep = 1,
                               uint64_t expert_mem_bytes = 0) {
  {
    std::lock_guard<std::mutex> lk(s_pending_mutex);
    s_pending_queue.push_back(PendingTask{jobId, op, pg, need, msgSize, root, k, ep, expert_mem_bytes});
  }
  {
    std::ostringstream ss;
    ss << "Scheduler: enqueued pending job " << jobId << " needing " << need
       << " GPUs (ep=" << ep << ", expert_mem_bytes=" << expert_mem_bytes << ")";
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
  auto task = SchedulePendingTasks();
  // std::cout << "Scheduler: ScheduleTask called at " << Simulator::Now().GetNanoSeconds() << "ns\n";
  if (task.jobId != 0) {
    // std::cout << "Scheduler: trying to schedule pending job " << task.jobId
    //           << " needing " << task.need << " GPUs\n";
    // try to allocate for the pending task
    int free_gpus = HasFreeGPUs(task.need);
    // cout << "Scheduler: job " << task.jobId << " needs " << task.need
    //           << " GPUs, free GPUs = " << free_gpus << "\n";
      if (free_gpus==0) {
      // remove from pending queue under lock
      auto allocated = AllocateGPUs(task.need, task.ep, task.expert_mem_bytes);
      if (allocated.size() != task.need) {
        ReleaseGPUs(allocated);
        {
          std::ostringstream ss;
          ss << "Scheduler: allocation policy returned " << allocated.size()
             << " GPUs for job " << task.jobId << ", need " << task.need;
          ccl::CclLog(ss.str());
        }
        return;
      }
      DequeuePendingTask();
      // assign experts to the allocated GPUs and store mapping
      auto expert_map = AssignExperts(task.jobId, task.ep, allocated);
      {
        std::ostringstream ss;
        ss << "Scheduler: assigned " << expert_map.size() << " experts for job " << task.jobId;
        ccl::CclLog(ss.str());
      }
      // submit the job (mnCCL handles further synchronization)
      mnccl::SubmitCollective(task.jobId, task.op, task.pg, allocated, task.msgSize, task.root, task.k, task.ep, task.expert_mem_bytes);
      {
        std::ostringstream ss;
        ss << "Scheduler: scheduled pending job " << task.jobId
           << " with allocated GPUs: "<< allocated.size();
        ccl::CclLog(ss.str());
      }
    }
    else {
      // std::cout << "Scheduler: failed to schedule pending job " << task.jobId
      //           << " due to insufficient free  "<< free_gpus << "  GPUs\n";
      return;
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
      return PendingTask{0, mnccl::CollectiveOp{}, 0, 0, 0, 0};
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
}

// Assign experts (0..ep-1) to the provided GPUs. Default policy: round-robin
// across the provided `gpus` vector. Stores mapping in `s_expert_alloc` under
// `jobId` and returns the mapping vector of size `ep`.
inline std::vector<std::vector<std::pair<uint32_t,uint32_t>>> AssignExperts(uint32_t jobId, uint32_t ep, const std::vector<std::pair<uint32_t,uint32_t>>& gpus) {
  std::vector<std::vector<std::pair<uint32_t,uint32_t>>> res;
  if (gpus.empty() || ep == 0)
    return res;
  {
    std::lock_guard<std::mutex> lk(s_mutex);
    // prepare per-expert vector
    res.assign(ep, {});
    uint32_t need = static_cast<uint32_t>(gpus.size());
    if (need >= ep) {
      // distribute GPUs to experts round-robin so an expert may have multiple GPUs
      for (uint32_t i = 0; i < need; ++i) {
        uint32_t expert_id = i % ep;
        res[expert_id].push_back(gpus[i]);
      }
    } else {
      // need < ep: ensure each GPU gets at least one expert, and assign remaining
      // experts to GPUs in round-robin so every expert is placed at least once.
      // First assign one expert per GPU
      uint32_t e = 0;
      for (; e < need && e < ep; ++e) {
        res[e].push_back(gpus[e]);
      }
      // Assign remaining experts to GPUs in round-robin (GPU index j)
      for (; e < ep; ++e) {
        uint32_t target_gpu = e % need;
        res[e].push_back(gpus[target_gpu]);
      }
    }
    s_expert_alloc[jobId] = res;
    // Log detailed mapping: expert -> list of node:gpu
    {
      std::ostringstream ss;
      ss << "Scheduler: job " << jobId << " expert->GPU mapping [";
      for (uint32_t ex = 0; ex < res.size(); ++ex) {
        if (ex) ss << ";";
        ss << ex << "->";
        for (size_t j = 0; j < res[ex].size(); ++j) {
          if (j) ss << ",";
          ss << res[ex][j].first << ":" << res[ex][j].second;
        }
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

//check if there are enough free GPUs for a new job
inline int HasFreeGPUs(uint32_t need) {
  std::lock_guard<std::mutex> lk(s_mutex);
  uint32_t free_count = 0;
  for (uint32_t node = 0; node < s_busy.size(); node++) {
    for (uint32_t g = 0; g < s_busy[node].size(); g++) {
      if (!s_busy[node][g]) {
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
inline std::vector<std::pair<uint32_t, uint32_t>> AllocateGPUs(uint32_t ngpus, uint32_t ep, uint64_t expert_mem_bytes) {
  std::lock_guard<std::mutex> lk(s_mutex);
  if (ngpus == 0)
    return {};
  // estimate experts assigned per GPU (ceiling)
  uint32_t ep_per_gpu = (ep + ngpus - 1) / ngpus;
  uint64_t needed_mem_per_gpu = ep_per_gpu * expert_mem_bytes;
  {
    std::ostringstream ss;
    ss << "Scheduler: AllocateGPUs request ngpus=" << ngpus << " ep=" << ep << " ep_per_gpu=" << ep_per_gpu << " needed_mem_per_gpu=" << needed_mem_per_gpu;
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
