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

// Forward-declare mnCCL SubmitAllReduce to avoid circular include with
// `mnCCL.h` (which itself includes this header). Only the symbol is needed
// here, so a forward declaration prevents the 'mnccl' not declared error.
namespace mnccl {
enum class CollectiveOp : uint8_t {
  AllReduce = 0,
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
                                 uint32_t root);
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
};

static std::deque<PendingTask> s_pending_queue;
static std::vector<std::vector<bool>> s_busy; // [node][gpu]

// Forward declarations for functions used before their full definitions.
inline void ScheduleTask();
inline PendingTask SchedulePendingTasks();
using AllocationPolicy =
    std::function<std::vector<std::pair<uint32_t, uint32_t>>(
        uint32_t,
        std::vector<std::vector<bool>>&,
        uint32_t,
        uint32_t)>;
inline std::vector<std::pair<uint32_t, uint32_t>> FillAllocateGPUs(
    uint32_t ngpus,
    std::vector<std::vector<bool>>& busy,
    uint32_t num_nodes,
    uint32_t gpus_per_server);
inline std::vector<std::pair<uint32_t, uint32_t>> AllocateGPUs(uint32_t ngpus);
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
                               uint32_t root = 0) {
  {
    std::lock_guard<std::mutex> lk(s_pending_mutex);
    s_pending_queue.push_back(PendingTask{jobId, op, pg, need, msgSize, root});
  }
  // std::cout << "Scheduler: enqueued pending job " << jobId << " needing " << need
  //           << " GPUs at " << Simulator::Now().GetNanoSeconds() << "ns\n";
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
      auto allocated = AllocateGPUs(task.need);
      if (allocated.size() != task.need) {
        ReleaseGPUs(allocated);
        std::cout << "Scheduler: allocation policy returned " << allocated.size()
                  << " GPUs for job " << task.jobId << ", need " << task.need << "\n";
        return;
      }
      DequeuePendingTask();
      // submit the job (mnCCL handles further synchronization)
      mnccl::SubmitCollective(task.jobId, task.op, task.pg, allocated, task.msgSize, task.root);
      std::cout << "Scheduler: scheduled pending job " << task.jobId
                << " with allocated GPUs: "<< allocated.size() << "\n";
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



inline void Init(uint32_t num_nodes, uint32_t gpus_per_server) {

  std::lock_guard<std::mutex> lk(s_mutex);
  s_num_nodes = num_nodes;
  s_gpus_per_server = gpus_per_server;
  s_busy.assign(num_nodes, std::vector<bool>(gpus_per_server, false));
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
    uint32_t gpus_per_server) {
  std::vector<std::pair<uint32_t, uint32_t>> res;
  for (uint32_t node = 0; node < num_nodes && res.size() < ngpus; node++) {
    for (uint32_t g = 0; g < gpus_per_server && res.size() < ngpus; g++) {
      if (!busy[node][g]) {
        busy[node][g] = true;
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

// Allocate GPUs through the configured allocation policy. Returns vector of <node, gpu_idx>.
inline std::vector<std::pair<uint32_t, uint32_t>> AllocateGPUs(uint32_t ngpus) {
  std::lock_guard<std::mutex> lk(s_mutex);
  return s_allocate_policy(ngpus, s_busy, s_num_nodes, s_gpus_per_server);
}

// Release GPUs (mark them free again).
inline void ReleaseGPUs(const std::vector<std::pair<uint32_t, uint32_t>>& gpus) {

  std::lock_guard<std::mutex> lk(s_mutex);
  for (auto &p : gpus) {
    uint32_t node = p.first;
    uint32_t g = p.second;
    if (node < s_busy.size() && g < s_busy[node].size())
      s_busy[node][g] = false;
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
