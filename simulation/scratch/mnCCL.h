// Minimal in-simulator collective (mnCCL) helpers.
// Header-only to avoid multiple-definition issues in scratch builds.
#ifndef MNCCL_H
#define MNCCL_H

#include <map>
#include <vector>
#include <mutex>
#include <iostream>
#include <cstdint>
#include "common.h"
#include "cclscheduler.h"


namespace mnccl {

// Map pg -> outstanding message count
static std::map<uint16_t, uint32_t> g_outstanding;
// Map pg -> allocated GPUs (vector of <node,gpu_idx>)
static std::map<uint16_t, std::vector<std::pair<uint32_t,uint32_t>>> g_alloc;
// Map encoded key (pg<<48|src<<32|dst<<16|port) -> JID
static std::map<uint64_t, uint16_t> g_key2JID;

// Mutex protecting the above maps
static std::mutex g_mutex;

struct CollectiveJob {
  uint32_t jobId;
  CollectiveOp op;
  uint16_t pg;
  uint32_t need;
  double submitTime;
  uint64_t msgSize;
  uint32_t root;
};

//forward declaration
inline void SubmitJob(uint32_t jobId, int need, double sim_time, uint64_t workload);
inline void SubmitColJob(const CollectiveJob& job);
inline uint16_t SubmitAllReduce(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize);
inline uint16_t SubmitBroadcast(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root = 0);
inline uint16_t SubmitReduce(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root = 0);
inline uint16_t SubmitGather(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root = 0);
inline uint16_t SubmitScatter(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root = 0);
inline uint16_t SubmitAllGather(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize);
inline uint16_t SubmitReduceScatter(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize);
inline uint16_t SubmitCollective(uint32_t jobId, CollectiveOp op, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root = 0);
inline void OnMessageFinish(FILE* fout, Ptr<RdmaQueuePair> q);

std::vector<std::pair<uint32_t,uint32_t>> participants;
uint64_t msgSize = 1024 * 1024 * 100; // 1 MB
uint32_t JID = 1; // 任务 ID，可以根据实际情况生成唯一 ID
uint16_t default_pg = 3; // 流优先级
int need = 128; // 需要的 GPU 数量
double sim_time = 0.0001; // 模拟时间，单位秒

inline void Init() {
  SubmitJob(JID, need, sim_time, msgSize);
  cout<< "mnCCL: initialized \n";
  Simulator::Schedule(Seconds(sim_time*2), [](){
    cclScheduler::ScheduleTask();
  });
}

inline void SubmitJob(uint32_t jobId, int need, double sim_time, uint64_t workload) {
  if (need <= 0)
    return;
    
  SubmitColJob(CollectiveJob{JID, CollectiveOp::AllReduce, default_pg, static_cast<uint32_t>(need), sim_time, workload, 0});
  JID = jobId + 1;
  need -= 16;
  cout<<"gen collective"<<JID<<endl;
  SubmitJob(JID, need, sim_time, workload);
}

inline void SubmitColJob(const CollectiveJob& job) {
  Simulator::Schedule(Seconds(job.submitTime), [job](){
    cclScheduler::EnqueuePendingTask(job.jobId, job.op, job.pg, job.need, job.msgSize, job.root);
    cclScheduler::ScheduleTask();
  });
}

inline const char* OpName(CollectiveOp op) {
  switch (op) {
    case CollectiveOp::AllReduce: return "allreduce";
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
                             const std::vector<uint64_t>& keys) {
  if (keys.empty()) {
    cclScheduler::ReleaseGPUs(gpus);
    return jobId;
  }

  {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_outstanding[jobId] = keys.size();
    g_alloc[jobId] = gpus;
    for (auto key : keys) {
      g_key2JID[key] = jobId;
    }
  }

  std::cout << "mnCCL: submitted " << OpName(op) << " JID=" << jobId
            << " participants=" << gpus.size() << " sends=" << keys.size() << "\n";
  return jobId;
}




// Submit a simple ring all-reduce: each participant sends one message to the
// next participant. `gpus` is vector of <node, gpu_idx>. Returns the pg
// (used to tag the QPs) assigned to this collective.
inline uint16_t SubmitAllReduce(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize) {
  if (gpus.size() < 2)
    return jobId;

  std::vector<uint64_t> keys;
  for (size_t i = 0; i < gpus.size(); ++i) {
    uint32_t src = gpus[i].first;
    uint32_t dst = gpus[(i + 1) % gpus.size()].first;
    SubmitFlow(pg, src, dst, msgSize, keys);
  }

  return FinishSubmit(jobId, CollectiveOp::AllReduce, gpus, keys);
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

inline uint16_t SubmitCollective(uint32_t jobId, CollectiveOp op, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root) {
  switch (op) {
    case CollectiveOp::AllReduce:
      return SubmitAllReduce(jobId, pg, gpus, msgSize);
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
  cclScheduler::ReleaseGPUs(gpus);
  return jobId;
}

// Called from NormalNetwork's message_finish callback to let mnCCL handle
// synchronization and resource release when an allreduce message completes.
inline void OnMessageFinish(FILE* /*fout*/, Ptr<RdmaQueuePair> q) {
    //通过pg,src,dst,port四元组映射到JID
    uint16_t pg = q->m_pg;
    uint32_t src = ip_to_node_id(q->sip);
    uint32_t dst = ip_to_node_id(q->dip);
    uint16_t port = q->sport;
    uint64_t key = MakeKey(pg, src, dst, port);

    uint16_t jobId = 0;
    std::vector<std::pair<uint32_t,uint32_t>> alloc;
    bool finished = false;

    {
      std::lock_guard<std::mutex> lk(g_mutex);
      auto it_key = g_key2JID.find(key);
      if (it_key == g_key2JID.end())
        return; // not found, maybe not an mnCCL message
      jobId = it_key->second;

      auto it = g_outstanding.find(jobId);
      if (it == g_outstanding.end())
        return;
      if (it->second == 0)
        return;
      it->second--;
      if (it->second == 0) {
        alloc = g_alloc[jobId];
        g_alloc.erase(jobId);
        g_outstanding.erase(jobId);
        g_key2JID.erase(key);
        finished = true;
      }
      else {
        g_key2JID.erase(key);
      }
    }

    if (finished) {
      cclScheduler::ReleaseGPUs(alloc);
      std::cout << "mnCCL: JID=" << jobId << " finished at " << Simulator::Now().GetNanoSeconds() << "ns\n";
    }
}

} // namespace mnccl

#endif // MNCCL_H
