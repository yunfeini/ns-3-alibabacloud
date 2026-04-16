// Minimal in-simulator collective (mnCCL) helpers.
// Header-only to avoid multiple-definition issues in scratch builds.
#ifndef MNCCL_H
#define MNCCL_H

#include <map>
#include <vector>
#include <mutex>
#include <iostream>
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

//forward declaration
inline void SubmitJob(uint32_t jobId, int need, double sim_time, uint64_t workload);
inline uint16_t SubmitAllReduce(uint32_t jobId, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize);
inline void OnMessageFinish(FILE* fout, Ptr<RdmaQueuePair> q);

std::vector<std::pair<uint32_t,uint32_t>> participants;
uint64_t msgSize = 1024 * 1024 * 100; // 1 MB
uint64_t JID = 1; // 任务 ID，可以根据实际情况生成唯一 ID
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
  // parse workload to get needed GPU count and msg size, for now we just use
  // the global `need` and `msgSize` for simplicity
  
  Simulator::Schedule(Seconds(sim_time), [jobId, need, workload](){
    cclScheduler::EnqueuePendingTask(jobId, need, workload);
  });
  // std::cout << "mnCCL: SubmitJob jobId=" << JID << " workload=" << workload << " sim_time=" << sim_time << "\n";

  need -= 16;
  sim_time += 0.0001;
  JID += 1;

  if (need <= 0) {
    // std::cout << "mnCCL: SubmitJob jobId=" << JID << " insufficient free GPUs, enqueueing\n";
    return;
  }
  SubmitJob(JID, need, sim_time, msgSize);
}




// Submit a simple ring all-reduce: each participant sends one message to the
// next participant. `gpus` is vector of <node, gpu_idx>. Returns the pg
// (used to tag the QPs) assigned to this collective.
inline uint16_t SubmitAllReduce(uint32_t jobId, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize) {
  if (gpus.size() < 2)
    return jobId;

  uint32_t sends = 0;
  uint16_t pg = 3; 

  // store keys to insert under lock
  std::vector<uint64_t> keys;
  for (size_t i = 0; i < gpus.size(); ++i) {
    uint32_t src = gpus[i].first;
    uint32_t dst = gpus[(i + 1) % gpus.size()].first;
    uint16_t port = portNumber[src][dst]++;

    // Use RdmaClientHelper like other flows. maxPacketCount == msgSize here.
    RdmaClientHelper clientHelper(
        pg,
        serverAddress[src],
        serverAddress[dst],
        port,
        0,
        msgSize,
        has_win ? (global_t == 1 ? maxBdp : pairBdp[n.Get(src)][n.Get(dst)]) : 0,
        global_t == 1 ? maxRtt : pairRtt[src][dst],
        nullptr,
        nullptr,
        1,
        src,
        dst);
    ApplicationContainer apps = clientHelper.Install(n.Get(src));
    apps.Start(Simulator::Now());
    ++sends;

    uint64_t key = ((uint64_t)pg << 48) | ((uint64_t)src << 32) | ((uint64_t)dst << 16) | port;
    keys.push_back(key);
  }

  {
    std::lock_guard<std::mutex> lk(g_mutex);
    g_outstanding[jobId] = sends;
    g_alloc[jobId] = gpus;
    for (auto key : keys) {
      g_key2JID[key] = jobId;
    }
  }

  // std::cout << "mnCCL: submitted allreduce jobId=" << jobId << " participants=" << gpus.size() << " sends=" << sends << "\n";
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
    uint64_t key = ((uint64_t)pg << 48) | ((uint64_t)src << 32) | ((uint64_t)dst << 16) | port;

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
        finished = true;
      }
    }

    if (finished) {
      cclScheduler::ReleaseGPUs(alloc);
      std::cout << "mnCCL: allreduce JID=" << jobId << " finished at " << Simulator::Now().GetNanoSeconds() << "ns\n";
    }
}

} // namespace mnccl

#endif // MNCCL_H
