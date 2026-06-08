// Minimal in-simulator collective (mnCCL) helpers.
// Header-only to avoid multiple-definition issues in scratch builds.
#ifndef MNCCL_H
#define MNCCL_H

#include <map>
#include <vector>
#include <mutex>
#include <random>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include "ccl_log.h"
#include "common.h"
#include "cclscheduler.h"


namespace mnccl {

// Map pg -> outstanding message count
static std::map<uint16_t, uint32_t> g_outstanding;
// Map pg -> allocated GPUs (vector of <node,gpu_idx>)
static std::map<uint16_t, std::vector<std::pair<uint32_t,uint32_t>>> g_alloc;
// Map encoded key (pg<<48|src<<32|dst<<16|port) -> JID
static std::map<uint64_t, uint16_t> g_key2JID;

// Map JID -> associated RdmaQueuePairs (one-to-many). Stored here for
// logging/inspection. Protected by g_mutex.
static std::map<uint32_t, std::vector<Ptr<RdmaQueuePair>>> g_JID2QPs;

// Mutex protecting the above maps
static std::mutex g_mutex;
// Path to write JID->QP mappings. Set by NormalNetwork via SetJidQpLogPath().
static std::string g_jid_qp_log_path;

struct CollectiveJob {
  uint32_t jobId;
  CollectiveOp op;
  uint16_t pg;
  uint32_t need;
  double submitTime;
  uint64_t msgSize;
  uint32_t root;
  uint32_t k;
  uint32_t ep;
  uint64_t expert_mem_bytes;
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
inline uint16_t SubmitAllToAll(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t k = 1, uint32_t ep = 1, uint64_t expert_mem_bytes = 0);
inline uint16_t SubmitCollective(uint32_t jobId, CollectiveOp op, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root, uint32_t k, uint32_t ep, uint64_t expert_mem_bytes);
inline void OnMessageFinish(FILE* fout, Ptr<RdmaQueuePair> q);

// Setter for log path (call from NormalNetwork to specify output file)
inline void SetJidQpLogPath(const std::string &path) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_jid_qp_log_path = path;
}

std::vector<std::pair<uint32_t,uint32_t>> participants;
uint64_t msgSize = 1024 * 1024 * 100; // 1 MB
uint32_t JID = 1; // 任务 ID，可以根据实际情况生成唯一 ID
uint16_t default_pg = 3; // 流优先级
int need = 128; // 需要的 GPU 数量
double sim_time = 0.0001; // 模拟时间，单位秒

// Map jobId -> per-job local ranks (0..n-1)
static std::map<uint32_t, std::vector<uint32_t>> g_job_ranks;

// Global, unique probability table used by all alltoall jobs. If empty,
// a uniform distribution will be used for the group size when needed.
static std::vector<double> g_prob_table;

inline void SetGlobalProbTable(const std::vector<double>& table) {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_prob_table = table;
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
    cclScheduler::EnqueuePendingTask(job.jobId, job.op, job.pg, job.need, job.msgSize, job.root, job.k, job.ep, job.expert_mem_bytes);
    cclScheduler::ScheduleTask();
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

  {
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

// Submit an all-to-all style job: each participant generates `k` flows
// whose destinations are sampled from a global probability table that is
// shifted/biased by the sender's local rank. The local ranks for the
// job are stored in `g_job_ranks[jobId]`.
inline uint16_t SubmitAllToAll(uint32_t jobId, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t k, uint32_t ep, uint64_t expert_mem_bytes) {
  if (gpus.size() < 2 || (k == 0 && ep == 0))
    return jobId;

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

  // For expert parallelism: there are `ep` expert types (0..ep-1).
  // Assign each expert type e to node (e % n). For each expert hosted on
  // a sender node, generate `k` flows whose destinations are sampled from
  // a probability table shifted by the expert id.
  for (size_t s = 0; s < n; ++s) {
    uint32_t src = gpus[s].first;
    // collect expert ids hosted on this sender
    std::vector<uint32_t> experts;
    for (uint32_t e = 0; e < ep; ++e) {
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

  return FinishSubmit(jobId, CollectiveOp::AllToAll, gpus, keys);
}

inline uint16_t SubmitCollective(uint32_t jobId, CollectiveOp op, uint16_t pg, const std::vector<std::pair<uint32_t,uint32_t>>& gpus, uint64_t msgSize, uint32_t root, uint32_t k, uint32_t ep, uint64_t expert_mem_bytes) {
  switch (op) {
    case CollectiveOp::AllReduce:
      return SubmitAllReduce(jobId, pg, gpus, msgSize);
    case CollectiveOp::AllToAll:
      return SubmitAllToAll(jobId, pg, gpus, msgSize, k, ep, expert_mem_bytes);
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

      // record this QP under the jobId for logging/inspection (avoid dupes)
      auto &vec = g_JID2QPs[jobId];
      bool found = false;
      for (auto &existing_q : vec) {
        if (existing_q == q) { found = true; break; }
      }
      if (!found) {
        vec.push_back(q);
      }

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
      cclScheduler::ReleaseExpertAlloc(jobId);
      // print and clear any recorded QPs for this JID
      {
        std::lock_guard<std::mutex> lk(g_mutex);
        auto itq = g_JID2QPs.find(jobId);
        if (itq != g_JID2QPs.end()) {
          if (!g_jid_qp_log_path.empty()) {
            std::ofstream ofs(g_jid_qp_log_path, std::ofstream::app);
              if (ofs) {
                // If file is new, write a header line
                bool need_header = false;
                // check if file was just created by trying to open for read
                std::ifstream ifs(g_jid_qp_log_path);
                if (!ifs.good())
                  need_header = true;
                ifs.close();
                if (need_header) {
                  ofs << "timestamp_ns,jobId,srcNode,dstNode,pg,sport,dport,qp_start_time_step\n";
                }
                // write each QP as CSV: timestamp_ns,jobId,node_src,node_dst,pg,sport,dport,qp_start_time_step
                for (auto &qp : itq->second) {
                  uint64_t ts = ns3::Simulator::Now().GetNanoSeconds();
                  uint32_t sid = ip_to_node_id(qp->sip);
                  uint32_t did = ip_to_node_id(qp->dip);
                  ofs << ts << "," << jobId << "," << sid << "," << did << "," << qp->m_pg << "," << qp->sport << "," << qp->dport << "," << qp->startTime.GetTimeStep() << "\n";
                }
                ofs.close();
              }
          }
          g_JID2QPs.erase(itq);
        }
      }
      {
        std::ostringstream ss;
        ss << "mnCCL: JID=" << jobId << " finished";
        ccl::CclLog(ss.str());
      }
    }
}

} // namespace mnccl

#endif // MNCCL_H
