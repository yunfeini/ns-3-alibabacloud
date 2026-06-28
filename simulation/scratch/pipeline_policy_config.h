#ifndef PIPELINE_POLICY_CONFIG_H
#define PIPELINE_POLICY_CONFIG_H

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "task_generator.h"

namespace pipelinePolicy {

struct PolicyConfig {
  bool enable_policy_group = true;
  bool print_submissions = true;
  double simulation_stop_time = 1000.0;
  uint16_t pg = mnccl::default_pg;
  uint32_t need_prefill = 128;
  uint32_t need_decode = 32;
  uint32_t expert_num = 64;
  uint32_t kflows = 2;
  uint32_t single_token_length = 2;
  uint64_t expert_mem_bytes = 64ULL * 1024ULL * 1024ULL;
  uint64_t token_msg_size = 2;
  uint64_t base_decode_compute_delay_ns = 1;
  uint32_t num_tasks = 3;
  uint32_t task_seed = 12345;
  uint32_t prefill_length_min = 8192;
  uint32_t prefill_length_max = 16384;
  uint32_t decode_length_min = 256;
  uint32_t decode_length_max = 2048;
  double first_submit_time = 0.00001;
  double submit_interval = 0.00001;
  uint16_t probe_high_pg = 0;
  uint16_t probe_low_pg = 7;
  uint64_t probe_bytes = 1;
  double prefill_rate_tokens_per_ns = 0.35;
  double decode_rate_tokens_per_ns = 0.03;
  uint32_t min_need = 1;
  uint32_t max_need = 0;
  uint32_t expert_per_gpu = 1;
  uint64_t cluster_monitor_interval_ns = 1000000;
  std::string need_placement_policy = "expert_capacity_l1";
  std::string expert_placement_policy = "l1_minmax_frequency";
  std::string same_rank_route_policy = "probe_rtt_delta";
  std::string local_flow_schedule_policy = "prefill_first_decode_sjf";
  std::string need_update_policy = "expert_variance";
  std::string pd_split_policy = "balance_pd_rate";
  std::vector<uint32_t> candidate_needs;
  std::vector<double> expert_access_freq;
};

struct DecisionState {
  PolicyConfig config;
  cclScheduler::NeedState current_need{1, 1, 1, 1};
  uint32_t selected_prefill_need = 1;
  uint32_t selected_decode_need = 1;
  uint32_t latest_split_iterations = 0;
  std::vector<std::vector<std::pair<uint32_t,uint32_t>>> latest_expert_plan;
  double latest_plan_max_l1_load = 0.0;
  double latest_plan_l1_variance = 0.0;
};

inline DecisionState& State() {
  static DecisionState state;
  return state;
}

inline std::string Trim(const std::string& s) {
  size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos)
    return "";
  size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

inline bool ParseBool(const std::string& value, bool fallback) {
  if (value == "1" || value == "true" || value == "TRUE" || value == "on")
    return true;
  if (value == "0" || value == "false" || value == "FALSE" || value == "off")
    return false;
  return fallback;
}

inline std::string Lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

inline bool PolicyHookEnabled(const std::string& value) {
  std::string v = Lower(Trim(value));
  return !(v.empty() || v == "0" || v == "off" || v == "none" || v == "default");
}

template <typename T>
inline std::vector<T> ParseVector(const std::string& value) {
  std::vector<T> out;
  std::istringstream iss(value);
  T v{};
  while (iss >> v)
    out.push_back(v);
  return out;
}

inline PolicyConfig LoadPolicyConfig(const std::string& path) {
  PolicyConfig cfg;
  std::ifstream in(path);
  if (!in) {
    ccl::CclLog("PipelinePolicy: config file not found, using built-in policy defaults");
    return cfg;
  }

  std::string line;
  while (std::getline(in, line)) {
    line = Trim(line);
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream iss(line);
    std::string key;
    iss >> key;
    std::string value;
    std::getline(iss, value);
    value = Trim(value);
    if (key == "ENABLE_POLICY_GROUP")
      cfg.enable_policy_group = ParseBool(value, cfg.enable_policy_group);
    else if (key == "PRINT_SUBMISSIONS")
      cfg.print_submissions = ParseBool(value, cfg.print_submissions);
    else if (key == "SIMULATION_STOP_TIME")
      cfg.simulation_stop_time = std::stod(value);
    else if (key == "PG")
      cfg.pg = static_cast<uint16_t>(std::stoul(value));
    else if (key == "NEED_PREFILL")
      cfg.need_prefill = static_cast<uint32_t>(std::stoul(value));
    else if (key == "NEED_DECODE")
      cfg.need_decode = static_cast<uint32_t>(std::stoul(value));
    else if (key == "EXPERT_NUM")
      cfg.expert_num = static_cast<uint32_t>(std::stoul(value));
    else if (key == "KFLOWS")
      cfg.kflows = static_cast<uint32_t>(std::stoul(value));
    else if (key == "SINGLE_TOKEN_LENGTH")
      cfg.single_token_length = static_cast<uint32_t>(std::stoul(value));
    else if (key == "EXPERT_MEM_BYTES")
      cfg.expert_mem_bytes = std::stoull(value);
    else if (key == "TOKEN_MSG_SIZE")
      cfg.token_msg_size = std::stoull(value);
    else if (key == "BASE_DECODE_COMPUTE_DELAY_NS")
      cfg.base_decode_compute_delay_ns = std::stoull(value);
    else if (key == "NUM_TASKS")
      cfg.num_tasks = static_cast<uint32_t>(std::stoul(value));
    else if (key == "TASK_SEED")
      cfg.task_seed = static_cast<uint32_t>(std::stoul(value));
    else if (key == "PREFILL_LENGTH_MIN")
      cfg.prefill_length_min = static_cast<uint32_t>(std::stoul(value));
    else if (key == "PREFILL_LENGTH_MAX")
      cfg.prefill_length_max = static_cast<uint32_t>(std::stoul(value));
    else if (key == "DECODE_LENGTH_MIN")
      cfg.decode_length_min = static_cast<uint32_t>(std::stoul(value));
    else if (key == "DECODE_LENGTH_MAX")
      cfg.decode_length_max = static_cast<uint32_t>(std::stoul(value));
    else if (key == "FIRST_SUBMIT_TIME")
      cfg.first_submit_time = std::stod(value);
    else if (key == "SUBMIT_INTERVAL")
      cfg.submit_interval = std::stod(value);
    else if (key == "PROBE_HIGH_PG")
      cfg.probe_high_pg = static_cast<uint16_t>(std::stoul(value));
    else if (key == "PROBE_LOW_PG")
      cfg.probe_low_pg = static_cast<uint16_t>(std::stoul(value));
    else if (key == "PROBE_BYTES")
      cfg.probe_bytes = std::stoull(value);
    else if (key == "PREFILL_RATE_TOKENS_PER_NS")
      cfg.prefill_rate_tokens_per_ns = std::stod(value);
    else if (key == "DECODE_RATE_TOKENS_PER_NS")
      cfg.decode_rate_tokens_per_ns = std::stod(value);
    else if (key == "MIN_NEED")
      cfg.min_need = static_cast<uint32_t>(std::stoul(value));
    else if (key == "MAX_NEED")
      cfg.max_need = static_cast<uint32_t>(std::stoul(value));
    else if (key == "EXPERT_PER_GPU")
      cfg.expert_per_gpu = static_cast<uint32_t>(std::stoul(value));
    else if (key == "CLUSTER_MONITOR_INTERVAL_NS")
      cfg.cluster_monitor_interval_ns = std::stoull(value);
    else if (key == "NEED_PLACEMENT_POLICY")
      cfg.need_placement_policy = value;
    else if (key == "EXPERT_PLACEMENT_POLICY")
      cfg.expert_placement_policy = value;
    else if (key == "SAME_RANK_ROUTE_POLICY")
      cfg.same_rank_route_policy = value;
    else if (key == "LOCAL_FLOW_SCHEDULE_POLICY")
      cfg.local_flow_schedule_policy = value;
    else if (key == "GLOBAL_NEED_UPDATE_POLICY")
      cfg.need_update_policy = value;
    else if (key == "PD_SPLIT_POLICY")
      cfg.pd_split_policy = value;
    else if (key == "CANDIDATE_NEEDS")
      cfg.candidate_needs = ParseVector<uint32_t>(value);
    else if (key == "EXPERT_ACCESS_FREQ")
      cfg.expert_access_freq = ParseVector<double>(value);
  }
  return cfg;
}

inline uint32_t NetworkNodeId(const std::pair<uint32_t,uint32_t>& gpu) {
  return gpu.first * std::max(1u, gpus_per_server) + gpu.second;
}

inline uint32_t L1GroupOfGpu(const std::pair<uint32_t,uint32_t>& gpu) {
  uint32_t node = NetworkNodeId(gpu);
  if (node < n.GetN()) {
    Ptr<Node> gpuNode = n.Get(node);
    for (const auto& kv : nbr2if[gpuNode]) {
      if (kv.first->GetNodeType() == 1)
        return kv.first->GetId();
    }
  }
  return gpu.first;
}

inline double ExpertFreq(uint32_t expertId) {
  const auto& freq = State().config.expert_access_freq;
  if (expertId < freq.size())
    return freq[expertId];
  return 1.0;
}

inline uint32_t ExpertsOnGpu(uint32_t expertNum,
                             uint32_t ngpus,
                             uint32_t /*idx*/,
                             uint32_t expertPerGpu) {
  return cclScheduler::ExpertsPerGpuForNeed(
      expertNum, ngpus, expertPerGpu, 1);
}

inline uint32_t ExpertsOnGpuForMem(uint32_t expertNum,
                                   uint32_t ngpus,
                                   uint32_t idx,
                                   uint64_t expertMemBytes,
                                   uint32_t expertPerGpu) {
  return cclScheduler::ExpertsOnGpuIndex(
      expertNum, ngpus, idx, expertMemBytes, expertPerGpu);
}

inline double Variance(const std::vector<double>& values) {
  if (values.empty())
    return 0.0;
  double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  double sum = 0.0;
  for (double v : values) {
    double d = v - mean;
    sum += d * d;
  }
  return sum / values.size();
}

inline std::vector<double> SortedL1Loads(const std::map<uint32_t, double>& l1Load) {
  std::vector<double> loads;
  for (const auto& kv : l1Load)
    loads.push_back(kv.second);
  std::sort(loads.begin(), loads.end());
  return loads;
}

inline double MaxL1Load(const std::map<uint32_t, double>& l1Load) {
  double maxLoad = 0.0;
  for (const auto& kv : l1Load)
    maxLoad = std::max(maxLoad, kv.second);
  return maxLoad;
}

inline double L1LoadVariance(const std::map<uint32_t, double>& l1Load) {
  return Variance(SortedL1Loads(l1Load));
}

inline double TopologyDistanceScore(
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus) {
  if (gpus.size() < 2)
    return 0.0;

  uint64_t sumRtt = 0;
  uint64_t pairs = 0;
  for (size_t i = 0; i < gpus.size(); ++i) {
    uint32_t a = NetworkNodeId(gpus[i]);
    for (size_t j = i + 1; j < gpus.size(); ++j) {
      uint32_t b = NetworkNodeId(gpus[j]);
      sumRtt += pairRtt[a][b];
      ++pairs;
    }
  }
  return pairs == 0 ? 0.0 : static_cast<double>(sumRtt) / static_cast<double>(pairs);
}

struct ExpertPlanScore {
  double maxL1Load = std::numeric_limits<double>::max();
  double l1Variance = std::numeric_limits<double>::max();
  double topologyCost = std::numeric_limits<double>::max();
};

struct NeedCandidateScore {
  bool feasible = false;
  uint32_t need = 0;
  ExpertPlanScore plan;
};

inline bool BetterPlanScore(const ExpertPlanScore& a, const ExpertPlanScore& b) {
  if (a.maxL1Load != b.maxL1Load)
    return a.maxL1Load < b.maxL1Load;
  if (a.l1Variance != b.l1Variance)
    return a.l1Variance < b.l1Variance;
  return a.topologyCost < b.topologyCost;
}

inline bool BetterNeedScore(const NeedCandidateScore& a, const NeedCandidateScore& b) {
  if (a.feasible != b.feasible)
    return a.feasible;
  if (!a.feasible)
    return false;
  if (a.plan.l1Variance != b.plan.l1Variance)
    return a.plan.l1Variance < b.plan.l1Variance;
  if (a.plan.maxL1Load != b.plan.maxL1Load)
    return a.plan.maxL1Load < b.plan.maxL1Load;
  if (a.plan.topologyCost != b.plan.topologyCost)
    return a.plan.topologyCost < b.plan.topologyCost;
  return a.need < b.need;
}

inline bool ExpertPlanReady(
    const std::vector<std::vector<std::pair<uint32_t,uint32_t>>>& plan,
    uint32_t expertNum) {
  if (plan.size() != expertNum)
    return false;
  for (const auto& replicas : plan) {
    if (replicas.empty())
      return false;
    std::set<std::pair<uint32_t,uint32_t>> uniqueReplicas;
    for (const auto& gpu : replicas) {
      if (!uniqueReplicas.insert(gpu).second)
        return false;
    }
  }
  return true;
}

inline std::vector<std::vector<std::pair<uint32_t,uint32_t>>> BuildExpertPlan(
    uint32_t expertNum,
    uint32_t expertPerGpu,
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
    ExpertPlanScore* scoreOut = nullptr) {
  std::vector<std::vector<std::pair<uint32_t,uint32_t>>> plan(expertNum);
  if (gpus.empty() || expertNum == 0)
    return plan;

  std::map<uint32_t, double> l1Load;
  std::map<std::pair<uint32_t,uint32_t>, uint32_t> expertsOnGpu;
  std::map<std::pair<uint32_t,uint32_t>, uint32_t> gpuCapacity;
  uint32_t expertsPerGpu = ExpertsOnGpu(
      expertNum, static_cast<uint32_t>(gpus.size()), 0, expertPerGpu);
  if (expertsPerGpu == 0)
    return plan;
  expertsPerGpu = std::min(expertsPerGpu, expertNum);
  uint32_t totalSlots = expertsPerGpu * static_cast<uint32_t>(gpus.size());
  std::map<std::pair<uint32_t,uint32_t>, std::set<uint32_t>> expertsAssignedOnGpu;
  for (const auto& gpu : gpus) {
    gpuCapacity[gpu] += expertsPerGpu;
    l1Load[L1GroupOfGpu(gpu)] += 0.0;
  }

  std::vector<uint32_t> expertsByFreq(expertNum);
  std::iota(expertsByFreq.begin(), expertsByFreq.end(), 0);
  std::sort(expertsByFreq.begin(), expertsByFreq.end(), [](uint32_t a, uint32_t b) {
    return ExpertFreq(a) > ExpertFreq(b);
  });

  std::vector<uint32_t> slots;
  slots.reserve(totalSlots);
  for (uint32_t i = 0; i < totalSlots; ++i)
    slots.push_back(expertsByFreq[i % expertsByFreq.size()]);

  for (uint32_t e : slots) {
    bool found = false;
    std::pair<uint32_t,uint32_t> bestGpu = gpus.front();
    ExpertPlanScore bestScore;
    for (const auto& gpu : gpus) {
      if (expertsOnGpu[gpu] >= gpuCapacity[gpu])
        continue;
      if (expertsAssignedOnGpu[gpu].count(e) != 0)
        continue;
      std::map<uint32_t, double> trialLoad = l1Load;
      trialLoad[L1GroupOfGpu(gpu)] += ExpertFreq(e);
      ExpertPlanScore trial{
          MaxL1Load(trialLoad),
          L1LoadVariance(trialLoad),
          static_cast<double>(expertsOnGpu[gpu])};
      if (!found || BetterPlanScore(trial, bestScore)) {
        found = true;
        bestScore = trial;
        bestGpu = gpu;
      }
    }
    if (!found)
      continue;
    plan[e].push_back(bestGpu);
    expertsAssignedOnGpu[bestGpu].insert(e);
    expertsOnGpu[bestGpu]++;
    l1Load[L1GroupOfGpu(bestGpu)] += ExpertFreq(e);
  }

  if (scoreOut != nullptr) {
    *scoreOut = ExpertPlanScore{
        MaxL1Load(l1Load),
        L1LoadVariance(l1Load),
        TopologyDistanceScore(gpus)};
  }
  return plan;
}

inline std::vector<std::vector<std::pair<uint32_t,uint32_t>>> BuildDomainBalancedExternalMinPlan(
    uint32_t expertNum,
    uint32_t expertPerGpu,
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
    ExpertPlanScore* scoreOut = nullptr) {
  std::vector<std::vector<std::pair<uint32_t,uint32_t>>> plan(expertNum);
  if (gpus.empty() || expertNum == 0)
    return plan;

  uint32_t expertsPerGpu = ExpertsOnGpu(
      expertNum, static_cast<uint32_t>(gpus.size()), 0, expertPerGpu);
  if (expertsPerGpu == 0)
    return plan;
  expertsPerGpu = std::min(expertsPerGpu, expertNum);
  uint32_t totalSlots = expertsPerGpu * static_cast<uint32_t>(gpus.size());
  if (totalSlots < expertNum)
    return plan;

  std::map<std::pair<uint32_t,uint32_t>, uint32_t> expertsOnGpu;
  std::map<std::pair<uint32_t,uint32_t>, std::set<uint32_t>> expertsAssignedOnGpu;
  std::map<uint32_t, std::vector<std::pair<uint32_t,uint32_t>>> gpusByL1;
  std::map<uint32_t, double> l1Load;
  std::map<uint32_t, std::set<uint32_t>> expertsByL1;
  for (const auto& gpu : gpus) {
    uint32_t l1 = L1GroupOfGpu(gpu);
    gpusByL1[l1].push_back(gpu);
    l1Load[l1] += 0.0;
  }

  std::vector<uint32_t> expertsByFreq(expertNum);
  std::iota(expertsByFreq.begin(), expertsByFreq.end(), 0);
  std::sort(expertsByFreq.begin(), expertsByFreq.end(), [](uint32_t a, uint32_t b) {
    if (ExpertFreq(a) != ExpertFreq(b))
      return ExpertFreq(a) > ExpertFreq(b);
    return a < b;
  });

  auto bestGpuInL1 = [&](uint32_t l1, uint32_t expert) {
    bool found = false;
    std::pair<uint32_t,uint32_t> bestGpu = gpus.front();
    uint32_t bestCount = std::numeric_limits<uint32_t>::max();
    for (const auto& gpu : gpusByL1[l1]) {
      if (expertsOnGpu[gpu] >= expertsPerGpu)
        continue;
      if (expertsAssignedOnGpu[gpu].count(expert) != 0)
        continue;
      uint32_t count = expertsOnGpu[gpu];
      if (!found || count < bestCount || (count == bestCount && gpu < bestGpu)) {
        found = true;
        bestCount = count;
        bestGpu = gpu;
      }
    }
    return std::make_pair(found, bestGpu);
  };

  auto placeExpert = [&](uint32_t expert, uint32_t l1) {
    auto chosen = bestGpuInL1(l1, expert);
    if (!chosen.first)
      return false;
    const auto& gpu = chosen.second;
    plan[expert].push_back(gpu);
    expertsAssignedOnGpu[gpu].insert(expert);
    expertsOnGpu[gpu]++;
    l1Load[l1] += ExpertFreq(expert);
    expertsByL1[l1].insert(expert);
    return true;
  };

  for (uint32_t expert : expertsByFreq) {
    bool found = false;
    uint32_t bestL1 = gpusByL1.begin()->first;
    ExpertPlanScore bestScore;
    for (const auto& kv : gpusByL1) {
      uint32_t l1 = kv.first;
      if (!bestGpuInL1(l1, expert).first)
        continue;
      std::map<uint32_t, double> trialLoad = l1Load;
      trialLoad[l1] += ExpertFreq(expert);
      ExpertPlanScore trial{
          MaxL1Load(trialLoad),
          L1LoadVariance(trialLoad),
          static_cast<double>(expertsByL1[l1].count(expert) == 0 ? 0 : 1)};
      if (!found || BetterPlanScore(trial, bestScore)) {
        found = true;
        bestScore = trial;
        bestL1 = l1;
      }
    }
    if (!found || !placeExpert(expert, bestL1))
      return std::vector<std::vector<std::pair<uint32_t,uint32_t>>>(expertNum);
  }

  uint32_t placed = expertNum;
  while (placed < totalSlots) {
    bool found = false;
    uint32_t bestExpert = 0;
    uint32_t bestL1 = gpusByL1.begin()->first;
    std::tuple<double, double, double, uint32_t, uint32_t> bestScore{
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max()};

    for (uint32_t expert : expertsByFreq) {
      double freq = ExpertFreq(expert);
      for (const auto& kv : gpusByL1) {
        uint32_t l1 = kv.first;
        if (expertsByL1[l1].count(expert) != 0)
          continue;
        if (!bestGpuInL1(l1, expert).first)
          continue;
        std::map<uint32_t, double> trialLoad = l1Load;
        trialLoad[l1] += freq;
        double domainsMissingExpert = 0.0;
        for (const auto& domain : gpusByL1) {
          if (domain.first != l1 && expertsByL1[domain.first].count(expert) == 0)
            domainsMissingExpert += 1.0;
        }
        double replicaCount = static_cast<double>(std::max<size_t>(1, plan[expert].size()));
        double marginalGain = freq / replicaCount;
        double externalCostAfter = domainsMissingExpert * freq;
        double replicaPenalty = replicaCount * replicaCount * 0.05;
        auto score = std::make_tuple(
            externalCostAfter * 0.01 - marginalGain + replicaPenalty,
            L1LoadVariance(trialLoad),
            MaxL1Load(trialLoad),
            static_cast<uint32_t>(plan[expert].size()),
            expert);
        if (!found || score < bestScore) {
          found = true;
          bestScore = score;
          bestExpert = expert;
          bestL1 = l1;
        }
      }
    }

    if (!found || !placeExpert(bestExpert, bestL1))
      break;
    placed++;
  }

  if (scoreOut != nullptr) {
    *scoreOut = ExpertPlanScore{
        MaxL1Load(l1Load),
        L1LoadVariance(l1Load),
        TopologyDistanceScore(gpus)};
  }
  return plan;
}

inline uint32_t MaxExpertsPerGpu() {
  return 9;
}

inline std::vector<std::pair<uint32_t,uint32_t>> PickPlacementForNeed(
    const cclScheduler::PlacementRequest& req,
    uint32_t need,
    const std::vector<std::vector<uint64_t>>& memUsed,
    const std::vector<std::vector<uint64_t>>& gpuMem,
    const std::vector<std::vector<uint32_t>>& expertUsed,
    uint32_t numNodes,
    uint32_t gpusPerServer,
    uint32_t maxExpertsPerGpu) {
  std::vector<std::pair<uint32_t,uint32_t>> candidates;
  for (uint32_t node = 0; node < numNodes; ++node) {
    for (uint32_t gpu = 0; gpu < gpusPerServer; ++gpu)
      candidates.emplace_back(node, gpu);
  }
  if (!candidates.empty()) {
    size_t split = (candidates.size() * 3) / 4;
    if (split == 0)
      split = 1;
    if (split >= candidates.size())
      split = candidates.size() - 1;
    if (req.kind == cclScheduler::PlacementKind::Prefill) {
      candidates.resize(split);
    } else if (req.kind == cclScheduler::PlacementKind::Decode) {
      candidates.erase(candidates.begin(), candidates.begin() + split);
    }
  }
  if (candidates.empty())
    return {};

  std::vector<std::pair<uint32_t,uint32_t>> chosen;
  std::set<std::pair<uint32_t,uint32_t>> chosenUnique;
  std::map<std::pair<uint32_t,uint32_t>, uint32_t> expertDelta;
  std::map<std::pair<uint32_t,uint32_t>, uint64_t> memDelta;
  std::map<uint32_t, double> l1Load;

  while (chosen.size() < need) {
    size_t best = candidates.size();
    std::tuple<double, uint32_t, double, std::pair<uint32_t,uint32_t>> bestScore{
        std::numeric_limits<double>::max(),
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<double>::max(),
        {std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max()}};
    bool preferUnique = chosenUnique.size() < candidates.size();
    for (size_t i = 0; i < candidates.size(); ++i) {
      const auto& candidate = candidates[i];
      if (preferUnique && chosenUnique.count(candidate) != 0)
        continue;
      uint32_t experts = ExpertsOnGpuForMem(
          req.expert_num,
          need,
          static_cast<uint32_t>(chosen.size()),
          req.expert_mem_bytes,
          req.expert_per_gpu);
      uint64_t memNeed = static_cast<uint64_t>(experts) * req.expert_mem_bytes;
      uint32_t node = candidate.first;
      uint32_t gpu = candidate.second;
      if (node >= memUsed.size() || gpu >= memUsed[node].size())
        continue;
      if (expertUsed[node][gpu] + expertDelta[candidate] + experts > maxExpertsPerGpu)
        continue;
      if (memUsed[node][gpu] + memDelta[candidate] + memNeed > gpuMem[node][gpu])
        continue;

      uint32_t l1 = L1GroupOfGpu(candidate);
      double projectedL1Load = l1Load[l1] + static_cast<double>(experts);
      double topoCost = chosen.empty() ? 0.0 : TopologyDistanceScore({chosen.front(), candidate});
      auto score = std::make_tuple(projectedL1Load,
                                   expertUsed[node][gpu] + expertDelta[candidate],
                                   topoCost,
                                   candidate);
      if (score < bestScore) {
        best = i;
        bestScore = score;
      }
    }
    if (best == candidates.size())
      break;
    auto gpu = candidates[best];
    chosen.push_back(gpu);
    chosenUnique.insert(gpu);
    uint32_t experts = ExpertsOnGpuForMem(
        req.expert_num,
        need,
        static_cast<uint32_t>(chosen.size() - 1),
        req.expert_mem_bytes,
        req.expert_per_gpu);
    expertDelta[gpu] += experts;
    memDelta[gpu] += static_cast<uint64_t>(experts) * req.expert_mem_bytes;
    l1Load[L1GroupOfGpu(gpu)] += static_cast<double>(experts);
  }
  return chosen.size() == need ? chosen : std::vector<std::pair<uint32_t,uint32_t>>{};
}

inline NeedCandidateScore ScoreNeedCandidate(uint32_t need,
                                             uint32_t expertNum,
                                             uint64_t expertMemBytes,
                                             uint32_t expertPerGpu,
                                             const std::vector<std::vector<uint64_t>>& memUsed,
                                             const std::vector<std::vector<uint64_t>>& gpuMem,
                                             const std::vector<std::vector<uint32_t>>& expertUsed,
                                             uint32_t numNodes,
                                             uint32_t gpusPerServer,
                                             uint32_t maxExpertsPerGpu) {
  NeedCandidateScore out;
  out.need = need;
  cclScheduler::PlacementRequest req{
      0, cclScheduler::PlacementKind::Decode, need, expertNum, expertMemBytes, expertPerGpu};
  auto placement = PickPlacementForNeed(req, need, memUsed, gpuMem, expertUsed, numNodes, gpusPerServer, maxExpertsPerGpu);
  if (placement.size() != need)
    return out;
  auto plan = BuildExpertPlan(expertNum, expertPerGpu, placement, &out.plan);
  out.feasible = ExpertPlanReady(plan, expertNum);
  return out;
}

// ===== POLICY GROUP: PD need placement within memory/expert constraints =====
inline cclScheduler::NeedPlacementPolicy NeedPlacementPolicy() {
  return [](const cclScheduler::PlacementRequest& req,
            const std::vector<std::vector<uint64_t>>& memUsed,
            const std::vector<std::vector<uint64_t>>& gpuMem,
            const std::vector<std::vector<uint32_t>>& expertUsed,
            uint32_t numNodes,
            uint32_t gpusPerServer,
            uint32_t maxExpertsPerGpu) {
    {
      std::ostringstream ss;
      ss << "PipelinePolicy: need placement start"
         << " owner=" << req.ownerId
         << " kind=" << cclScheduler::PlacementKindName(req.kind)
         << " need=" << req.need
         << " expert_num=" << req.expert_num
         << " expert_per_gpu=" << req.expert_per_gpu;
      ccl::CclLog(ss.str());
    }
    auto chosen = PickPlacementForNeed(
        req, req.need, memUsed, gpuMem, expertUsed, numNodes, gpusPerServer, maxExpertsPerGpu);
    {
      std::ostringstream ss;
      ss << "PipelinePolicy: need placement dispatch"
         << " owner=" << req.ownerId
         << " kind=" << cclScheduler::PlacementKindName(req.kind)
         << " selected_gpus=" << chosen.size();
      for (const auto& gpu : chosen)
        ss << " (" << gpu.first << "," << gpu.second << ")";
      ccl::CclLog(ss.str());
    }
    return chosen;
  };
}

// ===== POLICY GROUP: expert placement within selected PD GPUs =====
inline cclScheduler::ExpertPlacementPolicy ExpertPlacementPolicy() {
  return [](const cclScheduler::ExpertPlacementRequest& req) {
    {
      std::ostringstream ss;
      ss << "PipelinePolicy: expert placement start"
         << " owner=" << req.ownerId
         << " kind=" << cclScheduler::PlacementKindName(req.kind)
         << " selected_gpus=" << req.gpus.size()
         << " expert_num=" << req.expert_num
         << " expert_per_gpu=" << req.expert_per_gpu;
      ccl::CclLog(ss.str());
    }
    ExpertPlanScore score;
    std::string policy = Lower(Trim(State().config.expert_placement_policy));
    auto plan = policy == "domain_balanced_external_min"
                ? BuildDomainBalancedExternalMinPlan(
                      req.expert_num, req.expert_per_gpu, req.gpus, &score)
                : BuildExpertPlan(req.expert_num, req.expert_per_gpu, req.gpus, &score);
    State().latest_expert_plan = plan;
    State().latest_plan_max_l1_load = score.maxL1Load;
    State().latest_plan_l1_variance = score.l1Variance;
    {
      std::ostringstream ss;
      ss << "PipelinePolicy: expert placement dispatch"
         << " owner=" << req.ownerId
         << " kind=" << cclScheduler::PlacementKindName(req.kind)
         << " policy=" << State().config.expert_placement_policy
         << " experts=" << plan.size()
         << " l1_max_load=" << score.maxL1Load
         << " l1_variance=" << score.l1Variance;
      ccl::CclLog(ss.str());
    }
    return plan;
  };
}

inline uint64_t ProbeRtt(uint32_t srcNode, uint32_t dstNode, uint16_t pg, uint64_t bytes) {
  uint64_t base = pairRtt[srcNode][dstNode];
  uint64_t bw = pairBw[srcNode][dstNode];
  uint64_t tx = bw == 0 ? 0 : bytes * 8000000000lu / bw;
  uint64_t priorityPenalty = static_cast<uint64_t>(pg) * (base / 8 + 100);
  return base + tx + priorityPenalty;
}

// ===== POLICY GROUP: same-rank expert replica route selection =====
inline mnccl::ExpertRoutePolicy ExpertRoutePolicy() {
  return [](const mnccl::ExpertRouteRequest& req) {
    if (req.candidates.empty()) {
      ccl::CclLog("PipelinePolicy: same-rank route start with no candidates");
      return req.sourceGpu;
    }
    //##############################
    // {
    //   std::ostringstream ss;
    //   ss << "PipelinePolicy: same-rank route start"
    //      << " owner=" << req.expertPlacementOwner
    //      << " expert=" << req.expertId
    //      << " stage=" << mnccl::PipelineStageName(req.stage)
    //      << " kind=" << cclScheduler::PlacementKindName(req.pdKind)
    //      << " source=(" << req.sourceGpu.first << "," << req.sourceGpu.second << ")"
    //      << " candidates=" << req.candidates.size();
    //   ccl::CclLog(ss.str());
    // }
    std::pair<uint32_t,uint32_t> best = req.candidates.front();
    uint64_t bestDelta = std::numeric_limits<uint64_t>::max();
    uint32_t srcNode = NetworkNodeId(req.sourceGpu);
    for (const auto& candidate : req.candidates) {
      uint32_t dstNode = NetworkNodeId(candidate);
      uint64_t high = ProbeRtt(srcNode, dstNode, State().config.probe_high_pg, State().config.probe_bytes);
      uint64_t low = ProbeRtt(srcNode, dstNode, State().config.probe_low_pg, State().config.probe_bytes);
      uint64_t delta = low > high ? low - high : high - low;
      if (delta < bestDelta || (delta == bestDelta && candidate < best)) {
        bestDelta = delta;
        best = candidate;
      }
    }
    //###############################
    // {
    //   std::ostringstream ss;
    //   ss << "PipelinePolicy: same-rank route dispatch"
    //      << " owner=" << req.expertPlacementOwner
    //      << " expert=" << req.expertId
    //      << " selected=(" << best.first << "," << best.second << ")"
    //      << " rtt_delta=" << bestDelta;
    //   ccl::CclLog(ss.str());
    // }
    return best;
  };
}

inline size_t SelectPrefillFirstDecodeSjf(const std::vector<mnccl::LocalFlowTask>& queue) {
  auto priority = [](const mnccl::LocalFlowTask& task) {
    if (task.pdKind == cclScheduler::PlacementKind::Prefill)
      return 0;
    if (task.pdKind == cclScheduler::PlacementKind::Decode)
      return 1;
    return 2;
  };
  size_t best = 0;
  for (size_t i = 1; i < queue.size(); ++i) {
    const auto& a = queue[i];
    const auto& b = queue[best];
    int ap = priority(a);
    int bp = priority(b);
    if (ap != bp) {
      if (ap < bp)
        best = i;
      continue;
    }
    bool aDecode = a.pdKind == cclScheduler::PlacementKind::Decode;
    bool bDecode = b.pdKind == cclScheduler::PlacementKind::Decode;
    if (aDecode && bDecode && a.msgSize != b.msgSize) {
      if (a.msgSize < b.msgSize)
        best = i;
      continue;
    }
    if (a.arrivalTimeNs < b.arrivalTimeNs ||
        (a.arrivalTimeNs == b.arrivalTimeNs && a.sequence < b.sequence)) {
      best = i;
    }
  }
  return best;
}

inline size_t SelectDecodeFirstPrefillLater(const std::vector<mnccl::LocalFlowTask>& queue) {
  auto priority = [](const mnccl::LocalFlowTask& task) {
    if (task.pdKind == cclScheduler::PlacementKind::Decode)
      return 0;
    if (task.pdKind == cclScheduler::PlacementKind::Prefill)
      return 1;
    return 2;
  };
  size_t best = 0;
  for (size_t i = 1; i < queue.size(); ++i) {
    const auto& a = queue[i];
    const auto& b = queue[best];
    int ap = priority(a);
    int bp = priority(b);
    if (ap != bp) {
      if (ap < bp)
        best = i;
      continue;
    }
    bool aDecode = a.pdKind == cclScheduler::PlacementKind::Decode;
    bool bDecode = b.pdKind == cclScheduler::PlacementKind::Decode;
    if (aDecode && bDecode && a.msgSize != b.msgSize) {
      if (a.msgSize < b.msgSize)
        best = i;
      continue;
    }
    if (a.arrivalTimeNs < b.arrivalTimeNs ||
        (a.arrivalTimeNs == b.arrivalTimeNs && a.sequence < b.sequence)) {
      best = i;
    }
  }
  return best;
}

// ===== POLICY GROUP: per-GPU local flow queue scheduling =====
inline mnccl::LocalFlowSchedulePolicy LocalFlowSchedulePolicy() {
  return [](const std::vector<mnccl::LocalFlowTask>& queue, uint64_t /*nowNs*/) {
    // std::ostringstream start;
    // start << "PipelinePolicy: local flow queue schedule start"
    //       << " queue_size=" << queue.size()
    //       << " policy=" << State().config.local_flow_schedule_policy;
    // ccl::CclLog(start.str());
    size_t best = 0;
    std::string policy = Lower(Trim(State().config.local_flow_schedule_policy));
    if (policy == "decode_first_prefill_later" ||
        policy == "decode_first_prefill_last" ||
        policy == "decode_first")
      best = SelectDecodeFirstPrefillLater(queue);
    else
      best = SelectPrefillFirstDecodeSjf(queue);
    // std::ostringstream done;
    // done << "PipelinePolicy: local flow queue schedule dispatch"
    //      << " selected_index=" << best
    //      << " queue_size=" << queue.size()
    //      << " policy=" << State().config.local_flow_schedule_policy;
    // if (!queue.empty()) {
    //   done << " stage=" << mnccl::PipelineStageName(queue[best].stage)
    //        << " kind=" << cclScheduler::PlacementKindName(queue[best].pdKind)
    //        << " msg_size=" << queue[best].msgSize
    //        << " arrival_ns=" << queue[best].arrivalTimeNs;
    // }
    // ccl::CclLog(done.str());
    return best;
  };
}

inline uint32_t SplitIterations(uint32_t decodeIterations,
                                double prefillRate,
                                double decodeRate) {
  if (decodeIterations == 0 || prefillRate <= 0.0 || decodeRate <= 0.0)
    return 0;
  double ratio = decodeRate / (prefillRate + decodeRate);
  return static_cast<uint32_t>(std::min<double>(decodeIterations, std::round(decodeIterations * ratio)));
}

inline mnccl::PdSplitPolicy PdSplitPolicy() {
  return [](const mnccl::PdSplitRequest& req) {
    {
      std::ostringstream ss;
      ss << "PipelinePolicy: pd split start"
         << " task=" << req.task.taskId
         << " prefill_length=" << req.task.prefillLength
         << " decode_length=" << req.task.decodeLength
         << " decode_iterations=" << req.decodeIterations
         << " prefill_rate=" << State().config.prefill_rate_tokens_per_ns
         << " decode_rate=" << State().config.decode_rate_tokens_per_ns;
      ccl::CclLog(ss.str());
    }
    uint32_t split = SplitIterations(req.decodeIterations,
                                     State().config.prefill_rate_tokens_per_ns,
                                     State().config.decode_rate_tokens_per_ns);
    State().latest_split_iterations = split;
    {
      std::ostringstream ss;
      ss << "PipelinePolicy: pd split dispatch"
         << " task=" << req.task.taskId
         << " prefill_side_decode_iterations=" << split;
      ccl::CclLog(ss.str());
    }
    return split;
  };
}

inline std::vector<uint32_t> NeedCandidates(const cclScheduler::NeedState& state) {
  std::vector<uint32_t> candidates = State().config.candidate_needs;
  if (candidates.empty()) {
    uint32_t maxNeed = State().config.max_need == 0
                       ? std::max(state.need_prefill, state.need_decode)
                       : State().config.max_need;
    for (uint32_t need = std::max(1u, State().config.min_need); need <= maxNeed; need *= 2) {
      candidates.push_back(need);
      if (need > maxNeed / 2)
        break;
    }
  }
  if (candidates.empty())
    candidates.push_back(std::max(1u, state.need_decode));
  return candidates;
}

inline uint32_t SelectNeedByExpertPlacement(const cclScheduler::NeedState& state) {
  auto candidates = NeedCandidates(state);
  {
    std::ostringstream ss;
    ss << "PipelinePolicy: need candidate evaluation start"
       << " candidates=";
    for (uint32_t need : candidates)
      ss << " " << need;
    ss << " expert_num=" << state.expert_num
       << " expert_per_gpu=" << state.expert_per_gpu;
    ccl::CclLog(ss.str());
  }
  uint64_t expertMemBytes = mnccl::GetRuntimeConfig().expert_mem_bytes;
  uint32_t bestNeed = std::max(1u, state.need_decode);
  NeedCandidateScore bestScore;
  std::vector<std::vector<uint64_t>> memUsed;
  std::vector<std::vector<uint64_t>> gpuMem;
  std::vector<std::vector<uint32_t>> expertUsed;
  uint32_t numNodes = 0;
  uint32_t gpusPerServer = 0;
  uint32_t maxExpertsPerGpu = 0;
  cclScheduler::GetResourceSnapshot(memUsed, gpuMem, expertUsed, numNodes, gpusPerServer, maxExpertsPerGpu);
  for (uint32_t need : candidates) {
    if (need == 0)
      continue;
    NeedCandidateScore score = ScoreNeedCandidate(need,
                                                  state.expert_num,
                                                  expertMemBytes,
                                                  state.expert_per_gpu,
                                                  memUsed,
                                                  gpuMem,
                                                  expertUsed,
                                                  numNodes,
                                                  gpusPerServer,
                                                  maxExpertsPerGpu);
    if (BetterNeedScore(score, bestScore)) {
      bestScore = score;
      bestNeed = need;
    }
  }
  {
    std::ostringstream ss;
    ss << "PipelinePolicy: need candidate evaluation dispatch"
       << " selected_need=" << bestNeed
       << " max_l1_load=" << bestScore.plan.maxL1Load
       << " l1_variance=" << bestScore.plan.l1Variance;
    ccl::CclLog(ss.str());
  }
  return bestNeed;
}

// ===== POLICY GROUP: global need update on task dispatch/finish =====
inline cclScheduler::NeedUpdateCallback NeedUpdateCallback() {
  return [](cclScheduler::NeedEvent event,
            uint32_t taskId,
            uint32_t prefillLength,
            uint32_t decodeLength,
            cclScheduler::NeedState& state) {
    if (event != cclScheduler::NeedEvent::TaskDispatch)
      return;
    {
      std::ostringstream ss;
      ss << "PipelinePolicy: global need update start"
         << " event=" << (event == cclScheduler::NeedEvent::TaskDispatch ? "dispatch" : "finish")
         << " task=" << taskId
         << " current_need_prefill=" << state.need_prefill
         << " current_need_decode=" << state.need_decode
         << " expert_num=" << state.expert_num
         << " expert_per_gpu=" << state.expert_per_gpu;
      ccl::CclLog(ss.str());
    }
    auto& s = State();
    uint32_t bestNeed = SelectNeedByExpertPlacement(state);
    bestNeed = std::max(1u, bestNeed);
    s.selected_decode_need = bestNeed;
    s.selected_prefill_need = bestNeed;
    state.need_prefill = s.selected_prefill_need;
    state.need_decode = s.selected_decode_need;
    s.current_need = state;

    std::ostringstream ss;
    ss << "PipelinePolicy: global need update dispatch event="
       << (event == cclScheduler::NeedEvent::TaskDispatch ? "dispatch" : "finish")
       << " task=" << taskId
       << " prefill_length=" << prefillLength
       << " decode_length=" << decodeLength
       << " need_prefill=" << state.need_prefill
       << " need_decode=" << state.need_decode
       << " expert_num=" << state.expert_num
       << " expert_per_gpu=" << state.expert_per_gpu
       << " split_hint=" << s.latest_split_iterations
       << " expert_l1_max_load=" << s.latest_plan_max_l1_load
       << " expert_l1_variance=" << s.latest_plan_l1_variance;
    ccl::CclLog(ss.str());
  };
}

inline taskGenerator::PipelineWorkloadParams ApplyPolicyConfig(
    taskGenerator::PipelineWorkloadParams params,
    const PolicyConfig& cfg) {
  State().config = cfg;
  params.runtime.pg = cfg.pg;
  params.runtime.need_prefill = std::max(1u, cfg.need_prefill);
  params.runtime.need_decode = std::max(1u, cfg.need_decode);
  params.runtime.expert_num = std::max(1u, cfg.expert_num);
  params.runtime.expert_per_gpu = std::max(1u, cfg.expert_per_gpu);
  params.runtime.kflows = cfg.kflows;
  params.runtime.single_token_length = std::max(1u, cfg.single_token_length);
  params.runtime.expert_mem_bytes = cfg.expert_mem_bytes;
  params.runtime.token_msg_size = cfg.token_msg_size;
  params.runtime.base_decode_compute_delay_ns = cfg.base_decode_compute_delay_ns;
  params.distribution.num_tasks = cfg.num_tasks;
  params.distribution.seed = cfg.task_seed;
  params.distribution.prefill_length_min = cfg.prefill_length_min;
  params.distribution.prefill_length_max = cfg.prefill_length_max;
  params.distribution.decode_length_min = cfg.decode_length_min;
  params.distribution.decode_length_max = cfg.decode_length_max;
  params.distribution.first_submit_time = cfg.first_submit_time;
  params.distribution.submit_interval = cfg.submit_interval;
  params.simulation_stop_time = cfg.simulation_stop_time;
  params.print_submissions = cfg.print_submissions;
  State().current_need = cclScheduler::NeedState{
      params.runtime.need_prefill,
      params.runtime.need_decode,
      params.runtime.expert_num,
      params.runtime.expert_per_gpu};
  State().selected_prefill_need = params.runtime.need_prefill;
  State().selected_decode_need = params.runtime.need_decode;

  if (!cfg.enable_policy_group) {
    ccl::CclLog("PipelinePolicy: policy group disabled, using built-in defaults");
    return params;
  }

  ccl::CclLog("PipelinePolicy: policy config apply start");
  if (PolicyHookEnabled(cfg.need_update_policy)) {
    ccl::CclLog("PipelinePolicy: global need update policy dispatch name=" + cfg.need_update_policy);
    params.need_update_callback = NeedUpdateCallback();
  }
  if (PolicyHookEnabled(cfg.need_placement_policy)) {
    ccl::CclLog("PipelinePolicy: need placement policy dispatch name=" + cfg.need_placement_policy);
    params.need_placement_policy = NeedPlacementPolicy();
  }
  if (PolicyHookEnabled(cfg.expert_placement_policy)) {
    ccl::CclLog("PipelinePolicy: expert placement policy dispatch name=" + cfg.expert_placement_policy);
    params.expert_placement_policy = ExpertPlacementPolicy();
  }
  if (PolicyHookEnabled(cfg.local_flow_schedule_policy)) {
    ccl::CclLog("PipelinePolicy: local flow queue schedule policy dispatch name=" + cfg.local_flow_schedule_policy);
    params.local_flow_schedule_policy = LocalFlowSchedulePolicy();
  }
  if (PolicyHookEnabled(cfg.same_rank_route_policy)) {
    ccl::CclLog("PipelinePolicy: same-rank route policy dispatch name=" + cfg.same_rank_route_policy);
    params.expert_route_policy = ExpertRoutePolicy();
  }
  if (PolicyHookEnabled(cfg.pd_split_policy)) {
    ccl::CclLog("PipelinePolicy: pd split policy dispatch name=" + cfg.pd_split_policy);
    params.pd_split_policy = PdSplitPolicy();
  }
  ccl::CclLog("PipelinePolicy: policy config apply done");

  std::ostringstream ss;
  ss << "PipelinePolicy: hooks"
     << " need_update=" << cfg.need_update_policy
     << " need_placement=" << cfg.need_placement_policy
     << " expert_placement=" << cfg.expert_placement_policy
     << " local_flow_schedule=" << cfg.local_flow_schedule_policy
     << " same_rank_route=" << cfg.same_rank_route_policy
     << " pd_split=" << cfg.pd_split_policy;
  ccl::CclLog(ss.str());
  return params;
}

inline taskGenerator::PipelineWorkloadParams ApplyPolicyConfigFile(
    taskGenerator::PipelineWorkloadParams params,
    const std::string& path) {
  PolicyConfig cfg = LoadPolicyConfig(path);
  return ApplyPolicyConfig(params, cfg);
}

} // namespace pipelinePolicy

#endif // PIPELINE_POLICY_CONFIG_H
