#ifndef PIPELINE_POLICY_CONFIG_H
#define PIPELINE_POLICY_CONFIG_H

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
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
  bool print_expert_placement_detail = true;
  double simulation_stop_time = 1000.0;
  uint16_t pg = mnccl::default_pg;
  uint32_t need_prefill = 128;
  uint32_t need_decode = 32;
  uint32_t expert_num = 64;
  uint32_t kflows = 8;
  uint32_t single_token_length = 2;
  uint64_t expert_mem_bytes = 64ULL * 1024ULL * 1024ULL;
  uint64_t token_msg_size = 2;
  uint64_t base_decode_compute_delay_ns = 1;
  uint32_t dispatch_n = 1;
  uint32_t dispatch_n_min = 0;
  uint32_t dispatch_n_max = 0;
  uint64_t dispatch_expert_ffn_params = 1000000000ULL;
  uint32_t dispatch_precision_bytes = 1;
  uint32_t dispatch_batch_size = 16;
  uint32_t num_tasks = 3;
  bool enable_inference_workload = true;
  bool enable_training_workload = false;
  uint32_t train_num_tasks = 0;
  uint32_t train_seed = 54321;
  uint32_t train_need_min = 128;
  uint32_t train_need_max = 128;
  std::vector<uint32_t> train_need_set;
  uint64_t train_msg_size_min = 1048576;
  uint64_t train_msg_size_max = 1048576;
  uint64_t train_mem_bytes_per_gpu = 1073741824ULL;
  double train_first_submit_time = 0.00001;
  double train_submit_interval = 0.00002;
  uint16_t train_pg = mnccl::default_pg;
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
  uint32_t route_probe_max_inflight = 64;
  bool async_route_probe_enable = false;
  bool async_route_live_route = false;
  uint64_t async_route_probe_interval_ns = 50000;
  uint32_t async_route_probe_budget = 64;
  uint64_t async_route_probe_refresh_ns = 1000000;
  uint32_t async_route_probe_topk = 2;
  double prefill_rate_tokens_per_ns = 0.35;
  double decode_rate_tokens_per_ns = 0.03;
  uint32_t min_need = 1;
  uint32_t max_need = 0;
  uint32_t expert_per_gpu = 1;
  uint64_t cluster_monitor_interval_ns = 1000000;
  std::string need_placement_policy = "expert_capacity_l1";
  std::string expert_placement_policy = "probe_balanced_domain_spread";
  std::string same_rank_route_policy = "probe_rtt_delta";
  std::string local_flow_schedule_policy = "prefill_first_decode_sjf";
  std::string need_update_policy = "expert_variance";
  std::string pd_split_policy = "default";
  std::vector<uint32_t> candidate_needs;
  std::vector<double> expert_access_freq;
  std::vector<double> expert_access_freq_after;
  double expert_access_prior = 1.0;
  uint32_t expert_access_switch_task = 0;
  std::string placement_access_mode = "posterior";
  double placement_drift_threshold = 0.25;
  uint64_t placement_drift_min_samples = 128;
  double placement_external_weight = 4.0;
  double placement_replica_balance_weight = 2.0;
  double placement_l1_variance_weight = 1.0;
  double placement_l1_max_weight = 2.0;
  double placement_gpu_variance_weight = 1.0;
  double placement_probe_weight = 0.0;
  double placement_probe_delta_weight = 1.0;
  double placement_probe_rtt_weight = 0.25;
  std::string placement_adaptive_mode = "fixed";
  double placement_adaptive_learning_rate = 0.05;
  double placement_adaptive_sigma = 0.10;
  double placement_adaptive_obs_alpha = 0.30;
  double placement_adaptive_momentum = 0.70;
  double placement_adaptive_weight_sum = 0.0;
  double placement_adaptive_gradient_clip = 0.50;
  double placement_adaptive_dynamic_lr_gain = 2.0;
  double placement_adaptive_max_lr_multiplier = 4.0;
  double placement_adaptive_lr_update_threshold = 0.0;
  uint64_t placement_adaptive_lr_freeze_min_observations = 1;
  double placement_adaptive_min_weight = 0.0;
  double placement_adaptive_max_weight = 16.0;
  uint32_t placement_adaptive_seed = 20260714;
  double route_queue_weight = 0.0;
  double route_queue_norm = 16.0;
  uint64_t local_flow_preemptive_chunk_bytes = 0;
  uint64_t local_flow_pld_srpt_v_ns = 100000;
  uint64_t local_flow_pld_srpt_t0_ns = 100000;
  uint64_t local_flow_pld_srpt_d0_ns = 0;
  double local_flow_pld_srpt_kappa = 2.0;
  double local_flow_pld_srpt_beta = 0.0;
  uint64_t local_flow_pld_srpt_preempt_overhead_ns = 0;
  bool placement_force_new_l1_domain = true;
  bool trace_dispatch_enable = false;
  bool trace_stop_on_complete = true;
  bool trace_use_device_placement = true;
  bool trace_access_targets_as_expert_freq = true;
  bool trace_uniform_access_targets = false;
  uint64_t trace_uniform_access_target_count = 0;
  uint32_t trace_layer_id = 0;
  uint32_t trace_moe_layer_count = 0;
  uint32_t trace_access_divisor = 1;
  std::string trace_device_file = "./examples/HW/data/device.json";
  std::string trace_decode_dir = "./examples/HW/data";
  double crisp_alpha = 4.0;
  double crisp_beta = 2.0;
  double crisp_gamma = 3.0;
  double crisp_eta = 1.0;
  double crisp_min_frag_gain = 0.01;
  double crisp_max_interference = 8.0;
  double crisp_sla_urgent_threshold = 1.5;
  double crisp_training_boost_score = 0.01;
  double crisp_topology_mix = 0.35;
  double crisp_completion_floor = 0.35;
  double crisp_remaining_penalty = 0.25;
  uint32_t crisp_boost_max_outstanding = 4;
  uint32_t crisp_boost_need_divisor = 20;
  uint16_t crisp_boost_pg = mnccl::default_pg;
  uint16_t crisp_urgent_pg = mnccl::default_pg;
};

struct DecisionState {
  PolicyConfig config;
  cclScheduler::NeedState current_need{1, 1, 1, 1};
  uint32_t selected_prefill_need = 1;
  uint32_t selected_decode_need = 1;
  uint32_t latest_split_iterations = 0;
  std::vector<std::vector<std::pair<uint32_t,uint32_t>>> latest_expert_plan;
  std::vector<std::vector<uint32_t>> trace_device_experts;
  std::vector<std::vector<std::vector<uint32_t>>> trace_device_experts_by_layer;
  std::vector<uint64_t> trace_access_targets;
  std::vector<std::vector<uint64_t>> trace_access_targets_by_layer;
  std::vector<double> posterior_expert_access_freq;
  bool placement_drift_active = false;
  double latest_access_drift = 0.0;
  double latest_plan_max_l1_load = 0.0;
  double latest_plan_l1_variance = 0.0;
  bool placement_adaptive_initialized = false;
  uint64_t placement_adaptive_step = 0;
  uint32_t placement_adaptive_phase = 0;
  double placement_adaptive_plus_cost = 0.0;
  uint64_t placement_adaptive_observations = 0;
  double placement_adaptive_target_weight_sum = 0.0;
  std::vector<double> placement_adaptive_base_weights;
  std::vector<double> placement_adaptive_direction;
  std::vector<double> placement_adaptive_active_weights;
  std::vector<double> placement_adaptive_metric_ema;
  std::vector<double> placement_adaptive_gradient_momentum;
  std::vector<uint8_t> placement_adaptive_lr_frozen;
};

inline DecisionState& State() {
  static DecisionState state;
  return state;
}

inline void ResetRuntimeDecisionState() {
  auto& state = State();
  state.latest_split_iterations = 0;
  state.latest_expert_plan.clear();
  state.trace_device_experts.clear();
  state.trace_device_experts_by_layer.clear();
  state.trace_access_targets.clear();
  state.trace_access_targets_by_layer.clear();
  state.posterior_expert_access_freq.clear();
  state.placement_drift_active = false;
  state.latest_access_drift = 0.0;
  state.latest_plan_max_l1_load = 0.0;
  state.latest_plan_l1_variance = 0.0;
  state.placement_adaptive_initialized = false;
  state.placement_adaptive_step = 0;
  state.placement_adaptive_phase = 0;
  state.placement_adaptive_plus_cost = 0.0;
  state.placement_adaptive_observations = 0;
  state.placement_adaptive_target_weight_sum = 0.0;
  state.placement_adaptive_base_weights.clear();
  state.placement_adaptive_direction.clear();
  state.placement_adaptive_active_weights.clear();
  state.placement_adaptive_metric_ema.clear();
  state.placement_adaptive_gradient_momentum.clear();
  state.placement_adaptive_lr_frozen.clear();
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

inline bool TopologyGradientPolicyName(const std::string& value) {
  std::string v = Lower(Trim(value));
  return v == "topology_gradient_pg" ||
         v == "topo_gradient_pg" ||
         v == "l012_gradient_pg";
}

inline bool CrispPolicyName(const std::string& value) {
  std::string v = Lower(Trim(value));
  return v == "crisp" ||
         v == "crisp_pg" ||
         v == "proposed" ||
         v == "proposed_pg" ||
         v == "fragmentation_benefit" ||
         v == "fragmentation_benefit_pg" ||
         v == "frag_benefit" ||
         v == "frag_benefit_pg";
}

inline bool CrispPgOnlyPolicyName(const std::string& value) {
  std::string v = Lower(Trim(value));
  return v == "crisp_pg_only" ||
         v == "proposed_pg_only" ||
         v == "fragmentation_pg_only" ||
         v == "frag_pg_only";
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

inline bool ParseUnsignedAfterKey(const std::string& text,
                                  size_t keyPos,
                                  uint32_t* valueOut) {
  if (valueOut == nullptr)
    return false;
  size_t colon = text.find(':', keyPos);
  if (colon == std::string::npos)
    return false;
  size_t begin = colon + 1;
  while (begin < text.size() && !std::isdigit(static_cast<unsigned char>(text[begin])))
    ++begin;
  if (begin >= text.size())
    return false;
  size_t end = begin;
  while (end < text.size() && std::isdigit(static_cast<unsigned char>(text[end])))
    ++end;
  *valueOut = static_cast<uint32_t>(std::stoul(text.substr(begin, end - begin)));
  return true;
}

inline std::vector<uint32_t> ParseUnsignedArray(const std::string& text,
                                                size_t arrayPos) {
  std::vector<uint32_t> values;
  size_t begin = text.find('[', arrayPos);
  size_t end = begin == std::string::npos ? std::string::npos : text.find(']', begin);
  if (begin == std::string::npos || end == std::string::npos || end <= begin)
    return values;
  std::string body = text.substr(begin + 1, end - begin - 1);
  for (char& ch : body) {
    if (!std::isdigit(static_cast<unsigned char>(ch)))
      ch = ' ';
  }
  std::istringstream iss(body);
  uint32_t value = 0;
  while (iss >> value)
    values.push_back(value);
  return values;
}

inline std::vector<uint64_t> ParseCsvUint64Line(const std::string& line) {
  std::vector<uint64_t> values;
  std::string normalized = line;
  for (char& ch : normalized) {
    if (ch == ',')
      ch = ' ';
  }
  std::istringstream iss(normalized);
  uint64_t value = 0;
  while (iss >> value)
    values.push_back(value);
  return values;
}

inline std::string JoinPath(const std::string& dir, const std::string& file) {
  if (dir.empty())
    return file;
  char last = dir[dir.size() - 1];
  if (last == '/' || last == '\\')
    return dir + file;
  return dir + "/" + file;
}

inline uint32_t LoadTraceMoeLayerCount(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    ccl::CclLog("PipelinePolicy: trace device file open failed path=" + path);
    return 0;
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  std::string text = buffer.str();
  uint32_t layerCount = 0;
  size_t countPos = text.find("\"moe_layer_count\"");
  if (countPos != std::string::npos &&
      ParseUnsignedAfterKey(text, countPos, &layerCount) &&
      layerCount > 0) {
    return layerCount;
  }
  const std::string layerKey = "\"layer_id\"";
  uint32_t maxLayer = 0;
  uint32_t found = 0;
  bool any = false;
  size_t pos = 0;
  while ((pos = text.find(layerKey, pos)) != std::string::npos) {
    if (ParseUnsignedAfterKey(text, pos, &found)) {
      maxLayer = std::max(maxLayer, found);
      any = true;
    }
    pos += layerKey.size();
  }
  return any ? maxLayer + 1 : 0;
}

inline std::vector<std::vector<uint32_t>> LoadTraceDeviceLayerPlacement(
    const std::string& path,
    uint32_t layerId) {
  std::ifstream in(path);
  if (!in) {
    ccl::CclLog("PipelinePolicy: trace device file open failed path=" + path);
    return {};
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  std::string text = buffer.str();
  const std::string layerKey = "\"layer_id\"";
  const std::string deviceKey = "\"device_id\"";
  const std::string expertKey = "\"device_expert\"";

  size_t layerPos = 0;
  while ((layerPos = text.find(layerKey, layerPos)) != std::string::npos) {
    uint32_t foundLayer = 0;
    if (!ParseUnsignedAfterKey(text, layerPos, &foundLayer)) {
      layerPos += layerKey.size();
      continue;
    }
    size_t nextLayer = text.find(layerKey, layerPos + layerKey.size());
    std::string section = text.substr(
        layerPos,
        nextLayer == std::string::npos ? std::string::npos : nextLayer - layerPos);
    if (foundLayer != layerId) {
      layerPos += layerKey.size();
      continue;
    }

    std::vector<std::vector<uint32_t>> devices;
    size_t devicePos = 0;
    while ((devicePos = section.find(deviceKey, devicePos)) != std::string::npos) {
      uint32_t deviceId = 0;
      if (!ParseUnsignedAfterKey(section, devicePos, &deviceId)) {
        devicePos += deviceKey.size();
        continue;
      }
      size_t expertPos = section.find(expertKey, devicePos);
      size_t nextDevice = section.find(deviceKey, devicePos + deviceKey.size());
      if (expertPos == std::string::npos ||
          (nextDevice != std::string::npos && expertPos > nextDevice)) {
        devicePos += deviceKey.size();
        continue;
      }
      auto experts = ParseUnsignedArray(section, expertPos);
      if (deviceId >= devices.size())
        devices.resize(deviceId + 1);
      devices[deviceId] = experts;
      devicePos = expertPos + expertKey.size();
    }

    uint64_t slots = 0;
    for (const auto& experts : devices)
      slots += experts.size();
    std::ostringstream ss;
    ss << "PipelinePolicy: trace device placement loaded"
       << " path=" << path
       << " layer=" << layerId
       << " devices=" << devices.size()
       << " slots=" << slots;
    ccl::CclLog(ss.str());
    return devices;
  }
  ccl::CclLog("PipelinePolicy: trace device layer not found path=" +
              path + " layer=" + std::to_string(layerId));
  return {};
}

inline std::vector<std::vector<std::vector<uint32_t>>> LoadTraceDeviceAllLayerPlacements(
    const std::string& path,
    uint32_t requestedLayerCount,
    uint32_t fallbackLayerId) {
  uint32_t layerCount = requestedLayerCount;
  if (layerCount == 0)
    layerCount = LoadTraceMoeLayerCount(path);
  if (layerCount == 0)
    layerCount = fallbackLayerId + 1;

  std::vector<std::vector<std::vector<uint32_t>>> layers;
  layers.reserve(layerCount);
  for (uint32_t layer = 0; layer < layerCount; ++layer) {
    layers.push_back(LoadTraceDeviceLayerPlacement(path, layer));
  }
  uint32_t loadedLayers = 0;
  for (const auto& devices : layers) {
    if (!devices.empty())
      ++loadedLayers;
  }
  std::ostringstream ss;
  ss << "PipelinePolicy: trace device all-layer placement loaded"
     << " path=" << path
     << " requested_layers=" << layerCount
     << " loaded_layers=" << loadedLayers;
  ccl::CclLog(ss.str());
  return layers;
}

inline std::vector<uint64_t> LoadTraceDecodeLayerTargets(
    const std::string& dir,
    const std::vector<std::vector<uint32_t>>& deviceExperts,
    uint32_t layerId,
    uint32_t expertNum) {
  std::vector<uint64_t> targets(std::max(1u, expertNum), 0);
  uint64_t total = 0;
  uint32_t filesLoaded = 0;
  uint32_t nonzeroExperts = 0;
  for (uint32_t deviceId = 0;
       deviceId < static_cast<uint32_t>(deviceExperts.size());
       ++deviceId) {
    const auto& experts = deviceExperts[deviceId];
    if (experts.empty())
      continue;
    std::string path = JoinPath(dir, "decode_" + std::to_string(deviceId) + ".csv");
    std::ifstream in(path);
    if (!in) {
      ccl::CclLog("PipelinePolicy: trace decode file open failed path=" + path);
      continue;
    }
    std::string line;
    for (uint32_t row = 0; row <= layerId; ++row) {
      if (!std::getline(in, line)) {
        line.clear();
        break;
      }
    }
    if (line.empty()) {
      ccl::CclLog("PipelinePolicy: trace decode layer row missing path=" +
                  path + " layer=" + std::to_string(layerId));
      continue;
    }
    auto counts = ParseCsvUint64Line(line);
    size_t slots = std::min(counts.size(), experts.size());
    for (size_t slot = 0; slot < slots; ++slot) {
      uint32_t expert = experts[slot];
      if (expert >= targets.size())
        targets.resize(expert + 1, 0);
      targets[expert] += counts[slot];
      total += counts[slot];
    }
    ++filesLoaded;
  }
  for (uint64_t count : targets) {
    if (count > 0)
      ++nonzeroExperts;
  }
  std::ostringstream ss;
  ss << "PipelinePolicy: trace decode targets loaded"
     << " dir=" << dir
     << " layer=" << layerId
     << " files=" << filesLoaded
     << " experts=" << targets.size()
     << " nonzero_experts=" << nonzeroExperts
     << " total_accesses=" << total;
  ccl::CclLog(ss.str());
  return targets;
}

inline std::vector<std::vector<uint64_t>> LoadTraceDecodeAllLayerTargets(
    const std::string& dir,
    const std::vector<std::vector<std::vector<uint32_t>>>& deviceExpertsByLayer,
    uint32_t expertNum) {
  std::vector<std::vector<uint64_t>> targetsByLayer;
  targetsByLayer.reserve(deviceExpertsByLayer.size());
  uint64_t total = 0;
  uint32_t nonzeroLayers = 0;
  for (uint32_t layer = 0;
       layer < static_cast<uint32_t>(deviceExpertsByLayer.size());
       ++layer) {
    auto targets = LoadTraceDecodeLayerTargets(
        dir, deviceExpertsByLayer[layer], layer, expertNum);
    uint64_t layerTotal = std::accumulate(targets.begin(), targets.end(), uint64_t{0});
    if (layerTotal > 0)
      ++nonzeroLayers;
    total += layerTotal;
    targetsByLayer.push_back(std::move(targets));
  }
  std::ostringstream ss;
  ss << "PipelinePolicy: trace decode all-layer targets loaded"
     << " dir=" << dir
     << " layers=" << targetsByLayer.size()
     << " nonzero_layers=" << nonzeroLayers
     << " total_accesses=" << total;
  ccl::CclLog(ss.str());
  return targetsByLayer;
}

inline void ScaleTraceAccessTargets(std::vector<uint64_t>& targets,
                                    uint32_t divisor) {
  divisor = std::max(1u, divisor);
  if (divisor <= 1)
    return;
  uint64_t before = std::accumulate(targets.begin(), targets.end(), uint64_t{0});
  uint64_t nonzeroBefore = 0;
  for (uint64_t& count : targets) {
    if (count == 0)
      continue;
    ++nonzeroBefore;
    count = (count + divisor - 1) / divisor;
  }
  uint64_t after = std::accumulate(targets.begin(), targets.end(), uint64_t{0});
  uint64_t nonzeroAfter = 0;
  for (uint64_t count : targets) {
    if (count > 0)
      ++nonzeroAfter;
  }
  std::ostringstream ss;
  ss << "PipelinePolicy: trace access targets scaled"
     << " divisor=" << divisor
     << " total_before=" << before
     << " total_after=" << after
     << " nonzero_before=" << nonzeroBefore
     << " nonzero_after=" << nonzeroAfter;
  ccl::CclLog(ss.str());
}

inline void ScaleTraceAccessTargetsByLayer(
    std::vector<std::vector<uint64_t>>& targetsByLayer,
    uint32_t divisor) {
  for (auto& targets : targetsByLayer)
    ScaleTraceAccessTargets(targets, divisor);
}

inline void MakeTraceAccessTargetsUniform(std::vector<uint64_t>& targets,
                                          uint32_t expertNum,
                                          uint64_t targetCount) {
  uint64_t before = std::accumulate(targets.begin(), targets.end(), uint64_t{0});
  uint32_t count = std::max<uint32_t>(
      std::max(1u, expertNum),
      static_cast<uint32_t>(targets.size()));
  uint64_t perExpert = targetCount;
  if (perExpert == 0 && before > 0)
    perExpert = (before + count - 1) / count;
  targets.assign(count, perExpert);
  uint64_t after = std::accumulate(targets.begin(), targets.end(), uint64_t{0});
  std::ostringstream ss;
  ss << "PipelinePolicy: trace access targets uniformized"
     << " experts=" << count
     << " per_expert=" << perExpert
     << " total_before=" << before
     << " total_after=" << after
     << " explicit_count=" << targetCount;
  ccl::CclLog(ss.str());
}

inline void MakeTraceAccessTargetsUniformPreserveTotal(
    std::vector<uint64_t>& targets,
    uint32_t expertNum,
    uint64_t targetCount) {
  uint32_t count = std::max<uint32_t>(
      std::max(1u, expertNum), static_cast<uint32_t>(targets.size()));
  uint64_t originalTotal =
      std::accumulate(targets.begin(), targets.end(), uint64_t{0});
  uint64_t total = originalTotal;
  targets.assign(count, total / count);
  uint64_t rem = total % count;
  for (uint32_t expert = 0; expert < rem; ++expert)
    targets[expert] += 1;
  std::ostringstream ss;
  ss << "PipelinePolicy: trace access targets uniformized preserving layer total"
     << " experts=" << count
     << " original_total=" << originalTotal
     << " total_after=" << total
     << " explicit_target_count_ignored=" << targetCount;
  ccl::CclLog(ss.str());
}

inline void MakeTraceAccessTargetsUniformByLayer(
    std::vector<std::vector<uint64_t>>& targetsByLayer,
    uint32_t expertNum,
    uint64_t targetCount) {
  for (auto& targets : targetsByLayer) {
    MakeTraceAccessTargetsUniformPreserveTotal(
        targets, expertNum, targetCount);
  }
}

inline std::vector<uint64_t> AggregateTraceAccessTargetsByLayer(
    const std::vector<std::vector<uint64_t>>& targetsByLayer,
    uint32_t expertNum) {
  size_t count = std::max<size_t>(1, expertNum);
  for (const auto& targets : targetsByLayer)
    count = std::max(count, targets.size());
  std::vector<uint64_t> aggregate(count, 0);
  for (const auto& targets : targetsByLayer) {
    if (targets.size() > aggregate.size())
      aggregate.resize(targets.size(), 0);
    for (size_t expert = 0; expert < targets.size(); ++expert)
      aggregate[expert] += targets[expert];
  }
  return aggregate;
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
    else if (key == "PRINT_EXPERT_PLACEMENT_DETAIL")
      cfg.print_expert_placement_detail =
          ParseBool(value, cfg.print_expert_placement_detail);
    else if (key == "SIMULATION_STOP_TIME")
      cfg.simulation_stop_time = std::stod(value);
    else if (key == "PG")
      cfg.pg = static_cast<uint16_t>(std::stoul(value));
    else if (key == "DISPATCH_NEED") {
      cfg.need_prefill = static_cast<uint32_t>(std::stoul(value));
      cfg.need_decode = cfg.need_prefill;
    }
    else if (key == "NEED_PREFILL")
      cfg.need_prefill = static_cast<uint32_t>(std::stoul(value));
    else if (key == "NEED_DECODE")
      cfg.need_decode = static_cast<uint32_t>(std::stoul(value));
    else if (key == "EXPERT_NUM")
      cfg.expert_num = static_cast<uint32_t>(std::stoul(value));
    else if (key == "DISPATCH_TOPK")
      cfg.kflows = static_cast<uint32_t>(std::stoul(value));
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
    else if (key == "DISPATCH_N")
      cfg.dispatch_n = static_cast<uint32_t>(std::stoul(value));
    else if (key == "DISPATCH_N_MIN")
      cfg.dispatch_n_min = static_cast<uint32_t>(std::stoul(value));
    else if (key == "DISPATCH_N_MAX")
      cfg.dispatch_n_max = static_cast<uint32_t>(std::stoul(value));
    else if (key == "DISPATCH_EXPERT_FFN_PARAMS")
      cfg.dispatch_expert_ffn_params = std::stoull(value);
    else if (key == "DISPATCH_PRECISION_BYTES")
      cfg.dispatch_precision_bytes = static_cast<uint32_t>(std::stoul(value));
    else if (key == "DISPATCH_BATCH_SIZE")
      cfg.dispatch_batch_size = static_cast<uint32_t>(std::stoul(value));
    else if (key == "NUM_TASKS")
      cfg.num_tasks = static_cast<uint32_t>(std::stoul(value));
    else if (key == "ENABLE_INFERENCE_WORKLOAD")
      cfg.enable_inference_workload = ParseBool(value, cfg.enable_inference_workload);
    else if (key == "ENABLE_TRAINING_WORKLOAD")
      cfg.enable_training_workload = ParseBool(value, cfg.enable_training_workload);
    else if (key == "TRAIN_NUM_TASKS")
      cfg.train_num_tasks = static_cast<uint32_t>(std::stoul(value));
    else if (key == "TRAIN_SEED")
      cfg.train_seed = static_cast<uint32_t>(std::stoul(value));
    else if (key == "TRAIN_NEED_MIN")
      cfg.train_need_min = static_cast<uint32_t>(std::stoul(value));
    else if (key == "TRAIN_NEED_MAX")
      cfg.train_need_max = static_cast<uint32_t>(std::stoul(value));
    else if (key == "TRAIN_NEED_SET")
      cfg.train_need_set = ParseVector<uint32_t>(value);
    else if (key == "TRAIN_MSG_SIZE_MIN")
      cfg.train_msg_size_min = std::stoull(value);
    else if (key == "TRAIN_MSG_SIZE_MAX")
      cfg.train_msg_size_max = std::stoull(value);
    else if (key == "TRAIN_MEM_BYTES_PER_GPU")
      cfg.train_mem_bytes_per_gpu = std::stoull(value);
    else if (key == "TRAIN_FIRST_SUBMIT_TIME")
      cfg.train_first_submit_time = std::stod(value);
    else if (key == "TRAIN_SUBMIT_INTERVAL")
      cfg.train_submit_interval = std::stod(value);
    else if (key == "TRAIN_PG")
      cfg.train_pg = static_cast<uint16_t>(std::stoul(value));
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
    else if (key == "ROUTE_PROBE_MAX_INFLIGHT")
      cfg.route_probe_max_inflight = static_cast<uint32_t>(std::stoul(value));
    else if (key == "ASYNC_ROUTE_PROBE_ENABLE")
      cfg.async_route_probe_enable = ParseBool(value, cfg.async_route_probe_enable);
    else if (key == "ASYNC_ROUTE_LIVE_ROUTE")
      cfg.async_route_live_route = ParseBool(value, cfg.async_route_live_route);
    else if (key == "ASYNC_ROUTE_PROBE_INTERVAL_NS")
      cfg.async_route_probe_interval_ns = std::stoull(value);
    else if (key == "ASYNC_ROUTE_PROBE_BUDGET")
      cfg.async_route_probe_budget = static_cast<uint32_t>(std::stoul(value));
    else if (key == "ASYNC_ROUTE_PROBE_REFRESH_NS")
      cfg.async_route_probe_refresh_ns = std::stoull(value);
    else if (key == "ASYNC_ROUTE_PROBE_TOPK")
      cfg.async_route_probe_topk = static_cast<uint32_t>(std::stoul(value));
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
    else if (key == "LOCAL_FLOW_PREEMPTIVE_CHUNK_BYTES" ||
             key == "LOCAL_FLOW_QUANTUM_BYTES")
      cfg.local_flow_preemptive_chunk_bytes = std::stoull(value);
    else if (key == "LOCAL_FLOW_PLD_SRPT_V_NS")
      cfg.local_flow_pld_srpt_v_ns = std::stoull(value);
    else if (key == "LOCAL_FLOW_PLD_SRPT_T0_NS")
      cfg.local_flow_pld_srpt_t0_ns = std::stoull(value);
    else if (key == "LOCAL_FLOW_PLD_SRPT_D0_NS")
      cfg.local_flow_pld_srpt_d0_ns = std::stoull(value);
    else if (key == "LOCAL_FLOW_PLD_SRPT_KAPPA")
      cfg.local_flow_pld_srpt_kappa = std::stod(value);
    else if (key == "LOCAL_FLOW_PLD_SRPT_BETA")
      cfg.local_flow_pld_srpt_beta = std::stod(value);
    else if (key == "LOCAL_FLOW_PLD_SRPT_PREEMPT_OVERHEAD_NS")
      cfg.local_flow_pld_srpt_preempt_overhead_ns = std::stoull(value);
    else if (key == "GLOBAL_NEED_UPDATE_POLICY")
      cfg.need_update_policy = value;
    else if (key == "PD_SPLIT_POLICY")
      cfg.pd_split_policy = value;
    else if (key == "CANDIDATE_NEEDS")
      cfg.candidate_needs = ParseVector<uint32_t>(value);
    else if (key == "EXPERT_ACCESS_FREQ")
      cfg.expert_access_freq = ParseVector<double>(value);
    else if (key == "EXPERT_ACCESS_FREQ_AFTER")
      cfg.expert_access_freq_after = ParseVector<double>(value);
    else if (key == "EXPERT_ACCESS_PRIOR")
      cfg.expert_access_prior = std::stod(value);
    else if (key == "EXPERT_ACCESS_SWITCH_TASK")
      cfg.expert_access_switch_task = static_cast<uint32_t>(std::stoul(value));
    else if (key == "PLACEMENT_ACCESS_MODE")
      cfg.placement_access_mode = value;
    else if (key == "PLACEMENT_DRIFT_THRESHOLD")
      cfg.placement_drift_threshold = std::stod(value);
    else if (key == "PLACEMENT_DRIFT_MIN_SAMPLES")
      cfg.placement_drift_min_samples = std::stoull(value);
    else if (key == "PLACEMENT_EXTERNAL_WEIGHT")
      cfg.placement_external_weight = std::stod(value);
    else if (key == "PLACEMENT_REPLICA_BALANCE_WEIGHT")
      cfg.placement_replica_balance_weight = std::stod(value);
    else if (key == "PLACEMENT_L1_VARIANCE_WEIGHT")
      cfg.placement_l1_variance_weight = std::stod(value);
    else if (key == "PLACEMENT_L1_MAX_WEIGHT")
      cfg.placement_l1_max_weight = std::stod(value);
    else if (key == "PLACEMENT_GPU_VARIANCE_WEIGHT")
      cfg.placement_gpu_variance_weight = std::stod(value);
    else if (key == "PLACEMENT_PROBE_WEIGHT")
      cfg.placement_probe_weight = std::stod(value);
    else if (key == "PLACEMENT_PROBE_DELTA_WEIGHT")
      cfg.placement_probe_delta_weight = std::stod(value);
    else if (key == "PLACEMENT_PROBE_RTT_WEIGHT")
      cfg.placement_probe_rtt_weight = std::stod(value);
    else if (key == "PLACEMENT_ADAPTIVE_MODE")
      cfg.placement_adaptive_mode = value;
    else if (key == "PLACEMENT_ADAPTIVE_LEARNING_RATE")
      cfg.placement_adaptive_learning_rate = std::stod(value);
    else if (key == "PLACEMENT_ADAPTIVE_SIGMA")
      cfg.placement_adaptive_sigma = std::stod(value);
    else if (key == "PLACEMENT_ADAPTIVE_OBS_ALPHA")
      cfg.placement_adaptive_obs_alpha = std::stod(value);
    else if (key == "PLACEMENT_ADAPTIVE_MOMENTUM")
      cfg.placement_adaptive_momentum = std::stod(value);
    else if (key == "PLACEMENT_ADAPTIVE_WEIGHT_SUM")
      cfg.placement_adaptive_weight_sum = std::stod(value);
    else if (key == "PLACEMENT_ADAPTIVE_GRADIENT_CLIP")
      cfg.placement_adaptive_gradient_clip = std::stod(value);
    else if (key == "PLACEMENT_ADAPTIVE_DYNAMIC_LR_GAIN")
      cfg.placement_adaptive_dynamic_lr_gain = std::stod(value);
    else if (key == "PLACEMENT_ADAPTIVE_MAX_LR_MULTIPLIER")
      cfg.placement_adaptive_max_lr_multiplier = std::stod(value);
    else if (key == "PLACEMENT_ADAPTIVE_LR_UPDATE_THRESHOLD")
      cfg.placement_adaptive_lr_update_threshold = std::stod(value);
    else if (key == "PLACEMENT_ADAPTIVE_LR_FREEZE_MIN_OBSERVATIONS")
      cfg.placement_adaptive_lr_freeze_min_observations = std::stoull(value);
    else if (key == "PLACEMENT_ADAPTIVE_MIN_WEIGHT")
      cfg.placement_adaptive_min_weight = std::stod(value);
    else if (key == "PLACEMENT_ADAPTIVE_MAX_WEIGHT")
      cfg.placement_adaptive_max_weight = std::stod(value);
    else if (key == "PLACEMENT_ADAPTIVE_SEED")
      cfg.placement_adaptive_seed = static_cast<uint32_t>(std::stoul(value));
    else if (key == "ROUTE_QUEUE_WEIGHT")
      cfg.route_queue_weight = std::stod(value);
    else if (key == "ROUTE_QUEUE_NORM")
      cfg.route_queue_norm = std::stod(value);
    else if (key == "PLACEMENT_FORCE_NEW_L1_DOMAIN")
      cfg.placement_force_new_l1_domain = ParseBool(
          value, cfg.placement_force_new_l1_domain);
    else if (key == "TRACE_DISPATCH_ENABLE")
      cfg.trace_dispatch_enable = ParseBool(value, cfg.trace_dispatch_enable);
    else if (key == "TRACE_STOP_ON_COMPLETE")
      cfg.trace_stop_on_complete = ParseBool(value, cfg.trace_stop_on_complete);
    else if (key == "TRACE_USE_DEVICE_PLACEMENT")
      cfg.trace_use_device_placement = ParseBool(value, cfg.trace_use_device_placement);
    else if (key == "TRACE_ACCESS_TARGETS_AS_EXPERT_FREQ")
      cfg.trace_access_targets_as_expert_freq =
          ParseBool(value, cfg.trace_access_targets_as_expert_freq);
    else if (key == "TRACE_UNIFORM_ACCESS_TARGETS")
      cfg.trace_uniform_access_targets =
          ParseBool(value, cfg.trace_uniform_access_targets);
    else if (key == "TRACE_UNIFORM_ACCESS_TARGET_COUNT")
      cfg.trace_uniform_access_target_count = std::stoull(value);
    else if (key == "TRACE_LAYER_ID")
      cfg.trace_layer_id = static_cast<uint32_t>(std::stoul(value));
    else if (key == "TRACE_MOE_LAYER_COUNT")
      cfg.trace_moe_layer_count = static_cast<uint32_t>(std::stoul(value));
    else if (key == "TRACE_ACCESS_DIVISOR")
      cfg.trace_access_divisor =
          std::max(1u, static_cast<uint32_t>(std::stoul(value)));
    else if (key == "TRACE_DEVICE_FILE")
      cfg.trace_device_file = value;
    else if (key == "TRACE_DECODE_DIR")
      cfg.trace_decode_dir = value;
    else if (key == "CRISP_ALPHA")
      cfg.crisp_alpha = std::stod(value);
    else if (key == "CRISP_BETA")
      cfg.crisp_beta = std::stod(value);
    else if (key == "CRISP_GAMMA")
      cfg.crisp_gamma = std::stod(value);
    else if (key == "CRISP_ETA")
      cfg.crisp_eta = std::stod(value);
    else if (key == "CRISP_MIN_FRAG_GAIN")
      cfg.crisp_min_frag_gain = std::stod(value);
    else if (key == "CRISP_MAX_INTERFERENCE")
      cfg.crisp_max_interference = std::stod(value);
    else if (key == "CRISP_SLA_URGENT_THRESHOLD")
      cfg.crisp_sla_urgent_threshold = std::stod(value);
    else if (key == "CRISP_TRAINING_BOOST_SCORE")
      cfg.crisp_training_boost_score = std::stod(value);
    else if (key == "CRISP_TOPOLOGY_MIX")
      cfg.crisp_topology_mix = std::stod(value);
    else if (key == "CRISP_COMPLETION_FLOOR")
      cfg.crisp_completion_floor = std::stod(value);
    else if (key == "CRISP_REMAINING_PENALTY")
      cfg.crisp_remaining_penalty = std::stod(value);
    else if (key == "CRISP_BOOST_MAX_OUTSTANDING")
      cfg.crisp_boost_max_outstanding = static_cast<uint32_t>(std::stoul(value));
    else if (key == "CRISP_BOOST_NEED_DIVISOR")
      cfg.crisp_boost_need_divisor = static_cast<uint32_t>(std::stoul(value));
    else if (key == "CRISP_BOOST_PG")
      cfg.crisp_boost_pg = static_cast<uint16_t>(std::stoul(value));
    else if (key == "CRISP_URGENT_PG")
      cfg.crisp_urgent_pg = static_cast<uint16_t>(std::stoul(value));
  }
  return cfg;
}

inline uint32_t NetworkNodeId(const std::pair<uint32_t,uint32_t>& gpu) {
  return gpu.first * std::max(1u, gpus_per_server) + gpu.second;
}

inline std::vector<uint32_t> DirectSwitchNeighborIds(uint32_t nodeId) {
  std::vector<uint32_t> switches;
  if (nodeId >= n.GetN())
    return switches;

  Ptr<Node> gpuNode = n.Get(nodeId);
  auto it = nbr2if.find(gpuNode);
  if (it == nbr2if.end())
    return switches;

  for (const auto& kv : it->second) {
    if (!kv.second.up || kv.first->GetNodeType() != 1)
      continue;
    switches.push_back(kv.first->GetId());
  }
  std::sort(switches.begin(), switches.end());
  switches.erase(std::unique(switches.begin(), switches.end()), switches.end());
  return switches;
}

inline uint32_t SwitchSetDomainId(const std::vector<uint32_t>& switches,
                                  uint32_t fallback) {
  if (switches.empty())
    return fallback;
  if (switches.size() == 1)
    return switches.front();

  uint32_t hash = 2166136261u;
  for (uint32_t id : switches) {
    hash ^= id;
    hash *= 16777619u;
  }
  return 0x80000000u | (hash & 0x7fffffffu);
}

inline uint32_t L1GroupOfGpu(const std::pair<uint32_t,uint32_t>& gpu) {
  uint32_t node = NetworkNodeId(gpu);
  return SwitchSetDomainId(DirectSwitchNeighborIds(node), gpu.first);
}

inline double ExpertFreq(uint32_t expertId) {
  const auto& freq = State().posterior_expert_access_freq;
  if (expertId < freq.size())
    return freq[expertId];
  return 1.0;
}

inline void RefreshPosteriorExpertAccessFreq() {
  State().posterior_expert_access_freq =
      mnccl::GetExpertAccessEstimates(std::max(1u, State().config.expert_num));
}

inline std::vector<double> InitialExpertAccessFreq(uint32_t expertNum) {
  std::vector<double> freq(expertNum, 1.0);
  const auto& configured = State().config.expert_access_freq;
  for (uint32_t i = 0; i < expertNum && i < configured.size(); ++i)
    freq[i] = configured[i] > 0.0 ? configured[i] : 1.0;
  return freq;
}

inline double DistributionDrift(const std::vector<double>& a,
                                const std::vector<double>& b,
                                uint32_t expertNum) {
  if (expertNum == 0)
    return 0.0;
  double sumA = 0.0;
  double sumB = 0.0;
  for (uint32_t i = 0; i < expertNum; ++i) {
    sumA += i < a.size() ? std::max(0.0, a[i]) : 1.0;
    sumB += i < b.size() ? std::max(0.0, b[i]) : 1.0;
  }
  if (sumA <= 0.0 || sumB <= 0.0)
    return 0.0;
  double l1 = 0.0;
  for (uint32_t i = 0; i < expertNum; ++i) {
    double av = i < a.size() ? std::max(0.0, a[i]) : 1.0;
    double bv = i < b.size() ? std::max(0.0, b[i]) : 1.0;
    l1 += std::fabs(av / sumA - bv / sumB);
  }
  return 0.5 * l1;
}

inline void RefreshPlacementAccessFreqForPolicy() {
  uint32_t expertNum = std::max(1u, State().config.expert_num);
  std::string mode = Lower(Trim(State().config.placement_access_mode));
  auto initial = InitialExpertAccessFreq(expertNum);
  auto posterior = mnccl::GetExpertAccessEstimates(expertNum);
  double drift = DistributionDrift(initial, posterior, expertNum);
  uint64_t samples = mnccl::GetObservedExpertAccessSamples();

  State().latest_access_drift = drift;
  if (mode == "initial") {
    State().placement_drift_active = false;
    State().posterior_expert_access_freq = initial;
  } else if (mode == "initial_until_drift" || mode == "initial_hot_update") {
    bool active = samples >= State().config.placement_drift_min_samples &&
                  drift >= State().config.placement_drift_threshold;
    State().placement_drift_active = active;
    State().posterior_expert_access_freq = active ? posterior : initial;
  } else {
    State().placement_drift_active = false;
    State().posterior_expert_access_freq = posterior;
  }
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

struct L1GpuBalanceMetrics {
  double varianceNorm = 0.0;
  double maxNorm = 0.0;
};

inline L1GpuBalanceMetrics L1GpuBalanceForGpuLoads(
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
    const std::vector<double>& gpuLoads) {
  L1GpuBalanceMetrics out;
  if (gpus.empty() || gpuLoads.empty())
    return out;

  std::map<uint32_t, std::vector<double>> loadsByDomain;
  for (size_t i = 0; i < gpus.size() && i < gpuLoads.size(); ++i)
    loadsByDomain[L1GroupOfGpu(gpus[i])].push_back(std::max(0.0, gpuLoads[i]));

  double varNormSum = 0.0;
  uint32_t activeDomains = 0;
  for (const auto& kv : loadsByDomain) {
    const auto& loads = kv.second;
    if (loads.empty())
      continue;
    double sum = std::accumulate(loads.begin(), loads.end(), 0.0);
    double mean = sum / static_cast<double>(loads.size());
    if (mean <= 1e-12)
      continue;
    double var = Variance(loads);
    double maxLoad = *std::max_element(loads.begin(), loads.end());
    varNormSum += var / (mean * mean);
    out.maxNorm = std::max(out.maxNorm, maxLoad / mean);
    ++activeDomains;
  }
  if (activeDomains > 0)
    out.varianceNorm = varNormSum / static_cast<double>(activeDomains);
  return out;
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

inline uint64_t PairRttNs(uint32_t srcNode, uint32_t dstNode);
inline uint64_t ProbeRtt(uint32_t srcNode, uint32_t dstNode, uint16_t pg, uint64_t bytes);
inline std::vector<std::pair<uint32_t,uint32_t>> UniqueGpus(
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus);

inline std::vector<std::pair<uint32_t,uint32_t>> L1RepresentativeGpus(
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus) {
  std::map<uint32_t, std::pair<uint32_t,uint32_t>> reps;
  for (const auto& gpu : gpus) {
    uint32_t domain = L1GroupOfGpu(gpu);
    auto it = reps.find(domain);
    if (it == reps.end() || gpu < it->second)
      reps[domain] = gpu;
  }
  std::vector<std::pair<uint32_t,uint32_t>> out;
  out.reserve(reps.size());
  for (const auto& kv : reps)
    out.push_back(kv.second);
  return out;
}

inline double ProbeAwarePathCost(const std::pair<uint32_t,uint32_t>& src,
                                 const std::pair<uint32_t,uint32_t>& dst) {
  if (src == dst)
    return 0.0;

  const auto& cfg = State().config;
  uint32_t srcNode = NetworkNodeId(src);
  uint32_t dstNode = NetworkNodeId(dst);
  auto highSample = mnccl::GetRouteProbeSample(cfg.probe_high_pg, src, dst);
  auto lowSample = mnccl::GetRouteProbeSample(cfg.probe_low_pg, src, dst);
  double high = highSample.found
                    ? static_cast<double>(highSample.fctNs)
                    : static_cast<double>(
                          ProbeRtt(srcNode, dstNode, cfg.probe_high_pg, cfg.probe_bytes));
  double low = lowSample.found
                   ? static_cast<double>(lowSample.fctNs)
                   : static_cast<double>(
                         ProbeRtt(srcNode, dstNode, cfg.probe_low_pg, cfg.probe_bytes));
  if (high <= 0.0 && low <= 0.0) {
    uint64_t rtt = PairRttNs(srcNode, dstNode);
    if (rtt == 0)
      return 1e12;
    high = static_cast<double>(rtt);
    low = static_cast<double>(rtt);
  } else if (high <= 0.0) {
    high = low;
  } else if (low <= 0.0) {
    low = high;
  }
  double delta = std::fabs(low - high);
  double selected = std::min(high, low);
  return cfg.placement_probe_delta_weight * delta +
         cfg.placement_probe_rtt_weight * selected;
}

inline std::vector<std::vector<double>> BuildProbeCostByGpuAndSource(
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
    const std::vector<std::pair<uint32_t,uint32_t>>& sources) {
  std::vector<std::vector<double>> costs(
      gpus.size(), std::vector<double>(sources.size(), 0.0));
  for (size_t i = 0; i < gpus.size(); ++i) {
    for (size_t j = 0; j < sources.size(); ++j)
      costs[i][j] = ProbeAwarePathCost(sources[j], gpus[i]);
  }
  return costs;
}

inline double ProbeCostScale(const std::vector<std::vector<double>>& costs) {
  double scale = 0.0;
  for (const auto& row : costs) {
    for (double value : row) {
      if (std::isfinite(value))
        scale = std::max(scale, value);
    }
  }
  return std::max(1.0, scale);
}

inline double NormalizedProbeCost(double value, double scale) {
  if (!std::isfinite(value))
    return 1.0;
  return std::min(1.0, std::max(0.0, value / std::max(1.0, scale)));
}

inline double ProbePathCostForPlan(
    const std::vector<std::vector<std::pair<uint32_t,uint32_t>>>& plan,
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus) {
  if (plan.empty() || gpus.empty())
    return 0.0;

  std::vector<std::pair<uint32_t,uint32_t>> orderedGpus = UniqueGpus(gpus);
  std::vector<std::pair<uint32_t,uint32_t>> sources = orderedGpus;
  if (orderedGpus.empty() || sources.empty())
    return 0.0;

  std::map<std::pair<uint32_t,uint32_t>, size_t> gpuIndex;
  for (size_t i = 0; i < orderedGpus.size(); ++i)
    gpuIndex[orderedGpus[i]] = i;
  auto costs = BuildProbeCostByGpuAndSource(orderedGpus, sources);
  double scale = ProbeCostScale(costs);

  double total = 0.0;
  for (uint32_t expert = 0; expert < plan.size(); ++expert) {
    if (plan[expert].empty())
      continue;
    double expertCost = 0.0;
    for (size_t src = 0; src < sources.size(); ++src) {
      double best = std::numeric_limits<double>::max();
      for (const auto& gpu : plan[expert]) {
        auto it = gpuIndex.find(gpu);
        if (it == gpuIndex.end())
          continue;
        best = std::min(best, costs[it->second][src]);
      }
      expertCost += NormalizedProbeCost(best, scale);
    }
    total += ExpertFreq(expert) * expertCost / static_cast<double>(sources.size());
  }
  return total;
}

inline bool IsProbeBalancedPlacementPolicy(const std::string& policy) {
  return policy == "probe_balanced_domain_spread" ||
         policy == "probe_balanced_domain_spread_base" ||
         policy == "probe_balanced_hot_surplus" ||
         policy == "probe_balanced_probe_cost" ||
         policy == "probe_balanced_topology_diverse" ||
         policy == "probe_balanced_partition_decay" ||
         policy == "probe_balanced_full";
}

inline bool IsIncrementalPlacementPolicy(const std::string& policy) {
  return policy == "probe_balanced_incremental" ||
         policy == "probe_balanced_multiplicative" ||
         policy == "probe_balanced_topology_multiplicative" ||
         policy == "probe_balanced_topology_static_prior";
}

struct ExpertPlanScore {
  double maxL1Load = std::numeric_limits<double>::max();
  double l1Variance = std::numeric_limits<double>::max();
  double topologyCost = std::numeric_limits<double>::max();
  double externalAccessCost = std::numeric_limits<double>::max();
  double replicaDomainImbalance = std::numeric_limits<double>::max();
  double gpuLoadVariance = std::numeric_limits<double>::max();
  double probePathCost = std::numeric_limits<double>::max();
  double hop1Max = std::numeric_limits<double>::max();
  double hop1Mean = std::numeric_limits<double>::max();
  double hop2Max = std::numeric_limits<double>::max();
  double hop2Mean = std::numeric_limits<double>::max();
  double l1GpuBalanceVariance = std::numeric_limits<double>::max();
  double l1GpuBalanceMax = std::numeric_limits<double>::max();
  double objective = std::numeric_limits<double>::max();
};

struct NeedCandidateScore {
  bool feasible = false;
  uint32_t need = 0;
  ExpertPlanScore plan;
};

inline bool BetterPlanScore(const ExpertPlanScore& a, const ExpertPlanScore& b) {
  if (std::fabs(a.objective - b.objective) > 1e-12)
    return a.objective < b.objective;
  if (State().config.placement_probe_weight > 0.0 &&
      a.probePathCost != b.probePathCost)
    return a.probePathCost < b.probePathCost;
  if (a.externalAccessCost != b.externalAccessCost)
    return a.externalAccessCost < b.externalAccessCost;
  if (a.replicaDomainImbalance != b.replicaDomainImbalance)
    return a.replicaDomainImbalance < b.replicaDomainImbalance;
  if (a.l1GpuBalanceVariance != b.l1GpuBalanceVariance)
    return a.l1GpuBalanceVariance < b.l1GpuBalanceVariance;
  if (a.l1GpuBalanceMax != b.l1GpuBalanceMax)
    return a.l1GpuBalanceMax < b.l1GpuBalanceMax;
  if (a.l1Variance != b.l1Variance)
    return a.l1Variance < b.l1Variance;
  if (a.maxL1Load != b.maxL1Load)
    return a.maxL1Load < b.maxL1Load;
  if (a.gpuLoadVariance != b.gpuLoadVariance)
    return a.gpuLoadVariance < b.gpuLoadVariance;
  if (a.hop1Max != b.hop1Max)
    return a.hop1Max < b.hop1Max;
  if (a.hop1Mean != b.hop1Mean)
    return a.hop1Mean < b.hop1Mean;
  if (a.hop2Max != b.hop2Max)
    return a.hop2Max < b.hop2Max;
  if (a.hop2Mean != b.hop2Mean)
    return a.hop2Mean < b.hop2Mean;
  return a.topologyCost < b.topologyCost;
}

inline bool BetterNeedScore(const NeedCandidateScore& a, const NeedCandidateScore& b) {
  if (a.feasible != b.feasible)
    return a.feasible;
  if (!a.feasible)
    return false;
  if (a.plan.objective != b.plan.objective)
    return a.plan.objective < b.plan.objective;
  if (a.plan.replicaDomainImbalance != b.plan.replicaDomainImbalance)
    return a.plan.replicaDomainImbalance < b.plan.replicaDomainImbalance;
  if (a.plan.l1GpuBalanceVariance != b.plan.l1GpuBalanceVariance)
    return a.plan.l1GpuBalanceVariance < b.plan.l1GpuBalanceVariance;
  if (a.plan.l1GpuBalanceMax != b.plan.l1GpuBalanceMax)
    return a.plan.l1GpuBalanceMax < b.plan.l1GpuBalanceMax;
  if (a.plan.l1Variance != b.plan.l1Variance)
    return a.plan.l1Variance < b.plan.l1Variance;
  if (a.plan.maxL1Load != b.plan.maxL1Load)
    return a.plan.maxL1Load < b.plan.maxL1Load;
  if (a.plan.gpuLoadVariance != b.plan.gpuLoadVariance)
    return a.plan.gpuLoadVariance < b.plan.gpuLoadVariance;
  if (a.plan.hop1Max != b.plan.hop1Max)
    return a.plan.hop1Max < b.plan.hop1Max;
  if (a.plan.hop1Mean != b.plan.hop1Mean)
    return a.plan.hop1Mean < b.plan.hop1Mean;
  if (a.plan.hop2Max != b.plan.hop2Max)
    return a.plan.hop2Max < b.plan.hop2Max;
  if (a.plan.hop2Mean != b.plan.hop2Mean)
    return a.plan.hop2Mean < b.plan.hop2Mean;
  if (a.plan.topologyCost != b.plan.topologyCost)
    return a.plan.topologyCost < b.plan.topologyCost;
  return a.need < b.need;
}

inline std::vector<uint32_t> L1DomainsForGpus(
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus) {
  std::set<uint32_t> uniqueDomains;
  for (const auto& gpu : gpus)
    uniqueDomains.insert(L1GroupOfGpu(gpu));
  return std::vector<uint32_t>(uniqueDomains.begin(), uniqueDomains.end());
}

inline std::vector<std::pair<uint32_t,uint32_t>> UniqueGpus(
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus) {
  std::set<std::pair<uint32_t,uint32_t>> unique(gpus.begin(), gpus.end());
  return std::vector<std::pair<uint32_t,uint32_t>>(unique.begin(), unique.end());
}

struct HopHeatMetrics {
  std::array<double, 2> max{{0.0, 0.0}};
  std::array<double, 2> mean{{0.0, 0.0}};
};

inline std::vector<uint32_t> HopDistancesFromNode(uint32_t srcNode,
                                                  uint32_t maxHop) {
  uint32_t totalNodes = static_cast<uint32_t>(n.GetN());
  std::vector<uint32_t> dist(totalNodes, maxHop + 1);
  if (srcNode >= totalNodes)
    return dist;
  std::queue<uint32_t> q;
  dist[srcNode] = 0;
  q.push(srcNode);
  while (!q.empty()) {
    uint32_t nodeId = q.front();
    q.pop();
    if (dist[nodeId] >= maxHop)
      continue;
    Ptr<Node> node = n.Get(nodeId);
    for (const auto& kv : nbr2if[node]) {
      if (!kv.second.up)
        continue;
      uint32_t nextId = kv.first->GetId();
      if (nextId >= totalNodes || dist[nextId] <= dist[nodeId] + 1)
        continue;
      dist[nextId] = dist[nodeId] + 1;
      q.push(nextId);
    }
  }
  return dist;
}

inline uint64_t PairRttNs(uint32_t srcNode, uint32_t dstNode) {
  if (srcNode == dstNode)
    return 0;
  auto outer = pairRtt.find(srcNode);
  if (outer == pairRtt.end())
    return 0;
  auto inner = outer->second.find(dstNode);
  if (inner == outer->second.end())
    return 0;
  return inner->second;
}

inline uint64_t RttTierTolerance(uint64_t rtt) {
  return std::max<uint64_t>(1, rtt / 20);
}

inline std::vector<uint64_t> TopologyRttTierBoundaries(
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
    uint32_t maxHop) {
  std::vector<uint64_t> rtts;
  for (size_t i = 0; i < gpus.size(); ++i) {
    uint32_t src = NetworkNodeId(gpus[i]);
    for (size_t j = i + 1; j < gpus.size(); ++j) {
      uint64_t rtt = PairRttNs(src, NetworkNodeId(gpus[j]));
      if (rtt > 0)
        rtts.push_back(rtt);
    }
  }
  if (rtts.empty())
    return {};

  std::sort(rtts.begin(), rtts.end());
  std::vector<uint64_t> classes;
  for (uint64_t rtt : rtts) {
    if (classes.empty()) {
      classes.push_back(rtt);
      continue;
    }
    uint64_t tolerance = RttTierTolerance(classes.back());
    if (rtt > classes.back() + tolerance) {
      classes.push_back(rtt);
    } else {
      classes.back() = std::max(classes.back(), rtt);
    }
  }
  if (classes.size() > maxHop)
    classes.resize(maxHop);
  return classes;
}

inline uint32_t RttTierDistance(uint32_t srcNode,
                                uint32_t dstNode,
                                const std::vector<uint64_t>& boundaries,
                                uint32_t maxHop) {
  if (srcNode == dstNode)
    return 0;
  uint64_t rtt = PairRttNs(srcNode, dstNode);
  if (rtt == 0 || boundaries.empty())
    return maxHop + 1;
  for (size_t i = 0; i < boundaries.size() && i < maxHop; ++i) {
    if (rtt <= boundaries[i] + RttTierTolerance(boundaries[i]))
      return static_cast<uint32_t>(i + 1);
  }
  return maxHop + 1;
}

inline std::vector<std::vector<uint32_t>> BuildHopDistanceBySourceAndGpu(
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
    uint32_t maxHop = 2) {
  std::vector<std::vector<uint32_t>> distances(
      gpus.size(), std::vector<uint32_t>(gpus.size(), maxHop + 1));
  std::vector<uint64_t> rttBoundaries =
      TopologyRttTierBoundaries(gpus, maxHop);
  if (!rttBoundaries.empty()) {
    for (size_t src = 0; src < gpus.size(); ++src) {
      uint32_t srcNode = NetworkNodeId(gpus[src]);
      for (size_t dst = 0; dst < gpus.size(); ++dst) {
        distances[src][dst] = RttTierDistance(
            srcNode, NetworkNodeId(gpus[dst]), rttBoundaries, maxHop);
      }
    }
    return distances;
  }

  for (size_t src = 0; src < gpus.size(); ++src) {
    auto nodeDist = HopDistancesFromNode(NetworkNodeId(gpus[src]), maxHop);
    for (size_t dst = 0; dst < gpus.size(); ++dst) {
      uint32_t nodeId = NetworkNodeId(gpus[dst]);
      distances[src][dst] =
          nodeId < nodeDist.size() ? nodeDist[nodeId] : maxHop + 1;
    }
  }
  return distances;
}

inline HopHeatMetrics HopHeatMetricsFromHeat(
    const std::array<std::vector<double>, 2>& heatByHop,
    double normalizer) {
  HopHeatMetrics out;
  normalizer = std::max(1.0, normalizer);
  for (size_t hop = 0; hop < heatByHop.size(); ++hop) {
    const auto& heat = heatByHop[hop];
    if (heat.empty())
      continue;
    double sum = std::accumulate(heat.begin(), heat.end(), 0.0);
    double mean = sum / static_cast<double>(heat.size());
    if (mean <= 1e-12)
      continue;
    double maxHeat = *std::max_element(heat.begin(), heat.end());
    out.max[hop] = maxHeat / normalizer;
    out.mean[hop] = mean / normalizer;
  }
  return out;
}

inline HopHeatMetrics HopHeatMetricsForPlan(
    const std::vector<std::vector<std::pair<uint32_t,uint32_t>>>& plan,
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus) {
  std::vector<std::pair<uint32_t,uint32_t>> orderedGpus = UniqueGpus(gpus);
  std::array<std::vector<double>, 2> heatByHop;
  for (auto& heat : heatByHop)
    heat.assign(orderedGpus.size(), 0.0);
  if (plan.empty() || orderedGpus.empty())
    return HopHeatMetricsFromHeat(heatByHop, 1.0);

  std::map<std::pair<uint32_t,uint32_t>, size_t> gpuIndex;
  for (size_t i = 0; i < orderedGpus.size(); ++i)
    gpuIndex[orderedGpus[i]] = i;
  auto hopDistances = BuildHopDistanceBySourceAndGpu(orderedGpus, 2);

  double totalFreq = 0.0;
  for (uint32_t expert = 0; expert < plan.size(); ++expert) {
    if (plan[expert].empty())
      continue;
    double freq = ExpertFreq(expert);
    totalFreq += freq;
    for (size_t src = 0; src < orderedGpus.size(); ++src) {
      uint32_t bestHop = 3;
      for (const auto& gpu : plan[expert]) {
        auto it = gpuIndex.find(gpu);
        if (it == gpuIndex.end())
          continue;
        bestHop = std::min(bestHop, hopDistances[src][it->second]);
      }
      for (uint32_t hop = 1; hop <= 2; ++hop) {
        if (bestHop <= hop)
          heatByHop[hop - 1][src] += freq;
      }
    }
  }
  return HopHeatMetricsFromHeat(heatByHop, totalFreq);
}

inline double OfferedInferenceTokenRate(const PolicyConfig& cfg) {
  double activeWindow = std::max(1e-9, cfg.simulation_stop_time - cfg.first_submit_time);
  return static_cast<double>(cfg.num_tasks) *
         static_cast<double>(std::max(1u, cfg.dispatch_n)) / activeWindow;
}

inline bool MultiplicativePlacementScorePolicy() {
  std::string policy = Lower(Trim(State().config.expert_placement_policy));
  return policy == "probe_balanced_multiplicative" ||
         policy == "probe_balanced_topology_multiplicative";
}

inline bool TopologyStaticPriorPlacementPolicy() {
  std::string policy = Lower(Trim(State().config.expert_placement_policy));
  return policy == "probe_balanced_topology_static_prior";
}

struct TopologyStaticPriorStats {
  uint32_t gpuCount = 0;
  uint32_t domainCount = 0;
  double replicasPerExpert = 1.0;
  double coverageShortage = 0.0;
  double rttIntraMedian = 0.0;
  double rttInterMedian = 0.0;
  double rttGap = 0.0;
  double rttSpread = 0.0;
  double domainCv = 0.0;
  double intraBalanceNeed = 0.0;
  double externalLowerBound = 0.0;
  double replicaLowerBound = 0.0;
  double externalTau = 1.0;
  double replicaTau = 1.0;
  double l1VarianceTau = 1.0;
  double l1MaxTau = 1.0;
  double gpuVarianceTau = 1.0;
  double probeTau = 1.0;
  std::vector<double> weights;
};

inline double ClampUnit(double value) {
  if (!std::isfinite(value))
    return 0.0;
  return std::min(1.0, std::max(0.0, value));
}

inline double PercentileSorted(const std::vector<double>& sortedValues,
                               double quantile) {
  if (sortedValues.empty())
    return 0.0;
  quantile = ClampUnit(quantile);
  double pos = quantile * static_cast<double>(sortedValues.size() - 1);
  size_t lo = static_cast<size_t>(std::floor(pos));
  size_t hi = std::min(sortedValues.size() - 1, lo + 1);
  double frac = pos - static_cast<double>(lo);
  return sortedValues[lo] * (1.0 - frac) + sortedValues[hi] * frac;
}

inline double PlacementWeightBudget() {
  const auto& cfg = State().config;
  double sum = cfg.placement_replica_balance_weight +
               cfg.placement_external_weight +
               cfg.placement_l1_variance_weight +
               cfg.placement_l1_max_weight +
               cfg.placement_gpu_variance_weight +
               cfg.placement_probe_weight;
  return sum > 1e-12 ? sum : 24.0;
}

inline TopologyStaticPriorStats ComputeTopologyStaticPriorStats(
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
    uint32_t expertNum,
    uint32_t expertsPerGpu) {
  TopologyStaticPriorStats stats;
  std::vector<std::pair<uint32_t,uint32_t>> orderedGpus = UniqueGpus(gpus);
  stats.gpuCount = static_cast<uint32_t>(orderedGpus.size());
  std::vector<uint32_t> domains = L1DomainsForGpus(orderedGpus);
  stats.domainCount = static_cast<uint32_t>(domains.size());
  if (stats.gpuCount == 0 || stats.domainCount == 0 || expertNum == 0) {
    const auto& cfg = State().config;
    stats.weights = {
        cfg.placement_replica_balance_weight,
        cfg.placement_external_weight,
        cfg.placement_l1_variance_weight,
        cfg.placement_l1_max_weight,
        cfg.placement_gpu_variance_weight,
        cfg.placement_probe_weight};
    return stats;
  }

  stats.replicasPerExpert =
      static_cast<double>(stats.gpuCount) *
      static_cast<double>(std::max(1u, expertsPerGpu)) /
      static_cast<double>(expertNum);
  double coverageRatio = std::min(
      1.0, stats.replicasPerExpert / static_cast<double>(stats.domainCount));
  stats.coverageShortage = 1.0 - coverageRatio;

  std::map<uint32_t, double> domainGpuCounts;
  for (uint32_t domain : domains)
    domainGpuCounts[domain] = 0.0;
  for (const auto& gpu : orderedGpus)
    domainGpuCounts[L1GroupOfGpu(gpu)] += 1.0;
  std::vector<double> counts;
  counts.reserve(domainGpuCounts.size());
  for (const auto& kv : domainGpuCounts)
    counts.push_back(kv.second);
  double meanCount = std::accumulate(counts.begin(), counts.end(), 0.0) /
                     static_cast<double>(std::max<size_t>(1, counts.size()));
  stats.domainCv = meanCount > 1e-12
                       ? std::sqrt(std::max(0.0, Variance(counts))) / meanCount
                       : 0.0;

  std::vector<double> intraRtts;
  std::vector<double> interRtts;
  for (size_t i = 0; i < orderedGpus.size(); ++i) {
    uint32_t src = NetworkNodeId(orderedGpus[i]);
    uint32_t srcDomain = L1GroupOfGpu(orderedGpus[i]);
    for (size_t j = i + 1; j < orderedGpus.size(); ++j) {
      uint64_t rtt = PairRttNs(src, NetworkNodeId(orderedGpus[j]));
      if (rtt == 0)
        continue;
      if (srcDomain == L1GroupOfGpu(orderedGpus[j]))
        intraRtts.push_back(static_cast<double>(rtt));
      else
        interRtts.push_back(static_cast<double>(rtt));
    }
  }
  std::sort(intraRtts.begin(), intraRtts.end());
  std::sort(interRtts.begin(), interRtts.end());
  stats.rttIntraMedian = PercentileSorted(intraRtts, 0.5);
  stats.rttInterMedian = PercentileSorted(interRtts, 0.5);
  if (stats.rttIntraMedian <= 0.0 && stats.rttInterMedian > 0.0)
    stats.rttIntraMedian = stats.rttInterMedian;
  if (stats.rttInterMedian <= 0.0)
    stats.rttInterMedian = std::max(1.0, stats.rttIntraMedian);
  stats.rttGap = ClampUnit(
      (stats.rttInterMedian - stats.rttIntraMedian) /
      std::max(1.0, stats.rttInterMedian));
  double interP10 = PercentileSorted(interRtts, 0.10);
  double interP90 = PercentileSorted(interRtts, 0.90);
  stats.rttSpread = ClampUnit(
      ((interP90 - interP10) / std::max(1.0, stats.rttInterMedian)) / 2.0);

  double intraSize = static_cast<double>(stats.gpuCount) /
                     static_cast<double>(stats.domainCount);
  stats.intraBalanceNeed =
      ClampUnit(std::log2(std::max(1.0, intraSize)) / 3.0) *
      (1.0 - stats.rttGap);

  double r = stats.replicasPerExpert;
  double d = static_cast<double>(stats.domainCount);
  stats.externalLowerBound = std::max(0.0, d - std::min(d, r));
  double idealFrac = (d > 0.0) ? (r / d - std::floor(r / d)) : 0.0;
  stats.replicaLowerBound = idealFrac * (1.0 - idealFrac);
  stats.externalTau = std::max(1.0, d * std::max(0.10, stats.coverageShortage));
  stats.replicaTau = std::max(0.05, stats.replicaLowerBound + 1.0 / d);
  stats.l1VarianceTau = 1.0;
  stats.l1MaxTau = 1.0;
  stats.gpuVarianceTau = 1.0;
  stats.probeTau = std::max(0.5, 1.0 + stats.rttSpread);

  std::vector<double> raw{
      3.0 + 4.0 * stats.coverageShortage + 2.0 * stats.rttGap,
      3.0 + 8.0 * stats.rttGap + 6.0 * stats.coverageShortage +
          3.0 * stats.rttSpread,
      0.5 + 2.0 * stats.intraBalanceNeed + 1.0 * stats.domainCv,
      0.8 + 3.0 * stats.intraBalanceNeed + 2.0 * stats.domainCv,
      0.5 + 1.5 * stats.intraBalanceNeed + 1.0 * stats.domainCv,
      2.0 + 7.0 * stats.rttGap + 4.0 * stats.rttSpread +
          2.0 * stats.coverageShortage};

  if (stats.rttGap > 0.6 && raw.size() >= 6) {
    raw[1] = std::max(raw[1], raw[0]);
    raw[5] = std::max(raw[5], 0.6 * raw[1]);
    double balance = raw[2] + raw[3] + raw[4];
    double balanceLimit = 0.35 * (raw[0] + raw[1] + raw[5]);
    if (balance > balanceLimit && balance > 1e-12) {
      double scale = balanceLimit / balance;
      raw[2] *= scale;
      raw[3] *= scale;
      raw[4] *= scale;
    }
  }

  double rawSum = std::accumulate(raw.begin(), raw.end(), 0.0);
  double target = PlacementWeightBudget();
  if (rawSum > 1e-12) {
    for (double& value : raw)
      value *= target / rawSum;
  }
  stats.weights = raw;
  return stats;
}

inline double SaturatedViolation(double metric, double lowerBound, double tau) {
  if (!std::isfinite(metric))
    return 1.0;
  double x = std::max(0.0, metric - lowerBound) / std::max(1e-9, tau);
  return std::log1p(x);
}

inline double PlacementTopologyStaticPriorObjective(
    double replicaNorm,
    double externalNorm,
    double l1VarNorm,
    double l1MaxNorm,
    double gpuVarNorm,
    double probeNorm,
    const TopologyStaticPriorStats& stats) {
  const auto& cfg = State().config;
  return cfg.placement_replica_balance_weight *
             SaturatedViolation(replicaNorm, stats.replicaLowerBound, stats.replicaTau) +
         cfg.placement_external_weight *
             SaturatedViolation(externalNorm, stats.externalLowerBound, stats.externalTau) +
         cfg.placement_l1_variance_weight *
             SaturatedViolation(l1VarNorm, 0.0, stats.l1VarianceTau) +
         cfg.placement_l1_max_weight *
             SaturatedViolation(l1MaxNorm, 1.0, stats.l1MaxTau) +
         cfg.placement_gpu_variance_weight *
             SaturatedViolation(gpuVarNorm, 0.0, stats.gpuVarianceTau) +
         cfg.placement_probe_weight *
             SaturatedViolation(probeNorm, 0.0, stats.probeTau);
}

inline void ApplyTopologyStaticPriorWeights(
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
    uint32_t expertNum,
    uint32_t expertsPerGpu) {
  if (!TopologyStaticPriorPlacementPolicy())
    return;
  TopologyStaticPriorStats stats =
      ComputeTopologyStaticPriorStats(gpus, expertNum, expertsPerGpu);
  if (stats.weights.size() >= 6) {
    auto& cfg = State().config;
    cfg.placement_replica_balance_weight = stats.weights[0];
    cfg.placement_external_weight = stats.weights[1];
    cfg.placement_l1_variance_weight = stats.weights[2];
    cfg.placement_l1_max_weight = stats.weights[3];
    cfg.placement_gpu_variance_weight = stats.weights[4];
    cfg.placement_probe_weight = stats.weights[5];
  }

  std::ostringstream ss;
  ss << "PipelinePolicy: topology static prior weights"
     << " domains=" << stats.domainCount
     << " gpus=" << stats.gpuCount
     << " replicas_per_expert=" << stats.replicasPerExpert
     << " coverage_shortage=" << stats.coverageShortage
     << " rtt_intra_median=" << stats.rttIntraMedian
     << " rtt_inter_median=" << stats.rttInterMedian
     << " rtt_gap=" << stats.rttGap
     << " rtt_spread=" << stats.rttSpread
     << " domain_cv=" << stats.domainCv
     << " intra_balance_need=" << stats.intraBalanceNeed
     << " external_lb=" << stats.externalLowerBound
     << " replica_lb=" << stats.replicaLowerBound
     << " weights=[";
  for (size_t i = 0; i < stats.weights.size(); ++i) {
    if (i > 0)
      ss << ",";
    ss << stats.weights[i];
  }
  ss << "]";
  ccl::CclLog(ss.str());
}

inline double PlacementLinearObjective(double replicaNorm,
                                       double externalNorm,
                                       double l1VarNorm,
                                       double l1MaxNorm,
                                       double gpuVarNorm,
                                       double probeNorm) {
  const auto& cfg = State().config;
  return cfg.placement_replica_balance_weight * replicaNorm +
         cfg.placement_external_weight * externalNorm +
         cfg.placement_l1_variance_weight * l1VarNorm +
         cfg.placement_l1_max_weight * l1MaxNorm +
         cfg.placement_gpu_variance_weight * gpuVarNorm +
         cfg.placement_probe_weight * probeNorm;
}

inline double PlacementMultiplicativeObjective(double replicaNorm,
                                               double externalNorm,
                                               double l1VarNorm,
                                               double l1MaxNorm,
                                               double gpuVarNorm,
                                               double probeNorm) {
  const auto& cfg = State().config;
  auto logPenalty = [](double weight, double metric) {
    if (weight <= 0.0 || metric <= 0.0 || !std::isfinite(metric))
      return 0.0;
    return std::log1p(weight * metric);
  };
  return logPenalty(cfg.placement_probe_weight, probeNorm) +
         logPenalty(cfg.placement_replica_balance_weight, replicaNorm) +
         logPenalty(cfg.placement_external_weight, externalNorm) +
         logPenalty(cfg.placement_l1_variance_weight, l1VarNorm) +
         logPenalty(cfg.placement_l1_max_weight, l1MaxNorm) +
         logPenalty(cfg.placement_gpu_variance_weight, gpuVarNorm);
}

inline double PlacementObjective(double replicaNorm,
                                 double externalNorm,
                                 double l1VarNorm,
                                 double l1MaxNorm,
                                 double gpuVarNorm,
                                 double probeNorm) {
  if (MultiplicativePlacementScorePolicy()) {
    return PlacementMultiplicativeObjective(
        replicaNorm, externalNorm, l1VarNorm, l1MaxNorm, gpuVarNorm, probeNorm);
  }
  return PlacementLinearObjective(
      replicaNorm, externalNorm, l1VarNorm, l1MaxNorm, gpuVarNorm, probeNorm);
}

inline ExpertPlanScore ScoreExpertPlan(
    const std::vector<std::vector<std::pair<uint32_t,uint32_t>>>& plan,
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
    const std::map<std::pair<uint32_t,uint32_t>, uint32_t>* gpuLoadOverride = nullptr) {
  ExpertPlanScore score;
  if (gpus.empty() || plan.empty())
    return score;
  (void)gpuLoadOverride;

  std::vector<uint32_t> domains = L1DomainsForGpus(gpus);
  std::map<uint32_t, double> l1Load;
  for (uint32_t domain : domains)
    l1Load[domain] += 0.0;

  std::map<std::pair<uint32_t,uint32_t>, double> gpuLoad;
  for (const auto& gpu : gpus)
    gpuLoad[gpu] += 0.0;

  double totalFreq = 0.0;
  double externalAccessCost = 0.0;
  double replicaDomainImbalance = 0.0;

  for (uint32_t expert = 0; expert < plan.size(); ++expert) {
    double freq = ExpertFreq(expert);
    totalFreq += freq;
    std::map<uint32_t, double> domainReplicaCount;
    for (uint32_t domain : domains)
      domainReplicaCount[domain] += 0.0;

    for (const auto& gpu : plan[expert]) {
      uint32_t domain = L1GroupOfGpu(gpu);
      l1Load[domain] += freq;
      domainReplicaCount[domain] += 1.0;
      gpuLoad[gpu] += freq;
    }

    uint32_t coveredDomains = 0;
    std::vector<double> counts;
    counts.reserve(domains.size());
    for (uint32_t domain : domains) {
      double count = domainReplicaCount[domain];
      if (count > 0.0)
        coveredDomains++;
      counts.push_back(count);
    }
    if (!domains.empty()) {
      externalAccessCost += freq * static_cast<double>(domains.size() - coveredDomains);
      replicaDomainImbalance += freq * Variance(counts);
    }
  }

  std::vector<double> gpuLoads;
  gpuLoads.reserve(gpus.size());
  for (const auto& gpu : gpus)
    gpuLoads.push_back(gpuLoad[gpu]);
  L1GpuBalanceMetrics l1GpuBalance =
      L1GpuBalanceForGpuLoads(gpus, gpuLoads);

  score.maxL1Load = MaxL1Load(l1Load);
  score.l1Variance = L1LoadVariance(l1Load);
  score.topologyCost = TopologyDistanceScore(gpus);
  score.externalAccessCost = externalAccessCost;
  score.replicaDomainImbalance = replicaDomainImbalance;
  score.gpuLoadVariance = Variance(gpuLoads);
  score.probePathCost = ProbePathCostForPlan(plan, gpus);
  auto hopMetrics = HopHeatMetricsForPlan(plan, gpus);
  score.hop1Max = hopMetrics.max[0];
  score.hop1Mean = hopMetrics.mean[0];
  score.hop2Max = hopMetrics.max[1];
  score.hop2Mean = hopMetrics.mean[1];
  score.l1GpuBalanceVariance = l1GpuBalance.varianceNorm;
  score.l1GpuBalanceMax = l1GpuBalance.maxNorm;

  double externalNorm = totalFreq > 0.0
                            ? externalAccessCost / totalFreq
                            : 0.0;
  double replicaNorm = totalFreq > 0.0 ? replicaDomainImbalance / totalFreq : 0.0;
  double probeNorm = totalFreq > 0.0
                         ? score.probePathCost / totalFreq
                         : 0.0;
  double gpuMean = gpuLoads.empty()
                       ? 0.0
                       : std::accumulate(gpuLoads.begin(), gpuLoads.end(), 0.0) /
                             static_cast<double>(gpuLoads.size());
  double gpuVarNorm = gpuMean > 0.0 ? score.gpuLoadVariance / (gpuMean * gpuMean) : 0.0;

  if (TopologyStaticPriorPlacementPolicy()) {
    TopologyStaticPriorStats staticPriorStats =
        ComputeTopologyStaticPriorStats(
            gpus,
            static_cast<uint32_t>(plan.size()),
            State().config.expert_per_gpu);
    score.objective = PlacementTopologyStaticPriorObjective(
        replicaNorm,
        externalNorm,
        score.l1GpuBalanceVariance,
        score.l1GpuBalanceMax,
        gpuVarNorm,
        probeNorm,
        staticPriorStats);
  } else {
    score.objective = PlacementObjective(
        replicaNorm,
        externalNorm,
        score.l1GpuBalanceVariance,
        score.l1GpuBalanceMax,
        gpuVarNorm,
        probeNorm);
  }
  return score;
}

inline std::vector<double> PlacementWeightVectorFromConfig() {
  const auto& cfg = State().config;
  return {
      cfg.placement_replica_balance_weight,
      cfg.placement_external_weight,
      cfg.placement_l1_variance_weight,
      cfg.placement_l1_max_weight,
      cfg.placement_gpu_variance_weight,
      cfg.placement_probe_weight};
}

inline void ApplyPlacementWeightVector(const std::vector<double>& weights) {
  if (weights.size() < 6)
    return;
  auto& cfg = State().config;
  cfg.placement_replica_balance_weight = weights[0];
  cfg.placement_external_weight = weights[1];
  cfg.placement_l1_variance_weight = weights[2];
  cfg.placement_l1_max_weight = weights[3];
  cfg.placement_gpu_variance_weight = weights[4];
  cfg.placement_probe_weight = weights[5];
}

inline double ClampPlacementWeight(double value) {
  const auto& cfg = State().config;
  double lo = std::max(0.0, cfg.placement_adaptive_min_weight);
  double hi = std::max(lo, cfg.placement_adaptive_max_weight);
  return std::min(hi, std::max(lo, value));
}

inline std::vector<double> ClampPlacementWeightVector(std::vector<double> weights) {
  for (double& value : weights)
    value = ClampPlacementWeight(value);
  return weights;
}

inline std::vector<double> NormalizePlacementWeightVector(std::vector<double> weights,
                                                          double targetSum) {
  weights = ClampPlacementWeightVector(weights);
  if (targetSum <= 0.0)
    return weights;
  double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (sum <= 1e-12)
    return weights;
  double scale = targetSum / sum;
  for (double& value : weights)
    value *= scale;
  return ClampPlacementWeightVector(weights);
}

inline std::vector<double> NormalizePlacementWeightVectorWithFrozen(
    std::vector<double> weights,
    double targetSum,
    const std::vector<uint8_t>& frozen) {
  weights = ClampPlacementWeightVector(weights);
  if (targetSum <= 0.0 || frozen.size() != weights.size())
    return weights;

  double frozenSum = 0.0;
  double activeSum = 0.0;
  for (size_t i = 0; i < weights.size(); ++i) {
    if (frozen[i] != 0)
      frozenSum += weights[i];
    else
      activeSum += weights[i];
  }
  double activeTarget = targetSum - frozenSum;
  if (activeTarget <= 1e-12 || activeSum <= 1e-12)
    return weights;

  double scale = activeTarget / activeSum;
  for (size_t i = 0; i < weights.size(); ++i) {
    if (frozen[i] == 0)
      weights[i] = ClampPlacementWeight(weights[i] * scale);
  }
  return weights;
}

inline std::vector<double> PlacementAdaptiveDirection(uint64_t step) {
  uint64_t state = static_cast<uint64_t>(State().config.placement_adaptive_seed) ^
                   (step * 0x9e3779b97f4a7c15ULL);
  std::vector<double> direction(PlacementWeightVectorFromConfig().size(), 1.0);
  for (size_t i = 0; i < direction.size(); ++i) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    uint64_t mixed = state * 0x2545F4914F6CDD1DULL;
    direction[i] = (mixed & 1ULL) ? 1.0 : -1.0;
  }
  return direction;
}

inline std::vector<double> PerturbPlacementWeights(const std::vector<double>& base,
                                                   const std::vector<double>& direction,
                                                   double sign) {
  std::vector<double> out = base;
  double sigma = std::max(1e-9, State().config.placement_adaptive_sigma);
  for (size_t i = 0; i < out.size() && i < direction.size(); ++i)
    out[i] += sign * sigma * direction[i] * std::max(1.0, std::fabs(base[i]));
  return ClampPlacementWeightVector(out);
}

inline std::vector<double> PlacementMetricVector(
    const ExpertPlanScore& score,
    const std::vector<std::vector<std::pair<uint32_t,uint32_t>>>& plan,
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus) {
  if (plan.empty() || gpus.empty())
    return std::vector<double>(6, std::numeric_limits<double>::max());

  double totalFreq = 0.0;
  double totalL1Load = 0.0;
  for (uint32_t expert = 0; expert < plan.size(); ++expert) {
    double freq = ExpertFreq(expert);
    totalFreq += freq;
    totalL1Load += freq * static_cast<double>(plan[expert].size());
  }

  double l1VarNorm = std::isfinite(score.l1GpuBalanceVariance)
                         ? score.l1GpuBalanceVariance
                         : 0.0;
  double l1MaxNorm = std::isfinite(score.l1GpuBalanceMax)
                         ? score.l1GpuBalanceMax
                         : 0.0;
  double externalNorm = totalFreq > 0.0
                            ? score.externalAccessCost / totalFreq
                            : 0.0;
  double replicaNorm = totalFreq > 0.0
                           ? score.replicaDomainImbalance / totalFreq
                           : 0.0;
  double gpuMean = gpus.empty()
                       ? 0.0
                       : totalL1Load / static_cast<double>(gpus.size());
  double gpuVarNorm = gpuMean > 0.0
                          ? score.gpuLoadVariance / (gpuMean * gpuMean)
                          : 0.0;
  double probeNorm = totalFreq > 0.0
                         ? score.probePathCost / totalFreq
                         : 0.0;

  return {
      replicaNorm,
      externalNorm,
      l1VarNorm,
      l1MaxNorm,
      gpuVarNorm,
      probeNorm};
}

inline double PlacementProxyCost(
    const ExpertPlanScore& score,
    const std::vector<std::vector<std::pair<uint32_t,uint32_t>>>& plan,
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus) {
  auto metrics = PlacementMetricVector(score, plan, gpus);
  if (metrics.empty() || !std::isfinite(metrics[0]))
    return std::numeric_limits<double>::max();
  return std::accumulate(metrics.begin(), metrics.end(), 0.0);
}

inline bool PlacementGradientEmaMode() {
  std::string mode = Lower(Trim(State().config.placement_adaptive_mode));
  return mode == "gradient_ema" ||
         mode == "grad_ema" ||
         mode == "derivative_ema" ||
         mode == "partial_ema";
}

inline bool PlacementMovingAverageInverseMode() {
  std::string mode = Lower(Trim(State().config.placement_adaptive_mode));
  return mode == "moving_average_inverse_5" ||
         mode == "moving_avg_inverse_5" ||
         mode == "ma_inverse_5" ||
         mode == "ma5_inverse";
}

inline void EnsurePlacementAdaptiveInitialized() {
  auto& state = State();
  if (state.placement_adaptive_initialized)
    return;
  state.placement_adaptive_base_weights =
      ClampPlacementWeightVector(PlacementWeightVectorFromConfig());
  state.placement_adaptive_direction =
      PlacementAdaptiveDirection(state.placement_adaptive_step);
  state.placement_adaptive_active_weights =
      state.placement_adaptive_base_weights;
  state.placement_adaptive_phase = 0;
  state.placement_adaptive_plus_cost = 0.0;
  state.placement_adaptive_observations = 0;
  double initialSum = std::accumulate(
      state.placement_adaptive_base_weights.begin(),
      state.placement_adaptive_base_weights.end(),
      0.0);
  state.placement_adaptive_target_weight_sum =
      State().config.placement_adaptive_weight_sum > 0.0
          ? State().config.placement_adaptive_weight_sum
          : initialSum;
  state.placement_adaptive_metric_ema.assign(
      state.placement_adaptive_base_weights.size(), 0.0);
  state.placement_adaptive_gradient_momentum.assign(
      state.placement_adaptive_base_weights.size(), 0.0);
  state.placement_adaptive_lr_frozen.assign(
      state.placement_adaptive_base_weights.size(), 0);
  state.placement_adaptive_initialized = true;
}

inline void LogPlacementAdaptiveState(const std::string& event,
                                      double cost,
                                      const std::vector<double>& weights) {
  std::ostringstream ss;
  ss << "PipelinePolicy: placement adaptive " << event
     << " mode=" << State().config.placement_adaptive_mode
     << " step=" << State().placement_adaptive_step
     << " phase=" << State().placement_adaptive_phase
     << " proxy_cost=" << cost
     << " weights=[";
  for (size_t i = 0; i < weights.size(); ++i) {
    if (i > 0)
      ss << ",";
    ss << weights[i];
  }
  ss << "]";
  ccl::CclLog(ss.str());
}

inline bool PlacementAdaptiveEnabled() {
  std::string mode = Lower(Trim(State().config.placement_adaptive_mode));
  return mode == "spsa" || PlacementGradientEmaMode();
}

inline void PrepareMovingAverageInversePlacementWeights() {
  if (!PlacementMovingAverageInverseMode())
    return;
  EnsurePlacementAdaptiveInitialized();
  ApplyPlacementWeightVector(State().placement_adaptive_base_weights);
}

inline void LogPlacementMovingAverageInverseState(
    const std::vector<double>& metrics,
    const std::vector<double>& averages,
    const std::vector<double>& weights,
    const std::vector<uint8_t>& frozen) {
  double proxyCost = std::accumulate(metrics.begin(), metrics.end(), 0.0);
  std::ostringstream ss;
  ss << "PipelinePolicy: placement adaptive ma_inverse_update"
     << " mode=" << State().config.placement_adaptive_mode
     << " step=" << State().placement_adaptive_step
     << " observations=" << State().placement_adaptive_observations
     << " proxy_cost=" << proxyCost
     << " window=5"
     << " weights=[";
  for (size_t i = 0; i < weights.size(); ++i) {
    if (i > 0)
      ss << ",";
    ss << weights[i];
  }
  ss << "] partials=[";
  for (size_t i = 0; i < metrics.size(); ++i) {
    if (i > 0)
      ss << ",";
    ss << metrics[i];
  }
  ss << "] ema=[";
  for (size_t i = 0; i < averages.size(); ++i) {
    if (i > 0)
      ss << ",";
    ss << averages[i];
  }
  ss << "] frozen=[";
  for (size_t i = 0; i < frozen.size(); ++i) {
    if (i > 0)
      ss << ",";
    ss << static_cast<uint32_t>(frozen[i] != 0);
  }
  ss << "]";
  ccl::CclLog(ss.str());
}

inline void LogPlacementGradientEmaState(
    const std::vector<double>& metrics,
    const std::vector<double>& ema,
    const std::vector<double>& gradient,
    const std::vector<double>& learningRates,
    const std::vector<double>& weights,
    const std::vector<uint8_t>& frozen) {
  double proxyCost = std::accumulate(metrics.begin(), metrics.end(), 0.0);
  std::ostringstream ss;
  ss << "PipelinePolicy: placement adaptive gradient_ema_update"
     << " mode=" << State().config.placement_adaptive_mode
     << " step=" << State().placement_adaptive_step
     << " observations=" << State().placement_adaptive_observations
     << " proxy_cost=" << proxyCost
     << " lr_update_threshold="
     << State().config.placement_adaptive_lr_update_threshold
     << " lr_freeze_min_observations="
     << State().config.placement_adaptive_lr_freeze_min_observations
     << " weights=[";
  for (size_t i = 0; i < weights.size(); ++i) {
    if (i > 0)
      ss << ",";
    ss << weights[i];
  }
  ss << "] partials=[";
  for (size_t i = 0; i < metrics.size(); ++i) {
    if (i > 0)
      ss << ",";
    ss << metrics[i];
  }
  ss << "] ema=[";
  for (size_t i = 0; i < ema.size(); ++i) {
    if (i > 0)
      ss << ",";
    ss << ema[i];
  }
  ss << "] gradient=[";
  for (size_t i = 0; i < gradient.size(); ++i) {
    if (i > 0)
      ss << ",";
    ss << gradient[i];
  }
  ss << "] lr=[";
  for (size_t i = 0; i < learningRates.size(); ++i) {
    if (i > 0)
      ss << ",";
    ss << learningRates[i];
  }
  ss << "] frozen=[";
  for (size_t i = 0; i < frozen.size(); ++i) {
    if (i > 0)
      ss << ",";
    ss << static_cast<uint32_t>(frozen[i] != 0);
  }
  ss << "]";
  ccl::CclLog(ss.str());
}

inline void PrepareGradientEmaPlacementWeights() {
  if (!PlacementGradientEmaMode())
    return;
  EnsurePlacementAdaptiveInitialized();
  ApplyPlacementWeightVector(State().placement_adaptive_base_weights);
}

inline void ObserveGradientEmaPlacement(
    const ExpertPlanScore& score,
    const std::vector<std::vector<std::pair<uint32_t,uint32_t>>>& plan,
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus) {
  if (!PlacementGradientEmaMode())
    return;
  EnsurePlacementAdaptiveInitialized();
  auto metrics = PlacementMetricVector(score, plan, gpus);
  if (metrics.size() != State().placement_adaptive_base_weights.size() ||
      metrics.empty() ||
      !std::isfinite(metrics[0]))
    return;

  auto& state = State();
  if (state.placement_adaptive_metric_ema.size() != metrics.size())
    state.placement_adaptive_metric_ema.assign(metrics.size(), 0.0);
  if (state.placement_adaptive_gradient_momentum.size() != metrics.size())
    state.placement_adaptive_gradient_momentum.assign(metrics.size(), 0.0);
  if (state.placement_adaptive_lr_frozen.size() != metrics.size())
    state.placement_adaptive_lr_frozen.assign(metrics.size(), 0);

  double alpha = std::min(
      1.0, std::max(0.0, State().config.placement_adaptive_obs_alpha));
  if (state.placement_adaptive_observations == 0 || alpha >= 1.0) {
    state.placement_adaptive_metric_ema = metrics;
  } else {
    for (size_t i = 0; i < metrics.size(); ++i) {
      state.placement_adaptive_metric_ema[i] =
          (1.0 - alpha) * state.placement_adaptive_metric_ema[i] +
          alpha * metrics[i];
    }
  }
  state.placement_adaptive_observations++;

  std::vector<double> relative(metrics.size(), 0.0);
  double meanRelative = 0.0;
  size_t activeCount = 0;
  double eps = 1e-9;
  for (size_t i = 0; i < metrics.size(); ++i) {
    if (state.placement_adaptive_lr_frozen[i] != 0 ||
        state.placement_adaptive_base_weights[i] <= eps)
      continue;
    double baseline = std::max(eps, state.placement_adaptive_metric_ema[i]);
    double observed = std::max(eps, metrics[i]);
    relative[i] = std::log(observed / baseline);
    meanRelative += relative[i];
    activeCount++;
  }
  if (activeCount == 0)
    return;
  meanRelative /= static_cast<double>(activeCount);

  double momentum = std::min(
      0.999, std::max(0.0, State().config.placement_adaptive_momentum));
  double clip = std::max(0.0, State().config.placement_adaptive_gradient_clip);
  double baseLr = std::max(0.0, State().config.placement_adaptive_learning_rate);
  double lrGain = std::max(0.0, State().config.placement_adaptive_dynamic_lr_gain);
  double maxLrMultiplier = std::max(
      1.0, State().config.placement_adaptive_max_lr_multiplier);
  double lrUpdateThreshold = std::max(
      0.0, State().config.placement_adaptive_lr_update_threshold);
  uint64_t lrFreezeMinObservations =
      State().config.placement_adaptive_lr_freeze_min_observations;
  bool freezeEligible =
      lrUpdateThreshold > 0.0 &&
      state.placement_adaptive_observations >= lrFreezeMinObservations;
  std::vector<double> gradient(relative.size(), 0.0);
  std::vector<double> learningRates(relative.size(), baseLr);
  for (size_t i = 0; i < relative.size(); ++i) {
    if (state.placement_adaptive_lr_frozen[i] != 0) {
      gradient[i] = 0.0;
      state.placement_adaptive_gradient_momentum[i] = 0.0;
      learningRates[i] = 0.0;
      continue;
    }
    if (state.placement_adaptive_base_weights[i] <= eps) {
      state.placement_adaptive_gradient_momentum[i] = 0.0;
      learningRates[i] = 0.0;
      continue;
    }
    gradient[i] = relative[i] - meanRelative;
    if (clip > 0.0)
      gradient[i] = std::min(clip, std::max(-clip, gradient[i]));

    double multiplier = 1.0 + lrGain * std::fabs(gradient[i]);
    multiplier = std::min(maxLrMultiplier, multiplier);
    double lr = baseLr * multiplier;
    if (freezeEligible && lr < lrUpdateThreshold) {
      state.placement_adaptive_lr_frozen[i] = 1;
      gradient[i] = 0.0;
      state.placement_adaptive_gradient_momentum[i] = 0.0;
      learningRates[i] = 0.0;
      continue;
    }
    learningRates[i] = lr;
    state.placement_adaptive_gradient_momentum[i] =
        momentum * state.placement_adaptive_gradient_momentum[i] +
        (1.0 - momentum) * gradient[i];
  }

  std::vector<double> weights = state.placement_adaptive_base_weights;
  for (size_t i = 0; i < weights.size(); ++i) {
    if (state.placement_adaptive_lr_frozen[i] != 0) {
      weights[i] = state.placement_adaptive_base_weights[i];
      learningRates[i] = 0.0;
      continue;
    }
    if (state.placement_adaptive_base_weights[i] <= eps) {
      weights[i] = 0.0;
      learningRates[i] = 0.0;
      continue;
    }
    double seed = std::max(1e-3, weights[i]);
    weights[i] = seed * std::exp(
        learningRates[i] * state.placement_adaptive_gradient_momentum[i]);
  }
  weights = NormalizePlacementWeightVectorWithFrozen(
      weights,
      state.placement_adaptive_target_weight_sum,
      state.placement_adaptive_lr_frozen);

  state.placement_adaptive_base_weights = weights;
  state.placement_adaptive_active_weights = weights;
  state.placement_adaptive_step++;
  ApplyPlacementWeightVector(weights);
  LogPlacementGradientEmaState(
      metrics,
      state.placement_adaptive_metric_ema,
      state.placement_adaptive_gradient_momentum,
      learningRates,
      weights,
      state.placement_adaptive_lr_frozen);
}

inline void ObserveMovingAverageInversePlacement(
    const ExpertPlanScore& score,
    const std::vector<std::vector<std::pair<uint32_t,uint32_t>>>& plan,
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus) {
  if (!PlacementMovingAverageInverseMode())
    return;
  EnsurePlacementAdaptiveInitialized();
  auto metrics = PlacementMetricVector(score, plan, gpus);
  if (metrics.size() != State().placement_adaptive_base_weights.size() ||
      metrics.empty() ||
      !std::isfinite(metrics[0]))
    return;

  auto& state = State();
  constexpr uint64_t kWindow = 5;
  if (state.placement_adaptive_observations >= kWindow)
    return;
  if (state.placement_adaptive_metric_ema.size() != metrics.size())
    state.placement_adaptive_metric_ema.assign(metrics.size(), 0.0);
  if (state.placement_adaptive_lr_frozen.size() != metrics.size())
    state.placement_adaptive_lr_frozen.assign(metrics.size(), 0);

  uint64_t nextCount = state.placement_adaptive_observations + 1;
  for (size_t i = 0; i < metrics.size(); ++i) {
    double observed = std::isfinite(metrics[i]) ? std::max(0.0, metrics[i]) : 0.0;
    state.placement_adaptive_metric_ema[i] =
        (state.placement_adaptive_metric_ema[i] *
             static_cast<double>(state.placement_adaptive_observations) +
         observed) /
        static_cast<double>(nextCount);
  }

  std::vector<double> weights = state.placement_adaptive_base_weights;
  for (size_t i = 0; i < weights.size() && i < state.placement_adaptive_metric_ema.size(); ++i) {
    double avg = std::max(1e-9, state.placement_adaptive_metric_ema[i]);
    weights[i] = ClampPlacementWeight(1.0 / avg);
  }
  if (nextCount >= kWindow) {
    for (uint8_t& value : state.placement_adaptive_lr_frozen)
      value = 1;
  }

  state.placement_adaptive_observations = nextCount;
  state.placement_adaptive_step++;
  state.placement_adaptive_base_weights = weights;
  state.placement_adaptive_active_weights = weights;
  ApplyPlacementWeightVector(weights);
  LogPlacementMovingAverageInverseState(
      metrics,
      state.placement_adaptive_metric_ema,
      weights,
      state.placement_adaptive_lr_frozen);
}

inline void PrepareSpsaPlacementWeights() {
  std::string mode = Lower(Trim(State().config.placement_adaptive_mode));
  if (mode != "spsa")
    return;
  EnsurePlacementAdaptiveInitialized();
  auto& state = State();
  double sign = state.placement_adaptive_phase == 0 ? 1.0 : -1.0;
  state.placement_adaptive_active_weights = PerturbPlacementWeights(
      state.placement_adaptive_base_weights,
      state.placement_adaptive_direction,
      sign);
  ApplyPlacementWeightVector(state.placement_adaptive_active_weights);
}

inline void ObserveSpsaPlacementCost(double cost) {
  std::string mode = Lower(Trim(State().config.placement_adaptive_mode));
  if (mode != "spsa")
    return;
  EnsurePlacementAdaptiveInitialized();
  auto& state = State();
  if (state.placement_adaptive_phase == 0) {
    state.placement_adaptive_plus_cost = cost;
    state.placement_adaptive_phase = 1;
    LogPlacementAdaptiveState("spsa_plus_observed", cost,
                              state.placement_adaptive_active_weights);
    return;
  }

  double sigma = std::max(1e-9, State().config.placement_adaptive_sigma);
  double lr = std::max(0.0, State().config.placement_adaptive_learning_rate);
  double diff = state.placement_adaptive_plus_cost - cost;
  for (size_t i = 0; i < state.placement_adaptive_base_weights.size() &&
                     i < state.placement_adaptive_direction.size();
       ++i) {
    double scale = std::max(1.0, std::fabs(state.placement_adaptive_base_weights[i]));
    double grad = diff / (2.0 * sigma * scale) *
                  state.placement_adaptive_direction[i];
    state.placement_adaptive_base_weights[i] =
        ClampPlacementWeight(state.placement_adaptive_base_weights[i] - lr * grad);
  }
  state.placement_adaptive_step++;
  state.placement_adaptive_direction =
      PlacementAdaptiveDirection(state.placement_adaptive_step);
  state.placement_adaptive_phase = 0;
  state.placement_adaptive_active_weights =
      state.placement_adaptive_base_weights;
  ApplyPlacementWeightVector(state.placement_adaptive_base_weights);
  LogPlacementAdaptiveState("spsa_update", cost,
                            state.placement_adaptive_base_weights);
}

class IncrementalExpertPlanScorer {
 public:
  IncrementalExpertPlanScorer(
      uint32_t expertNum,
      const std::vector<std::pair<uint32_t,uint32_t>>& orderedGpus,
      uint32_t expertsPerGpu)
      : m_gpus(orderedGpus),
        m_l1Loads(L1DomainsForGpus(orderedGpus).size(), 0.0),
        m_gpuLoads(orderedGpus.size(), 0.0) {
    m_domains = L1DomainsForGpus(orderedGpus);
    m_l1Loads.assign(m_domains.size(), 0.0);
    for (size_t i = 0; i < m_domains.size(); ++i)
      m_domainIndex[m_domains[i]] = i;
    m_domainGpuCounts.assign(m_domains.size(), 0);
    m_l1GpuLoadSquareSums.assign(m_domains.size(), 0.0);
    m_l1GpuMaxLoads.assign(m_domains.size(), 0.0);
    m_gpuDomainIndex.assign(orderedGpus.size(), 0);
    for (size_t i = 0; i < orderedGpus.size(); ++i) {
      m_gpuIndex[orderedGpus[i]] = i;
      uint32_t domain = L1GroupOfGpu(orderedGpus[i]);
      auto domainIt = m_domainIndex.find(domain);
      size_t domainIdx = domainIt == m_domainIndex.end() ? 0 : domainIt->second;
      m_gpuDomainIndex[i] = domainIdx;
      if (domainIdx < m_domainGpuCounts.size())
        m_domainGpuCounts[domainIdx] += 1;
    }

    m_expertDomainCounts.assign(
        expertNum, std::vector<uint32_t>(m_domains.size(), 0));
    m_expertDomainSums.assign(expertNum, 0.0);
    m_expertDomainSquareSums.assign(expertNum, 0.0);
    m_sourceGpus = orderedGpus;
    m_probeCostByGpuAndSource =
        BuildProbeCostByGpuAndSource(orderedGpus, m_sourceGpus);
    m_probeCostScale = ProbeCostScale(m_probeCostByGpuAndSource);
    m_expertReplicaCounts.assign(expertNum, 0);
    m_expertBestProbeBySource.assign(
        expertNum,
        std::vector<double>(
            m_sourceGpus.size(), std::numeric_limits<double>::max()));
    m_hopDistanceBySourceAndGpu = BuildHopDistanceBySourceAndGpu(orderedGpus, 2);
    m_expertBestHopBySource.assign(
        expertNum,
        std::vector<uint32_t>(orderedGpus.size(), 3));
    for (auto& heat : m_hopHeatBySource)
      heat.assign(orderedGpus.size(), 0.0);
    for (uint32_t expert = 0; expert < expertNum; ++expert) {
      double freq = ExpertFreq(expert);
      m_totalFreq += freq;
      m_externalAccessCost += freq * static_cast<double>(m_domains.size());
    }
    m_topologyCost = TopologyDistanceScore(orderedGpus);
    m_useTopologyStaticPrior = TopologyStaticPriorPlacementPolicy();
    if (m_useTopologyStaticPrior) {
      m_topologyStaticPriorStats =
          ComputeTopologyStaticPriorStats(orderedGpus, expertNum, expertsPerGpu);
    }
  }

  void Place(uint32_t expert, const std::pair<uint32_t,uint32_t>& gpu) {
    size_t domainIdx = DomainIndex(gpu);
    size_t gpuIdx = GpuIndex(gpu);
    double freq = ExpertFreq(expert);

    double oldVar = ExpertDomainVariance(expert);
    if (m_expertDomainCounts[expert][domainIdx] == 0) {
      m_externalAccessCost -= freq;
    }
    m_expertDomainCounts[expert][domainIdx] += 1;
    m_expertDomainSums[expert] += 1.0;
    m_expertDomainSquareSums[expert] +=
        static_cast<double>(m_expertDomainCounts[expert][domainIdx] *
                            m_expertDomainCounts[expert][domainIdx]) -
        static_cast<double>((m_expertDomainCounts[expert][domainIdx] - 1) *
                            (m_expertDomainCounts[expert][domainIdx] - 1));
    double newVar = ExpertDomainVariance(expert);
    m_replicaDomainImbalance += freq * (newVar - oldVar);

    double oldL1 = m_l1Loads[domainIdx];
    m_l1Loads[domainIdx] += freq;
    m_totalL1Load += freq;
    m_l1LoadSquareSum += m_l1Loads[domainIdx] * m_l1Loads[domainIdx] - oldL1 * oldL1;
    m_maxL1Load = std::max(m_maxL1Load, m_l1Loads[domainIdx]);

    double oldGpu = m_gpuLoads[gpuIdx];
    m_gpuLoads[gpuIdx] += freq;
    double newGpu = m_gpuLoads[gpuIdx];
    m_totalGpuLoad += freq;
    m_gpuLoadSquareSum += newGpu * newGpu - oldGpu * oldGpu;
    if (domainIdx < m_l1GpuLoadSquareSums.size()) {
      m_l1GpuLoadSquareSums[domainIdx] += newGpu * newGpu - oldGpu * oldGpu;
      m_l1GpuMaxLoads[domainIdx] =
          std::max(m_l1GpuMaxLoads[domainIdx], newGpu);
    }

    double probeDelta = ProbeDeltaAfterPlace(expert, gpuIdx);
    m_probePathCost += freq * probeDelta;
    if (expert < m_expertReplicaCounts.size())
      m_expertReplicaCounts[expert] += 1;
    if (expert < m_expertBestProbeBySource.size() &&
        gpuIdx < m_probeCostByGpuAndSource.size()) {
      for (size_t src = 0; src < m_sourceGpus.size(); ++src) {
        m_expertBestProbeBySource[expert][src] = std::min(
            m_expertBestProbeBySource[expert][src],
            m_probeCostByGpuAndSource[gpuIdx][src]);
      }
    }
    ApplyHopHeatPlacement(expert, gpuIdx, freq);
  }

  ExpertPlanScore ScoreAfterPlace(
      uint32_t expert,
      const std::pair<uint32_t,uint32_t>& gpu) const {
    size_t domainIdx = DomainIndex(gpu);
    size_t gpuIdx = GpuIndex(gpu);
    double freq = ExpertFreq(expert);

    double external = m_externalAccessCost;
    if (m_expertDomainCounts[expert][domainIdx] == 0)
      external -= freq;

    double replica = m_replicaDomainImbalance;
    double oldVar = ExpertDomainVariance(expert);
    double newVar = ExpertDomainVarianceAfterIncrement(expert, domainIdx);
    replica += freq * (newVar - oldVar);

    return ComposeScore(expert, domainIdx, freq, gpuIdx, external, replica);
  }

  ExpertPlanScore CurrentScore() const {
    return ComposeScore(
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<size_t>::max(),
        0.0,
        std::numeric_limits<size_t>::max(),
        m_externalAccessCost,
        m_replicaDomainImbalance);
  }

  bool CoversNewDomain(uint32_t expert, const std::pair<uint32_t,uint32_t>& gpu) const {
    size_t domainIdx = DomainIndex(gpu);
    if (expert >= m_expertDomainCounts.size() || domainIdx >= m_domains.size())
      return false;
    return m_expertDomainCounts[expert][domainIdx] == 0;
  }

 private:
  size_t DomainIndex(const std::pair<uint32_t,uint32_t>& gpu) const {
    uint32_t domain = L1GroupOfGpu(gpu);
    auto it = m_domainIndex.find(domain);
    return it == m_domainIndex.end() ? 0 : it->second;
  }

  size_t GpuIndex(const std::pair<uint32_t,uint32_t>& gpu) const {
    auto it = m_gpuIndex.find(gpu);
    return it == m_gpuIndex.end() ? 0 : it->second;
  }

  double ExpertDomainVariance(uint32_t expert) const {
    return ExpertDomainVarianceAfterIncrement(
        expert, std::numeric_limits<size_t>::max());
  }

  double ExpertDomainVarianceAfterIncrement(uint32_t expert, size_t addDomain) const {
    if (m_domains.empty())
      return 0.0;
    double sum = m_expertDomainSums[expert];
    double squareSum = m_expertDomainSquareSums[expert];
    if (addDomain < m_domains.size()) {
      double oldCount = static_cast<double>(m_expertDomainCounts[expert][addDomain]);
      double newCount = oldCount + 1.0;
      sum += 1.0;
      squareSum += newCount * newCount - oldCount * oldCount;
    }
    double mean = sum / static_cast<double>(m_domains.size());
    return squareSum / static_cast<double>(m_domains.size()) - mean * mean;
  }

  double ProbeDeltaAfterPlace(uint32_t expert, size_t addGpu) const {
    if (expert >= m_expertBestProbeBySource.size() ||
        addGpu >= m_probeCostByGpuAndSource.size() ||
        m_sourceGpus.empty())
      return 0.0;

    bool hasReplica =
        expert < m_expertReplicaCounts.size() &&
        m_expertReplicaCounts[expert] > 0;
    double oldTotal = 0.0;
    double newTotal = 0.0;
    for (size_t src = 0; src < m_sourceGpus.size(); ++src) {
      double oldRaw = hasReplica
                          ? m_expertBestProbeBySource[expert][src]
                          : 0.0;
      double candidateRaw = m_probeCostByGpuAndSource[addGpu][src];
      double newRaw = hasReplica ? std::min(oldRaw, candidateRaw)
                                 : candidateRaw;
      oldTotal += NormalizedProbeCost(oldRaw, m_probeCostScale);
      newTotal += NormalizedProbeCost(newRaw, m_probeCostScale);
    }
    return (newTotal - oldTotal) / static_cast<double>(m_sourceGpus.size());
  }

  void ApplyHopHeatPlacement(uint32_t expert, size_t addGpu, double freq) {
    if (expert >= m_expertBestHopBySource.size() ||
        addGpu >= m_gpus.size() ||
        m_hopDistanceBySourceAndGpu.empty())
      return;
    for (size_t src = 0; src < m_gpus.size(); ++src) {
      uint32_t oldHop = m_expertBestHopBySource[expert][src];
      uint32_t candidateHop = m_hopDistanceBySourceAndGpu[src][addGpu];
      uint32_t newHop = std::min(oldHop, candidateHop);
      if (newHop < oldHop) {
        for (uint32_t hop = 1; hop <= 2; ++hop) {
          if (oldHop > hop && newHop <= hop)
            m_hopHeatBySource[hop - 1][src] += freq;
        }
      }
      m_expertBestHopBySource[expert][src] = newHop;
    }
  }

  HopHeatMetrics HopHeatMetricsAfterPlace(uint32_t expert,
                                          size_t addGpu,
                                          double freq) const {
    auto heat = m_hopHeatBySource;
    if (expert >= m_expertBestHopBySource.size() ||
        addGpu >= m_gpus.size() ||
        m_hopDistanceBySourceAndGpu.empty())
      return HopHeatMetricsFromHeat(heat, m_totalFreq);
    for (size_t src = 0; src < m_gpus.size(); ++src) {
      uint32_t oldHop = m_expertBestHopBySource[expert][src];
      uint32_t candidateHop = m_hopDistanceBySourceAndGpu[src][addGpu];
      uint32_t newHop = std::min(oldHop, candidateHop);
      if (newHop < oldHop) {
        for (uint32_t hop = 1; hop <= 2; ++hop) {
          if (oldHop > hop && newHop <= hop)
            heat[hop - 1][src] += freq;
        }
      }
    }
    return HopHeatMetricsFromHeat(heat, m_totalFreq);
  }

  L1GpuBalanceMetrics L1GpuBalanceAfterPlace(size_t addGpu,
                                             double addFreq) const {
    L1GpuBalanceMetrics out;
    double varNormSum = 0.0;
    uint32_t activeDomains = 0;
    size_t addDomain = addGpu < m_gpuDomainIndex.size()
                           ? m_gpuDomainIndex[addGpu]
                           : std::numeric_limits<size_t>::max();

    for (size_t domain = 0; domain < m_domains.size(); ++domain) {
      uint32_t gpuCount = domain < m_domainGpuCounts.size()
                              ? m_domainGpuCounts[domain]
                              : 0;
      if (gpuCount == 0)
        continue;
      double sum = domain < m_l1Loads.size() ? m_l1Loads[domain] : 0.0;
      double squareSum = domain < m_l1GpuLoadSquareSums.size()
                             ? m_l1GpuLoadSquareSums[domain]
                             : 0.0;
      double maxLoad = domain < m_l1GpuMaxLoads.size()
                           ? m_l1GpuMaxLoads[domain]
                           : 0.0;
      if (domain == addDomain && addGpu < m_gpuLoads.size()) {
        double oldLoad = m_gpuLoads[addGpu];
        double newLoad = oldLoad + addFreq;
        sum += addFreq;
        squareSum += newLoad * newLoad - oldLoad * oldLoad;
        maxLoad = std::max(maxLoad, newLoad);
      }
      double mean = sum / static_cast<double>(gpuCount);
      if (mean <= 1e-12)
        continue;
      double var = squareSum / static_cast<double>(gpuCount) - mean * mean;
      varNormSum += std::max(0.0, var) / (mean * mean);
      out.maxNorm = std::max(out.maxNorm, maxLoad / mean);
      ++activeDomains;
    }
    if (activeDomains > 0)
      out.varianceNorm = varNormSum / static_cast<double>(activeDomains);
    return out;
  }

  ExpertPlanScore ComposeScore(uint32_t expert,
                               size_t addDomain,
                               double addFreq,
                               size_t addGpu,
                               double external,
                               double replica) const {
    ExpertPlanScore score;
    if (m_domains.empty() || m_gpus.empty())
      return score;

    double l1Sum = m_totalL1Load + addFreq;
    double l1Mean = l1Sum / static_cast<double>(m_l1Loads.size());
    double l1SquareSum = m_l1LoadSquareSum;
    double maxL1 = m_maxL1Load;
    if (addDomain < m_l1Loads.size()) {
      double oldLoad = m_l1Loads[addDomain];
      double newLoad = oldLoad + addFreq;
      l1SquareSum += newLoad * newLoad - oldLoad * oldLoad;
      maxL1 = std::max(maxL1, newLoad);
    }
    double l1Var = l1SquareSum / static_cast<double>(m_l1Loads.size()) -
                   l1Mean * l1Mean;

    double gpuSum = m_totalGpuLoad + (addGpu < m_gpuLoads.size() ? addFreq : 0.0);
    double gpuMean = gpuSum / static_cast<double>(m_gpuLoads.size());
    double gpuSquareSum = m_gpuLoadSquareSum;
    if (addGpu < m_gpuLoads.size()) {
      double oldLoad = m_gpuLoads[addGpu];
      double newLoad = oldLoad + addFreq;
      gpuSquareSum += newLoad * newLoad - oldLoad * oldLoad;
    }
    double gpuVar = gpuSquareSum / static_cast<double>(m_gpuLoads.size()) -
                    gpuMean * gpuMean;
    L1GpuBalanceMetrics l1GpuBalance =
        L1GpuBalanceAfterPlace(addGpu, addFreq);

    score.maxL1Load = maxL1;
    score.l1Variance = std::max(0.0, l1Var);
    score.topologyCost = m_topologyCost;
    score.externalAccessCost = external;
    score.replicaDomainImbalance = replica;
    score.gpuLoadVariance = std::max(0.0, gpuVar);
    score.probePathCost = m_probePathCost;
    if (addGpu < m_gpuLoads.size())
      score.probePathCost += addFreq * ProbeDeltaAfterPlace(expert, addGpu);
    score.probePathCost = std::max(0.0, score.probePathCost);
    HopHeatMetrics hopMetrics = HopHeatMetricsAfterPlace(expert, addGpu, addFreq);
    score.hop1Max = hopMetrics.max[0];
    score.hop1Mean = hopMetrics.mean[0];
    score.hop2Max = hopMetrics.max[1];
    score.hop2Mean = hopMetrics.mean[1];
    score.l1GpuBalanceVariance = l1GpuBalance.varianceNorm;
    score.l1GpuBalanceMax = l1GpuBalance.maxNorm;

    double externalNorm = m_totalFreq > 0.0
                              ? external / m_totalFreq
                              : 0.0;
    double replicaNorm = m_totalFreq > 0.0 ? replica / m_totalFreq : 0.0;
    double probeNorm = m_totalFreq > 0.0
                           ? score.probePathCost / m_totalFreq
                           : 0.0;
    double gpuVarNorm = gpuMean > 0.0 ? score.gpuLoadVariance / (gpuMean * gpuMean) : 0.0;

    if (m_useTopologyStaticPrior) {
      score.objective = PlacementTopologyStaticPriorObjective(
          replicaNorm,
          externalNorm,
          score.l1GpuBalanceVariance,
          score.l1GpuBalanceMax,
          gpuVarNorm,
          probeNorm,
          m_topologyStaticPriorStats);
    } else {
      score.objective = PlacementObjective(
          replicaNorm,
          externalNorm,
          score.l1GpuBalanceVariance,
          score.l1GpuBalanceMax,
          gpuVarNorm,
          probeNorm);
    }
    return score;
  }

  std::vector<std::pair<uint32_t,uint32_t>> m_gpus;
  std::vector<uint32_t> m_domains;
  std::map<uint32_t, size_t> m_domainIndex;
  std::map<std::pair<uint32_t,uint32_t>, size_t> m_gpuIndex;
  std::vector<double> m_l1Loads;
  std::vector<double> m_gpuLoads;
  std::vector<size_t> m_gpuDomainIndex;
  std::vector<uint32_t> m_domainGpuCounts;
  std::vector<double> m_l1GpuLoadSquareSums;
  std::vector<double> m_l1GpuMaxLoads;
  std::vector<std::pair<uint32_t,uint32_t>> m_sourceGpus;
  std::vector<std::vector<double>> m_probeCostByGpuAndSource;
  double m_probeCostScale = 1.0;
  std::vector<std::vector<uint32_t>> m_hopDistanceBySourceAndGpu;
  std::vector<std::vector<uint32_t>> m_expertBestHopBySource;
  std::array<std::vector<double>, 2> m_hopHeatBySource;
  std::vector<std::vector<uint32_t>> m_expertDomainCounts;
  std::vector<double> m_expertDomainSums;
  std::vector<double> m_expertDomainSquareSums;
  std::vector<uint32_t> m_expertReplicaCounts;
  std::vector<std::vector<double>> m_expertBestProbeBySource;
  double m_totalFreq = 0.0;
  double m_totalL1Load = 0.0;
  double m_l1LoadSquareSum = 0.0;
  double m_maxL1Load = 0.0;
  double m_totalGpuLoad = 0.0;
  double m_gpuLoadSquareSum = 0.0;
  double m_externalAccessCost = 0.0;
  double m_replicaDomainImbalance = 0.0;
  double m_topologyCost = 0.0;
  double m_probePathCost = 0.0;
  bool m_useTopologyStaticPrior = false;
  TopologyStaticPriorStats m_topologyStaticPriorStats;
};

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

inline std::vector<std::vector<std::pair<uint32_t,uint32_t>>> BuildTraceDeviceExpertPlan(
    uint32_t expertNum,
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
    uint32_t traceLayerId = 0,
    ExpertPlanScore* scoreOut = nullptr) {
  std::vector<std::vector<std::pair<uint32_t,uint32_t>>> plan(expertNum);
  const auto* deviceExpertsPtr = &State().trace_device_experts;
  if (traceLayerId < State().trace_device_experts_by_layer.size() &&
      !State().trace_device_experts_by_layer[traceLayerId].empty()) {
    deviceExpertsPtr = &State().trace_device_experts_by_layer[traceLayerId];
  }
  const auto& deviceExperts = *deviceExpertsPtr;
  if (gpus.empty() || expertNum == 0 || deviceExperts.empty())
    return plan;
  if (gpus.size() < deviceExperts.size()) {
    std::ostringstream ss;
    ss << "PipelinePolicy: trace device placement failed selected_gpus="
       << gpus.size()
       << " required_devices=" << deviceExperts.size();
    ccl::CclLog(ss.str());
    return std::vector<std::vector<std::pair<uint32_t,uint32_t>>>(expertNum);
  }

  std::map<std::pair<uint32_t,uint32_t>, std::set<uint32_t>> expertsOnGpu;
  uint32_t templateDevices = static_cast<uint32_t>(deviceExperts.size());
  uint32_t fullRepeats = templateDevices == 0
                             ? 0
                             : static_cast<uint32_t>(gpus.size()) / templateDevices;
  uint32_t remainder = templateDevices == 0
                           ? 0
                           : static_cast<uint32_t>(gpus.size()) % templateDevices;
  for (uint32_t gpuIdx = 0;
       gpuIdx < static_cast<uint32_t>(gpus.size());
       ++gpuIdx) {
    uint32_t deviceId = gpuIdx % templateDevices;
    const auto& gpu = gpus[gpuIdx];
    for (uint32_t expert : deviceExperts[deviceId]) {
      if (expert >= expertNum)
        continue;
      if (!expertsOnGpu[gpu].insert(expert).second)
        continue;
      plan[expert].push_back(gpu);
    }
  }
  {
    std::ostringstream ss;
    ss << "PipelinePolicy: trace device placement tiled"
       << " trace_layer=" << traceLayerId
       << " template_devices=" << templateDevices
       << " selected_gpus=" << gpus.size()
       << " full_repeats=" << fullRepeats
       << " remainder=" << remainder;
    ccl::CclLog(ss.str());
  }

  uint32_t missing = 0;
  for (const auto& replicas : plan) {
    if (replicas.empty())
      ++missing;
  }
  if (missing > 0) {
    ccl::CclLog("PipelinePolicy: trace device placement missing_experts=" +
                std::to_string(missing));
  }
  if (scoreOut != nullptr)
    *scoreOut = ScoreExpertPlan(plan, gpus);
  return plan;
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
    *scoreOut = ScoreExpertPlan(plan, gpus, &expertsOnGpu);
  }
  return plan;
}

inline std::vector<std::vector<std::pair<uint32_t,uint32_t>>> BuildProbeBalancedDomainSpreadPlan(
    uint32_t expertNum,
    uint32_t expertPerGpu,
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
    ExpertPlanScore* scoreOut = nullptr) {
  std::vector<std::vector<std::pair<uint32_t,uint32_t>>> plan(expertNum);
  if (gpus.empty() || expertNum == 0)
    return plan;

  std::string policy = Lower(Trim(State().config.expert_placement_policy));
  bool useHotSurplus = policy == "probe_balanced_domain_spread" ||
                       policy == "probe_balanced_hot_surplus" ||
                       policy == "probe_balanced_probe_cost" ||
                       policy == "probe_balanced_full";
  bool useProbeCost = policy == "probe_balanced_probe_cost" ||
                      policy == "probe_balanced_full";
  bool useTopologyDiversity = policy == "probe_balanced_topology_diverse" ||
                              policy == "probe_balanced_full";

  uint32_t expertsPerGpu = ExpertsOnGpu(
      expertNum, static_cast<uint32_t>(gpus.size()), 0, expertPerGpu);
  if (expertsPerGpu == 0)
    return plan;
  expertsPerGpu = std::min(expertsPerGpu, expertNum);

  std::map<std::pair<uint32_t,uint32_t>, uint32_t> gpuCapacity;
  for (const auto& gpu : gpus)
    gpuCapacity[gpu] = std::min(expertNum, gpuCapacity[gpu] + expertsPerGpu);
  uint32_t totalSlots = 0;
  for (const auto& kv : gpuCapacity)
    totalSlots += kv.second;
  if (totalSlots < expertNum)
    return plan;

  std::vector<std::pair<uint32_t,uint32_t>> orderedGpus = UniqueGpus(gpus);
  std::stable_sort(orderedGpus.begin(), orderedGpus.end(),
                   [](const auto& a, const auto& b) {
                     uint32_t al1 = L1GroupOfGpu(a);
                     uint32_t bl1 = L1GroupOfGpu(b);
                     if (al1 != bl1)
                       return al1 < bl1;
                     return a < b;
                   });

  auto domains = L1DomainsForGpus(orderedGpus);
  std::map<uint32_t, size_t> domainIndex;
  for (size_t i = 0; i < domains.size(); ++i)
    domainIndex[domains[i]] = i;

  std::map<std::pair<uint32_t,uint32_t>, size_t> gpuIndex;
  for (size_t i = 0; i < orderedGpus.size(); ++i)
    gpuIndex[orderedGpus[i]] = i;
  std::vector<std::pair<uint32_t,uint32_t>> sourceGpus =
      L1RepresentativeGpus(orderedGpus);
  std::vector<std::vector<double>> probeCostByGpuAndSource =
      useProbeCost ? BuildProbeCostByGpuAndSource(orderedGpus, sourceGpus)
                   : std::vector<std::vector<double>>();
  std::vector<std::vector<double>> bestProbeByExpertAndSource(
      expertNum,
      std::vector<double>(
          sourceGpus.size(), std::numeric_limits<double>::max()));

  std::vector<uint32_t> expertsByFreq(expertNum);
  std::iota(expertsByFreq.begin(), expertsByFreq.end(), 0);
  std::sort(expertsByFreq.begin(), expertsByFreq.end(), [](uint32_t a, uint32_t b) {
    if (ExpertFreq(a) != ExpertFreq(b))
      return ExpertFreq(a) > ExpertFreq(b);
    return a < b;
  });

  std::map<std::pair<uint32_t,uint32_t>, std::set<uint32_t>> expertsAssignedOnGpu;
  std::map<std::pair<uint32_t,uint32_t>, uint32_t> expertsOnGpu;
  std::vector<uint32_t> replicaCount(expertNum, 0);
  std::vector<std::vector<uint32_t>> domainReplicaCount(
      expertNum, std::vector<uint32_t>(domains.size(), 0));
  std::map<uint32_t, double> l1Load;
  for (uint32_t domain : domains)
    l1Load[domain] = 0.0;

  auto canPlace = [&](uint32_t expert, const std::pair<uint32_t,uint32_t>& gpu) {
    if (expertsOnGpu[gpu] >= gpuCapacity[gpu])
      return false;
    return expertsAssignedOnGpu[gpu].count(expert) == 0;
  };

  auto place = [&](uint32_t expert, const std::pair<uint32_t,uint32_t>& gpu) {
    uint32_t domain = L1GroupOfGpu(gpu);
    size_t d = domainIndex[domain];
    plan[expert].push_back(gpu);
    expertsAssignedOnGpu[gpu].insert(expert);
    expertsOnGpu[gpu] += 1;
    replicaCount[expert] += 1;
    domainReplicaCount[expert][d] += 1;
    l1Load[domain] += ExpertFreq(expert);
    if (useProbeCost && !sourceGpus.empty()) {
      auto it = gpuIndex.find(gpu);
      if (it != gpuIndex.end()) {
        size_t g = it->second;
        for (size_t src = 0; src < sourceGpus.size(); ++src) {
          bestProbeByExpertAndSource[expert][src] = std::min(
              bestProbeByExpertAndSource[expert][src],
              probeCostByGpuAndSource[g][src]);
        }
      }
    }
  };

  auto probeBestAfterPlace = [&](uint32_t expert,
                                 const std::pair<uint32_t,uint32_t>& gpu) {
    if (!useProbeCost || sourceGpus.empty())
      return 0.0;
    auto it = gpuIndex.find(gpu);
    if (it == gpuIndex.end())
      return 0.0;
    double total = 0.0;
    size_t g = it->second;
    bool hasReplica = replicaCount[expert] > 0;
    for (size_t src = 0; src < sourceGpus.size(); ++src) {
      double candidate = probeCostByGpuAndSource[g][src];
      if (hasReplica)
        candidate = std::min(candidate, bestProbeByExpertAndSource[expert][src]);
      total += candidate;
    }
    return total / static_cast<double>(sourceGpus.size());
  };

  auto topologyDiversityTie = [&](uint32_t expert,
                                  const std::pair<uint32_t,uint32_t>& gpu) {
    if (!useTopologyDiversity || plan[expert].empty())
      return 0.0;
    uint32_t node = NetworkNodeId(gpu);
    uint64_t minDistance = std::numeric_limits<uint64_t>::max();
    for (const auto& replica : plan[expert]) {
      uint32_t replicaNode = NetworkNodeId(replica);
      minDistance = std::min(minDistance, pairRtt[node][replicaNode]);
    }
    return -static_cast<double>(
        minDistance == std::numeric_limits<uint64_t>::max() ? 0 : minDistance);
  };

  auto bestGpuForCoverage = [&](uint32_t expert,
                                std::pair<uint32_t,uint32_t>* bestGpu) {
    bool found = false;
    std::tuple<uint32_t, double, double, double, uint32_t, uint32_t, uint32_t> bestTie{
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max()};
    for (const auto& gpu : orderedGpus) {
      if (!canPlace(expert, gpu))
        continue;
      uint32_t domain = L1GroupOfGpu(gpu);
      size_t d = domainIndex[domain];
      auto tie = std::make_tuple(
          domainReplicaCount[expert][d],
          l1Load[domain] + ExpertFreq(expert),
          probeBestAfterPlace(expert, gpu),
          topologyDiversityTie(expert, gpu),
          expertsOnGpu[gpu],
          gpu.first,
          gpu.second);
      if (!found || tie < bestTie) {
        found = true;
        bestTie = tie;
        *bestGpu = gpu;
      }
    }
    return found;
  };

  for (uint32_t expert : expertsByFreq) {
    std::pair<uint32_t,uint32_t> bestGpu;
    if (!bestGpuForCoverage(expert, &bestGpu))
      return std::vector<std::vector<std::pair<uint32_t,uint32_t>>>(expertNum);
    place(expert, bestGpu);
  }

  uint32_t placedSlots = expertNum;
  for (const auto& gpu : orderedGpus) {
    while (placedSlots < totalSlots && expertsOnGpu[gpu] < gpuCapacity[gpu]) {
      uint32_t domain = L1GroupOfGpu(gpu);
      size_t d = domainIndex[domain];
      bool found = false;
      uint32_t bestExpert = 0;
      std::tuple<uint32_t, double, double, double, double, double, uint32_t> bestTie{
          std::numeric_limits<uint32_t>::max(),
          std::numeric_limits<double>::max(),
          std::numeric_limits<double>::max(),
          std::numeric_limits<double>::max(),
          std::numeric_limits<double>::max(),
          std::numeric_limits<double>::max(),
          std::numeric_limits<uint32_t>::max()};
      for (uint32_t expert : expertsByFreq) {
        if (!canPlace(expert, gpu))
          continue;
        double freq = std::max(1.0, ExpertFreq(expert));
        double weightedReplicaPressure =
            useHotSurplus
                ? static_cast<double>(replicaCount[expert]) / std::sqrt(freq)
                : static_cast<double>(replicaCount[expert]);
        auto tie = std::make_tuple(
            domainReplicaCount[expert][d],
            probeBestAfterPlace(expert, gpu),
            topologyDiversityTie(expert, gpu),
            weightedReplicaPressure,
            l1Load[domain] + ExpertFreq(expert),
            -ExpertFreq(expert),
            expert);
        if (!found || tie < bestTie) {
          found = true;
          bestTie = tie;
          bestExpert = expert;
        }
      }
      if (!found)
        break;
      place(bestExpert, gpu);
      placedSlots++;
    }
  }

  if (scoreOut != nullptr)
    *scoreOut = ScoreExpertPlan(plan, orderedGpus, &expertsOnGpu);
  return plan;
}

inline std::vector<std::vector<std::pair<uint32_t,uint32_t>>> BuildProbeBalancedPartitionDecayPlan(
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

  std::map<std::pair<uint32_t,uint32_t>, uint32_t> gpuCapacity;
  for (const auto& gpu : gpus)
    gpuCapacity[gpu] += expertsPerGpu;
  uint32_t totalSlots = 0;
  for (auto& kv : gpuCapacity) {
    kv.second = std::min(kv.second, expertNum);
    totalSlots += kv.second;
  }
  if (totalSlots < expertNum)
    return plan;

  std::vector<std::pair<uint32_t,uint32_t>> orderedGpus = UniqueGpus(gpus);
  std::stable_sort(orderedGpus.begin(), orderedGpus.end(),
                   [](const auto& a, const auto& b) {
                     uint32_t al1 = L1GroupOfGpu(a);
                     uint32_t bl1 = L1GroupOfGpu(b);
                     if (al1 != bl1)
                       return al1 < bl1;
                     return a < b;
                   });

  std::vector<uint32_t> domains = L1DomainsForGpus(orderedGpus);
  if (domains.empty())
    return plan;

  std::map<uint32_t, size_t> domainIndex;
  std::vector<std::vector<std::pair<uint32_t,uint32_t>>> gpusByDomain(domains.size());
  for (size_t i = 0; i < domains.size(); ++i)
    domainIndex[domains[i]] = i;
  for (const auto& gpu : orderedGpus) {
    auto it = domainIndex.find(L1GroupOfGpu(gpu));
    if (it == domainIndex.end())
      continue;
    gpusByDomain[it->second].push_back(gpu);
  }

  std::map<std::pair<uint32_t,uint32_t>, size_t> gpuIndex;
  for (size_t i = 0; i < orderedGpus.size(); ++i)
    gpuIndex[orderedGpus[i]] = i;

  std::vector<uint32_t> expertsByFreq(expertNum);
  std::iota(expertsByFreq.begin(), expertsByFreq.end(), 0);
  std::stable_sort(expertsByFreq.begin(), expertsByFreq.end(),
                   [](uint32_t a, uint32_t b) {
                     if (ExpertFreq(a) != ExpertFreq(b))
                       return ExpertFreq(a) > ExpertFreq(b);
                     return a < b;
                   });

  std::map<std::pair<uint32_t,uint32_t>, std::set<uint32_t>> expertsAssignedOnGpu;
  std::map<std::pair<uint32_t,uint32_t>, uint32_t> expertsOnGpu;
  std::vector<double> partitionHeat(domains.size(), 0.0);
  std::vector<uint32_t> partitionAssignedCount(domains.size(), 0);
  std::vector<double> gpuHeat(orderedGpus.size(), 0.0);
  std::vector<uint32_t> expertReplicaCount(expertNum, 0);
  std::vector<double> expertWorkingHeat(expertNum, 0.0);
  for (uint32_t expert = 0; expert < expertNum; ++expert)
    expertWorkingHeat[expert] = ExpertFreq(expert);

  struct CandidateChoice {
    bool found = false;
    size_t domainIdx = 0;
    size_t gpuIdx = 0;
    std::pair<uint32_t,uint32_t> gpu{0, 0};
    double projectedHeat = 0.0;
    uint32_t domainAssignedCount = 0;
    uint32_t gpuAssignedCount = 0;
    double gpuProjectedHeat = 0.0;
  };

  auto choiceKey = [](const CandidateChoice& choice) {
    return std::make_tuple(choice.projectedHeat,
                           choice.domainAssignedCount,
                           choice.gpuAssignedCount,
                           choice.gpuProjectedHeat,
                           choice.gpu.first,
                           choice.gpu.second);
  };

  auto canPlace = [&](uint32_t expert, const std::pair<uint32_t,uint32_t>& gpu) {
    auto capIt = gpuCapacity.find(gpu);
    if (capIt == gpuCapacity.end() || expertsOnGpu[gpu] >= capIt->second)
      return false;
    return expertsAssignedOnGpu[gpu].count(expert) == 0;
  };

  auto bestGpuInDomain = [&](uint32_t expert,
                             size_t domainIdx,
                             CandidateChoice* out) {
    bool found = false;
    CandidateChoice best;
    double currentHeat = expertWorkingHeat[expert];
    for (const auto& gpu : gpusByDomain[domainIdx]) {
      if (!canPlace(expert, gpu))
        continue;
      auto gpuIt = gpuIndex.find(gpu);
      if (gpuIt == gpuIndex.end())
        continue;
      CandidateChoice choice;
      choice.found = true;
      choice.domainIdx = domainIdx;
      choice.gpuIdx = gpuIt->second;
      choice.gpu = gpu;
      choice.domainAssignedCount = partitionAssignedCount[domainIdx];
      choice.gpuAssignedCount = expertsOnGpu[gpu];
      choice.projectedHeat = partitionHeat[domainIdx] + currentHeat;
      choice.gpuProjectedHeat = gpuHeat[choice.gpuIdx] + currentHeat;
      if (!found || choiceKey(choice) < choiceKey(best)) {
        found = true;
        best = choice;
      }
    }
    if (found && out != nullptr)
      *out = best;
    return found;
  };

  auto chooseBestCandidate = [&](uint32_t expert,
                                 bool coveragePhase,
                                 CandidateChoice* out) {
    bool hasUncovered = false;
    if (coveragePhase) {
      for (size_t domainIdx = 0; domainIdx < gpusByDomain.size(); ++domainIdx) {
        if (partitionAssignedCount[domainIdx] != 0)
          continue;
        CandidateChoice probe;
        if (bestGpuInDomain(expert, domainIdx, &probe)) {
          hasUncovered = true;
          break;
        }
      }
    }

    bool found = false;
    CandidateChoice best;
    for (size_t domainIdx = 0; domainIdx < gpusByDomain.size(); ++domainIdx) {
      if (hasUncovered && partitionAssignedCount[domainIdx] != 0)
        continue;
      CandidateChoice choice;
      if (!bestGpuInDomain(expert, domainIdx, &choice))
        continue;
      if (!found || choiceKey(choice) < choiceKey(best)) {
        found = true;
        best = choice;
      }
    }
    if (found && out != nullptr)
      *out = best;
    return found;
  };

  auto place = [&](uint32_t expert, const CandidateChoice& choice) {
    double currentHeat = expertWorkingHeat[expert];
    plan[expert].push_back(choice.gpu);
    expertsAssignedOnGpu[choice.gpu].insert(expert);
    expertsOnGpu[choice.gpu] += 1;
    partitionHeat[choice.domainIdx] += currentHeat;
    partitionAssignedCount[choice.domainIdx] += 1;
    gpuHeat[choice.gpuIdx] += currentHeat;
    expertReplicaCount[expert] += 1;
    double replicas = static_cast<double>(expertReplicaCount[expert]);
    // Local-only marginal heat: after n replicas, future contribution is H/(n+1).
    expertWorkingHeat[expert] *= replicas / (replicas + 1.0);
  };

  uint32_t stage1Slots = std::min<uint32_t>(
      static_cast<uint32_t>(domains.size()), expertNum);
  for (uint32_t i = 0; i < stage1Slots; ++i) {
    CandidateChoice choice;
    if (!chooseBestCandidate(expertsByFreq[i], true, &choice))
      return std::vector<std::vector<std::pair<uint32_t,uint32_t>>>(expertNum);
    place(expertsByFreq[i], choice);
  }

  uint32_t placedSlots = stage1Slots;
  while (placedSlots < totalSlots) {
    bool progressed = false;
    for (uint32_t expert : expertsByFreq) {
      if (placedSlots >= totalSlots)
        break;
      CandidateChoice choice;
      if (!chooseBestCandidate(expert, false, &choice))
        continue;
      place(expert, choice);
      ++placedSlots;
      progressed = true;
    }
    if (!progressed)
      break;
  }

  if (scoreOut != nullptr)
    *scoreOut = ScoreExpertPlan(plan, orderedGpus, &expertsOnGpu);
  return plan;
}

inline std::vector<std::vector<std::pair<uint32_t,uint32_t>>> BuildProbeBalancedPlacementPlan(
    uint32_t expertNum,
    uint32_t expertPerGpu,
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
    ExpertPlanScore* scoreOut = nullptr) {
  std::string policy = Lower(Trim(State().config.expert_placement_policy));
  if (policy == "probe_balanced_partition_decay")
    return BuildProbeBalancedPartitionDecayPlan(
        expertNum, expertPerGpu, gpus, scoreOut);
  return BuildProbeBalancedDomainSpreadPlan(
      expertNum, expertPerGpu, gpus, scoreOut);
}

inline std::vector<std::vector<std::pair<uint32_t,uint32_t>>> BuildIncrementalExpertPlan(
    uint32_t expertNum,
    uint32_t expertPerGpu,
    const std::vector<std::pair<uint32_t,uint32_t>>& gpus,
    ExpertPlanScore* scoreOut = nullptr,
    bool fastSurplusFill = false) {
  std::vector<std::vector<std::pair<uint32_t,uint32_t>>> plan(expertNum);
  if (gpus.empty() || expertNum == 0)
    return plan;

  uint32_t expertsPerGpu = ExpertsOnGpu(
      expertNum, static_cast<uint32_t>(gpus.size()), 0, expertPerGpu);
  if (expertsPerGpu == 0)
    return plan;
  expertsPerGpu = std::min(expertsPerGpu, expertNum);

  std::map<std::pair<uint32_t,uint32_t>, uint32_t> gpuCapacity;
  for (const auto& gpu : gpus)
    gpuCapacity[gpu] += expertsPerGpu;
  uint32_t totalSlots = 0;
  for (auto& kv : gpuCapacity) {
    kv.second = std::min(kv.second, expertNum);
    totalSlots += kv.second;
  }
  if (totalSlots < expertNum)
    return plan;

  std::vector<std::pair<uint32_t,uint32_t>> orderedGpus = UniqueGpus(gpus);
  std::stable_sort(orderedGpus.begin(), orderedGpus.end(),
                   [](const auto& a, const auto& b) {
                     uint32_t al1 = L1GroupOfGpu(a);
                     uint32_t bl1 = L1GroupOfGpu(b);
                     if (al1 != bl1)
                       return al1 < bl1;
                     return a < b;
                   });

  std::vector<uint32_t> expertsByFreq(expertNum);
  std::iota(expertsByFreq.begin(), expertsByFreq.end(), 0);
  std::stable_sort(expertsByFreq.begin(), expertsByFreq.end(),
                   [](uint32_t a, uint32_t b) {
                     if (ExpertFreq(a) != ExpertFreq(b))
                       return ExpertFreq(a) > ExpertFreq(b);
                     return a < b;
                   });

  ApplyTopologyStaticPriorWeights(orderedGpus, expertNum, expertsPerGpu);
  IncrementalExpertPlanScorer scorer(expertNum, orderedGpus, expertsPerGpu);
  std::map<std::pair<uint32_t,uint32_t>, std::set<uint32_t>> expertsAssignedOnGpu;
  std::map<std::pair<uint32_t,uint32_t>, uint32_t> expertsOnGpu;

  auto canPlace = [&](uint32_t expert, const std::pair<uint32_t,uint32_t>& gpu) {
    auto capIt = gpuCapacity.find(gpu);
    if (capIt == gpuCapacity.end() || expertsOnGpu[gpu] >= capIt->second)
      return false;
    return expertsAssignedOnGpu[gpu].count(expert) == 0;
  };

  auto place = [&](uint32_t expert, const std::pair<uint32_t,uint32_t>& gpu) {
    plan[expert].push_back(gpu);
    expertsAssignedOnGpu[gpu].insert(expert);
    expertsOnGpu[gpu] += 1;
    scorer.Place(expert, gpu);
  };

  for (uint32_t expert : expertsByFreq) {
    bool found = false;
    std::pair<uint32_t,uint32_t> bestGpu;
    ExpertPlanScore bestScore;
    for (const auto& gpu : orderedGpus) {
      if (!canPlace(expert, gpu))
        continue;
      ExpertPlanScore score = scorer.ScoreAfterPlace(expert, gpu);
      if (!found || BetterPlanScore(score, bestScore) ||
          (std::fabs(score.objective - bestScore.objective) <= 1e-12 &&
           gpu < bestGpu)) {
        found = true;
        bestGpu = gpu;
        bestScore = score;
      }
    }
    if (!found)
      return std::vector<std::vector<std::pair<uint32_t,uint32_t>>>(expertNum);
    place(expert, bestGpu);
  }

  uint32_t placedSlots = expertNum;
  if (fastSurplusFill) {
    while (placedSlots < totalSlots) {
      bool progressed = false;
      for (const auto& gpu : orderedGpus) {
        auto capIt = gpuCapacity.find(gpu);
        if (capIt == gpuCapacity.end() || expertsOnGpu[gpu] >= capIt->second)
          continue;

        bool found = false;
        uint32_t bestExpert = 0;
        ExpertPlanScore bestScore;
        for (uint32_t expert : expertsByFreq) {
          if (!canPlace(expert, gpu))
            continue;
          ExpertPlanScore score = scorer.ScoreAfterPlace(expert, gpu);
          if (!found || BetterPlanScore(score, bestScore) ||
              (std::fabs(score.objective - bestScore.objective) <= 1e-12 &&
               std::make_pair(-ExpertFreq(expert), expert) <
                   std::make_pair(-ExpertFreq(bestExpert), bestExpert))) {
            found = true;
            bestExpert = expert;
            bestScore = score;
          }
        }
        if (!found)
          continue;
        place(bestExpert, gpu);
        ++placedSlots;
        progressed = true;
        if (placedSlots >= totalSlots)
          break;
      }
      if (!progressed)
        break;
    }
  } else {
    while (placedSlots < totalSlots) {
      bool found = false;
      uint32_t bestExpert = 0;
      std::pair<uint32_t,uint32_t> bestGpu;
      ExpertPlanScore bestScore;
      for (uint32_t expert : expertsByFreq) {
        for (const auto& gpu : orderedGpus) {
          if (!canPlace(expert, gpu))
            continue;
          ExpertPlanScore score = scorer.ScoreAfterPlace(expert, gpu);
          if (!found || BetterPlanScore(score, bestScore) ||
              (std::fabs(score.objective - bestScore.objective) <= 1e-12 &&
               std::make_pair(-ExpertFreq(expert), std::make_pair(expert, gpu)) <
                   std::make_pair(-ExpertFreq(bestExpert), std::make_pair(bestExpert, bestGpu)))) {
            found = true;
            bestExpert = expert;
            bestGpu = gpu;
            bestScore = score;
          }
        }
      }
      if (!found)
        break;
      place(bestExpert, bestGpu);
      ++placedSlots;
    }
  }

  if (scoreOut != nullptr)
    *scoreOut = ScoreExpertPlan(plan, orderedGpus, &expertsOnGpu);
  return plan;
}

inline uint32_t MaxExpertsPerGpu() {
  return cclScheduler::kMaxExpertsPerGpu;
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
      0, cclScheduler::PlacementKind::Generic, need, expertNum, expertMemBytes, expertPerGpu};
  auto placement = PickPlacementForNeed(req, need, memUsed, gpuMem, expertUsed, numNodes, gpusPerServer, maxExpertsPerGpu);
  if (placement.size() != need)
    return out;
  std::string policy = Lower(Trim(State().config.expert_placement_policy));
  auto plan = IsIncrementalPlacementPolicy(policy)
                  ? BuildIncrementalExpertPlan(
                        expertNum, expertPerGpu, placement, &out.plan)
                  : (IsProbeBalancedPlacementPolicy(policy)
                         ? BuildProbeBalancedPlacementPlan(
                               expertNum, expertPerGpu, placement, &out.plan)
                         : BuildExpertPlan(expertNum, expertPerGpu, placement, &out.plan));
  out.feasible = ExpertPlanReady(plan, expertNum);
  return out;
}

inline std::vector<std::vector<std::pair<uint32_t,uint32_t>>> BuildExpertPlanForPolicyRaw(
    const cclScheduler::ExpertPlacementRequest& req,
    const std::string& policy,
    bool useTraceDevicePlacement,
    ExpertPlanScore* scoreOut = nullptr) {
  if (useTraceDevicePlacement)
    return BuildTraceDeviceExpertPlan(req.expert_num, req.gpus, req.trace_layer_id, scoreOut);
  if (IsIncrementalPlacementPolicy(policy))
    return BuildIncrementalExpertPlan(
        req.expert_num,
        req.expert_per_gpu,
        req.gpus,
        scoreOut,
        true);
  if (IsProbeBalancedPlacementPolicy(policy))
    return BuildProbeBalancedPlacementPlan(
        req.expert_num, req.expert_per_gpu, req.gpus, scoreOut);
  return BuildExpertPlan(req.expert_num, req.expert_per_gpu, req.gpus, scoreOut);
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
    RefreshPlacementAccessFreqForPolicy();
    if (State().config.trace_dispatch_enable &&
        req.trace_layer_id < State().trace_access_targets_by_layer.size()) {
      const auto& layerTargets = State().trace_access_targets_by_layer[req.trace_layer_id];
      if (!layerTargets.empty()) {
        std::vector<double> layerFreq(
            std::max<size_t>(State().config.expert_num, layerTargets.size()),
            1.0);
        for (size_t i = 0; i < layerTargets.size(); ++i)
          layerFreq[i] = layerTargets[i] > 0 ? static_cast<double>(layerTargets[i]) : 1.0;
        State().posterior_expert_access_freq = std::move(layerFreq);
      }
    }
    {
      std::ostringstream ss;
      ss << "PipelinePolicy: expert placement start"
         << " owner=" << req.ownerId
         << " kind=" << cclScheduler::PlacementKindName(req.kind)
         << " trace_layer=" << req.trace_layer_id
         << " selected_gpus=" << req.gpus.size()
         << " expert_num=" << req.expert_num
         << " expert_per_gpu=" << req.expert_per_gpu;
      ccl::CclLog(ss.str());
    }
    ExpertPlanScore score;
    std::string policy = Lower(Trim(State().config.expert_placement_policy));
    bool useTraceDevicePlacement =
        State().config.trace_dispatch_enable &&
        State().config.trace_use_device_placement &&
        !State().trace_device_experts.empty();
    std::string adaptiveMode = Lower(Trim(State().config.placement_adaptive_mode));
    std::vector<std::vector<std::pair<uint32_t,uint32_t>>> plan;
    PrepareSpsaPlacementWeights();
    PrepareGradientEmaPlacementWeights();
    PrepareMovingAverageInversePlacementWeights();
    bool adaptiveInitialRoundRobin =
        PlacementAdaptiveEnabled() &&
        IsIncrementalPlacementPolicy(policy) &&
        !useTraceDevicePlacement &&
        State().latest_expert_plan.empty();
    std::string effectivePolicy =
        useTraceDevicePlacement ? "trace_device_layer" : State().config.expert_placement_policy;
    if (adaptiveInitialRoundRobin) {
      plan = cclScheduler::RoundRobinExpertPlacementPolicy(req);
      if (ExpertPlanReady(plan, req.expert_num)) {
        score = ScoreExpertPlan(plan, req.gpus);
        effectivePolicy = "adaptive_initial_roundrobin";
      } else {
        ccl::CclLog("PipelinePolicy: adaptive initial round-robin placement failed, falling back to configured policy");
        plan = BuildExpertPlanForPolicyRaw(
            req, policy, useTraceDevicePlacement, &score);
      }
    } else {
      plan = BuildExpertPlanForPolicyRaw(
          req, policy, useTraceDevicePlacement, &score);
    }
    bool skipAdaptiveObservation =
        adaptiveInitialRoundRobin &&
        effectivePolicy == "adaptive_initial_roundrobin";
    if (adaptiveMode == "spsa" && !skipAdaptiveObservation) {
      double proxyCost = PlacementProxyCost(score, plan, req.gpus);
      ObserveSpsaPlacementCost(proxyCost);
    } else if (PlacementGradientEmaMode() && !skipAdaptiveObservation) {
      ObserveGradientEmaPlacement(score, plan, req.gpus);
    } else if (PlacementMovingAverageInverseMode() && !skipAdaptiveObservation) {
      ObserveMovingAverageInversePlacement(score, plan, req.gpus);
    }
    State().latest_expert_plan = plan;
    State().latest_plan_max_l1_load = score.maxL1Load;
    State().latest_plan_l1_variance = score.l1Variance;
    {
      std::vector<uint32_t> domains = L1DomainsForGpus(req.gpus);
      std::vector<uint64_t> rttTiers =
          TopologyRttTierBoundaries(UniqueGpus(req.gpus), 2);
      std::ostringstream ss;
      ss << "PipelinePolicy: expert placement dispatch"
         << " owner=" << req.ownerId
         << " kind=" << cclScheduler::PlacementKindName(req.kind)
         << " trace_layer=" << req.trace_layer_id
         << " policy=" << (useTraceDevicePlacement
                              ? "trace_device_layer"
                              : State().config.expert_placement_policy)
         << " effective_policy=" << effectivePolicy
         << " offered_token_rate=" << OfferedInferenceTokenRate(State().config)
         << " posterior_access_samples=" << mnccl::GetObservedExpertAccessSamples()
         << " placement_access_mode=" << State().config.placement_access_mode
         << " placement_adaptive_mode=" << State().config.placement_adaptive_mode
         << " placement_weights=["
         << State().config.placement_replica_balance_weight << ","
         << State().config.placement_external_weight << ","
         << State().config.placement_l1_variance_weight << ","
         << State().config.placement_l1_max_weight << ","
         << State().config.placement_gpu_variance_weight << ","
         << State().config.placement_probe_weight << "]"
         << " topology_domains=" << domains.size()
         << " rtt_tiers=[";
      for (size_t i = 0; i < rttTiers.size(); ++i) {
        if (i != 0)
          ss << ",";
        ss << rttTiers[i];
      }
      ss << "]"
         << " access_drift=" << State().latest_access_drift
         << " drift_hot_update=" << (State().placement_drift_active ? 1 : 0)
         << " experts=" << plan.size()
         << " objective=" << score.objective
         << " l1_max_load=" << score.maxL1Load
         << " l1_variance=" << score.l1Variance
         << " l1_gpu_balance_variance=" << score.l1GpuBalanceVariance
         << " l1_gpu_balance_max=" << score.l1GpuBalanceMax
         << " external_access_cost=" << score.externalAccessCost
         << " replica_domain_imbalance=" << score.replicaDomainImbalance
         << " hop1_max=" << score.hop1Max
         << " hop1_mean=" << score.hop1Mean
         << " hop2_max=" << score.hop2Max
         << " hop2_mean=" << score.hop2Mean
         << " gpu_load_variance=" << score.gpuLoadVariance
         << " probe_path_cost=" << score.probePathCost;
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
    double bestScore = std::numeric_limits<double>::max();
    uint32_t srcNode = NetworkNodeId(req.sourceGpu);
    for (const auto& candidate : req.candidates) {
      auto highSample = mnccl::GetRouteProbeSample(State().config.probe_high_pg,
                                                   req.sourceGpu,
                                                   candidate);
      auto lowSample = mnccl::GetRouteProbeSample(State().config.probe_low_pg,
                                                  req.sourceGpu,
                                                  candidate);
      uint32_t dstNode = NetworkNodeId(candidate);
      bool hasProbeSample = highSample.found || lowSample.found;
      double queuePressure = mnccl::GetLocalGpuQueuePressure(candidate);
      double queueNorm = std::max(1.0, State().config.route_queue_norm);
      double score = 0.0;
      if (!hasProbeSample) {
        double staticRtt = static_cast<double>(
            mnccl::StaticRouteRttNs(req.sourceGpu, candidate));
        score = staticRtt +
                State().config.route_queue_weight *
                    (queuePressure / queueNorm) * std::max(1.0, staticRtt);
      } else {
        uint64_t high = highSample.found
                            ? highSample.fctNs
                            : ProbeRtt(srcNode,
                                       dstNode,
                                       State().config.probe_high_pg,
                                       State().config.probe_bytes);
        uint64_t low = lowSample.found
                           ? lowSample.fctNs
                           : ProbeRtt(srcNode,
                                      dstNode,
                                      State().config.probe_low_pg,
                                      State().config.probe_bytes);
        uint64_t delta = low > high ? low - high : high - low;
        double baseline = std::max(
            1.0,
            static_cast<double>(
                ProbeRtt(srcNode,
                         dstNode,
                         State().config.probe_high_pg,
                         State().config.probe_bytes)));
        double selectedRtt = static_cast<double>(std::min(high, low));
        score = (static_cast<double>(delta) / baseline) +
                0.25 * (selectedRtt / baseline) +
                State().config.route_queue_weight *
                    (queuePressure / queueNorm);
      }
      if (score < bestScore ||
          (std::fabs(score - bestScore) <= 1e-12 && candidate < best)) {
        bestScore = score;
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
    //      << " score=" << bestScore;
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

inline double FragmentationReleaseBenefit(const mnccl::LocalFlowTask& task);
inline double TopologyGradientReleaseBenefit(const mnccl::LocalFlowTask& task);

inline double FragmentationReleaseBenefit(const mnccl::LocalFlowTask& task) {
  std::vector<std::vector<uint64_t>> memUsed;
  std::vector<std::vector<uint64_t>> gpuMem;
  std::vector<std::vector<uint32_t>> expertUsed;
  uint32_t numNodes = 0;
  uint32_t gpusPerServer = 0;
  uint32_t maxExpertsPerGpu = 0;
  cclScheduler::GetResourceSnapshot(
      memUsed, gpuMem, expertUsed, numNodes, gpusPerServer, maxExpertsPerGpu);

  std::set<uint32_t> releaseNodes;
  auto snapshot = mnccl::GetPipelineTaskPolicySnapshotForJob(task.jobId);
  if (snapshot.found) {
    if (!snapshot.prefillPlacementReleased) {
      for (const auto& gpu : snapshot.prefillGpus)
        releaseNodes.insert(NetworkNodeId(gpu));
    }
    if (!snapshot.decodePlacementReleased) {
      for (const auto& gpu : snapshot.decodeGpus)
        releaseNodes.insert(NetworkNodeId(gpu));
    }
  } else {
    auto collective = mnccl::GetCollectivePolicySnapshot(task.jobId);
    if (!collective.found)
      return 0.0;
    for (const auto& gpu : collective.gpus)
      releaseNodes.insert(NetworkNodeId(gpu));
  }

  if (releaseNodes.empty())
    return 0.0;

  double before = mnccl::DocumentCnFragmentation(
      memUsed, expertUsed, numNodes, gpusPerServer);
  double after = mnccl::DocumentCnFragmentation(
      memUsed, expertUsed, numNodes, gpusPerServer, releaseNodes);
  return std::max(0.0, before - after);
}

inline std::set<uint32_t> ReleaseNodesForTask(const mnccl::LocalFlowTask& task) {
  std::set<uint32_t> releaseNodes;
  auto snapshot = mnccl::GetPipelineTaskPolicySnapshotForJob(task.jobId);
  if (snapshot.found) {
    if (!snapshot.prefillPlacementReleased) {
      for (const auto& gpu : snapshot.prefillGpus)
        releaseNodes.insert(NetworkNodeId(gpu));
    }
    if (!snapshot.decodePlacementReleased) {
      for (const auto& gpu : snapshot.decodeGpus)
        releaseNodes.insert(NetworkNodeId(gpu));
    }
    return releaseNodes;
  }

  auto collective = mnccl::GetCollectivePolicySnapshot(task.jobId);
  if (collective.found) {
    for (const auto& gpu : collective.gpus)
      releaseNodes.insert(NetworkNodeId(gpu));
  }
  return releaseNodes;
}

inline double ReleaseScopeWeight(const mnccl::LocalFlowTask& task) {
  auto releaseNodes = ReleaseNodesForTask(task);
  if (releaseNodes.empty())
    return 0.0;
  return 1.0 / std::sqrt(static_cast<double>(releaseNodes.size()));
}

inline uint32_t L0DomainOfNetworkNode(uint32_t node) {
  uint32_t stride = std::max(1u, gpus_per_server);
  if (stride == 1)
    return node / 8;
  return node / stride;
}

inline uint32_t L1DomainOfNetworkNode(uint32_t node) {
  uint32_t l1 = mnccl::L1GroupOfNetworkNode(node);
  if (l1 == node)
    l1 = L0DomainOfNetworkNode(node);
  return l1;
}

inline std::set<uint32_t> ExpandDomainNodes(uint32_t domainLevel,
                                            const std::set<uint32_t>& seedNodes,
                                            uint32_t numNodes,
                                            uint32_t gpusPerServer) {
  std::set<uint32_t> out;
  if (seedNodes.empty())
    return out;
  if (domainLevel >= 2) {
    for (uint32_t node = 0; node < numNodes; ++node) {
      for (uint32_t gpu = 0; gpu < gpusPerServer; ++gpu)
        out.insert(node * std::max(1u, gpusPerServer) + gpu);
    }
    return out;
  }

  std::set<uint32_t> domains;
  for (uint32_t node : seedNodes) {
    domains.insert(domainLevel == 0 ? L0DomainOfNetworkNode(node)
                                    : L1DomainOfNetworkNode(node));
  }
  for (uint32_t node = 0; node < numNodes; ++node) {
    for (uint32_t gpu = 0; gpu < gpusPerServer; ++gpu) {
      uint32_t networkNode = node * std::max(1u, gpusPerServer) + gpu;
      uint32_t domain = domainLevel == 0 ? L0DomainOfNetworkNode(networkNode)
                                         : L1DomainOfNetworkNode(networkNode);
      if (domains.count(domain))
        out.insert(networkNode);
    }
  }
  return out;
}

inline double DomainCnFragmentation(
    const std::vector<std::vector<uint64_t>>& memUsed,
    const std::vector<std::vector<uint32_t>>& expertUsed,
    uint32_t numNodes,
    uint32_t gpusPerServer,
    const std::set<uint32_t>& domainNodes,
    const std::set<uint32_t>& releaseNodes = {}) {
  if (domainNodes.empty())
    return 0.0;
  auto domainRelease = releaseNodes;
  std::vector<std::vector<uint64_t>> maskedMem = memUsed;
  std::vector<std::vector<uint32_t>> maskedExpert = expertUsed;
  for (uint32_t node = 0; node < numNodes; ++node) {
    for (uint32_t gpu = 0; gpu < gpusPerServer; ++gpu) {
      uint32_t networkNode = node * std::max(1u, gpusPerServer) + gpu;
      if (domainNodes.count(networkNode))
        continue;
      if (node < maskedMem.size() && gpu < maskedMem[node].size())
        maskedMem[node][gpu] = std::numeric_limits<uint64_t>::max() / 4;
      if (node < maskedExpert.size() && gpu < maskedExpert[node].size())
        maskedExpert[node][gpu] = std::numeric_limits<uint32_t>::max() / 4;
      domainRelease.erase(networkNode);
    }
  }
  return mnccl::DocumentCnFragmentation(
      maskedMem, maskedExpert, numNodes, gpusPerServer, domainRelease);
}

inline double TopologyGradientReleaseBenefit(const mnccl::LocalFlowTask& task) {
  std::vector<std::vector<uint64_t>> memUsed;
  std::vector<std::vector<uint64_t>> gpuMem;
  std::vector<std::vector<uint32_t>> expertUsed;
  uint32_t numNodes = 0;
  uint32_t gpusPerServer = 0;
  uint32_t maxExpertsPerGpu = 0;
  cclScheduler::GetResourceSnapshot(
      memUsed, gpuMem, expertUsed, numNodes, gpusPerServer, maxExpertsPerGpu);

  std::set<uint32_t> releaseNodes = ReleaseNodesForTask(task);
  if (releaseNodes.empty())
    return 0.0;

  const double weights[3] = {0.25, 0.35, 0.40};
  double score = 0.0;
  for (uint32_t level = 0; level < 3; ++level) {
    auto domainNodes = ExpandDomainNodes(level, releaseNodes, numNodes, gpusPerServer);
    double before = DomainCnFragmentation(
        memUsed, expertUsed, numNodes, gpusPerServer, domainNodes);
    double after = DomainCnFragmentation(
        memUsed, expertUsed, numNodes, gpusPerServer, domainNodes, releaseNodes);
    score += weights[level] * std::max(0.0, before - after);
  }
  return score;
}

inline double RemainingWorkEstimate(const mnccl::LocalFlowTask& task) {
  auto snapshot = mnccl::GetPipelineTaskPolicySnapshotForJob(task.jobId);
  double flowCost = static_cast<double>(std::max<uint64_t>(1, task.msgSize)) / 1048576.0;
  if (!snapshot.found) {
    auto collective = mnccl::GetCollectivePolicySnapshot(task.jobId);
    if (collective.found)
      return flowCost * static_cast<double>(std::max(1u, collective.outstanding));
    return flowCost;
  }

  uint32_t remainingDecode = snapshot.decodeIterations > snapshot.decodeIteration
                             ? snapshot.decodeIterations - snapshot.decodeIteration
                             : 0;
  double stageCost = 1.0;
  if (task.stage == mnccl::PipelineStage::PrefillAllReduce ||
      task.stage == mnccl::PipelineStage::PrefillAllToAll ||
      task.stage == mnccl::PipelineStage::KvCacheTransfer)
    stageCost += static_cast<double>(snapshot.prefillLength) / 1024.0;
  if (task.stage == mnccl::PipelineStage::DecodeAllReduce ||
      task.stage == mnccl::PipelineStage::DecodeAllToAll)
    stageCost += static_cast<double>(std::max(1u, remainingDecode)) *
                 static_cast<double>(std::max(1u, snapshot.decodeLength)) / 64.0;
  return flowCost * stageCost;
}

inline double SlaPressureScore(const mnccl::LocalFlowTask& task) {
  if (task.pdKind != cclScheduler::PlacementKind::Decode)
    return 0.0;
  auto snapshot = mnccl::GetPipelineTaskPolicySnapshotForJob(task.jobId);
  if (!snapshot.found)
    return 0.5;
  uint32_t remainingDecode = snapshot.decodeIterations > snapshot.decodeIteration
                             ? snapshot.decodeIterations - snapshot.decodeIteration
                             : 0;
  double progress = snapshot.decodeIterations == 0
                    ? 1.0
                    : static_cast<double>(snapshot.decodeIteration) /
                      static_cast<double>(snapshot.decodeIterations);
  double tailPressure = remainingDecode <= 1 ? 1.0 : 0.0;
  return 1.0 + progress + tailPressure;
}

inline double InterferenceCost(const mnccl::LocalFlowTask& task) {
  double msgMiB = static_cast<double>(task.msgSize) / 1048576.0;
  double pathPenalty = 0.0;
  uint64_t rtt = mnccl::PairRttOrMax(task.srcNode, task.dstNode);
  if (maxRtt > 0)
    pathPenalty = static_cast<double>(rtt) / static_cast<double>(maxRtt);
  double priorityPenalty = static_cast<double>(task.pg) * 0.05;
  return msgMiB + pathPenalty + priorityPenalty;
}

inline bool IsTrainingCollective(const mnccl::LocalFlowTask& task) {
  if (task.pdKind != cclScheduler::PlacementKind::Generic)
    return false;
  auto collective = mnccl::GetCollectivePolicySnapshot(task.jobId);
  return collective.found;
}

inline double CrispReleaseBenefit(const mnccl::LocalFlowTask& task) {
  std::string policy = Lower(Trim(State().config.local_flow_schedule_policy));
  const auto& cfg = State().config;
  double frag = FragmentationReleaseBenefit(task);
  double topo = TopologyGradientReleaseBenefit(task);
  double scopeWeight = ReleaseScopeWeight(task);
  if (scopeWeight <= 0.0)
    return 0.0;
  double release = frag + cfg.crisp_topology_mix * topo;
  if (TopologyGradientPolicyName(policy))
    release = topo + cfg.crisp_topology_mix * frag;
  return release * scopeWeight;
}

inline double CompletionProximityScore(const mnccl::LocalFlowTask& task) {
  double remaining = RemainingWorkEstimate(task);
  if (remaining <= 0.0)
    return 1.0;
  double normalized = std::log1p(remaining);
  return std::clamp(1.0 / (1.0 + normalized), 0.0, 1.0);
}

inline double NormalizedReleaseScore(const mnccl::LocalFlowTask& task) {
  double release = CrispReleaseBenefit(task);
  if (release <= 0.0)
    return 0.0;
  double completion = CompletionProximityScore(task);
  double remaining = RemainingWorkEstimate(task);
  double remainingPenalty = 1.0 / (1.0 + std::log1p(std::max(1.0, remaining)));
  return release * completion * remainingPenalty;
}

inline double CrispScore(const mnccl::LocalFlowTask& task, uint64_t nowNs) {
  const auto& cfg = State().config;
  double completion = CompletionProximityScore(task);
  double releaseScore = NormalizedReleaseScore(task);
  double sla = SlaPressureScore(task);
  double interference = InterferenceCost(task);
  double waitMs = nowNs >= task.arrivalTimeNs
                  ? static_cast<double>(nowNs - task.arrivalTimeNs) / 1000000.0
                  : 0.0;
  double age = std::log1p(waitMs);
  double stageBias = task.stage == mnccl::PipelineStage::KvCacheTransfer ? 0.35 : 0.0;
  double remainingPenalty = 1.0 - completion;

  return cfg.crisp_alpha * releaseScore +
         cfg.crisp_beta * sla -
         cfg.crisp_gamma * interference +
         cfg.crisp_completion_floor * completion +
         stageBias -
         cfg.crisp_remaining_penalty * remainingPenalty -
         cfg.crisp_eta * age +
         0.0;
}

inline double LocalEstimatedFctUs(const mnccl::LocalFlowTask& task) {
  uint64_t fctNs = mnccl::CalculateStandaloneFct(task.srcNode, task.dstNode, task.msgSize);
  return static_cast<double>(std::max<uint64_t>(1, fctNs)) / 1000.0;
}

inline double LocalEstimatedServiceNs(const mnccl::LocalFlowTask& task,
                                      uint64_t bytes) {
  uint64_t fctNs = mnccl::CalculateStandaloneFct(
      task.srcNode, task.dstNode, std::max<uint64_t>(1, bytes));
  return static_cast<double>(std::max<uint64_t>(1, fctNs));
}

inline size_t SelectPreemptiveLyapunovSrpt(const std::vector<mnccl::LocalFlowTask>& queue,
                                           uint64_t nowNs) {
  if (queue.empty())
    return 0;

  struct FlowGroup {
    uint64_t remainingBytes = 0;
    uint64_t arrivalNs = std::numeric_limits<uint64_t>::max();
    size_t firstIndex = 0;
    uint64_t firstSequence = std::numeric_limits<uint64_t>::max();
  };

  std::map<uint64_t, FlowGroup> groups;
  for (size_t i = 0; i < queue.size(); ++i) {
    const auto& task = queue[i];
    uint64_t groupKey = task.flowGroupKey == 0 ? task.key : task.flowGroupKey;
    uint64_t groupArrival = task.flowGroupArrivalTimeNs == 0
                            ? task.arrivalTimeNs
                            : task.flowGroupArrivalTimeNs;
    auto& group = groups[groupKey];
    group.remainingBytes += task.msgSize;
    group.arrivalNs = std::min(group.arrivalNs, groupArrival);
    if (task.sequence < group.firstSequence) {
      group.firstSequence = task.sequence;
      group.firstIndex = i;
    }
  }

  const auto& cfg = State().config;
  double vNs = static_cast<double>(std::max<uint64_t>(1, cfg.local_flow_pld_srpt_v_ns));
  double t0Ns = static_cast<double>(std::max<uint64_t>(1, cfg.local_flow_pld_srpt_t0_ns));
  double d0Ns = static_cast<double>(cfg.local_flow_pld_srpt_d0_ns);
  double kappa = std::max(0.0, cfg.local_flow_pld_srpt_kappa);
  double beta = std::max(0.0, cfg.local_flow_pld_srpt_beta);
  double overheadNs = static_cast<double>(cfg.local_flow_pld_srpt_preempt_overhead_ns);

  size_t best = 0;
  double bestScore = -std::numeric_limits<double>::infinity();
  double bestRemainingNs = std::numeric_limits<double>::infinity();
  uint64_t bestArrivalNs = std::numeric_limits<uint64_t>::max();
  uint64_t bestSeq = std::numeric_limits<uint64_t>::max();

  for (const auto& kv : groups) {
    const auto& group = kv.second;
    const auto& representative = queue[group.firstIndex];
    double remainingNs = LocalEstimatedServiceNs(representative, group.remainingBytes);
    double ageNs = nowNs >= group.arrivalNs
                   ? static_cast<double>(nowNs - group.arrivalNs)
                   : 0.0;
    double debtNs = std::max(0.0, ageNs - d0Ns - kappa * remainingNs);
    double numerator = vNs + beta * debtNs + 0.5 * debtNs * debtNs / t0Ns;
    double denominator = std::max(1.0, remainingNs + overheadNs);
    double score = numerator / denominator;
    if (!std::isfinite(score))
      score = -std::numeric_limits<double>::infinity();

    bool better = score > bestScore;
    if (!better && score == bestScore) {
      if (remainingNs < bestRemainingNs)
        better = true;
      else if (remainingNs == bestRemainingNs &&
               (group.arrivalNs < bestArrivalNs ||
                (group.arrivalNs == bestArrivalNs &&
                 group.firstSequence < bestSeq)))
        better = true;
    }
    if (better) {
      best = group.firstIndex;
      bestScore = score;
      bestRemainingNs = remainingNs;
      bestArrivalNs = group.arrivalNs;
      bestSeq = group.firstSequence;
    }
  }
  return best;
}

inline size_t SelectSlaThenMinLocalCompletion(const std::vector<mnccl::LocalFlowTask>& queue,
                                              uint64_t nowNs) {
  size_t best = 0;
  auto classOf = [](const mnccl::LocalFlowTask& task) {
    if (task.pdKind == cclScheduler::PlacementKind::Decode)
      return 0;
    if (task.stage == mnccl::PipelineStage::KvCacheTransfer)
      return 1;
    if (task.pdKind == cclScheduler::PlacementKind::Prefill)
      return 2;
    return 3;
  };
  auto score = [&](const mnccl::LocalFlowTask& task) {
    double waitUs = nowNs >= task.arrivalTimeNs
                    ? static_cast<double>(nowNs - task.arrivalTimeNs) / 1000.0
                    : 0.0;
    double localCompletion = waitUs + LocalEstimatedFctUs(task);
    return std::make_tuple(classOf(task), localCompletion, task.arrivalTimeNs, task.sequence);
  };
  auto bestScore = score(queue.front());
  for (size_t i = 1; i < queue.size(); ++i) {
    auto cur = score(queue[i]);
    if (cur < bestScore) {
      bestScore = cur;
      best = i;
    }
  }
  return best;
}

inline size_t SelectFragmentationBenefit(const std::vector<mnccl::LocalFlowTask>& queue,
                                         uint64_t nowNs) {
  size_t fallback = SelectDecodeFirstPrefillLater(queue);
  size_t best = fallback;
  double bestScore = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < queue.size(); ++i) {
    const auto& task = queue[i];
    double score = CrispScore(task, nowNs);
    if (task.pdKind == cclScheduler::PlacementKind::Generic &&
        NormalizedReleaseScore(task) < State().config.crisp_min_frag_gain)
      score -= 1.0;
    if (score > bestScore ||
        (score == bestScore &&
         (task.arrivalTimeNs < queue[best].arrivalTimeNs ||
          (task.arrivalTimeNs == queue[best].arrivalTimeNs &&
           task.sequence < queue[best].sequence)))) {
      bestScore = score;
      best = i;
    }
  }
  const auto& cfg = State().config;
  const auto& selected = queue[best];
  const auto& fallbackTask = queue[fallback];
  double selectedRelease = NormalizedReleaseScore(selected);
  double fallbackRelease = NormalizedReleaseScore(fallbackTask);
  double selectedSla = SlaPressureScore(selected);
  double fallbackSla = SlaPressureScore(fallbackTask);
  double requiredGain = std::max(cfg.crisp_min_frag_gain * 2.0, 0.02);
  if (best == fallback ||
      selectedSla >= std::max(0.0, fallbackSla + 0.15) ||
      selectedRelease >= fallbackRelease + requiredGain)
    return best;
  return fallback;
}

// ===== POLICY GROUP: per-GPU local flow queue scheduling =====
inline mnccl::LocalFlowSchedulePolicy LocalFlowSchedulePolicy() {
  return [](const std::vector<mnccl::LocalFlowTask>& queue, uint64_t nowNs) {
    // std::ostringstream start;
    // start << "PipelinePolicy: local flow queue schedule start"
    //       << " queue_size=" << queue.size()
    //       << " policy=" << State().config.local_flow_schedule_policy;
    // ccl::CclLog(start.str());
    size_t best = 0;
    std::string policy = Lower(Trim(State().config.local_flow_schedule_policy));
    if (policy == "decode_first_prefill_later" ||
        policy == "decode_first_prefill_last" ||
        policy == "decode_first" ||
        CrispPgOnlyPolicyName(policy))
      best = SelectDecodeFirstPrefillLater(queue);
    else if (policy == "pld_srpt" ||
             policy == "lyapunov_srpt" ||
             policy == "preemptive_lyapunov_srpt")
      best = SelectPreemptiveLyapunovSrpt(queue, nowNs);
    else if (CrispPolicyName(policy))
      best = SelectFragmentationBenefit(queue, nowNs);
    else if (TopologyGradientPolicyName(policy))
      best = SelectSlaThenMinLocalCompletion(queue, nowNs);
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

inline mnccl::LocalFlowPgPolicy LocalFlowPgPolicy() {
  return [](const mnccl::LocalFlowTask& task, uint64_t nowNs) {
    std::string policy = Lower(Trim(State().config.local_flow_schedule_policy));
    if (!(CrispPolicyName(policy) || CrispPgOnlyPolicyName(policy) ||
          TopologyGradientPolicyName(policy)))
      return task.pg;

    if (task.op != mnccl::CollectiveOp::AllReduce &&
        task.op != mnccl::CollectiveOp::AllToAll)
      return task.pg;

    const auto& cfg = State().config;
    double sla = SlaPressureScore(task);
    if (sla >= cfg.crisp_sla_urgent_threshold)
      return cfg.crisp_urgent_pg;
    double completion = CompletionProximityScore(task);
    double releaseScore = NormalizedReleaseScore(task);
    if (completion < cfg.crisp_completion_floor)
      return task.pg;
    if (releaseScore < cfg.crisp_min_frag_gain)
      return task.pg;

    auto collective = mnccl::GetCollectivePolicySnapshot(task.jobId);
    auto snapshot = mnccl::GetPipelineTaskPolicySnapshotForJob(task.jobId);
    if (collective.found && collective.need > 0) {
      uint32_t divisor = std::max(1u, cfg.crisp_boost_need_divisor);
      uint32_t boostWindow = std::max(1u, collective.need / divisor);
      boostWindow = std::min<uint32_t>(
          boostWindow, std::max(1u, cfg.crisp_boost_max_outstanding));
      if (collective.outstanding > boostWindow)
        return task.pg;
    } else if (!snapshot.found) {
      return task.pg;
    }

    double interference = InterferenceCost(task);
    double score = CrispScore(task, nowNs);
    if (interference > cfg.crisp_max_interference)
      return task.pg;
    return score > cfg.crisp_training_boost_score ? cfg.crisp_boost_pg : task.pg;
  };
}

inline void RegisterTrainingAllReduceWorkload(const PolicyConfig& cfg) {
  if (!cfg.enable_training_workload || cfg.train_num_tasks == 0)
    return;
  uint32_t needMin = std::min(cfg.train_need_min, cfg.train_need_max);
  uint32_t needMax = std::max(cfg.train_need_min, cfg.train_need_max);
  uint64_t msgMin = std::min(cfg.train_msg_size_min, cfg.train_msg_size_max);
  uint64_t msgMax = std::max(cfg.train_msg_size_min, cfg.train_msg_size_max);

  std::mt19937 rng(cfg.train_seed);
  std::uniform_int_distribution<uint32_t> needDist(needMin, needMax);
  std::uniform_int_distribution<uint64_t> msgDist(msgMin, msgMax);
  std::vector<uint32_t> trainNeedSet;
  for (uint32_t need : cfg.train_need_set) {
    if (need >= 2)
      trainNeedSet.push_back(need);
  }
  std::uniform_int_distribution<size_t> needSetDist(
      0, trainNeedSet.empty() ? 0 : trainNeedSet.size() - 1);
  for (uint32_t i = 0; i < cfg.train_num_tasks; ++i) {
    uint32_t jobId = 0x40000000u + i;
    uint32_t need = trainNeedSet.empty()
                    ? std::max(2u, needDist(rng))
                    : trainNeedSet[needSetDist(rng)];
    uint64_t msgSize = std::max<uint64_t>(1, msgDist(rng));
    double submitTime = cfg.train_first_submit_time + cfg.train_submit_interval * i;
    mnccl::SubmitTrainingAllReduce(jobId,
                                   submitTime,
                                   cfg.train_pg,
                                   need,
                                   msgSize,
                                   cfg.train_mem_bytes_per_gpu);
    if (cfg.print_submissions) {
      std::ostringstream ss;
      ss << "PipelinePolicy: scheduled training allreduce"
         << " job=" << jobId
         << " need=" << need
         << " msg_size=" << msgSize
         << " submit_time=" << submitTime
         << " pg=" << cfg.train_pg;
      ccl::CclLog(ss.str());
    }
  }
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
  RefreshPosteriorExpertAccessFreq();
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
  uint32_t bestNeed = std::max(1u, state.need_prefill);
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
       << " objective=" << bestScore.plan.objective
       << " max_l1_load=" << bestScore.plan.maxL1Load
       << " l1_variance=" << bestScore.plan.l1Variance
       << " l1_gpu_balance_variance=" << bestScore.plan.l1GpuBalanceVariance
       << " l1_gpu_balance_max=" << bestScore.plan.l1GpuBalanceMax
       << " external_access_cost=" << bestScore.plan.externalAccessCost
       << " replica_domain_imbalance=" << bestScore.plan.replicaDomainImbalance
       << " hop1_max=" << bestScore.plan.hop1Max
       << " hop1_mean=" << bestScore.plan.hop1Mean
       << " hop2_max=" << bestScore.plan.hop2Max
       << " hop2_mean=" << bestScore.plan.hop2Mean;
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
         << " dispatch_N=" << prefillLength
         << " current_dispatch_need=" << state.need_prefill
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
       << " dispatch_N=" << prefillLength
       << " dispatch_need=" << state.need_prefill
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
  ResetRuntimeDecisionState();
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
  params.runtime.dispatch_n = std::max(1u, cfg.dispatch_n);
  params.runtime.dispatch_expert_ffn_params =
      std::max<uint64_t>(1, cfg.dispatch_expert_ffn_params);
  params.runtime.dispatch_precision_bytes =
      std::max(1u, cfg.dispatch_precision_bytes);
  params.runtime.dispatch_batch_size =
      std::max(1u, cfg.dispatch_batch_size);
  params.distribution.num_tasks = cfg.num_tasks;
  params.distribution.seed = cfg.task_seed;
  params.distribution.dispatch_n = std::max(1u, cfg.dispatch_n);
  params.distribution.dispatch_n_min = cfg.dispatch_n_min;
  params.distribution.dispatch_n_max = cfg.dispatch_n_max;
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
  State().trace_device_experts.clear();
  State().trace_device_experts_by_layer.clear();
  State().trace_access_targets.clear();
  State().trace_access_targets_by_layer.clear();
  mnccl::ClearDispatchTraceTargets();
  if (cfg.trace_dispatch_enable) {
    State().trace_device_experts =
        LoadTraceDeviceLayerPlacement(cfg.trace_device_file, cfg.trace_layer_id);
    State().trace_device_experts_by_layer =
        LoadTraceDeviceAllLayerPlacements(cfg.trace_device_file,
                                          cfg.trace_moe_layer_count,
                                          cfg.trace_layer_id);
    State().trace_access_targets_by_layer =
        LoadTraceDecodeAllLayerTargets(cfg.trace_decode_dir,
                                       State().trace_device_experts_by_layer,
                                       params.runtime.expert_num);
    if (State().trace_access_targets_by_layer.empty() &&
        !State().trace_device_experts.empty()) {
      State().trace_access_targets_by_layer.push_back(
          LoadTraceDecodeLayerTargets(cfg.trace_decode_dir,
                                      State().trace_device_experts,
                                      cfg.trace_layer_id,
                                      params.runtime.expert_num));
    }
    ScaleTraceAccessTargetsByLayer(State().trace_access_targets_by_layer,
                                   cfg.trace_access_divisor);
    if (cfg.trace_uniform_access_targets) {
      MakeTraceAccessTargetsUniformByLayer(
          State().trace_access_targets_by_layer,
          params.runtime.expert_num,
          cfg.trace_uniform_access_target_count);
    }
    State().trace_access_targets =
        AggregateTraceAccessTargetsByLayer(State().trace_access_targets_by_layer,
                                           params.runtime.expert_num);
    if (!State().trace_access_targets.empty()) {
      params.runtime.expert_num = std::max<uint32_t>(
          params.runtime.expert_num,
          static_cast<uint32_t>(State().trace_access_targets.size()));
      State().config.expert_num = params.runtime.expert_num;
      State().current_need.expert_num = params.runtime.expert_num;
      if (cfg.trace_access_targets_as_expert_freq) {
        std::vector<double> traceFreq(State().trace_access_targets.size(), 1.0);
        for (size_t i = 0; i < State().trace_access_targets.size(); ++i)
          traceFreq[i] = State().trace_access_targets[i] > 0
                             ? static_cast<double>(State().trace_access_targets[i])
                             : 1.0;
        State().config.expert_access_freq = traceFreq;
      }
    }
    uint64_t totalTarget = std::accumulate(State().trace_access_targets.begin(),
                                           State().trace_access_targets.end(),
                                           uint64_t{0});
    params.trace_dispatch_enabled =
        !State().trace_access_targets.empty() && totalTarget > 0;
    if (params.trace_dispatch_enabled) {
      uint64_t accessesPerTask =
          static_cast<uint64_t>(std::max(1u, params.runtime.need_prefill)) *
          static_cast<uint64_t>(std::max(1u, params.runtime.kflows));
      uint64_t estimatedTasks =
          accessesPerTask == 0 ? 1 : (totalTarget + accessesPerTask - 1) / accessesPerTask;
      double estimatedStop =
          params.distribution.first_submit_time +
          params.distribution.submit_interval * static_cast<double>(estimatedTasks + 1) +
          10.0;
      params.simulation_stop_time =
          std::max(params.simulation_stop_time, estimatedStop);
      std::ostringstream ss;
      ss << "PipelinePolicy: trace dispatch enabled"
         << " placement_layer=" << cfg.trace_layer_id
         << " moe_layers=" << State().trace_access_targets_by_layer.size()
         << " target_accesses=" << totalTarget
         << " estimated_tasks=" << estimatedTasks
         << " safety_stop_time=" << params.simulation_stop_time;
      ccl::CclLog(ss.str());
    }
  }
  uint32_t traceLayerCount = static_cast<uint32_t>(State().trace_device_experts_by_layer.size());
  if (traceLayerCount == 0 && cfg.trace_moe_layer_count > 0)
    traceLayerCount = cfg.trace_moe_layer_count;
  if (traceLayerCount == 0 && !State().trace_device_experts.empty())
    traceLayerCount = 1;
  uint32_t maxExpertsPerGpu =
      cfg.trace_dispatch_enable
          ? std::max(1u, cfg.expert_per_gpu) * std::max(1u, traceLayerCount)
          : 9;
  cclScheduler::SetMaxExpertsPerGpu(maxExpertsPerGpu);
  {
    std::ostringstream ss;
    ss << "PipelinePolicy: expert capacity configured"
       << " max_experts_per_gpu=" << maxExpertsPerGpu
       << " trace_layers=" << std::max(1u, traceLayerCount)
       << " expert_per_gpu=" << cfg.expert_per_gpu;
    ccl::CclLog(ss.str());
  }
  mnccl::SetRouteProbeMaxInflight(cfg.route_probe_max_inflight);
  mnccl::SetAsyncRouteProbeConfig(cfg.async_route_probe_enable,
                                  cfg.async_route_live_route,
                                  cfg.probe_high_pg,
                                  cfg.probe_low_pg,
                                  cfg.probe_bytes,
                                  cfg.async_route_probe_interval_ns,
                                  cfg.async_route_probe_budget,
                                  cfg.async_route_probe_refresh_ns,
                                  cfg.async_route_probe_topk);
  mnccl::SetLocalFlowPreemptiveChunkBytes(cfg.local_flow_preemptive_chunk_bytes);
  cclScheduler::SetPrintExpertPlacementDetail(cfg.print_expert_placement_detail);
  mnccl::SetGlobalProbTable(State().config.expert_access_freq);
  mnccl::ResetExpertAccessStats(params.runtime.expert_num, cfg.expert_access_prior);
  if (params.trace_dispatch_enabled) {
    mnccl::SetDispatchTraceTargets(State().trace_access_targets_by_layer,
                                   cfg.trace_stop_on_complete);
  }
  if (!cfg.expert_access_freq_after.empty()) {
    double switchTime = cfg.first_submit_time +
                        cfg.submit_interval *
                            static_cast<double>(cfg.expert_access_switch_task);
    auto nextFreq = cfg.expert_access_freq_after;
    Simulator::Schedule(Seconds(std::max(0.0, switchTime)), [nextFreq]() {
      mnccl::SetGlobalProbTable(nextFreq);
      ccl::CclLog("PipelinePolicy: expert access distribution switched for drift experiment");
    });
  }
  RefreshPlacementAccessFreqForPolicy();

  if (cfg.trace_dispatch_enable &&
      cfg.trace_use_device_placement &&
      !State().trace_device_experts.empty()) {
    ccl::CclLog("PipelinePolicy: trace device expert placement policy dispatch name=trace_device_layer");
    params.expert_placement_policy = ExpertPlacementPolicy();
  }

  if (!cfg.enable_policy_group) {
    ccl::CclLog("PipelinePolicy: policy group disabled, using built-in defaults");
    return params;
  }

  ccl::CclLog("PipelinePolicy: policy config apply start");
  {
    std::ostringstream ss;
    ss << "PipelinePolicy: local flow preemption config"
       << " chunk_bytes=" << cfg.local_flow_preemptive_chunk_bytes
       << " pld_srpt_v_ns=" << cfg.local_flow_pld_srpt_v_ns
       << " pld_srpt_t0_ns=" << cfg.local_flow_pld_srpt_t0_ns
       << " pld_srpt_d0_ns=" << cfg.local_flow_pld_srpt_d0_ns
       << " pld_srpt_kappa=" << cfg.local_flow_pld_srpt_kappa
       << " pld_srpt_beta=" << cfg.local_flow_pld_srpt_beta
       << " pld_srpt_preempt_overhead_ns="
       << cfg.local_flow_pld_srpt_preempt_overhead_ns;
    ccl::CclLog(ss.str());
  }
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
  if (PolicyHookEnabled(cfg.local_flow_schedule_policy)) {
    ccl::CclLog("PipelinePolicy: local flow PG policy dispatch name=" + cfg.local_flow_schedule_policy);
    mnccl::SetLocalFlowPgPolicy(LocalFlowPgPolicy());
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
