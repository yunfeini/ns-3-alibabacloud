#ifndef TASK_GENERATOR_H
#define TASK_GENERATOR_H

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <random>
#include <vector>

#include "cclscheduler.h"
#include "mnCCL.h"

namespace taskGenerator {

struct PipelineRuntimeParams {
  uint16_t pg = mnccl::default_pg;
  uint32_t need_prefill = 128;
  uint32_t need_decode = 32;
  uint32_t expert_num = 64;
  uint32_t expert_per_gpu = 1;
  uint32_t kflows = 8;
  uint32_t single_token_length = 2;
  uint64_t expert_mem_bytes = 64ULL * 1024ULL * 1024ULL;
  uint64_t token_msg_size = 2;
  uint64_t base_decode_compute_delay_ns = 1;
  uint32_t dispatch_n = 1;
  uint64_t dispatch_expert_ffn_params = 1000000000ULL;
  uint32_t dispatch_precision_bytes = 1;
  uint32_t dispatch_batch_size = 16;
};

struct PipelineTaskDistributionParams {
  uint32_t num_tasks = 3;
  uint32_t seed = 12345;
  uint32_t dispatch_n = 1;
  uint32_t dispatch_n_min = 0;
  uint32_t dispatch_n_max = 0;
  uint32_t prefill_length_min = 8192;
  uint32_t prefill_length_max = 16384;
  uint32_t decode_length_min = 256;
  uint32_t decode_length_max = 2048;
  double first_submit_time = 0.00001;
  double submit_interval = 0.00001;
};

struct PipelineWorkloadParams {
  PipelineRuntimeParams runtime;
  PipelineTaskDistributionParams distribution;

  // ===== SCHEDULING POLICY INTERFACE: global need update on task events =====
  cclScheduler::NeedUpdateCallback need_update_callback;

  // ===== SCHEDULING POLICY INTERFACE: dispatch GPU placement =====
  cclScheduler::NeedPlacementPolicy need_placement_policy;

  // ===== SCHEDULING POLICY INTERFACE: expert placement within selected GPUs =====
  cclScheduler::ExpertPlacementPolicy expert_placement_policy;

  // ===== SCHEDULING POLICY INTERFACE: per-GPU local flow queue scheduling =====
  mnccl::LocalFlowSchedulePolicy local_flow_schedule_policy;

  // ===== SCHEDULING POLICY INTERFACE: same-rank expert replica route selection =====
  mnccl::ExpertRoutePolicy expert_route_policy;

  // Legacy hook retained for compatibility; dispatch-only workload does not use it.
  mnccl::PdSplitPolicy pd_split_policy;

  double simulation_stop_time = 1000.0;
  bool print_submissions = true;
  bool trace_dispatch_enabled = false;
  std::ostream* output = &std::cout;
};

inline cclScheduler::NeedUpdateCallback DefaultNeedUpdateCallback() {
  return [](cclScheduler::NeedEvent /*event*/,
            uint32_t /*taskId*/,
            uint32_t /*prefillLength*/,
            uint32_t /*decodeLength*/,
            cclScheduler::NeedState& /*state*/) {
    // Hook for external policies. In dispatch-only mode, state.need_prefill is
    // interpreted as dispatch_need; state.need_decode is kept as a compatibility
    // mirror for older policy code.
  };
}

inline mnccl::RuntimeConfig MakeRuntimeConfig(const PipelineRuntimeParams& params) {
  return mnccl::RuntimeConfig{
      params.pg,
      params.need_prefill,
      params.need_decode,
      params.expert_num,
      params.expert_per_gpu,
      params.kflows,
      params.single_token_length,
      params.expert_mem_bytes,
      params.token_msg_size,
      params.base_decode_compute_delay_ns,
      params.dispatch_n,
      params.dispatch_expert_ffn_params,
      params.dispatch_precision_bytes,
      params.dispatch_batch_size};
}

inline void ConfigurePipelineRuntime(const PipelineRuntimeParams& params) {
  mnccl::ConfigureRuntime(MakeRuntimeConfig(params));
}

inline std::vector<mnccl::PipelineTask> GenerateCurrentPipelineTaskDistribution(
    const PipelineTaskDistributionParams& params) {
  uint32_t prefill_min = std::min(params.prefill_length_min, params.prefill_length_max);
  uint32_t prefill_max = std::max(params.prefill_length_min, params.prefill_length_max);
  uint32_t decode_min = std::min(params.decode_length_min, params.decode_length_max);
  uint32_t decode_max = std::max(params.decode_length_min, params.decode_length_max);

  std::mt19937 rng(params.seed);
  std::uniform_int_distribution<uint32_t> dist_prefill_len(prefill_min, prefill_max);
  std::uniform_int_distribution<uint32_t> dist_decode_len(decode_min, decode_max);

  std::vector<mnccl::PipelineTask> tasks;
  tasks.reserve(params.num_tasks);
  for (uint32_t t = 0; t < params.num_tasks; ++t) {
    tasks.push_back(mnccl::PipelineTask{
        mnccl::TID++,
        params.first_submit_time + params.submit_interval * t,
        dist_prefill_len(rng),
        dist_decode_len(rng)});
  }
  return tasks;
}

inline std::vector<mnccl::DispatchTask> GenerateCurrentDispatchTaskDistribution(
    const PipelineTaskDistributionParams& params) {
  std::vector<mnccl::DispatchTask> tasks;
  tasks.reserve(params.num_tasks);
  uint32_t dispatchMin = params.dispatch_n_min == 0
                         ? std::max(1u, params.dispatch_n)
                         : params.dispatch_n_min;
  uint32_t dispatchMax = params.dispatch_n_max == 0
                         ? std::max(1u, params.dispatch_n)
                         : params.dispatch_n_max;
  if (dispatchMin > dispatchMax)
    std::swap(dispatchMin, dispatchMax);
  std::mt19937 rng(params.seed);
  std::uniform_int_distribution<uint32_t> dist_dispatch_n(dispatchMin, dispatchMax);
  for (uint32_t t = 0; t < params.num_tasks; ++t) {
    tasks.push_back(mnccl::DispatchTask{
        mnccl::TID++,
        params.first_submit_time + params.submit_interval * t,
        dist_dispatch_n(rng)});
  }
  return tasks;
}

inline void SubmitPipelineTasks(const std::vector<mnccl::PipelineTask>& tasks,
                                bool print_submissions,
                                std::ostream* output) {
  for (const auto& task : tasks) {
    mnccl::SubmitPipelineTask(task);
    if (print_submissions && output != nullptr) {
      *output << "NormalNetwork: scheduled task " << task.taskId
              << " prefill_length=" << task.prefillLength
              << " decode_length=" << task.decodeLength
              << " at " << task.submitTime << "s" << std::endl;
    }
  }
}

inline void SubmitDispatchTasks(const std::vector<mnccl::DispatchTask>& tasks,
                                bool print_submissions,
                                std::ostream* output) {
  for (const auto& task : tasks) {
    mnccl::SubmitDispatchTask(task);
    if (print_submissions && output != nullptr) {
      *output << "NormalNetwork: scheduled dispatch task " << task.taskId
              << " dispatch_N=" << task.dispatchN
              << " at " << task.submitTime << "s" << std::endl;
    }
  }
}

inline PipelineWorkloadParams DefaultPipelineWorkloadParams() {
  return PipelineWorkloadParams{};
}

inline void RegisterPipelineWorkload(const PipelineWorkloadParams& params) {
  ccl::CclLog("PipelinePolicy: register workload start");
  ConfigurePipelineRuntime(params.runtime);
  ccl::CclLog("PipelinePolicy: dispatch hooks install start");
  cclScheduler::SetNeedUpdateCallback(
      params.need_update_callback ? params.need_update_callback : DefaultNeedUpdateCallback());
  cclScheduler::SetNeedPlacementPolicy(params.need_placement_policy);
  cclScheduler::SetExpertPlacementPolicy(params.expert_placement_policy);
  mnccl::SetLocalFlowSchedulePolicy(params.local_flow_schedule_policy);
  mnccl::SetExpertRoutePolicy(params.expert_route_policy);
  mnccl::SetPdSplitPolicy(params.pd_split_policy);
  ccl::CclLog("PipelinePolicy: dispatch hooks install done");
  if (params.trace_dispatch_enabled) {
    ccl::CclLog("PipelinePolicy: trace dispatch generation start");
    mnccl::StartTraceDispatchWorkload(params.distribution.first_submit_time,
                                      params.distribution.submit_interval,
                                      params.distribution.dispatch_n,
                                      params.print_submissions,
                                      params.output);
    ccl::CclLog("PipelinePolicy: trace dispatch generation armed");
    return;
  }
  auto tasks = GenerateCurrentDispatchTaskDistribution(params.distribution);
  ccl::CclLog("PipelinePolicy: task generation start num_tasks=" +
              std::to_string(tasks.size()));
  SubmitDispatchTasks(tasks, params.print_submissions, params.output);
  ccl::CclLog("PipelinePolicy: task generation and dispatch done");
}

} // namespace taskGenerator

#endif // TASK_GENERATOR_H
