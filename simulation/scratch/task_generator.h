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
  uint32_t need_prefill = 64;
  uint32_t need_decode = 64;
  uint32_t expert_num = 128;
  uint32_t kflows = 2;
  uint32_t single_token_length = 64;
  uint64_t expert_mem_bytes = 64ULL * 1024ULL * 1024ULL;
  uint64_t prefill_unit_msg_size = 1ULL * 1024ULL * 1024ULL;
  uint64_t prefill_alltoall_msg_size = 512ULL * 1024ULL;
  uint64_t token_msg_size = 512ULL * 1024ULL;
  uint64_t base_decode_compute_delay_ns = 50000;
};

struct PipelineTaskDistributionParams {
  uint32_t num_tasks = 5;
  uint32_t seed = 12345;
  uint32_t prefill_length_min = 512;
  uint32_t prefill_length_max = 2048;
  uint32_t decode_length_min = 64;
  uint32_t decode_length_max = 192;
  double first_submit_time = 0.00001;
  double submit_interval = 0.00001;
};

struct PipelineWorkloadParams {
  PipelineRuntimeParams runtime;
  PipelineTaskDistributionParams distribution;
  cclScheduler::NeedUpdateCallback need_update_callback;
  bool print_submissions = true;
  std::ostream* output = &std::cout;
};

inline cclScheduler::NeedUpdateCallback DefaultNeedUpdateCallback() {
  return [](cclScheduler::NeedEvent /*event*/,
            uint32_t /*taskId*/,
            uint32_t /*prefillLength*/,
            uint32_t /*decodeLength*/,
            cclScheduler::NeedState& /*state*/) {
    // Hook for external policies. Mutate state.need_prefill /
    // state.need_decode / state.expert_num here.
  };
}

inline mnccl::RuntimeConfig MakeRuntimeConfig(const PipelineRuntimeParams& params) {
  return mnccl::RuntimeConfig{
      params.pg,
      params.need_prefill,
      params.need_decode,
      params.expert_num,
      params.kflows,
      params.single_token_length,
      params.expert_mem_bytes,
      params.prefill_unit_msg_size,
      params.prefill_alltoall_msg_size,
      params.token_msg_size,
      params.base_decode_compute_delay_ns};
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

inline void SubmitPipelineTasks(const std::vector<mnccl::PipelineTask>& tasks,
                                bool print_submissions,
                                std::ostream* output) {
  for (const auto& task : tasks) {
    mnccl::SubmitPipelineTask(task);
    if (print_submissions && output != nullptr) {
      *output << "NormalNetwork: scheduled task " << task.taskId
              << " prefill_length=" << task.prefillLength
              << " decode_length=" << task.decodeLength
              << " at " << task.submitTime << "s\n";
    }
  }
}

inline PipelineWorkloadParams DefaultPipelineWorkloadParams() {
  return PipelineWorkloadParams{};
}

inline void RegisterPipelineWorkload(const PipelineWorkloadParams& params) {
  ConfigurePipelineRuntime(params.runtime);
  cclScheduler::SetNeedUpdateCallback(
      params.need_update_callback ? params.need_update_callback : DefaultNeedUpdateCallback());
  auto tasks = GenerateCurrentPipelineTaskDistribution(params.distribution);
  SubmitPipelineTasks(tasks, params.print_submissions, params.output);
}

} // namespace taskGenerator

#endif // TASK_GENERATOR_H
