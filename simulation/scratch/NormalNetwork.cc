#include <execinfo.h>
#include <stdio.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include "common.h"
#include "mnCCL.h"
#include "cclscheduler.h"
#include "task_generator.h"
#include "pipeline_policy_config.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#ifdef NS3_MTP
#include "ns3/mtp-interface.h"
#endif
#ifdef NS3_MPI
#include <mpi.h>
#include "ns3/mpi-interface.h"
#endif

using namespace std;
using namespace ns3;

extern uint32_t node_num, switch_num, link_num, trace_num, nvswitch_num,
    gpus_per_server;

extern std::unordered_map<uint32_t, unordered_map<uint32_t, uint16_t>>
    portNumber;

extern std::ifstream flowf;
extern FlowInput flow_input;

uint32_t flow_num_finished = 0;

void qp_finish_normal(FILE* fout, Ptr<RdmaQueuePair> q) {
  uint32_t did = ip_to_node_id(q->dip);
#ifdef NS3_MTP
  MtpInterface::explicitCriticalSection cs;
#endif
  Ptr<Node> dstNode = n.Get(did);
  Ptr<RdmaDriver> rdma = dstNode->GetObject<RdmaDriver>();
  rdma->m_rdma->DeleteRxQp(q->sip.Get(), q->m_pg, q->sport);
#ifdef NS3_MTP
  cs.ExitSection();
#endif
}

void send_finish_normal(FILE* fout, Ptr<RdmaQueuePair> q) {
}

void message_finish_normal(FILE* fout, Ptr<RdmaQueuePair> q, uint64_t msgSize){
  bool collective_finished = mnccl::OnMessageFinish(fout, q, msgSize);
  if (collective_finished)
    cclScheduler::ScheduleTask();
  uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
  uint32_t pg = q->m_pg;
  uint64_t size = msgSize;
  uint64_t standalone_fct = mnccl::CalculateStandaloneFct(sid, did, size);
  fprintf(
      fout,
      "%u %u %u %u %lu %lu %lu %lu %u\n",
      sid,
      did,
      q->sport,
      q->dport,
      size,
      q->startTime.GetTimeStep(),
      (Simulator::Now() - q->startTime).GetTimeStep(),
      standalone_fct,
      pg);
  fflush(fout);
}

int main(int argc, char* argv[]) {
#ifdef NS3_MTP
  MtpInterface::Enable(16);
#endif

#ifdef NS3_MPI
  ns3::MpiInterface::Enable(&argc, &argv);
#endif

  float comm_scale = 1;
  std::string network_config_file = "./examples/HW/test.sh";
  std::string policy_config_file = "./scratch/pipeline_policy.conf";

  CommandLine cmd;
  cmd.AddValue("commscale", "Communication Scale", comm_scale);
  cmd.AddValue("networkConfig", "Network topology/config file consumed by ReadConf", network_config_file);
  cmd.AddValue("policyConfig", "Pipeline scheduling policy config file", policy_config_file);
  cmd.Parse(argc, argv);

  clock_t begint, endt;
  begint = clock();

  if (!network_config_file.empty()) {
    std::vector<std::string> conf_args = {argv[0], network_config_file};
    std::vector<char*> conf_argv;
    conf_argv.reserve(conf_args.size());
    for (auto& arg : conf_args)
      conf_argv.push_back(arg.data());
    if (!ReadConf(static_cast<int>(conf_argv.size()), conf_argv.data()))
      return -1;
  } else if (!ReadConf(1, argv)) {
    return -1;
  }
  SetConfig();
  SetupNetwork(qp_finish_normal, send_finish_normal, message_finish_normal);

  uint32_t gpu_num = node_num - switch_num - nvswitch_num;
  uint32_t scheduler_nodes = std::max(1u, gpu_num / std::max(1u, gpus_per_server));
  cclScheduler::Init(scheduler_nodes, std::max(1u, gpus_per_server));
  mnccl::Init();

  const std::string mncc_log = "mncc.log";
  const std::string mncc_flow_finish_log = "mncc_flow_finish.csv";
  const std::string mncc_cluster_timeseries_log = "mncc_cluster_timeseries.csv";
  mnccl::SetFlowFinishLogPath(mncc_flow_finish_log);
  mnccl::SetClusterMonitorLogPath(mncc_cluster_timeseries_log);
  ccl::SetCclLogPath(mncc_log);
  {
    std::ofstream ofs(mncc_log, std::ofstream::trunc);
  }
  {
    std::ofstream ofs(mncc_flow_finish_log, std::ofstream::trunc);
    ofs << "finish_time_ns,jobId,srcNode,dstNode,pg,sport,dport,msg_size,"
        << "qp_start_time_step,actual_fct_ns,standalone_fct_ns\n";
  }
  {
    std::ofstream ofs(mncc_cluster_timeseries_log, std::ofstream::trunc);
    ofs << "time_ns,active_tasks,outstanding_collectives,queued_local_flows,"
        << "active_local_gpus,total_gpus,used_gpus,total_expert_slots,"
        << "used_expert_slots,free_expert_slots,max_free_expert_slots,"
        << "gpu_utilization,expert_slot_utilization,expert_fragmentation,"
        << "free_slot_variance,placement_gpu_utilization,used_gpu_mem_bytes,"
        << "total_gpu_mem_bytes,cn_fragmentation,sched_success_rate,"
        << "fragment_block_rate,complete_8gpu_hosts,same_tor_8gpu_blocks,"
        << "free_gpus\n";
  }

  auto workload_params = taskGenerator::DefaultPipelineWorkloadParams();
  auto policy_cfg = pipelinePolicy::LoadPolicyConfig(policy_config_file);
  workload_params = pipelinePolicy::ApplyPolicyConfig(workload_params, policy_cfg);
  if (policy_cfg.enable_inference_workload)
    taskGenerator::RegisterPipelineWorkload(workload_params);
  pipelinePolicy::RegisterTrainingAllReduceWorkload(policy_cfg);
  mnccl::StartClusterTimeSeriesMonitor(policy_cfg.cluster_monitor_interval_ns);

  Simulator::Stop(Seconds(workload_params.simulation_stop_time));
  printf("Simulation Start.\n");
  Simulator::Run();
  mnccl::LogPipelineSummariesAtSimulationEnd();
  printf("Simulation Complete.\n");
  Simulator::Destroy();

  endt = clock();
  (void)begint;
  (void)endt;
  return 0;
}
