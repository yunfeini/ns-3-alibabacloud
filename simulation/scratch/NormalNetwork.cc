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

  CommandLine cmd;
  cmd.AddValue("commscale", "Communication Scale", comm_scale);
  cmd.Parse(argc, argv);

  clock_t begint, endt;
  begint = clock();

  if (!ReadConf(argc, argv))
    return -1;
  SetConfig();
  SetupNetwork(qp_finish_normal, send_finish_normal, message_finish_normal);

  uint32_t gpu_num = node_num - switch_num - nvswitch_num;
  uint32_t scheduler_nodes = std::max(1u, gpu_num / std::max(1u, gpus_per_server));
  cclScheduler::Init(scheduler_nodes, std::max(1u, gpus_per_server));
  mnccl::Init();

  const std::string mncc_log = "mncc.log";
  const std::string mncc_flow_finish_log = "mncc_flow_finish.csv";
  mnccl::SetFlowFinishLogPath(mncc_flow_finish_log);
  ccl::SetCclLogPath(mncc_log);
  {
    std::ofstream ofs(mncc_log, std::ofstream::trunc);
  }
  {
    std::ofstream ofs(mncc_flow_finish_log, std::ofstream::trunc);
    ofs << "finish_time_ns,jobId,srcNode,dstNode,pg,sport,dport,msg_size,"
        << "qp_start_time_step,actual_fct,standalone_fct\n";
  }

  auto workload_params = taskGenerator::DefaultPipelineWorkloadParams();
  taskGenerator::RegisterPipelineWorkload(workload_params);

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
