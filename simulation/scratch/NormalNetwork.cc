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


// void ReadFlowInput() {
//   if (flow_input.idx < flow_num) {
//     flowf >> flow_input.src >> flow_input.dst >> flow_input.pg >>
//         flow_input.dport >> flow_input.maxPacketCount >> flow_input.start_time;
//     NS_ASSERT(
//         n.Get(flow_input.src)->GetNodeType() == 0 &&
//         n.Get(flow_input.dst)->GetNodeType() == 0);
//   }
// }
// void ScheduleFlowInputs() {
//   while (flow_input.idx < flow_num &&
//          Seconds(flow_input.start_time) == Simulator::Now()) {
//     uint32_t port =
//         portNumber[flow_input.src][flow_input.dst]++; // get a new port number
//     // re_priority(flow_input);

//     RdmaClientHelper clientHelper(
//         (uint16_t)flow_input.pg,
//         serverAddress[flow_input.src],
//         serverAddress[flow_input.dst],
//         port,
//         flow_input.dport,
//         flow_input.maxPacketCount,
//         has_win ? (global_t == 1
//                        ? maxBdp
//                        : pairBdp[n.Get(flow_input.src)][n.Get(flow_input.dst)])
//                 : 0,
//         global_t == 1 ? maxRtt : pairRtt[flow_input.src][flow_input.dst],
//         nullptr,
//         nullptr,
//         1,
//         flow_input.src,
//         flow_input.dst);
//     ApplicationContainer appCon = clientHelper.Install(n.Get(flow_input.src));
//     appCon.Start(Time(0));
//     // std::cout << "注册流" << flow_input.idx << " src: " << flow_input.src
//     //           << " dst: " << flow_input.dst << " pg: " << flow_input.pg
//     //           << " sport: " << port
//     //           << " dport: " << flow_input.dport
//     //           << " maxPacketCount: " << flow_input.maxPacketCount //虽然叫maxPacketCount，但是其实是字节数
//     //           << " start_time: " << flow_input.start_time << std::endl;

//     // get the next flow input
//     flow_input.idx++;
//     ReadFlowInput();
//   }

//   // schedule the next time to run this function
//   if (flow_input.idx < flow_num) {
//     Simulator::Schedule(
//         Seconds(flow_input.start_time) - Simulator::Now(), ScheduleFlowInputs);
//   } else { // no more flows, close the file
//     flowf.close();
//   }
// }

void qp_finish_normal(FILE* fout, Ptr<RdmaQueuePair> q) {
  uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
  #ifdef NS3_MTP
  MtpInterface::explicitCriticalSection cs;
  #endif
  Ptr<Node> dstNode = n.Get(did);
  Ptr<RdmaDriver> rdma = dstNode->GetObject<RdmaDriver>();
  rdma->m_rdma->DeleteRxQp(q->sip.Get(), q->m_pg, q->sport);
  // std::cout << "at "<< Simulator::Now().GetNanoSeconds()<<"ns, qp finish, src: " << sid << " did: " << did
            // << " port: " << q->sport << std::endl;
  #ifdef NS3_MTP
  cs.ExitSection();
  #endif
  {  
    cclScheduler::OnSendFinish(fout, q);
    mnccl::OnMessageFinish(fout, q);
    cclScheduler::ScheduleTask();
  }
}

void send_finish_normal(FILE* fout, Ptr<RdmaQueuePair> q) {
  // uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
  // fprintf(fout, "%u %0u %u %u %lu %lu %lu\n", sid, did, q->sport,
  //         q->dport, q->snd_nxt, q->startTime.GetTimeStep(), Simulator::Now().GetTimeStep());
  // fflush(fout);
}

void message_finish_normal(FILE* fout, Ptr<RdmaQueuePair> q, uint64_t msgSize){
  uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
  uint32_t pg = q->m_pg;
  uint64_t base_rtt = pairRtt[sid][did], b = pairBw[sid][did];
  uint32_t packet_payload_size =
      get_config_value_ns3<uint64_t>("ns3::RdmaHw::Mtu");
  uint64_t size = msgSize;
  uint32_t total_bytes = size +
      ((size - 1) / packet_payload_size + 1) *
          (CustomHeader::GetStaticWholeHeaderSize() -
           IntHeader::GetStaticSize()); // translate to the minimum bytes
                                        // required (with header but no INT)
  uint64_t standalone_fct = base_rtt + total_bytes * 8000000000lu / b;
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
// GlobalValue::Bind ("SimulatorImplementationType",
//                    StringValue ("ns3::DistributedSimulatorImpl"));
#endif

  // MPI_Init(&argc, &argv);
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

  // initialize mnCCL and scheduler (use values read by SetupNetwork)
  uint32_t gpu_num = node_num - switch_num - nvswitch_num;
  cclScheduler::Init(gpu_num, 1);
  mnccl::Init();

  Simulator::Stop(Seconds(1)); // ensure all events are processed before Destroy
  printf("Simulation Start.\n");
  Simulator::Run(); 
  printf("Simulation Complete.\n");
  Simulator::Destroy();

  endt = clock();
  // std:://cout << (double)(endt - begint) / CLOCKS_PER_SEC << "\n";
  return 0;
}