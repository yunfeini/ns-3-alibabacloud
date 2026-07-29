# ./ns3 configure -d default --enable-mtp
# ./ns3 run 'scratch/EPNet ./examples/rdma-test/test_ondpm.sh'
# ./ns3 run 'scratch/NormalNetwork ./examples/rdma-test/test_ondpm.sh'
# ./ns3 run 'scratch/NormalNetwork --networkConfig=./examples/HW/test.sh --policyConfig=./scratch/pipeline_policy.conf'

# python3 run_dispatch_l012_topology_compare_visual.py \
#   --topology-file ./examples/HW/topo2_2_8.txt \
#   --methods uniform_trace full \
#   --output ./results/topo2_2_8_uniform_full_l012_moe58_400g_2300ns \
#   --dispatch-expert-ffn-params 7168 \
#   --no-placement-detail

# python3 run_dispatch_l012_topology_compare_visual.py \
#   --topology-file ./examples/HW/topo1_4_8.txt \
#   --methods uniform_trace full \
#   --output ./results/topo1_4_8_uniform_full_l012_moe58_400g_2300ns \
#   --dispatch-expert-ffn-params 7168 \
#   --no-placement-detail

python3 run_dispatch_l012_topology_compare_visual.py \
  --topology-file ./examples/HW/topo1_8_4.txt \
  --methods uniform_trace full \
  --output ./results/topo1_8_4_uniform_full_l012_moe58_400g_2300ns \
  --dispatch-expert-ffn-params 7168 \
  --no-placement-detail

python3 run_dispatch_l012_topology_compare_visual.py \
  --topology-file ./examples/HW/topo1_16_2.txt \
  --methods uniform_trace full \
  --output ./results/topo1_16_2_uniform_full_l012_moe58_400g_2300ns \
  --dispatch-expert-ffn-params 7168 \
  --no-placement-detail