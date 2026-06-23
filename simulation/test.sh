# ./ns3 configure -d default --enable-mtp
# ./ns3 run 'scratch/EPNet ./examples/rdma-test/test_ondpm.sh'
# ./ns3 run 'scratch/NormalNetwork ./examples/rdma-test/test_ondpm.sh'
./ns3 run 'scratch/NormalNetwork --networkConfig=./examples/HW/test.sh --policyConfig=./scratch/pipeline_policy.conf'
