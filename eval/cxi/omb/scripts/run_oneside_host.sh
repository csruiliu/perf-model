#!/bin/bash
#SBATCH -J OMB_1sided_host
#SBATCH -o ../results/OMB_%j/OMB_1sided_host-%j.out
#SBATCH -N 2
#SBATCH -C cpu
#SBATCH -q sow
#SBATCH -t 00:30:00
#SBATCH -A nstaff
#SBATCH --exclusive
##SBATCH -w nid[004074,004138]
#
#The -w option specifies which nodes to use for the test,
#thus controling the number of network hops between them.
#It should be modified for each system because
#the nid-topology differs with the system architechture.
#The nodes identified above are maximally distant
#on Perlmutter's Slingshot network.

#The number of NICs(j) and CPU cores (k) per node
#should be specified here.
j=1   #NICs per node
k=128 #Cores per node

#The paths to OMB and its point-to-point benchmarks
#should be specified here
OMB_DIR=/pscratch/sd/r/ruiliu/osu-micro-benchmarks-7.1-1/libexec/osu-micro-benchmarks
OMB_PT2PT=${OMB_DIR}/mpi/pt2pt
OMB_1SIDE=${OMB_DIR}/mpi/one-sided

export RESULTS_DIR=../results/OMB_${SLURM_JOB_ID}
mkdir -p $RESULTS_DIR

export IPM_LOGDIR=/pscratch/sd/r/ruiliu/ipm-logs/omb
export IPM_LOG=full
export IPM_REPORT=full

# =============================================================
# Explicit counter lists — defined here, exported as
# space-separated strings so cxi_snapshot.sh and cxi_monitor.sh
# can access them. Reconstruct in child scripts with:
#   read -ra TX_COUNTERS <<< "${TX_COUNTERS_STR}"
#   read -ra RX_COUNTERS <<< "${RX_COUNTERS_STR}"
# =============================================================
export TX_COUNTERS_STR="\
hni_tx_ok_64 \
hni_tx_ok_65_to_127 \
hni_tx_ok_128_to_255 \
hni_tx_ok_256_to_511 \
hni_tx_ok_512_to_1023 \
hni_tx_ok_1024_to_2047 \
hni_tx_ok_2048_to_4095 \
hni_tx_ok_4096_to_8191 \
hni_pkts_sent_by_tc_0 \
hni_pkts_sent_by_tc_1"

export RX_COUNTERS_STR="\
hni_rx_ok_64 \
hni_rx_ok_65_to_127 \
hni_rx_ok_128_to_255 \
hni_rx_ok_256_to_511 \
hni_rx_ok_512_to_1023 \
hni_rx_ok_1024_to_2047 \
hni_rx_ok_2048_to_4095 \
hni_rx_ok_4096_to_8191 \
hni_pkts_recv_by_tc_0 \
hni_pkts_recv_by_tc_1"

MESSAGE_SIZE=4194304

ITER=10
WARMUP_ITER=0
WINDOW_SIZE=1

# Pre-create per-node directories so cxi_snapshot.sh can write immediately
echo "Creating per-node directory structure under ${RESULTS_DIR}..."
for node in $(scontrol show hostnames ${SLURM_NODELIST}); do
    mkdir -p ${RESULTS_DIR}/${node}
    echo "  Created: ${RESULTS_DIR}/${node}"
done

# Collect baseline counters BEFORE benchmarks
echo "Collecting baseline telemetry ..."
srun -N 2 --ntasks-per-node=1 ./cxi_snapshot.sh before

srun -N 2 -n 2 ${OMB_1SIDE}/osu_put_bw -m $MESSAGE_SIZE:$MESSAGE_SIZE -i $ITER -x $WARMUP_ITER -W $WINDOW_SIZE H H

# Collect final counters AFTER benchmarks
echo "Collecting final telemetry..."
srun -N 2 --ntasks-per-node=1 ./cxi_snapshot.sh after

elapsed=$(printf "%s - %s\n" $end $start | bc -l)

# Create runtime.out with node assignment info

echo "" >>$RESULTS_DIR/runtime.out
echo "MESSAGE_SIZE: ${MESSAGE_SIZE} Byte(s)" >>$RESULTS_DIR/runtime.out
echo "ITERATIONS: $ITER" >>$RESULTS_DIR/runtime.out
echo "SAMPLE_INTERVAL: $SAMPLE_INTERVAL" >>$RESULTS_DIR/runtime.out
printf "Elapsed Time: %.2f seconds\n" $elapsed >>$RESULTS_DIR/runtime.out
