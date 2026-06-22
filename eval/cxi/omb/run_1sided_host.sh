#!/bin/bash
#SBATCH -J OMB_1sided_host
#SBATCH -o /pscratch/sd/r/ruiliu/osu-micro-benchmarks/results/OMB_%j/OMB_1sided_host-%j.out
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
OMB_DIR=../libexec/osu-micro-benchmarks
OMB_PT2PT=${OMB_DIR}/mpi/pt2pt
OMB_1SIDE=${OMB_DIR}/mpi/one-sided

export RESULTS_DIR=/pscratch/sd/r/ruiliu/osu-micro-benchmarks/results/OMB_${SLURM_JOB_ID}
mkdir -p $RESULTS_DIR

export MPICH_GNI_MAX_EAGER_MSG_SIZE=10485760
export FI_CXI_RDZV_THRESHOLD=10485760

# Time windows for before/after collection (in seconds)
BEFORE_DURATION=10
AFTER_DURATION=10

MESSAGE_SIZE=1
#MESSAGE_SIZE=1048576
#MESSAGE_SIZE=5243000

export SAMPLE_INTERVAL=1

ITER=500000
WARMUP_ITER=0
WINDOW_SIZE=1

# Collect baseline counters BEFORE benchmarks
echo "Collecting baseline telemetry for ${BEFORE_DURATION} seconds..."
srun -N 2 --ntasks-per-node=1 ./cxi_snapshot.sh before ${BEFORE_DURATION}

echo "=== Node Assignment ===" >$RESULTS_DIR/runtime.out

start=$(date +%s.%N)

srun -N 2 -n 2 ./cxi_monitor.sh ${OMB_1SIDE}/osu_put_bw -m $MESSAGE_SIZE:$MESSAGE_SIZE -i $ITER -x $WARMUP_ITER -W $WINDOW_SIZE H H

end=$(date +%s.%N)

echo "======================================" >>$RESULTS_DIR/runtime.out

# Collect final counters AFTER benchmarks
echo "Collecting final telemetry for ${AFTER_DURATION} seconds..."
srun -N 2 --ntasks-per-node=1 ./cxi_snapshot.sh after ${AFTER_DURATION}

elapsed=$(printf "%s - %s\n" $end $start | bc -l)

# Create runtime.out with node assignment info

echo "" >>$RESULTS_DIR/runtime.out
echo "MESSAGE_SIZE: ${MESSAGE_SIZE} Byte(s)" >>$RESULTS_DIR/runtime.out
echo "ITERATIONS: $ITER" >>$RESULTS_DIR/runtime.out
echo "SAMPLE_INTERVAL: $SAMPLE_INTERVAL" >>$RESULTS_DIR/runtime.out
printf "Elapsed Time: %.2f seconds\n" $elapsed >>$RESULTS_DIR/runtime.out
