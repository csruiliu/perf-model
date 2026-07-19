#!/bin/bash
#SBATCH --qos=sow
#SBATCH --account=nstaff
#SBATCH --job-name=lmp_small
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -C gpu&hbm40g
#SBATCH -G 2
#SBATCH --gpu-bind=none
#SBATCH --perf=generic
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH -o /global/homes/r/ruiliu/perf-model/eval/cxi/lammps/pm/results/LPS_SMALL_FP32_CTR_%j/%j.out

# Load necessary modules for GPU and CUDA support
module load PrgEnv-gnu
module load cudatoolkit
module load craype-accel-nvidia80

# Start DCGM container on all nodes before srun
#srun -N 2 --ntasks-per-node=1 podman-hpc run -d -it --name dcgm-container-${SLURM_JOB_ID} --rm --gpu \
#    --cap-add SYS_ADMIN -p 5555:5555 \
#    nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04

# Wait a moment for the DCGM container to start and initialize
#sleep 5

# IPM Path and Settings
export IPM_HOME=/pscratch/sd/r/ruiliu/IPM/install
export LD_PRELOAD=${IPM_HOME}/lib/libipm.so
export IPM_LOG=full
export IPM_REPORT=full

# LAMMPS Path
export LAMMPS_DIR="/pscratch/sd/r/ruiliu/lammps-pm-a100-fp32"
export LAMMPS_COMM="/global/homes/r/ruiliu/perf-model/eval/cxi/lammps/common"
export LAMMPS_PM="/global/homes/r/ruiliu/perf-model/eval/cxi/lammps/pm"

# Results directory for this job
export RESULTS_DIR="${LAMMPS_PM}/results/LPS_SMALL_FP32_CTR_${SLURM_JOB_ID}"

export IPM_LOGDIR=${RESULTS_DIR}/ipm-logs

# OMP Settings for LAMMPS
export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

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
hni_pkts_recv_by_tc_1 \
lpe_net_match_overflow_0 \
lpe_net_match_priority_0"

# sample interval
export SAMPLE_INTERVAL=1
export DCGM_DELAY=1000

# GPU-aware MPI settings for Cray Slingshot (Perlmutter)
export MPICH_GPU_SUPPORT_ENABLED=0

# Create folder for IPM logs if not exists
mkdir -p $IPM_LOGDIR

# Create results directory and go into it
mkdir -p ${RESULTS_DIR}
cd    ${RESULTS_DIR}

# Create symlinks to common files and scripts in results directory
ln -s ${LAMMPS_COMM} .
ln -s ${LAMMPS_PM}/wrap_dcgmi_container.sh .
ln -s ${LAMMPS_PM}/cxi_snapshot.sh .
#ln -s ${LAMMPS_PM}/cxi_monitor.sh .

# Pre-create per-node directories so cxi_snapshot.sh can write immediately
echo "Creating per-node directory structure under ${RESULTS_DIR}..."
for node in $(scontrol show hostnames ${SLURM_NODELIST}); do
    mkdir -p ${RESULTS_DIR}/${node}
    echo "  Created: ${RESULTS_DIR}/${node}"
done

# LAMMPS input specification
spec=small
nn=256
BENCH_SPEC="\
        -in common/in.snap.test \
        -var snapdir common/2J8_W.SNAP \
        -var nx $nn -var ny $nn -var nz $nn \
        -var nsteps 100"

# LAMMPS executable and input settings
EXE="${LAMMPS_DIR}/install_lammps/bin/lmp"

# -k on g 1: 1 GPU per MPI rank (correct for 2 ranks x 1 GPU each)
input="-k on g 1 -sf kk -pk kokkos newton on neigh half ${BENCH_SPEC}"

# Collect baseline counters BEFORE benchmark
echo "Collecting baseline telemetry before running benchmark..."
srun -N 2 --ntasks-per-node=1 ./cxi_snapshot.sh before

echo "=== Node Assignment for LAMMPS ===" > ${RESULTS_DIR}/runtime.out

start=$(date +%s.%N)

srun -N 2 -n 2 -c $SLURM_CPUS_PER_TASK --gpus-per-task=1 --cpu-bind=cores $EXE $input

end=$(date +%s.%N)

echo "======================================" >> ${RESULTS_DIR}/runtime.out

# Collect final counters AFTER benchmark
echo "Collecting final telemetry..."
srun -N 2 --ntasks-per-node=1 ./cxi_snapshot.sh after

elapsed=$(printf "%s - %s\n" $end $start | bc -l)

echo "" >> ${RESULTS_DIR}/runtime.out
echo "SPEC: ${spec}" >> ${RESULTS_DIR}/runtime.out
echo "GRID: ${nn} x ${nn} x ${nn}" >> ${RESULTS_DIR}/runtime.out
echo "NSTEPS: 100" >> ${RESULTS_DIR}/runtime.out
echo "SAMPLE_INTERVAL: ${SAMPLE_INTERVAL}" >> ${RESULTS_DIR}/runtime.out
printf "Elapsed Time: %.2f seconds\n" $elapsed >> ${RESULTS_DIR}/runtime.out

#srun -N 2 --ntasks-per-node=1 podman-hpc stop dcgm-container-${SLURM_JOB_ID} 2>/dev/null || true

unlink common
unlink wrap_dcgmi_container.sh
unlink cxi_snapshot.sh
#unlink cxi_monitor.sh
