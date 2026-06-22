#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu&hbm40g
#SBATCH -G 1
#SBATCH -q sow
#SBATCH -t 00:05:00
#SBATCH -A nstaff
#SBATCH --perf=generic
#SBATCH -o ../results/GEMM_BURSTY_%j/GEMM_BURSTY_%j.out
#SBATCH --exclusive

podman-hpc run -d -it --name dcgm-container --rm --gpu --cap-add SYS_ADMIN -p 5555:5555 nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# create results directory if not exist
if [ ! -d "../results" ]; then
  mkdir ../results
fi

export RESULTS_DIR=../results/GEMM_BURSTY_${SLURM_JOBID}

export DCGM_SAMPLE_RATE=10000

#run the application:
dcgm_delay=${DCGM_SAMPLE_RATE} \
srun --cpu_bind=cores --gpu-bind=single:0 ./wrap_dcgmi.sh ./gemm_bursty.x 4 4096 4096 300000 \
	> ${RESULTS_DIR}/gemm_bursty-${SLURM_JOBID}.dcgmi

