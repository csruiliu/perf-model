#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu&hbm40g
#SBATCH -G 1
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -A nstaff
#SBATCH --perf=generic
#SBATCH -o ../results/GEMM_INTERLEAVE_%j/GEMM_INTERLEAVE_%j.out
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

export RESULTS_DIR=../results/GEMM_INTERLEAVE_${SLURM_JOBID}

export DCGM_SAMPLE_RATE=10000

#run the application:
dcgm_delay=${DCGM_SAMPLE_RATE} \
srun --cpu_bind=cores --gpu-bind=single:0 ./wrap_dcgmi.sh ./gemm_interleave.x 4 4096 4096 300000 \
	> ${RESULTS_DIR}/gemm_interleave-${SLURM_JOBID}.dcgmi

