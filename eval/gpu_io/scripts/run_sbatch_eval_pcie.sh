#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu&hbm40g
#SBATCH -G 1
#SBATCH --cpus-per-task=2
#SBATCH -q sow
#SBATCH -t 00:30:00
#SBATCH -A nstaff
#SBATCH --exclusive
#SBATCH --perf=generic
#SBATCH -o ../results/GPU_UTIL_%j/GPU_UTIL_%j.out

podman-hpc run -d -it --name dcgm-container --rm --gpu --cap-add SYS_ADMIN -p 5555:5555 nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# create results directory if not exist
if [ ! -d "../results" ]; then
  mkdir ../results
fi

export RESULTS_DIR=../results/GPU_UTIL_${SLURM_JOBID}

export DCGM_SAMPLE_RATE=100

#gemm.x args
# 1: matrix size
# 2: repeats
# 3: alpha
# 4: beta
# 5: precision
dcgm_delay=${DCGM_SAMPLE_RATE} \
	srun -n 1 -c 1 --cpu_bind=cores -G 1 --gpu-bind=single:1 \
	./wrap_dcgmi_container.sh \
	./eval_io_pcie.x --size 32 --copies 20 --pinned 0 --cuda-events 0\
	> $RESULTS_DIR/gpu_util_eval_io-$SLURM_JOBID.dcgmi
