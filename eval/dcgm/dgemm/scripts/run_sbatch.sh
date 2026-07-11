#!/bin/bash
#SBATCH --qos=sow
#SBATCH -C gpu&hbm40g
#SBATCH -G 1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -t 00:30:00
#SBATCH -A nstaff
#SBATCH --exclusive
#SBATCH --perf=generic
#SBATCH -o ../results/DGEMM_%j/DGEMM_%j.out

module load python
module load cudatoolkit

conda create -n py-dgemm python=3.10 -y
conda activate py-dgemm
conda install numpy -y
conda install -c conda-forge cupy cuda-version=12.9 -y

podman-hpc run -d -it --name dcgm-container --rm --gpu --cap-add SYS_ADMIN -p 5556:5556 nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=true

# create results directory if not exist
if [ ! -d "../results" ]; then
  mkdir ../results
fi

export RESULTS_DIR=../results/DGEMM_${SLURM_JOBID}

export DCGM_DELAY=1000

start_time=$(date +%s.%N)
srun ./wrap_dcgmi_container.sh python py-dgemm.py --accelerator > ${RESULTS_DIR}/DGEMM_${SLURM_JOBID}.out
end_time=$(date +%s.%N)
elapsed=$(printf "%s - %s\n" $end_time $start_time | bc -l)

printf "Elapsed Time: %.2f seconds\n" $elapsed > ${RESULTS_DIR}/runtime.out


