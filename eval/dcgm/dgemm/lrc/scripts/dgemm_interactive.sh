#!/bin/bash

# Allocating resources first, the following command is an example
# salloc -p es2 -A pc_perfume -q es2_normal --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:H100:1 -t 12:00:00

# Start Singularity server first
# singularity instance start --fakeroot --nv --writable-tmpfs --bind /tmp:/tmp --bind /global/home/users/rliu5:/global/home/users/rliu5 --network=none docker://nvidia/dcgm:4.4.1-2-ubuntu22.04 dcgm-instance

# singularity exec instance://dcgm-instance nv-hostengine -n &

# Create Conda VENV

# go to the compute node!

# module load nvhpc
# module load miniconda3
# module load miniforge3

# conda create -n py-dgemm python=3.10 -y
# conda init (you may need to logout the computing node by exit and re-login the compute node by ssh)

#conda activate py-dgemm
#conda install numpy -y
#conda install -c conda-forge cupy cuda-version=12.9 -y

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# create results directory if not exist
if [ ! -d "../results" ]; then
  mkdir ../results
fi

export RESULTS_DIR=../results/DGEMM_${SLURM_JOB_ID}

mkdir -p $RESULTS_DIR

export DCGM_DELAY=1000
PYTHON_EXE=../common/py-dgemm.py

start_time=$(date +%s.%N)
srun ./wrap_dcgmi_container.sh python $PYTHON_EXE --niterations 1000 --nsize 16384 --accelerator > ${RESULTS_DIR}/DGEMM_${SLURM_JOB_ID}.out
end_time=$(date +%s.%N)
elapsed=$(printf "%s - %s\n" $end_time $start_time | bc -l)

printf "Elapsed Time: %.2f seconds\n" $elapsed > ${RESULTS_DIR}/runtime.out

