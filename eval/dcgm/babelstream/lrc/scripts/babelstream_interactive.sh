#!/bin/bash

# Allocating resources first, the following command is an example
# salloc -p es2 -A pc_perfume -q es2_normal --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:H100:1 -t 12:00:00

# Start Singularity server first
# singularity instance start --fakeroot --nv --writable-tmpfs --bind /tmp:/tmp --bind /global/home/users/rliu5:/global/home/users/rliu5 --network=none docker://nvidia/dcgm:4.4.1-2-ubuntu22.04 dcgm-instance

# singularity exec instance://dcgm-instance nv-hostengine -n &

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# create results directory if not exist
if [ ! -d "../results" ]; then
  mkdir ../results
fi

export RESULTS_DIR=../results/BABELSTREAM_${SLURM_JOB_ID}

export DCGM_DELAY=1000

#Array size must be a multiple of 1024
export ARRAYSIZE=268435456
export NUMTIMES=5000 
export BABELSTREAM="/pscratch/sd/r/ruiliu/BabelStream-5.0/build/cuda-stream"

mkdir -p $RESULTS_DIR

start=$(date +%s.%N)
srun --nodes=1 --ntasks=1 --cpus-per-task=2 ./wrap_dcgmi_container.sh $BABELSTREAM -s $ARRAYSIZE -n $NUMTIMES > ${RESULTS_DIR}/babelstream-${SLURM_JOB_ID}.out
end=$(date +%s.%N)
elapsed=$(printf "%s - %s\n" $end $start | bc -l)

printf "Elapsed Time: %.2f seconds\n" $elapsed > ${RESULTS_DIR}/runtime.out

