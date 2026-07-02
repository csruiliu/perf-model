#!/bin/bash

# Allocating resources first, the following command is an example
# salloc -p es2 -A pc_perfume -q es2_normal --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:H100:1 -t 12:00:00

# Start Singularity server first
# singularity instance start --fakeroot --nv --writable-tmpfs --bind /tmp:/tmp --bind /global/home/users/rliu5:/global/home/users/rliu5 --network=none docker://nvidia/dcgm:4.4.1-2-ubuntu22.04 dcgm-instance

# singularity exec instance://dcgm-instance nv-hostengine -n &

# the input specification
spec=small
nn=256
BENCH_SPEC="\
        -in common/in.snap.test \
        -var snapdir common/2J8_W.SNAP \
        -var nx $nn -var ny $nn -var nz $nn \
        -var nsteps 100"

export RESULTS_DIR="/global/home/users/rliu5/perf-model/eval/dcgm/lammps/lrc/results/LPS_SMALL_FP64_${SLURM_JOB_ID}"

LAMMPS_DIR="/global/scratch/users/rliu5/lammps-lrc-h100-fp64"

LAMMPS_COMM="/global/home/users/rliu5/perf-model/eval/dcgm/lammps/common"

LAMMPS_LRC="/global/home/users/rliu5/perf-model/eval/dcgm/lammps/lrc"

DCGM_PATH="${LAMMPS_LRC}/wrap_dcgmi_container.sh"

mkdir -p ${RESULTS_DIR}
cd ${RESULTS_DIR}
ln -s ${LAMMPS_COMM} .

# This is needed if LAMMPS is built using cmake.
#install_dir="../../../install_PM"
#export LD_LIBRARY_PATH=${install_dir}/lib64:$LD_LIBRARY_PATH
EXE="${LAMMPS_DIR}/install_lammps/bin/lmp"

# Match the build env.
export MPICH_GPU_SUPPORT_ENABLED=1

input="-k on g 1 -sf kk -pk kokkos newton on neigh half ${BENCH_SPEC} "

# export the variable for wrap_dcgmi_container.sh 
export DCGM_DELAY=1000

start_time=$(date +%s.%N)
srun -N 1 -n 1 -c 16 --gpus-per-node=1 --cpu-bind=cores ${DCGM_PATH} $EXE $input > ${RESULTS_DIR}/${SLURM_JOB_ID}.out
end_time=$(date +%s.%N)
elapsed=$(printf "%s - %s\n" $end_time $start_time | bc -l)

printf "Elapsed Time: %.2f seconds\n" $elapsed > ${RESULTS_DIR}/runtime.out

unlink common