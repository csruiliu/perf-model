#!/bin/bash
#SBATCH --qos=sow
#SBATCH --account=nstaff
#SBATCH --job-name=lmp_small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH -C gpu&hbm40g
#SBATCH -G 1
#SBATCH --gpu-bind=none
#SBATCH --perf=generic
#SBATCH -t 00:30:00
#SBATCH -o /global/homes/r/ruiliu/perf-model-dcgm/lammps/pm/results/LPS_SMALL_FP64_%j/%j.out

podman-hpc run -d -it --name dcgm-container --rm --gpu --cap-add SYS_ADMIN -p 5555:5555 nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04

# the input specification
spec=small
nn=256
BENCH_SPEC="\
        -in common/in.snap.test \
        -var snapdir common/2J8_W.SNAP \
        -var nx $nn -var ny $nn -var nz $nn \
        -var nsteps 100"

LAMMPS_DIR="/pscratch/sd/r/ruiliu/lammps-pm-a100-fp64"

LAMMPS_COMM="/global/homes/r/ruiliu/perf-model-dcgm/lammps/common"

LAMMPS_PM="/global/homes/r/ruiliu/perf-model-dcgm/lammps/pm"

RESULTS_DIR="${LAMMPS_PM}/results/LPS_SMALL_FP64_${SLURM_JOB_ID}"

mkdir -p ${RESULTS_DIR}
cd    ${RESULTS_DIR}
ln -s ${LAMMPS_COMM} .
ln -s ${LAMMPS_PM}/wrap_dcgmi_container.sh .

# This is needed if LAMMPS is built using cmake.
#install_dir="../../../install_PM"
#export LD_LIBRARY_PATH=${install_dir}/lib64:$LD_LIBRARY_PATH
EXE="${LAMMPS_DIR}/install_lammps/bin/lmp"

# For different cluster, num_threads could be different
export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Match the build env.
module load PrgEnv-gnu
module load cudatoolkit
module load craype-accel-nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1

input="-k on g 1 -sf kk -pk kokkos newton on neigh half ${BENCH_SPEC} " 

# export these two variables for wrap_dcgmi_container.sh 
export RESULTS_DIR
export DCGM_DELAY=1000

start=$(date +%s.%N)
srun -N 1 -n 1 -c $SLURM_CPUS_PER_TASK --gpus-per-node=1 --cpu-bind=cores ./wrap_dcgmi_container.sh $EXE $input
end=$(date +%s.%N)
elapsed=$(printf "%s - %s\n" $end $start | bc -l)

printf "Elapsed Time: %.2f seconds\n" $elapsed > ${RESULTS_DIR}/runtime.out

unlink common
unlink wrap_dcgmi_container.sh

