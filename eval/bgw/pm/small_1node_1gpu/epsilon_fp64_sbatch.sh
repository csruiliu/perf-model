#!/bin/bash
#SBATCH -t 00:20:00
#SBATCH --qos=sow
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -A nstaff
#SBATCH -J bgw_eps_Si214
#SBATCH -C gpu&hbm40g
#SBATCH --perf=generic
#SBATCH -o /global/homes/r/ruiliu/perf-model-dcgm/bgw/pm/results/EPS_SMALL_FP64_%j/%j.out

podman-hpc run -d -it --name dcgm-container --rm --gpu --cap-add SYS_ADMIN nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04

N10_BGW="/pscratch/sd/r/ruiliu/bgw-pm-a100-fp64"
if [[ -z "${N10_BGW}" ]]; then
    echo "The N10_BGW variable is not defined."
    echo "Please set N10_BGW in site_path_config.sh and try again."
    exit 0
fi

N10_BGW_EXEC="${N10_BGW}/BerkeleyGW-n10/bin"

BGW_PM="/global/homes/r/ruiliu/perf-model-dcgm/bgw/pm"

BGW_COMM="/global/homes/r/ruiliu/perf-model-dcgm/bgw/common"

Si_WFN_folder=${N10_BGW}/Si_WFN_folder

Si214_WFN_folder=${Si_WFN_folder}/Si214/WFN_file

RESULTS_DIR="${BGW_PM}/results/EPS_SMALL_FP64_${SLURM_JOB_ID}"

mkdir -p ${RESULTS_DIR}
#stripe_large $RESULTS_DIR
cd    ${RESULTS_DIR}
ln -s ${N10_BGW_EXEC}/epsilon.cplx.x .
ln -s ${BGW_COMM}/epsilon-si214.inp epsilon.inp
ln -sfn  ${Si214_WFN_folder}/WFNq.h5      .
ln -sfn  ${Si214_WFN_folder}/WFN_out.h5   ./WFN.h5

ulimit -s unlimited

export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# BerkeleyGW specific
export HDF5_USE_FILE_LOCKING=FALSE
export BGW_HDF5_WRITE_REDIST=1
export BGW_WFN_HDF5_INDEPENDENT=1

DCGM_PATH="${BGW_PM}/wrap_dcgmi_container.sh"

# export these two variables for wrap_dcgmi_container.sh 
export RESULTS_DIR
export DCGM_DELAY=1000

start=$(date +%s.%N)
srun -N 1 -c 32 --ntasks-per-node=1 --gpus-per-node=1 ${DCGM_PATH} ./epsilon.cplx.x
end=$(date +%s.%N)
elapsed=$(printf "%s - %s\n" $end $start | bc -l)

printf "Elapsed Time: %.2f seconds\n" $elapsed > ${RESULTS_DIR}/runtime.out

unlink epsilon.cplx.x
unlink epsilon.inp
unlink WFN.h5
unlink WFNq.h5
rm -f eps0mat.h5
