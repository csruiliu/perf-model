#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu&hbm80g
#SBATCH -t 4:30:00
#SBATCH -A nstaff
#SBATCH --job-name=milc-tiny
#SBATCH -q sow
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --perf=generic
#SBATCH -o /global/homes/r/ruiliu/perf-model-dcgm/milc/pm/results/MILC_TINY_FP64_%j/%j.out

podman-hpc run -d -it --name dcgm-container --rm --gpu --cap-add SYS_ADMIN -p 5555:5555 nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04

#module load PrgEnv-gnu
#module load cudatoolkit
#module load craype-accel-nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1

MILC_DIR="/pscratch/sd/r/ruiliu/milc-pm-a100-fp64"
MILC_QCD_DIR=${MILC_DIR}/milc_qcd
LATTICE_DIR=${MILC_DIR}/lattices

MILC_COMM="/global/homes/r/ruiliu/perf-model-dcgm/milc/common"
MILC_PM="/global/homes/r/ruiliu/perf-model-dcgm/milc/pm"

# Tuning results are stored in qudatune_dir.
qudatune_dir="$PWD/qudatune-generation-fp64"
export QUDA_RESOURCE_PATH=${qudatune_dir}
if [ ! -d ${qudatune_dir} ]; then
    mkdir ${qudatune_dir}
fi

RESULTS_DIR="${MILC_PM}/results/MILC_TINY_FP64_${SLURM_JOB_ID}"
mkdir -p ${RESULTS_DIR}
cd ${RESULTS_DIR}

ln -s $LATTICE_DIR .
ln -s ${MILC_PM}/wrap_dcgmi_container.sh .
ln -s ${MILC_COMM}/input_4864_1node ./input_4864
ln -s ${MILC_COMM}/rat.m001907m05252m6382 .

bind="${MILC_COMM}/bind4-perlmutter.sh"
exe="${MILC_QCD_DIR}/ks_imp_rhmc/su3_rhmd_hisq"
input=input_4864

export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

export QUDA_ENABLE_GDR=1
export QUDA_MILC_HISQ_RECONSTRUCT=13
export QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=9

# export these two variables for wrap_dcgmi_container.sh 
export RESULTS_DIR
export DCGM_DELAY=1000

start=$(date +%s.%N)
srun -N 1 -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK --gpus-per-node=1 --cpu-bind=cores ./wrap_dcgmi_container.sh $exe $input
end=$(date +%s.%N)
elapsed=$(printf "%s - %s\n" $end $start | bc -l)

printf "Elapsed Time: %.2f seconds\n" $elapsed > ${RESULTS_DIR}/runtime.out

unlink wrap_dcgmi_container.sh
unlink input_4864
unlink rat.m001907m05252m6382
unlink lattices

rm l4864f211b600m001907m05252m6382i.420x
rm l4864f211b600m001907m05252m6382i.420x.info