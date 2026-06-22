#!/bin/bash
export MPICH_GPU_SUPPORT_ENABLED=1

N10_MILC="/global/scratch/users/rliu5/milc-lrc-h100-fp64"
MILC_QCD_DIR="${N10_MILC}/milc_qcd"
LATTICE_DIR="${N10_MILC}/lattices"

MILC_COMM="/global/home/users/rliu5/perf-model-dcgm/milc/common"
MILC_LRC="/global/home/users/rliu5/perf-model-dcgm/milc/lrc"

DCGM_PATH="${MILC_LRC}/wrap_dcgmi_container.sh"

# Tuning results are stored in qudatune_dir.
qudatune_dir="$PWD/qudatune-generation-h100-fp64"
export QUDA_RESOURCE_PATH=${qudatune_dir}
if [ ! -d ${qudatune_dir} ]; then
    mkdir ${qudatune_dir}
fi

RESULTS_DIR="${MILC_LRC}/results/MILC_TINY_FP64_${SLURM_JOB_ID}"
mkdir -p ${RESULTS_DIR}
cd ${RESULTS_DIR}

ln -s $LATTICE_DIR .
ln -s ${MILC_COMM}/input_4864_1node ./input_4864
ln -s ${MILC_COMM}/rat.m001907m05252m6382 .

#bind="${MILC_COMM}/bind4-perlmutter.sh"
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
srun -N 1 -n 1 -c 16 --gpus-per-node=1 --cpu-bind=cores ${DCGM_PATH} $exe $input > ${RESULTS_DIR}/${SLURM_JOB_ID}.out
end=$(date +%s.%N)
elapsed=$(printf "%s - %s\n" $end $start | bc -l)

printf "Elapsed Time: %.2f seconds\n" $elapsed > ${RESULTS_DIR}/runtime.out

unlink input_4864
unlink rat.m001907m05252m6382
unlink lattices

rm l4864f211b600m001907m05252m6382i.420x
rm l4864f211b600m001907m05252m6382i.420x.info