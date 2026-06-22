#!/bin/bash
#SBATCH --qos=debug
#SBATCH -C gpu&hbm40g
#SBATCH -G 4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -A nstaff
#SBATCH --exclusive
#SBATCH --perf=generic
#SBATCH -o ../results/GEMM_MPI_%j/GEMM_MPI_%j.out

podman-hpc run -d -it --name dcgm-container --rm --gpu --cap-add SYS_ADMIN -p 5555:5555 nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04

#OpenMP settings:
export OMP_NUM_THREADS=4
export OMP_PLACES=threads
export OMP_PROC_BIND=true

# create results directory if not exist
if [ ! -d "../results" ]; then
  mkdir ../results
fi

export RESULTS_DIR=../results/GEMM_MPI_${SLURM_JOBID}

export DCGM_SAMPLE_RATE=1000

for prec in D S H; do
#run the application:
start=$(date +%s.%N)
dcgm_delay=${DCGM_SAMPLE_RATE} srun --cpu_bind=cores ./wrap_dcgmi_mpi.sh gemm_mpi.x 32768 700 1.0 1.0 $prec \
	> ${RESULTS_DIR}/"$prec"gemm_mpi-${SLURM_JOBID}.dcgmi
end=$(date +%s.%N)
elapsed=$(printf "%s - %s\n" $end $start | bc -l)
printf "Elapsed Time: %.2f seconds\n" $elapsed > ${RESULTS_DIR}/"$prec"gemm_mpi_d${DCGM_SAMPLE_RATE}_runtime.out
done



