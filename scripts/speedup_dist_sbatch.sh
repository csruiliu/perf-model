#!/bin/bash
#SBATCH --job-name=speedup_dist
#SBATCH --account=nstaff
#SBATCH --constraint=cpu
#SBATCH --qos=sow
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=logs-%j/slurm-%x-%j.out
#SBATCH --error=logs-%j/slurm-%x-%j.err

set -u

# --- Environment (adjust to however you set this up interactively) ---
module load python
# conda activate my_env

export OMP_NUM_THREADS=1        # keep BLAS/OpenMP from fighting your 32 workers
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# --- Configuration ---
INPUT_DIR=/pscratch/sd/r/ruiliu/system-wide-analysis
RESULT_DIR=/global/homes/r/ruiliu/perf-model/tgt-h100
LOG_DIR=/global/homes/r/ruiliu/perf-model/logs-${SLURM_JOB_ID}

mkdir -p "$RESULT_DIR" "$LOG_DIR"

# --- Main loop ---
for n in $(seq 0 127); do
    i=$(printf "%03d" "$n")

    infile="$INPUT_DIR/all_jobs_${i}of128.pkl"
    outfile="$RESULT_DIR/speedup_dist_${i}of128.parquet"
    logfile="$LOG_DIR/job_${i}.log"
    donefile="$RESULT_DIR/.done_${i}"

    if [ -f "$donefile" ]; then
        echo "=== Skipping job $i (already done) ==="
        continue
    fi

    if [ ! -f "$infile" ]; then
        echo "!!! Input file not found: $infile — skipping"
        continue
    fi

    echo "=== Processing job $i ($(date)) ==="

    python3 -m counter_model.dcgm.launcher \
        --job_mode multi --num_gpu 1 \
        --dcgm_input "$infile" \
        -d 10000 -rg A100-SXM-40 -tg H100-SXM -rh Perlmutter -th Perlmutter \
        --cores_alloc same --max_workers 32 \
        --agg_results_dir "$RESULT_DIR" \
        > "$logfile" 2>&1
    status=$?

    if [ $status -ne 0 ]; then
        echo "!!! Job $i failed (exit code $status). See $logfile"
        continue
    fi

    if [ -f "$RESULT_DIR/speedup_dist.parquet" ]; then
        mv "$RESULT_DIR/speedup_dist.parquet" "$outfile"
        touch "$donefile"
        echo "    -> saved $outfile"
    else
        echo "!!! Job $i finished but speedup_dist.parquet was not found"
    fi
done

echo "=== All done ==="
