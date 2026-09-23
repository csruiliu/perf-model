#!/bin/bash
# run_speedup_dist.sh
# Runs the DCGM launcher over all 128 job files (000..127),
# renaming the generated speedup_dist.parquet after each run.

set -u  # treat unset variables as errors

# --- Configuration ---
INPUT_DIR=/pscratch/sd/r/ruiliu/system-wide-analysis
RESULT_DIR=tgt-h100-ng2
LOG_DIR=logs

mkdir -p "$RESULT_DIR" "$LOG_DIR"

# --- Main loop ---
for n in $(seq 0 127); do
    i=$(printf "%03d" "$n")

    infile="$INPUT_DIR/all_jobs_${i}of128.pkl"
    outfile="$RESULT_DIR/speedup_dist_${i}of128.parquet"
    logfile="$LOG_DIR/job_${i}.log"
    donefile="$RESULT_DIR/.done_${i}"

    # Skip if already completed (resume support)
    if [ -f "$donefile" ]; then
        echo "=== Skipping job $i (already done) ==="
        continue
    fi

    # Sanity check that the input file exists
    if [ ! -f "$infile" ]; then
        echo "!!! Input file not found: $infile — skipping"
        continue
    fi

    echo "=== Processing job $i ==="

    python3 -m counter_model.dcgm.launcher \
        --job_mode multi --num_gpu 1 \
        --dcgm_input "$infile" \
        -d 10000 -rg A100-SXM-40 -tg H100-SXM-NG2 -rh Perlmutter -th Perlmutter-NG2 \
        --cores_alloc same --max_workers 32 \
        --agg_results_dir $RESULT_DIR \
        > "$logfile" 2>&1

    status=$?

    if [ $status -ne 0 ]; then
        echo "!!! Job $i failed (exit code $status). See $logfile"
        continue
    fi

    # Rename the output; confirm it exists first
    if [ -f $RESULT_DIR/speedup_dist.parquet ]; then
        mv $RESULT_DIR/speedup_dist.parquet "$outfile"
        touch "$donefile"
        echo "    -> saved $outfile"
    else
        echo "!!! Job $i finished but speedup_dist.parquet was not found"
    fi
done

echo "=== All done ==="
