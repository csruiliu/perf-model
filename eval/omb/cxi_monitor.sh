#!/bin/bash

# =============================================================
# cxi_monitor.sh
#
# TX/RX counter names are defined in my_run_p2p_host.sh and
# passed via TX_COUNTERS_STR and RX_COUNTERS_STR exports.
# Do not define counter names here.
#
# Usage: ./cxi_monitor.sh <benchmark_command> [args...]
# Requires: RESULTS_DIR, TX_COUNTERS_STR, RX_COUNTERS_STR
# =============================================================

: ${SAMPLE_INTERVAL:=1}

# Validate required exports
if [ -z "${RESULTS_DIR}" ] || [ ! -d "${RESULTS_DIR}" ]; then
    echo "Error: RESULTS_DIR is not set or does not exist: ${RESULTS_DIR}"
    exit 1
fi

if [ -z "${TX_COUNTERS_STR}" ] || [ -z "${RX_COUNTERS_STR}" ]; then
    echo "Error: TX_COUNTERS_STR or RX_COUNTERS_STR is not set."
    echo "  Define and export them in my_run_p2p_host.sh"
    exit 1
fi

# Reconstruct arrays from exported strings
read -ra TX_COUNTERS <<< "${TX_COUNTERS_STR}"
read -ra RX_COUNTERS <<< "${RX_COUNTERS_STR}"

NODE=${SLURMD_NODENAME:-$(hostname -s)}
NODE_DIR="${RESULTS_DIR}/${NODE}"

TELEM_PATH="/sys/class/cxi/cxi0/device/telemetry"
MONITOR_FILE="${NODE_DIR}/monitor_${NODE}.csv"


# =============================================================
# Helper: write one timestamped sample of all counters
# Direction is known explicitly — no inference needed
# =============================================================
write_monitor_sample() {
    local timestamp=$1

    # TX counters
    for counter in "${TX_COUNTERS[@]}"; do
        local telem_file="${TELEM_PATH}/${counter}"
        local value=0
        if [ -f "${telem_file}" ]; then
            value=$(cat "${telem_file}" 2>/dev/null || echo 0)
        fi
        echo "${timestamp},${counter},TX,${value}"
    done >> "${MONITOR_FILE}"

    # RX counters
    for counter in "${RX_COUNTERS[@]}"; do
        local telem_file="${TELEM_PATH}/${counter}"
        local value=0
        if [ -f "${telem_file}" ]; then
            value=$(cat "${telem_file}" 2>/dev/null || echo 0)
        fi
        echo "${timestamp},${counter},RX,${value}"
    done >> "${MONITOR_FILE}"
}


# =============================================================
# Determine role from SLURM rank
# =============================================================
if [ "${SLURM_PROCID:-0}" -eq 0 ]; then
    ROLE="SENDER"
else
    ROLE="RECEIVER"
fi

echo "${ROLE}: Rank ${SLURM_PROCID} on ${NODE}" >> ${RESULTS_DIR}/runtime.out
echo "SAMPLE_INTERVAL: ${SAMPLE_INTERVAL}"

# =============================================================
# Start continuous monitoring (rank-local process only)
# =============================================================
if [ "${SLURM_LOCALID}" -eq 0 ]; then

    if [ -d "${TELEM_PATH}" ]; then
        mkdir -p "${NODE_DIR}"

        echo "timestamp,counter_name,direction,value" > "${MONITOR_FILE}"
        echo "Starting monitor on ${NODE} → ${MONITOR_FILE}"
        echo "  TX counters: ${#TX_COUNTERS[@]}, RX counters: ${#RX_COUNTERS[@]}"

        (
            while true; do
                timestamp=$(date +%s.%N)
                write_monitor_sample "${timestamp}"
                sleep "${SAMPLE_INTERVAL}"
            done
        ) &

        telemetry_pid=$!
        echo "  Monitor PID: ${telemetry_pid}"
    else
        echo "Warning: Telemetry path not found on ${NODE}: ${TELEM_PATH}"
    fi

fi

# =============================================================
# Run the actual benchmark
# =============================================================
"$@"

# =============================================================
# Stop monitoring after benchmark completes
# =============================================================
if [ "${SLURM_LOCALID}" -eq 0 ]; then
    if [ -n "${telemetry_pid}" ]; then
        kill -9 "${telemetry_pid}"
        wait "${telemetry_pid}" 2>/dev/null
        echo "Monitor stopped on ${NODE}"
        echo "  Time-series log: ${MONITOR_FILE}"
    fi
fi