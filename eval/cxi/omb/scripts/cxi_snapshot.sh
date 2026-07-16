#!/bin/bash

# =============================================================
# cxi_snapshot.sh
#
# TX/RX counter names are defined in my_run_p2p_host.sh and
# passed via TX_COUNTERS_STR and RX_COUNTERS_STR exports.
# Do not define counter names here.
#
# Usage: ./cxi_snapshot.sh <prefix> <duration>
# Requires: RESULTS_DIR, TX_COUNTERS_STR, RX_COUNTERS_STR
# =============================================================

PREFIX=$1
DURATION=${2:-5}

# Validate arguments
if [ -z "$PREFIX" ]; then
    echo "Usage: $0 <prefix> [duration_seconds]"
    exit 1
fi

if [ "$PREFIX" != "before" ] && [ "$PREFIX" != "after" ]; then
    echo "Error: prefix must be 'before' or 'after', got '${PREFIX}'"
    exit 1
fi

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

# Node name — Level 2 directory
NODE=${SLURMD_NODENAME:-$(hostname -s)}
NODE_DIR="${RESULTS_DIR}/${NODE}"

# One NIC per node
TELEM_PATH="/sys/class/cxi/cxi0/device/telemetry"


# =============================================================
# Helper: take a single point-in-time snapshot
# Writes CSV: counter_name,direction,value
# Direction is known explicitly — TX list and RX list separate
# =============================================================
write_snapshot() {
    local output_file=$1

    echo "counter_name,direction,value" > "${output_file}"

    # TX counters
    for counter in "${TX_COUNTERS[@]}"; do
        local telem_file="${TELEM_PATH}/${counter}"
        local value=0
        if [ -f "${telem_file}" ]; then
            value=$(cat "${telem_file}" 2>/dev/null || echo 0)
        else
            echo "  [WARN] Counter not found: ${telem_file}"
        fi
        echo "${counter},TX,${value}"
    done >> "${output_file}"

    # RX counters
    for counter in "${RX_COUNTERS[@]}"; do
        local telem_file="${TELEM_PATH}/${counter}"
        local value=0
        if [ -f "${telem_file}" ]; then
            value=$(cat "${telem_file}" 2>/dev/null || echo 0)
        else
            echo "  [WARN] Counter not found: ${telem_file}"
        fi
        echo "${counter},RX,${value}"
    done >> "${output_file}"
}


# =============================================================
# Helper: compute delta (after - before) → counters.csv
# =============================================================
compute_delta() {
    local before_file=$1
    local after_file=$2
    local delta_file=$3

    awk -F',' '
        NR == FNR {
            if (FNR == 1) next
            before[$1] = $3
            next
        }
        FNR == 1 {
            print
            next
        }
        {
            counter   = $1
            direction = $2
            after_val  = $3 + 0
            before_val = (counter in before) ? before[counter] + 0 : 0
            delta = after_val - before_val
            if (delta < 0) delta = 0
            print counter "," direction "," delta
        }
    ' "${before_file}" "${after_file}" > "${delta_file}"
}


# =============================================================
# Main
# =============================================================
echo "=============================================="
echo "cxi_snapshot.sh"
echo "  Node       : ${NODE}"
echo "  Prefix     : ${PREFIX}"
echo "  Duration   : ${DURATION}s"
echo "  RESULTS_DIR: ${RESULTS_DIR}"
echo "  TX counters: ${#TX_COUNTERS[@]}"
echo "  RX counters: ${#RX_COUNTERS[@]}"
echo "=============================================="

if [ ! -d "${TELEM_PATH}" ]; then
    echo "Error: Telemetry path not found on ${NODE}: ${TELEM_PATH}"
    exit 1
fi

mkdir -p "${NODE_DIR}"

echo "Waiting ${DURATION} seconds for stability on ${NODE}..."
sleep "${DURATION}"

OUTPUT_FILE="${NODE_DIR}/${PREFIX}_counters.csv"
echo "Taking '${PREFIX}' snapshot on ${NODE}..."
write_snapshot "${OUTPUT_FILE}"

N_COUNTERS=$(( $(wc -l < "${OUTPUT_FILE}") - 1 ))
echo "  [OK] ${NODE}/${PREFIX}_counters.csv (${N_COUNTERS} counters)"

if [ "${PREFIX}" == "after" ]; then

    BEFORE_FILE="${NODE_DIR}/before_counters.csv"
    DELTA_FILE="${NODE_DIR}/counters.csv"

    if [ ! -f "${BEFORE_FILE}" ]; then
        echo "  [WARN] before_counters.csv missing on ${NODE}"
        echo "  [WARN] Copying after snapshot as counters.csv"
        cp "${OUTPUT_FILE}" "${DELTA_FILE}"
    else
        compute_delta "${BEFORE_FILE}" "${OUTPUT_FILE}" "${DELTA_FILE}"
        echo "  [OK] ${NODE}/counters.csv (delta = after - before)"
    fi

    echo ""
    echo "Files written to ${NODE_DIR}:"
    find "${NODE_DIR}" -name "*.csv" | sort | \
        sed 's|'"${RESULTS_DIR}"'/||' | \
        awk '{print "  " $0}'
fi

echo "Snapshot '${PREFIX}' complete on ${NODE}"