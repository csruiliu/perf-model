#!/bin/bash

# =============================================================
# cxi_snapshot.sh
#
# Usage: ./cxi_snapshot.sh <prefix> <duration>
# =============================================================

# prefix: "before" or "after" (used in output filename)
PREFIX=$1
# Assign default duration of 10 seconds if not provided as second argument
DURATION=${2:-10}

# Validate arguments
if [ -z "$PREFIX" ]; then
    echo "Usage: $0 <prefix> [duration_seconds]"
    exit 1
fi

if [ "$PREFIX" != "before" ] && [ "$PREFIX" != "after" ]; then
    echo "Error: prefix must be 'before' or 'after', got '${PREFIX}'"
    exit 1
fi

# Check required environment variables
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
# Use SLURMD_NODENAME if available (set by srun), otherwise fallback to hostname
NODE=${SLURMD_NODENAME:-$(hostname -s)}
NODE_DIR="${RESULTS_DIR}/${NODE}"

# One NIC per node
TELEM_PATH="/sys/class/cxi/cxi0/device/telemetry"


# =============================================================
# Take a single point-in-time snapshot
# Writes CSV: counter_name,direction,value
# =============================================================
write_snapshot() {
    local output_file=$1
    local counter telem_file value direction

    {
        echo "counter_name,direction,value"
        # Loop over TX and RX counters separately, using the direction
        for direction in TX RX; do
            local -n counters="${direction}_COUNTERS"
            for counter in "${counters[@]}"; do
                telem_file="${TELEM_PATH}/${counter}"
                value=0
                if [[ -f "${telem_file}" ]]; then
                    # read the value from the telemetry file, defaulting to 0 if read fails
                    # the format is value@timestamp, so we extract the value before the '@'
                    IFS='@' read -r value _ < "${telem_file}" || value=0
                else
                    echo "  [WARN] Counter not found: ${telem_file}" >&2
                fi
                echo "${counter},${direction},${value}"
            done
        done
    } > "${output_file}"
}

# =============================================================
# Compute delta (after - before) → counters.csv
# Output: counter_name,direction,delta_value in each line
# =============================================================
compute_delta() {
    local before_file=$1
    local after_file=$2
    local delta_file=$3

    awk -F',' '
        BEGIN { OFS = "," }
        NR == FNR {
            if (FNR > 1) before[$1,$2] = $3 + 0
            next
        }
        FNR == 1 { print; next }
        {
            delta = ($3 + 0) - before[$1,$2]
            print $1, $2, (delta > 0 ? delta : 0)
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

OUTPUT_FILE="${NODE_DIR}/${PREFIX}_counters.csv"

if [ "${PREFIX}" == "before" ]; then
    # "before" path: snapshot first, then sleep for stability
    echo "Taking '${PREFIX}' snapshot on ${NODE} (before stability wait)..."
    write_snapshot "${OUTPUT_FILE}"

    # counts the number of lines in the file (subtract 1 for header)
    N_COUNTERS=$(( $(wc -l < "${OUTPUT_FILE}") - 1 ))
    echo "  [OK] ${NODE}/${PREFIX}_counters.csv (${N_COUNTERS} counters)"

    # Sleep after snapshot to let traffic settle before "after" snapshot
    echo "Waiting ${DURATION} seconds for stability on ${NODE}..."
    sleep "${DURATION}"

else
    # "after" path: sleep first to let traffic settle, then snapshot
    echo "Waiting ${DURATION} seconds for stability on ${NODE}..."
    sleep "${DURATION}"
    echo "Taking '${PREFIX}' snapshot on ${NODE}..."
    write_snapshot "${OUTPUT_FILE}"

    # counts the number of lines in the file (subtract 1 for header)
    N_COUNTERS=$(( $(wc -l < "${OUTPUT_FILE}") - 1 ))
    echo "  [OK] ${NODE}/${PREFIX}_counters.csv (${N_COUNTERS} counters)"

    BEFORE_FILE="${NODE_DIR}/before_counters.csv"
    DELTA_FILE="${NODE_DIR}/counters.csv"

    if [ ! -f "${BEFORE_FILE}" ]; then
        echo "  [WARN] before_counters.csv missing on ${NODE}"
        echo "  [WARN] Copying after snapshot as counters.csv"
        cp "${OUTPUT_FILE}" "${DELTA_FILE}"
    else
        # compute delta and write to counters.csv
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
