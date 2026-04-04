#!/bin/bash

# =============================================================
# cxi_snapshot.sh
#
# Usage: ./cxi_snapshot.sh <prefix> <duration>
#
# Supports multiple CXI NICs. For each NIC discovered under
# /sys/class/cxi/cxiN/, snapshots are written to a per-NIC
# subdirectory: ${RESULTS_DIR}/${NODE}/cxiN/
# =============================================================

PREFIX=$1
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

NODE=${SLURMD_NODENAME:-$(hostname -s)}
NODE_DIR="${RESULTS_DIR}/${NODE}"

# =============================================================
# Discover all CXI NICs that have a telemetry directory.
# Populates the global CXI_DEVICES array, sorted for
# consistent ordering across runs.
# =============================================================
discover_cxi_devices() {
    local -a devices=()
    local nic_path nic_name

    for nic_path in /sys/class/cxi/cxi*/; do
        [ -d "${nic_path}" ] || continue
        nic_name=$(basename "${nic_path}")
        if [ -d "${nic_path}/device/telemetry" ]; then
            devices+=("${nic_name}")
        fi
    done

    mapfile -t CXI_DEVICES < <(printf '%s\n' "${devices[@]}" | sort)
}

# =============================================================
# Take a point-in-time snapshot for one NIC.
# Args:
#   $1  output_file  path to write CSV to
#   $2  telem_path   /sys/class/cxi/cxiN/device/telemetry
# Output CSV: counter_name,direction,value
# =============================================================
write_snapshot() {
    local output_file=$1
    local telem_path=$2
    local counter telem_file value direction

    {
        echo "counter_name,direction,value"
        for direction in TX RX; do
            local -n counters="${direction}_COUNTERS"
            for counter in "${counters[@]}"; do
                telem_file="${telem_path}/${counter}"
                value=0
                if [[ -f "${telem_file}" ]]; then
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
# Compute delta (after - before) and write to delta_file.
# Negative deltas (counter wrap / reset) are clamped to 0.
# Output CSV: counter_name,direction,delta_value
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
discover_cxi_devices

if [ ${#CXI_DEVICES[@]} -eq 0 ]; then
    echo "Error: No CXI NICs with a telemetry directory found under /sys/class/cxi/"
    exit 1
fi

echo "=============================================="
echo "cxi_snapshot.sh"
echo "  Node       : ${NODE}"
echo "  Prefix     : ${PREFIX}"
echo "  Duration   : ${DURATION}s"
echo "  RESULTS_DIR: ${RESULTS_DIR}"
echo "  TX counters: ${#TX_COUNTERS[@]}"
echo "  RX counters: ${#RX_COUNTERS[@]}"
echo "  NICs found : ${CXI_DEVICES[*]}"
echo "=============================================="

mkdir -p "${NODE_DIR}"

# ------------------------------------------------------------------
if [ "${PREFIX}" == "before" ]; then
# ------------------------------------------------------------------
    echo "Taking '${PREFIX}' snapshot on ${NODE}..."

    for nic in "${CXI_DEVICES[@]}"; do
        telem_path="/sys/class/cxi/${nic}/device/telemetry"
        nic_dir="${NODE_DIR}/${nic}"
        mkdir -p "${nic_dir}"

        output_file="${nic_dir}/${PREFIX}_counters.csv"
        write_snapshot "${output_file}" "${telem_path}"

        N_COUNTERS=$(( $(wc -l < "${output_file}") - 1 ))
        echo "  [OK] ${NODE}/${nic}/${PREFIX}_counters.csv (${N_COUNTERS} counters)"
    done

# ------------------------------------------------------------------
else  # PREFIX == "after"
# ------------------------------------------------------------------
    echo "Waiting ${DURATION} seconds for stability on ${NODE}..."
    sleep "${DURATION}"
    echo "Taking '${PREFIX}' snapshot on ${NODE}..."
    echo ""

    for nic in "${CXI_DEVICES[@]}"; do
        telem_path="/sys/class/cxi/${nic}/device/telemetry"
        nic_dir="${NODE_DIR}/${nic}"
        mkdir -p "${nic_dir}"

        output_file="${nic_dir}/${PREFIX}_counters.csv"
        write_snapshot "${output_file}" "${telem_path}"

        N_COUNTERS=$(( $(wc -l < "${output_file}") - 1 ))
        echo "  [OK] ${NODE}/${nic}/${PREFIX}_counters.csv (${N_COUNTERS} counters)"

        before_file="${nic_dir}/before_counters.csv"
        delta_file="${nic_dir}/counters.csv"

        if [ ! -f "${before_file}" ]; then
            echo "  [WARN] before_counters.csv missing for ${nic} on ${NODE}"
            echo "  [WARN] Copying after snapshot as counters.csv"
            cp "${output_file}" "${delta_file}"
        else
            compute_delta "${before_file}" "${output_file}" "${delta_file}"
            echo "  [OK] ${NODE}/${nic}/counters.csv (delta = after - before)"
        fi
        echo ""
    done

    echo "Files written to ${NODE_DIR}:"
    find "${NODE_DIR}" -name "*.csv" | sort | \
        sed 's|'"${RESULTS_DIR}"'/||' | \
        awk '{print "  " $0}'
fi

echo ""
echo "Snapshot '${PREFIX}' complete on ${NODE}"