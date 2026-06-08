#!/bin/bash

#sampling interval (ms)
: ${DCGM_DELAY:=42}
echo "DCGM_DELAY: $DCGM_DELAY"

#use `dcgmi dmon -l` to get the list of available metrics and their field_id
#comment/uncomment metrics as deeded
dcgm_metrics=""

#dcgm_metrics+="110,"  #sm_app_clock
#dcgm_metrics+="140,"  #memory_temp
#dcgm_metrics+="150,"  #gpu_temp
#dcgm_metrics+="155,"  #power_utilization
#dcgm_metrics+="156,"  #total_energy_consumption
#dcgm_metrics+="164,"  #enforced_power_limit
#dcgm_metrics+="190,"  #pstate

#dcgm_metrics+="200,"  #pcie_tx_throughput
#dcgm_metrics+="201,"  #pcie_rx_throughput
#dcgm_metrics+="203,"  #gpu_utilization
#dcgm_metrics+="204,"  #mem_copy_utilization
#dcgm_metrics+="206,"  #enc_utilization
#dcgm_metrics+="207,"  #dec_utilization
#dcgm_metrics+="210,"  #mem_util_samples
#dcgm_metrics+="211,"  #gpu_util_samples

dcgm_metrics+="1001," #gr_engine_active
#dcgm_metrics+="1002," #sm_active
dcgm_metrics+="1003," #sm_occupancy
dcgm_metrics+="1004," #tensor_active
dcgm_metrics+="1005," #dram_active
dcgm_metrics+="1006," #fp64_active
dcgm_metrics+="1007," #fp32_active
dcgm_metrics+="1008," #fp16_active

dcgm_metrics+="1009," #pcie_tx_bytes
dcgm_metrics+="1010," #pcie_rx_bytes
dcgm_metrics+="1011," #nvlink_tx_bytes
dcgm_metrics+="1012," #nvlink_rx_bytes

#dcgm_metrics+="1013," #tensor_imma_active
#dcgm_metrics+="1014," #tensor_hmma_active
#dcgm_metrics+="1015," #tensor_dfma_active
#dcgm_metrics+="1016," #integer_active

dcgm_metrics="${dcgm_metrics%?}"

dcgm_outfile=dcgm.d$DCGM_DELAY.$SLURM_JOB_ID.$SLURM_STEP_ID-$SLURM_NODEID.out

if [[ $SLURM_LOCALID -eq 0 ]]; then
    # For Perlmutter
    podman-hpc exec -it dcgm-container dcgmi dmon -d $DCGM_DELAY -i 0 -e $dcgm_metrics > $RESULTS_DIR/$dcgm_outfile &
    
    dcgmi_pid=$!
fi

$@

if [[ $SLURM_LOCALID -eq 0 ]]; then
    kill -9 $dcgmi_pid
    wait $! 2>/dev/null
fi

