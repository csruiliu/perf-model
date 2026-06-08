# GPU Utilization Performance Profiling

This repo provides codes for evaluating fundamental GPU performance.

**Profiling GPU Initialization**

`eval_init.cpp`: Measures GPU initialization overhead and startup performance metrics

**Profiling PCIe Data Transfer**

`eval_io_pcie.cpp`: Profiles host-to-GPU data transfer performance over PCIe interface for single GPU configurations

**NVLink Data Transfer**

`eval_io_nvlink.cpp`: Profiles host-to-GPU data transfer performance in multi-GPU configurations utilizing NVLink interconnect technology


```bash
# enter src folder
cd src

# build binary code *.x 
make 

# copy binary code to scripts folder
cp *.x ../scripts

# enter scripts folder
cd scripts

# using slurm to run scripts and get performance results in `results` folder
sbatch run_eval_init.sh
sbatch run_eval_io_pcie.sh
sbatch run_eval_io_nvlink.sh
```
