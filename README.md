# Performance Modeling and Prediction

We use collected Nvidia DCGM and HPE CXI data for performance modeling and prediction. 

We evaluate the model via various benchmarking applications including GEMM, BableStream, BerkeleyGW, LAMMPS, MILC, etc.

## Installation

```bash
git clone git@github.com:csruiliu/perf-model.git
cd perf_model
# better to be in a virtual environment
pip install -e .
```

## Quick Start

One exmaple to quickly run the DCGM-based model using the collected counters of BerkeleyGW is shown below, 

```bash
# in the root folder
python3 -m counter_model.dcgm.launcher \
    -job_mode single \
    --num_gpu 1 \
    --dcgm_input <dcgm_out_file> \
    -d 1000 \
    -o 932000 \
    -rg A100-40 \
    -tg H100-SXM \
    -rh Perlmutter \
    -th Einsteinium-H100 \
    --cores_alloc same
```


One exmaple to quickly run the CXI-based model using the collected counters of LAMMPS is shown below, 

```bash
# in the root folder
python3 -m counter_model.cxi.modeling_cxi \
    --counter_dir=<counter_file_folder> \
    --msg_set fine \
    -rh Perlmutter
```
