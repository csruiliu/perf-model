# Performance Modeling and Prediction

We use collected Nvidia DCGM and HPE CXI data for performance modeling and prediction. 

We evaluate the model via various benchmarking applications including GEMM, BableStream, BerkeleyGW, LAMMPS, MILC, etc.

## Installation

```bash
git clone https://github.com/username/project.git
cd project
# better to be in a virtual environment
pip install -e .
```

## Quick Start

One exmaple to quickly run the DCGM-based model using the collected counters of BerkeleyGW is shown below, 

```bash
# in the root folder
python3 -m counter_model.dcgm.modeling_dcgm_sg -f ./eval/bgw/results/BGW_EPSILON_45934168-a100-fp64/dcgm.d1000.45934168.0-0.out -d 1000 -o 932000 -rg A100-40 -tg H100-SXM -rh Perlmutter -th Einsteinium-H100 -ca same --metrics GRACT,SMOCC,TENSO,DRAMA,FP64A,FP32A,FP16A,PCITX,PCIRX,NVLTX,NVLRX
```


One exmaple to quickly run the CXI-based model using the collected counters of LAMMPS is shown below, 

```bash
# in the root folder
python3 -m counter_model.cxi.modeling_cxi --counter_dir=./eval/lammps/results/LPS_SMALL_FP32_CTR_51907293 --msg_set fine -rh Perlmutter
```
