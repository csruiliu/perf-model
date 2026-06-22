# OSU Micro-Benchmark

## Build OMB on Perlmutter

The OMB source code is distributed by the MVAPICH website. It can be downloaded and unpacked using the commands:
```bash
wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.1-1.tar.gz
tar -xzf osu-micro-benchmarks-7.1-1.tar.gz
```

Compiling the OMB tests for CPUs on Perlmutter follows the common configure-make procedure:
```bash
./configure CC=cc CXX=CC --prefix=`pwd`
make
make install
```

Compiling the OMB tests for GPUs on Perlmutter follows the common configure-make procedure:

```bash
./configure CC=cc CXX=CC --prefix=`pwd` --enable-cuda=basic --with-cuda=$CUDA_HOME
make
make install
```