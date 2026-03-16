# Performance Modeling and Prediction for GPU-Workloads Using DCGM

We use collected Nvidia DCGM data for performance modeling and prediction. We evaluate the model via various benchmarking applications including GEMM, BableStream, BerkeleyGW, LAMMPS, MILC, etc.

## Containerized DCGM deployment on Perlmutter

The deploy containerized DCGM on Perlmutter, we need to add following option in sbatch scripts.

```
#SBATCH --perf=generic
```

Also, we need to use `podman-hpc` to start containerized dcgm. Usually, we should add it in the sbatch script or interactive script as well.

```
podman-hpc run -d -it --name dcgm-container --rm \
    --gpu --cap-add SYS_ADMIN -p 5555:5555 \
    nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04
```

## Containerized DCGM deployment on Lawrencium/Einsteinium

DCGM is not currently installed on Lawrencium/Einsteinium systems. To deploy DCGM on these clusters, we'll use a container-based approach with Singularity, which is the default container tool. Use the following commands for deployment:

Start dcgm-instance using 4.4.1-2-ubuntu22.04 (the latest version at the time of writing), and `--fakeroot` is the key option.

```bash
singularity instance start \
  --fakeroot \
  --nv \
  --writable-tmpfs \
  --bind /tmp:/tmp \
  --network=none \
  docker://nvidia/dcgm:4.4.1-2-ubuntu22.04 \
  dcgm-instance
```

Starting DCGM engine in background

```bash
singularity exec instance://dcgm-instance nv-hostengine -n &
```

Using container-based `dcgmi dmon`, which usually defined in `wrap-dcgmi.sh`.

```bash
singularity exec instance://dcgm-instance \
    dcgmi dmon -d $dcgm_delay -i 0 -e $dcgm_metrics \
    > $RESULTS_DIR/$dcgm_outfile &
```

## Profiling Tools for MPI Time Modeling

### IPM (Integrated Performance Monitoring)

Using the following script to build IPM on Perlmutter

```bash
# Get IPM
git clone https://github.com/nerscadmin/IPM.git

# Regenerate all build system files from scratch, and install any missing helper files.
autoreconf -fi

# install the IPM
./configure --prefix=/pscratch/sd/r/ruiliu/IPM/install FC=ftn F77=ftn CC=cc
make 
make install
```

Adding the corresponding environmental variable

```bash
export IPM_HOME="/pscratch/sd/r/ruiliu/IPM/install"
export LD_LIBRARY_PATH=$IPM_HOME/lib:$LD_LIBRARY_PATH
export PATH=$IPM_HOME/bin:$PATH
```

Running IPM for OMB--Static Linking

1. Recompile OMB with IPM

```bash
cd /pscratch/sd/r/ruiliu/osu-micro-benchmarks-IPM

export IPM_HOME=/pscratch/sd/r/ruiliu/IPM/install

./configure CC=cc CXX=CC FC=ftn \
    --prefix=`pwd` \
    LDFLAGS="-L$IPM_HOME/lib -lipm" \
    CFLAGS="-I$IPM_HOME/include"

make
make install
```

2. Run Sbatch Scripts

```bash
# add the following to your sbatch
export IPM_HOME=/pscratch/sd/r/ruiliu/IPM/install
export IPM_LOGDIR=/pscratch/sd/r/ruiliu/ipm-logs
export IPM_LOG=full
export IPM_REPORT=terse

mkdir -p $IPM_LOGDIR
```

3. Parse the IPM Output

```bash
cd $IPM_LOGDIR

# Parse IPM output
$IPM_HOME/bin/ipm_parse -html osu_bw.username.hash.xml

$IPM_HOME/bin/ipm_parse -full osu_bw.username.hash.xml
```

### Darshan

Build Darshan

```bash
# Get darshan 3.5.0
wget https://github.com/darshan-hpc/darshan/releases/download/3.5.0/darshan-3.5.0.tar.gz
tar -xvf darshan-3.5.0.tar.gz
cd darshan-3.5.0

# build darshan
./configure --prefix=$HOME/software/darshan --with-log-path=$SCRATCH/darshan-logs --with-jobid-env=SLURM_JOB_ID CC=cc
make 
make install

# set darshan environment variables 
DARSHAN_HOME="/global/homes/r/ruiliu/software/darshan"
export LD_LIBRARY_PATH=$DARSHAN_HOME/lib:$LD_LIBRARY_PATH
export PATH=$DARSHAN_HOME/bin:$PATH
```

Recompile OMB with Darshan

```bash
./configure CC=cc CXX=CC CFLAGS="-I$DARSHAN_HOME/include" LDFLAGS="-L$DARSHAN_HOME/lib -ldarshan -lz -ldl -Wl,--export-dynamic" --prefix=`pwd`
make
make install
```

Run Sbatch 

```bash
# add following to your sbatch scripts

export DARSHAN_LOG_PATH=$SCRATCH/darshan-logs
mkdir -p $DARSHAN_LOG_PATH

darshan-mk-log-dirs.pl $DARSHAN_LOG_PATH
```

Parse Darshan output

```bash
# quick plain output
darshan-parser <your>/<path>/<your_log>.darshan

# generate html
# Install PyDarshan if not already installed
pip install --user darshan

# Generate HTML report directly
python -m darshan summary <your>/<path>/<your_log>.darshan
```


