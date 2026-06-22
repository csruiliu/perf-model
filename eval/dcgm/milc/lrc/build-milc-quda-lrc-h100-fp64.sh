#!/bin/bash

BASE_DIRECTORY=$(pwd)
QUDA_SOURCE_DIRECTORY=${BASE_DIRECTORY}/quda_src
QUDA_INSTALL_DIRECTORY=${BASE_DIRECTORY}/quda_install
QUDA_BUILD_DIRECTORY=${BASE_DIRECTORY}/quda_build
MILC_DIRECTORY=${BASE_DIRECTORY}/milc_qcd


# Hack to find the CUDA directory
#CUDA_DIRECTORY=$(which nvcc)
#CUDA_DIRECTORY=${CUDA_DIRECTORY/\bin\/nvcc/}

pushd .

#############################################
# Part 1: Build QUDA
#############################################

cd $BASE_DIRECTORY
if [ ! -d ${QUDA_SOURCE_DIRECTORY} ]
then
  git clone --branch develop https://github.com/lattice/quda ${QUDA_SOURCE_DIRECTORY}
fi

cd ${QUDA_SOURCE_DIRECTORY}
git checkout c75b77c731eb9ad16c93b4fc312e80225a84f1ea

mkdir -p ${QUDA_BUILD_DIRECTORY}
mkdir -p ${QUDA_INSTALL_DIRECTORY}

cd ${QUDA_BUILD_DIRECTORY}

cmake ${QUDA_SOURCE_DIRECTORY} \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=${QUDA_INSTALL_DIRECTORY} \
  -DQUDA_GPU_ARCH=sm_90 \
  -DQUDA_DIRAC_DEFAULT_OFF=ON \
  -DQUDA_DIRAC_STAGGERED=ON \
  -DQUDA_QMP=ON \
  -DQUDA_QIO=ON \
  -DQUDA_DOWNLOAD_USQCD=ON && \
cmake --build ${QUDA_BUILD_DIRECTORY} --target all -- -j && \
cmake --build ${QUDA_BUILD_DIRECTORY} --target install -- -j

# Check if QUDA build was successful
if [ $? -ne 0 ]; then
  echo "QUDA build failed, exiting..."
  popd
  exit 1
fi

#############################################
# Part 2: Build MILC (Generation)
#############################################

cd $BASE_DIRECTORY
if [ ! -d ${MILC_DIRECTORY} ]
then
  git clone --branch develop https://github.com/milc-qcd/milc_qcd ${MILC_DIRECTORY}
fi

cd ${MILC_DIRECTORY}
git checkout f803f4bf

cd ${MILC_DIRECTORY}/ks_imp_rhmc

if [ ! -f "./Makefile" ]
then
  cp ../Makefile .
fi

if [ -f "./su3_rhmd_hisq" ]
then
  rm ./su3_rhmd_hisq
fi

COMPILER="gnu" \
CTIME="-DCGTIME -DFFTIME -DGATIME -DGFTIME -DREMAP -DPRTIME -DIOTIME" \
CGEOM="-DFIX_NODE_GEOM -DFIX_IONODE_GEOM" \
MY_CC=mpicc \
MY_CXX=mpicxx \
CUDA_HOME=${CUDA_HOME} \
QUDA_HOME=${QUDA_INSTALL_DIRECTORY} \
WANTQUDA=true \
WANT_FN_CG_GPU=true \
WANT_FL_GPU=true \
WANT_GF_GPU=true \
WANT_FF_GPU=true \
WANT_GA_GPU=true \
WANT_MIXED_PRECISION_GPU=0 \
PRECISION=2 \
MPP=true \
OMP=true \
WANTQIO=true \
WANTQMP=true \
QIOPAR=${QUDA_INSTALL_DIRECTORY} \
QMPPAR=${QUDA_INSTALL_DIRECTORY} \
LDFLAGS="-L${CUDA_MATH}/lib64 -Wl,-rpath,${CUDA_MATH}/lib64" \
make -j 1 su3_rhmd_hisq

popd