#!/bin/bash

module load gcc/11.4.0

BASE_DIRECTORY=/global/scratch/users/rliu5/lammps-lrc-h100-fp32
SOURCE_DIRECTORY=${BASE_DIRECTORY}/lammps
BUILD_DIRECTORY=${BASE_DIRECTORY}/build_lammps
INSTALL_DIRECTORY=${BASE_DIRECTORY}/install_lammps

cd $BASE_DIRECTORY
if [ ! -d ${SOURCE_DIRECTORY} ]
then
  git clone https://github.com/lammps/lammps
  git checkout 372c8cdabab05268417ae7627b0a3005a9da2642
fi

mkdir -p ${BUILD_DIRECTORY}
mkdir -p ${INSTALL_DIRECTORY}

cd ${BUILD_DIRECTORY}

cmake ${SOURCE_DIRECTORY}/cmake \
    -D BUILD_SHARED_LIBS=OFF \
    -D CMAKE_INSTALL_PREFIX=${INSTALL_DIRECTORY} \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_VERBOSE_MAKEFILE=ON \
    -D CMAKE_CXX_COMPILER=${SOURCE_DIRECTORY}/lib/kokkos/bin/nvcc_wrapper \
    -D MPI_C_COMPILER=mpicc \
    -D MPI_CXX_COMPILER=mpicxx \
    -D BUILD_MPI=yes \
    -D FFT=KISS \
    -D FFT_KOKKOS=CUFFT \
    -D PKG_ML-SNAP=yes \
    -D PKG_KOKKOS=yes \
    -D KOKKOS_PREC=single \
    -D Kokkos_ENABLE_CUDA=yes \
    -D Kokkos_ENABLE_SERIAL=yes \
    -D Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF \
    -D Kokkos_ARCH_HOPPER90=ON && \
cmake --build ${BUILD_DIRECTORY} --target all -- -j4 && \
cmake --build ${BUILD_DIRECTORY} --target install -- -j4
