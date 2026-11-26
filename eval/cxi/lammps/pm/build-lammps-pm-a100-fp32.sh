#!/bin/bash
#
# Build LAMMPS (KOKKOS + ML-SNAP, single precision) for Perlmutter A100 GPUs.

set -euo pipefail

ml cmake

# ---- Configuration -----------------------------------------------------------
LAMMPS_REPO=https://github.com/lammps/lammps
LAMMPS_COMMIT=372c8cdabab05268417ae7627b0a3005a9da2642
BUILD_JOBS=${BUILD_JOBS:-8}

BASE_DIRECTORY=/pscratch/sd/r/ruiliu/lammps-pm-a100-fp32
SOURCE_DIRECTORY=${BASE_DIRECTORY}/lammps
BUILD_DIRECTORY=${BASE_DIRECTORY}/build_lammps
INSTALL_DIRECTORY=${BASE_DIRECTORY}/install_lammps

# ---- Fresh start -------------------------------------------------------------
rm -rf "${BASE_DIRECTORY:?}"
mkdir -p "${BASE_DIRECTORY}" "${BUILD_DIRECTORY}" "${INSTALL_DIRECTORY}"

# ---- Fetch source at the pinned commit ---------------------------------------
git clone "${LAMMPS_REPO}" "${SOURCE_DIRECTORY}"
git -C "${SOURCE_DIRECTORY}" checkout "${LAMMPS_COMMIT}"

# ---- Configure ---------------------------------------------------------------
cmake -S "${SOURCE_DIRECTORY}/cmake" -B "${BUILD_DIRECTORY}" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX="${INSTALL_DIRECTORY}" \
    -D CMAKE_VERBOSE_MAKEFILE=ON \
    -D BUILD_SHARED_LIBS=OFF \
    -D CMAKE_CXX_COMPILER="${SOURCE_DIRECTORY}/lib/kokkos/bin/nvcc_wrapper" \
    -D BUILD_MPI=yes \
    -D MPI_C_COMPILER="$(command -v cc)" \
    -D MPI_CXX_COMPILER="$(command -v CC)" \
    -D FFT=KISS \
    -D FFT_KOKKOS=CUFFT \
    -D PKG_ML-SNAP=yes \
    -D PKG_KOKKOS=yes \
    -D KOKKOS_PREC=single \
    -D Kokkos_ENABLE_SERIAL=yes \
    -D Kokkos_ENABLE_CUDA=yes \
    -D Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF \
    -D Kokkos_ARCH_AMPERE80=ON

# ---- Build and install -------------------------------------------------------
cmake --build "${BUILD_DIRECTORY}" --parallel "${BUILD_JOBS}"
cmake --install "${BUILD_DIRECTORY}"

echo "LAMMPS installed to ${INSTALL_DIRECTORY}"