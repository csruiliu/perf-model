COMPFLAG  = -DNVHPC -DNVHPC_API -DNVIDIA_GPU
PARAFLAG  = -DMPI  -DOMP
MATHFLAG  = -DUSESCALAPACK -DUNPACKED -DUSEFFTW3 -DHDF5 -DOPENACC -DOMP_TARGET # -DUSEPRIMME -DUSEELPA # -DOMP_TARGET
# DEBUGFLAG = -DDEBUG -DNVTX
#

NVCC=nvcc 
NVCCOPT= -O3 -use_fast_math
CUDA_DIR=
CUDALIB= -lcufft -lcublasLt -lcublas -lcudart -lcuda -lnvToolsExt

FCPP    = /usr/bin/cpp  -C   -nostdinc   #  -C  -P  -E -ansi  -nostdinc  /usr/bin/cpp
#F90free = ftn -Mfree -acc -mp=multicore,gpu -gpu=cc80  -cudalib=cublas,cufft -traceback -Minfo=all,mp,accel -gopt -traceback
#LINK    = ftn        -acc -mp=multicore,gpu -gpu=cc80  -cudalib=cublas,cufft -Minfo=mp,accel # -lnvToolsExt  
F90free = mpifort -Mfree -acc -mp=multicore,gpu -gpu=cc90  -cudalib=cublas,cufft -traceback -Minfo=all,mp,accel -gopt -traceback -tp x86-64-v3
LINK    = mpifort        -acc -mp=multicore,gpu -gpu=cc90  -cudalib=cublas,cufft -Minfo=mp,accel # -lnvToolsExt  
FOPTS   = -fast -Mfree -Mlarge_arrays
# F90free = ftn -Mfree -acc=sync,wait -mp=multicore,gpu -gpu=cc80  -cudalib=cublas,cufft -traceback -Minfo=all,mp,acc -gopt -traceback
# LINK    = ftn        -acc=sync,wait -mp=multicore,gpu -gpu=cc80  -cudalib=cublas,cufft -Minfo=mp,acc # -lnvToolsExt
# FOPTS   = -O0 -Mfree # -Mlarge_arrays # FF epsilon hangs
FNOOPTS = $(FOPTS)
MOD_OPT = -module  
INCFLAG = -I #./

#C_PARAFLAG  = -DPARA -DMPICH_IGNORE_CXX_SEEK
C_PARAFLAG  = -DPARA
CC_COMP = mpicxx 
C_COMP  = mpicc
C_LINK  = mpicc -lstdc++ # ${CUDALIB} -lstdc++
C_OPTS  = -fast -mp 
C_DEBUGFLAG =

REMOVE  = /bin/rm -f

#FFT_CUDALIB=/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/nvhpc-23.11-gh5cygvdqksy6mxuy2xgoibowwxi3w7t/Linux_x86_64/23.11/math_libs/lib64/libcufft.so
FFTW_DIR=${SCRATCH}/local/fftw_3.3.10-pl/nvhpc_23.11_ompi_4.1.6
FFTW_LIB=${FFTW_DIR}/lib
FFTW_INC=${FFTW_DIR}/include
FFTWLIB      = $(FFTW_LIB)/libfftw3.so \
               $(FFTW_LIB)/libfftw3_threads.so \
               $(FFTW_LIB)/libfftw3_omp.so \
               ${CUDALIB}  -lstdc++
#FFTWLIB      = $(FFTW_DIR)/libfftw3.so \
#               $(FFTW_DIR)/libfftw3_threads.so \
#               $(FFTW_DIR)/libfftw3_omp.so \
#               ${CUDALIB}  -lstdc++
FFTWINCLUDE  = ${FFTW_INC}
PERFORMANCE  = 

SCALAPACKLIB = ${SCRATCH}/local/scalapack_2.2.2-pl/lib/libscalapack.so
#LAPACKLIB = /usr/lib64/libopenblasp.so.0
#LAPACKLIB = ${HOME}/local/openblas/install/openblas_0.3.29/gcc_13.2.0_omp_spr/lib64/libopenblas.so
LAPACKLIB = /global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/nvhpc-23.11-gh5cygvdqksy6mxuy2xgoibowwxi3w7t/Linux_x86_64/23.11/compilers/lib/liblapack_lp64.so.0 /global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/nvhpc-23.11-gh5cygvdqksy6mxuy2xgoibowwxi3w7t/Linux_x86_64/23.11/compilers/lib/libblas_lp64.so.0

HDF5_DIR=${SCRATCH}/local/hdf5_1.14.6-pl/nvhpc_23.11_ompi_4.1.6
HDF5_LDIR    =  ${HDF5_DIR}/lib
#HDF5LIB      = -L$(HDF5_LDIR) -lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -lz -ldl
HDF5LIB      = -L$(HDF5_LDIR) -lhdf5_hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -lz $(SCALAPACKLIB) $(LAPACKLIB) -ldl
HDF5INCLUDE  = ${HDF5_DIR}/include
