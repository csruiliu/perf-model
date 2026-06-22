COMPFLAG  = -DNVHPC -DNVHPC_API -DNVIDIA_GPU
PARAFLAG  = -DMPI  -DOMP
MATHFLAG  = -DUSESCALAPACK -DUNPACKED -DUSEFFTW3 -DHDF5 -DOPENACC -DOMP_TARGET # -DUSEPRIMME -DUSEELPA # -DOMP_TARGET
# DEBUGFLAG = -DDEBUG -DNVTX
#
NVCC=nvcc 
NVCCOPT= -O3 -use_fast_math
# CUDALIB= -lcufft -lcublasLt -lcublas -lcudart -lcuda -lnvToolsExt
FCPP    = /usr/bin/cpp  -C   -nostdinc   #  -C  -P  -E -ansi  -nostdinc  /usr/bin/cpp
F90free = ftn -Mfree -acc -mp=multicore,gpu -gpu=cc80  -cudalib=cublas,cufft -traceback -Minfo=all,mp,accel -gopt -traceback
LINK    = ftn        -acc -mp=multicore,gpu -gpu=cc80  -cudalib=cublas,cufft -Minfo=mp,accel # -lnvToolsExt  
FOPTS   = -fast -Mfree -Mlarge_arrays
# F90free = ftn -Mfree -acc=sync,wait -mp=multicore,gpu -gpu=cc80  -cudalib=cublas,cufft -traceback -Minfo=all,mp,acc -gopt -traceback
# LINK    = ftn        -acc=sync,wait -mp=multicore,gpu -gpu=cc80  -cudalib=cublas,cufft -Minfo=mp,acc # -lnvToolsExt
# FOPTS   = -O0 -Mfree # -Mlarge_arrays # FF epsilon hangs
FNOOPTS = $(FOPTS)
MOD_OPT = -module  
INCFLAG = -I #./
C_PARAFLAG  = -DPARA -DMPICH_IGNORE_CXX_SEEK
CC_COMP = CC
C_COMP  = cc
C_LINK  = cc -lstdc++ # ${CUDALIB} -lstdc++
C_OPTS  = -fast -mp 
C_DEBUGFLAG =
REMOVE  = /bin/rm -f
# FFTW_DIR=
FFTWLIB      = $(FFTW_DIR)/libfftw3.so \
               $(FFTW_DIR)/libfftw3_threads.so \
               $(FFTW_DIR)/libfftw3_omp.so \
               ${CUDALIB}  -lstdc++
FFTWINCLUDE  = $(FFTW_INC)
PERFORMANCE  = 
SCALAPACKLIB = 
LAPACKLIB = 
HDF5_LDIR    =  ${HDF5_DIR}/lib/
HDF5LIB      = -L$(HDF5_LDIR)/ -lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -lz -ldl
HDF5INCLUDE  = ${HDF5_DIR}/include/
