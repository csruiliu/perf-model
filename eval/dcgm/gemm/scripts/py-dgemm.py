#!/usr/bin/env python3

import time
import argparse
import numpy as np
import math

##Hardware-specific python modules (including drop-in Numpy subtitutes) here
##substitute your own!
##NumPy substitutes should be aliased to "xp"
try:
    import cupy 
    xp = cupy
except:
    pass

#// -----
#// Function: numpy_initializer
#// Switch between numpy(np) or accelerated-numpy(xp).
#// Device or library initialization may be added to this function if needed.
#// -----
def numpy_initializer( shownumpy ):

    global xp
    if accelerator:
        try:
            print("Using accelerated numpy:\n  {}\n".format( xp ) )
        except:
            print("The --accelerator option was used, but no accelerated numpy (e.g. cupy) was found.")
            exit(1)
    else:
        xp = np

    if shownumpy:
        print( xp.show_config() )

    return xp

#// -----
#// Function: synchronize_host_accel
#// This is a no-op if running on the host
#// May be modified for non-cupy
#// -----
def synchronize_host_accel():
    if accelerator:
        cupy.cuda.runtime.deviceSynchronize()
        

#// -----
#// Function: initialize_accel_arrays
#// Initialize matrices using accelerator memory/processor
#// Here, cupy arrays are used to run this custom kernel on 
#// May be modified for non-cupy
#// -----
def initialize_accel_arrays( nsize, A, B ):

    @cupy.fuse()
    def cupy_fuse_kernel(j, k):
        a = j*cupy.sin(j) + k*cupy.cos(k)
        b = k*cupy.cos(j) + j*cupy.sin(k)
        return a, b

    j, k = cupy.mgrid[0:nsize, 0:nsize]
    A[:], B[:] = cupy_fuse_kernel(j, k)
    cupy.cuda.runtime.deviceSynchronize()


#// ----
#// Function: copy_array_accel_to_host
#//This may be modified for non-cupy.
#// ----
def copy_array_accel_to_host( Ad, Ah ):
    Ah= Ad.get()
    

#// -----
#// CODE BELOW THIS LINE SHOULD NOT BE MODIFIED
#// -----

#// -----
#// Function: initialize_host_arrays
#// Initialize matrices using host memory/processor
#// -----
def initialize_host_arrays( nsize, A, B ):
    for j in range(nsize):
        for k in range(nsize):
            A[j, k] = j*math.sin(j) + k*math.cos(k)
            B[j, k] = k*math.cos(j) + j*math.sin(k)


#// -----
#// Function: create_arrays
#// allocate matrices and call their initialization functions
#// -----
def create_arrays(nsize, xp ):

    def memory_string( memory_bytes ):
        units = ' kMGTPX'
        iunit = int( math.log( memory_bytes, 1024 ) )
        memory_units = memory_bytes / 1024**iunit
        memory_str = "{:.3f} {}B".format( memory_units, units[iunit] )
        return memory_str

    print("Preparing Matrix arrays")
    print("Memory required: {}".format( memory_string( 3 * nsize * nsize * 8 ) ) )

    t_start = time.time()
    A = xp.zeros((nsize,nsize))
    B = xp.zeros((nsize,nsize))
    C = xp.zeros((nsize,nsize))
    t_end = time.time()
    deltat = t_end - t_start
    print("Time for Array Allocation (sec): {:.6f}".format( deltat ) )
    

    t_start = time.time()
    if accelerator:
        initialize_accel_arrays( nsize, A, B )
    else:
        initialize_host_arrays( nsize, A, B )
    t_end = time.time()
    deltat = t_end - t_start
    print("Time for Array Initialization (sec): {:.3f}".format( deltat ) )

    return A, B, C



        
#// ------------------------------------------------------- //
#// Function: matmul_loop
#//
#// Run & time matmul iterations.
#// This function should not be modified.
#// The call to xp.matmul should have the same signature, even if xp != Numpy
#// ------------------------------------------------------- //
def matmul_loop(niterations, A, B, C, xp ):


    print("Running matmul...")

    tstart = time.time()
    synchronize_host_accel()
    tend = time.time()
    deltat = tend - tstart
    print("Synchronization Overhead (sec): {:.2e}".format( deltat ) )
       
    deltat = np.zeros( niterations )
    for i in range(niterations):

        synchronize_host_accel()
        tstart = time.time()

        xp.matmul(A, B, out=C )

        synchronize_host_accel()
        tend = time.time()

        deltat[i] = tend - tstart

        if( i==0 ):
            print("First of {:d} iterations (sec): {:.6f}".format( niterations, deltat[0] ) )

    # sanity check array type
    #print("type of C:", type(C))

    return deltat



#// ------------------------------------------------------- //
#// Function: check_correctness
#// Sample a number (ntest) of matrix elements to compare
#// Test against pythonic dot product
#// ------------------------------------------------------- //    
def check_correctness( nsize, A, B, C, testseed ):

    print("Running correctness test...")
    
    C_test = C
    if accelerator:
        C_test = np.zeros((nsize, nsize))
        copy_array_accel_to_host( C, C_test )

    ntest = 1024
    is_correct = True
    rng = np.random.default_rng(testseed)
    for itest in range( ntest ):

        i = rng.integers( nsize, size=1 )[0]
        j = rng.integers( nsize, size=1 )[0]
        c_test = C[i,j]
        c_ref = sum( [ (i*math.sin(i) + k * math.cos(k)) *
                       (j*math.cos(k) + k * math.sin(j))
                       for k in range( nsize ) ] )
        itest_correct = math.isclose( c_ref, c_test )
        
        if not itest_correct:
            msg = "Error Found at row {:d}, col {:d}. Expected: {:e}, Found: {:e}"
            print( msg.format( i, j, c_ref, c_test ) )
            is_correct = False

    print()
    print("Correctness test: {}".format(( "Passed" if is_correct  else "Failed")) )
    return is_correct

    
#// Function: report_performance
def report_performance(niterations, nsize, deltat_matmul ):
  

    flops = 2 * (nsize**3) + 2 * nsize**2    
    gflops = [ flops / t / 1.0e9 for t in deltat_matmul ]

    print_all_iterations = False
    if( print_all_iterations ):
        print("FlopCount: {}".format( flops ) )
        for i in range( niterations ):
            print("iter: {:2d}   time: {:.6f}   gflops: {: 7.2f}".format( i, deltat_matmul[i], gflops[i] ) )
        print("")

        
    ind = { "First":0,
            "Last":niterations-1,
            "Best":np.argmin( deltat_matmul ) }
    
    print("FlopCount: {:e}".format( flops ) )
    print("{:15s}   {:7s} {:7s}".format("Iteration (int)","Time(s)","Gflop/s"))
    for s in ["First", "Last", "Best"]:
        i = ind[s]
        si = "{:s} ({:d})".format( s, i )
        print("{:15s}   {:7.5f} {:7.1f}".format( si, deltat_matmul[i], gflops[i] ) )


#// -----
#// Function: get_args
#// Parse and print the command line arguments
#// -----
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--niterations", type=int, required=False, default=10, help="number of iterations")
    parser.add_argument("--nsize", type=int, required=False, default=8000, help="dimension of square matrix")
    parser.add_argument("--accelerator", required=False, action='store_true', help="option to use accelerator")
    parser.add_argument("--shownumpy", required=False, action='store_true', help="show numpy configuration")
    parser.add_argument("--testseed", type=int, required=False, default=None, help="random seed for sampling matrix elements to validate (integer).")
    args = parser.parse_args()

    print("Requested Arguments:")
    print("  {:12s}: {}".format( "niterations", args.niterations ))
    print("  {:12s}: {}".format( "nsize",       args.nsize       ))
    print("  {:12s}: {}".format( "accelerator", args.accelerator ))
    print("  {:12s}: {}".format( "testseed",    args.testseed    ))
    
    return args

#// ------------------------------------------------------- //
#// Function: main
#// -----
def main():

    #retreive command line arguments
    args = get_args()

    #stores accelerator as a global variable
    global accelerator
    accelerator = args.accelerator
    niterations = args.niterations
    nsize       = args.nsize
    testseed    = args.testseed
    
    #choose the appropriate numpy-like interface:
    xp = numpy_initializer( args.shownumpy )

    #create working arrays on the target processor ( host or accelerator )
    [ A, B, C ] = create_arrays( nsize, xp )
          
    # do matmul (dgemm) 
    deltat_matmul = matmul_loop( niterations, A, B, C, xp )

    # check against source of truth
    is_correct = check_correctness( nsize, A, B, C, testseed )
    assert( is_correct )

    # if correctness test has passed, report performance
    report_performance( niterations, nsize, deltat_matmul )
    
if __name__ == '__main__':
    main()


