#!/usr/bin/env python3

import sys

def usage():
    print("validate.sh: test output correctness for the NERSC-10 LAMMPS benchmark.")
    print("Usage: validate.sh <output_file>")
#usage

def parse_output( Fname ):

    Natom = None
    E100  = None
    T100  = None

    F = open( Fname, 'r' )
    while( not ( Natom and E100 and T100 ) ):

        L = F.readline()

        if( L=='' ): break

        if("Created" in L ):
            S = L.split()
            if( S[0]=="Created" and S[2]=="atoms" ):
                Natom = int(S[1])

        if("Step" in L ):
            while( not E100 ):
                S = F.readline().split()
                if( S[0]=='100' ):
                    E100 = float( S[4] )
                    break

        if("Loop time of" in L ):
            T100 = float( L.split()[3] )
            
    #end while
    F.close()
            
    if( not Natom ):
        print("Error: could not find number of atoms in {:}".format( Fname ) )
        print("       should have found \"Created ##### atoms\"" )
        exit(1)

    if( not E100 ):
        print("Error: could not find Step-100 energy in {:}".format( Fname ) )
        print("       should be located shortly after the following line:" )
        print("      \"Step          Temp          E_pair         E_mol          TotEng         Press\"" )
        exit(1)

    if( not T100 ):
        print("Error: could not find walltime  in {:}".format( Fname ) )
        print("       should have found \"Loop time of # on # procs for 100 steps with # atoms\"" )
        exit(1)

    return Natom, E100, T100
#parse_output

def check_Natom( Natom ):

    ref_Natom = {
              65536:(0,'0_nano',),
             524288:(1,'1_micro'),
            4194304:(2,'2_tiny'),
           33554432:(3,'3_small'),
          268435456:(4,'4_medium'),
         2147483648:(5,'5_reference'),
        17179869184:(6,'6_target')
    }

    if( Natom in ref_Natom ):
        print("Found size:", ref_Natom[Natom][1] )
        i_size= ref_Natom[Natom][0]

    else:
        print("Error: atom count not recognized. Found:", Natom )
        print("       expected one of:", ref_Natom.keys() )
        exit(1)

    return i_size
#check_Natom

def check_result( i_size, E_measured ):

    E_ref =-8.7467391

    E_tol = 1e-4 * pow( 2, -i_size )
    E_err = abs( ( E_measured - E_ref ) / E_ref )
    status_passed = ( E_err < E_tol )

    #if( True ):
    if( not status_passed ):    
        print("  Measured: ", E_measured )
        print("  Expected: ", E_ref )
        print("  RelError: ", E_err )
        print("  Tolerance:", E_tol )
        print("  Result:   ", ("PASSED" if status_passed else "FAILED") )

    return status_passed
#check_result

def main():

    if( len(sys.argv) != 2 ):
        usage()
        exit(1)

    if( sys.argv[1] == "--help" ):
        usage()
        exit(0)
            
    Fname = sys.argv[1]

    Natom, E_measured, T_measured = parse_output( Fname )

    i_size = check_Natom( Natom )

    status_passed = check_result( i_size, E_measured )

    if( status_passed ):
        print("Validation:", "PASSED")
    else:
        print("Validation:", "FAILED")

    print("BenchmarkTime(sec):", T_measured )
#main

main()
