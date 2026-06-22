#! /bin/awk -f
#Usage: check-generation.sh <output>

BEGIN{

    nx = 0;
    ny = 0;
    nz = 0;
    nt = 0;
    
    p_count = 0;
    p_value = 0;

    traj_ctr = 0;
}


function define_expectations( nx, ny, nz, nt ){

    p_expect = 0;
    
    if( nx==48 && ny==48 && nz==48 && nt==64 ){
	print "Size:           tiny"
	p_expect=-1.418101e+08
	return;
    }
      
    if( nx==64 && ny==64 && nz==64 && nt==96 ){
        print "Size:           small"
	p_expect=-5.555243e+08
	return;q
    }

    if( nx==96 && ny==96 && nz==96 && nt==192 ){
        print "Size:           medium"
	p_expect=-4.216653e+09
	return;
    }

    if( nx==144 && ny==144 && nz==144 && nt==288 ){
        print "Size:           reference"
	p_expect=-2.288500e+10
	return;
    }

    if( nx==192 && ny==192 && nz==192 && nt==384 ){
        print "Size: target"
	#p_expect=-7.705928e+10
	p_expect = 1.0
	print "Fiduciary value not set for target problem."
	return;
    }

    print "Error: Lattice size not found. n[x,y,z,t]: ", nx, ny, nz, nt
    print "Expected one of:"
    print "  tiny:       48^3 x  64"
    print "  small:      64^3 x  96"
    print "  medium:     96^3 x 192"
    print "  reference: 144^3 x 288"
    print "  target:    192^3 x 384"
    exit 1
}

function test_result ( tr_testname, tr_measured, tr_expected, tr_tolerance ){

    tr_error = ( tr_measured - tr_expected ) / tr_expected;
    tr_error = ( tr_error > 0 ? tr_error : -tr_error );
    tr_errno = ( tr_error > tr_tolerance )
    
    if( tr_errno ){
	print tr_testname
	print "  Measured: ", tr_measured;
	print "  Expected: ", tr_expected;
	print "  RelError: ", tr_error;
	print "  Tolerance:", tr_tolerance;
	print "  Result:   FAILED"
    }

    return tr_errno
}

#get lattice parameters
/nx /{ if($1=="nx"){nx=$2;} }
/ny /{ if($1=="ny"){ny=$2;} }
/nz /{ if($1=="nz"){nz=$2;} }
/nt /{ if($1=="nt"){nt=$2;} }

#get_measurements
/PLAQUETTE ACTION:/{ p_count++; if( p_count==4 ){ p_value=$3; } }

#get timing
/Time to reload gauge configuration/{ read_time = $7; }
/Time to save/{ write_time = $5;}
/Time = /{ if( $1 != "Warmup"){ total_time = $3; } }
/Warmup Time = /{ warm_time = $4; }
/StepTime/{
    step_ctr = $2;
    if( step_ctr == 2 ){ traj_ctr +=1; }
    if( traj_ctr==1 && step_ctr==2 ){ traj1_step2_time = $5; }
    if( traj_ctr==1 && step_ctr==4 ){ traj1_step4_time = $5; }
    if( traj_ctr==2 && step_ctr==2 ){ traj2_step2_time = $5; }
    if( traj_ctr==2 && step_ctr==4 ){ traj2_step4_time = $5; }
}

END{
    errno = 0
    define_expectations( nx, ny, nz, nt );
    errno += test_result( "Plaquette action", p_value, p_expect, 1.0e-6 );
    print "Validation:    ", ( errno==0 ? "PASSED" : "FAILED" )
    
    #bench_time rationalle:
    #bench_time models a run with two trajectories, with eighty steps per trajectory
    #
    #setup time is a slightly misleading name it includes everything that is neither I/O, nor a step
    #    it includes things that are done once per trajectory (not just once per job)
    #    regardless of whether those things occur before or after the steps
    #
    #step time scaling:
    #for each trajectory, estimate the time to perform 80 steps
    #   step2_time: the first two steps in a trajectory may be a little slower, so measure those separately
    #   step4_time / 2 * ( 80 - 2 ):
    #       the StepTime measures two steps at once, so divide by two to get the time per step
    #       total number of steps is 80, but the step2_time has already done two steps
    #
    step_time_all = traj1_step2_time + traj1_step4_time + traj2_step2_time + traj2_step4_time
    traj1_80step_time = traj1_step2_time + traj1_step4_time / 2 * ( 80 - 2 );
    traj2_80step_time = traj2_step2_time + traj2_step4_time / 2 * ( 80 - 2 );

    setup_time = total_time - read_time - write_time - step_time_all;
    bench_time = setup_time + traj1_80step_time + traj2_80step_time;
    
    printf "Total Time:     %6.2f\n", total_time
    printf "Read  Time:     %6.2f\n",  read_time
    printf "Write Time:     %6.2f\n", write_time
    printf "Step  Time:     %6.2f\n", step_time_all
    printf "Benchmark Time: %6.2f\n", bench_time
}

