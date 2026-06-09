#!/bin/bash

Usage () {
  echo "BGW_validate.sh: test output correctness for the NERSC-10 BerkeleyGW benchmark."
  echo " Usage: BGW_validate.sh <app> <size> <output_file>"
  echo " Allowed apps: [ epsilon, sigma ]"
  echo " Allowed sizes: [ small, medium, reference, target ]"
  echo " Example: BGW_validate.sh epsilon small BGW_EPSILON.out"
}

validate_args () {

  #validate CL arguments
  #defines the global variables: app, size, outfile

  #provide help if requested
  if [ $# -ge 1 ]; then
    if [ $1 == "--help" ]; then
      Usage
      exit
    fi
  fi

  #validate argc
  if [ $# -ne 3 ]; then
    echo "Three arguments are required, but $# were provided."
    Usage
    exit 1
  fi

  app=$1
  size=$2
  outfile=$3

  #validate app
  case "$app" in
  epsilon);;
  sigma);;
  *)
    echo "Error: the requested app ($1) is not supported."
    Usage
    exit 1
  esac

  #validate size
  case "$size" in
  small);;
  medium);;
  reference);;
  target);;
  *)
    echo "Error: the requested size ($2) is not supported."
    Usage
    exit 1
  esac

#end validate_args()
}

define_expectations () {

  #set some defaults
  expected_1=0.0
  expected_2=0.0
  tolerance=0.0
  sigstr=''
    
  case "$1:$2" in

  epsilon:small)
    expected_1=1.777506988066533E+01
    expected_2=9.276513200265800E+00
    tolerance=1.0E-10
    ;;
  epsilon:medium)
    expected_1=1.524681719435080E+01
    expected_2=1.070898941170535E+01
    tolerance=1.0E-10
    ;;
  epsilon:reference)
    expected_1=1.416652250097137E+01
    expected_2=1.151002087971401E+01
    tolerance=1.0E-10
    ;;
  epsilon:target)
    echo "Warning: epsilon:target validation not yet implemented; validation will fail."
    expected_1=0.0
    expected_2=0.0
    tolerance=1.0E-10
    ;;


  sigma:small)
    sigstr='       1     429'
    expected_1=6.558148597 
    expected_2=6.160450464
    tolerance=0.0
    ;;
  sigma:medium)
    sigstr='       1    1021'
    expected_1=6.617391171
    expected_2=6.075106894
    tolerance=0.0
    ;;
  sigma:reference)
    sigstr='       1    1997'
    expected_1=6.597934369
    expected_2=6.043978050
    tolerance=0.0
    ;;
  sigma:target)
    echo "Warning: sigma:target validation not yet implemented; validation will fail."
    sigstr=''
    expected_1=0.0
    expected_2=0.0
    tolerance=0.0
    ;;

  esac
#end define_expectations
}

gather_measurements () {

  #defines global variables [testval_1, testval_2, IO_TIME, TOTAL_TIME ]
  #set defaults
  testval_1=0.0
  testval_2=0.0
  IO_TIME=0.0
  TOTAL_TIME=0.0
        
  gm_app=$1
  gm_outfile=$2

  #read the newly created values and timers
  #there is a lot of quotation-mark ugliness to debug so lets "set -x"
  #set -x
  case "$gm_app" in

  epsilon)
  epsstr='Head of Epsilon         ='
  test_file=$gm_outfile
  testval_1=`grep "$epsstr" $test_file | awk {'print $7'}`
  epsstr='Epsilon(2,2)            ='
  testval_2=`grep "$epsstr" $test_file | awk {'print $5'}`
  ;;

  sigma)
  # test_file=`dirname $outfile`/eqp1.dat
  test_file=`echo $outfile | sed 's/.out//g'`/eqp1.dat
  if [ -f $test_file ]; then
      testval_1=`grep "$sigstr" $test_file | awk {'print $3'}`
      testval_2=`grep "$sigstr" $test_file | awk {'print $4'}`
  else
      echo "The file $test_file is needed for validation, but could not be found"
      echo "Proceeding, but validation will fail."
  fi
  ;;
  esac

  TOTAL_TIME=`grep '\- TOTAL '     $gm_outfile | awk {'print $6'}`
  IO_TIME=`   grep '\- I/O TOTAL ' $gm_outfile | awk {'print $7'}`
  BENCH_TIME=`echo "scale=10; $TOTAL_TIME - $IO_TIME" | bc`
  
  #set +x
#end gather_measurements
}

test_result () {

  tr_testname=$1
  tr_measured=$2
  tr_expected=$3
  tr_tolerance=$4

  trm=$(E_notation_bc $tr_measured)
  trx=$(E_notation_bc $tr_expected)
  trt=$(E_notation_bc $tr_tolerance)
  
  tr_error_t="AbsError"
  tr_error=`echo "scale=10; x=($trm - $trx);     if(x<0) x=-x; x" | bc`
  tr_bool=` echo "scale=10; $tr_error <= $trt" | bc`

  #echo "trm=$trm"
  #echo "trx=$trx"
  #echo "tr_error=$tr_error"
  #echo "trt=$trt"
  #echo "tr_tolerance=$tr_tolerance"
  #echo "tr_bool=$tr_bool"

  if [ $tr_bool -eq 0 ]; then
    echo "$tr_testname"
    echo "  Measured: $tr_measured"
    echo "  Expected: $tr_expected"
    echo "  $tr_error_t: $tr_error"
    echo "  Tol:      $tr_tolerance"
    echo "  Result:   $(result_str $tr_bool)"
  fi

  return $tr_bool
}

result_str () {
  if [ $1 -eq 0 ]; then
    echo "FAILED"
  else
    echo "PASSED"
  fi
}

E_notation_bc () {
  #convert scientific notation to a format that bc will recognize
  echo $1 | sed s/e/*10^/ | sed s/E/*10^/ | sed s/+//
}

__main__ () {

  #sets variables [ app, size, outfile ]
  validate_args "$@" 

  #sets variables [ expected_1, expected_2, tolerance, <sigstr> ]
  define_expectations $app $size

  #sets variables [testval_1, testval_2, IO_TIME, TOTAL_TIME ]
  gather_measurements $app $outfile

  validation_status=1
  test_result "Test_1"  $testval_1  $expected_1 $tolerance
  if [ $? -ne 1 ]; then validation_status=0; fi

  test_result "Test_2"  $testval_2  $expected_2 $tolerance
  if [ $? -ne 1 ]; then validation_status=0; fi

  echo "Testing $app $size"
  echo " Validation:    $(result_str $validation_status)"
  echo " Total Time:     $TOTAL_TIME"
  echo " I/O Time:       $IO_TIME"
  echo " Benchmark Time: $BENCH_TIME"
  #end __main__()
}

__main__ "$@"

