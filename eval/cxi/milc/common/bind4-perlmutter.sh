#!/bin/bash
# Steven Gottlieb, September 2, 2021.  Based on Summit script with this history:
# Evan Weinberg, evansweinberg@gmail.com
# Binding script for 6 GPUs per node. Based on a script given to me by Kate, which I believe was based on something from Steve, which may have been based on something originally by Kate...

lrank=$(($SLURM_LOCALID % 4))

export MPICH_OFI_NIC_POLICY GPU
APP=$*

case ${lrank} in
 [0])
 #numactl --physcpubind=0-15,64-79 --membind=0 $APP
 numactl --physcpubind=0-15 --membind=0 $APP
 ;;
 
 [1])
 #numactl --physcpubind=16-31,80-95 --membind=1 $APP
 numactl --physcpubind=16-31 --membind=1 $APP
 ;;
 
 [2])
 #numactl --physcpubind=32-47,96-111 --membind=2 $APP
 numactl --physcpubind=32-47 --membind=2 $APP
 ;;
 
 [3])
 numactl --physcpubind=48-63,112-127 --membind=3 $APP
 ;;
esac
