#!/bin/sh

args=("$@")
number_of_arguments=$#

program_name=${args[0]}

for (( c=1; c<=${number_of_arguments}; c++ ))
do
   program_args+=(${args[c]})
done

export OMP_NUM_THREADS=96
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PATH=../build/:$PATH # for all GB_Kun binaries
export LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH # for TBB

${program_name} ${program_args[@]}