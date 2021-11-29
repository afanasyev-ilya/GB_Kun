#!/bin/bash

args=("$@")
number_of_arguments=$#

program_name=${args[0]}

for (( c=1; c<=${number_of_arguments}; c++ ))
do
   program_args+=(${args[c]})
done

export OMP_NUM_THREADS=128
export OMP_PROC_BIND=close

echo common launch of application ${program_name}

${program_name} ${program_args[@]}