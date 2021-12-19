#!/bin/bash

if [ -e ../vars_global.bash ]; then
    echo ../vars_global.bash exists
    source ../vars_global.bash
fi
if [ -e ./vars.bash ]; then
    echo ./vars.bash exists
    source ./vars.bash
fi

#cmake -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ .
#make clean
#make

export OMP_NUM_THREADS=48
export OMP_PROC_BIND=close
export OMP_PLACES=cores

#export OMP_NUM_THREADS=12,4
#export OMP_PLACES="{0:1}:48:1"
#export OMP_PROC_BIND=spread,close
#export OMP_NESTED=true
#export OMP_MAX_ACTIVE_LEVELS=2
#export OMP_DYNAMIC=true

#./spmv -s $1 -e 32 -type $2 -format $3

#./spmv -s 19 -e 32 -type RU -format CSR

./spmv -graph RW $1 -format $2
#./spmv -graph $1 -s $2 -e 16 -format $3

#./spmv -s 20 -e 32 -type RU -format CSR
#./spmv -graph RW ./lj.mtx -format CSR
#./spmv -graph RW ./lj.mtx -format CSR_SEG
#./spmv -graph RW ./lj.mtx -format SIGMA

#./spmv -s 20 -e 32 -type RU -format CSR_SEG

#./spmv -s 21 -e 27 -type RU -format CSR_SEG