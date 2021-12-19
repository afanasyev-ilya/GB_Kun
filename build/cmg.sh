#!/bin/bash

if [ -e ../vars_global.bash ]; then
    echo ../vars_global.bash exists
    source ../vars_global.bash
fi
if [ -e ./vars.bash ]; then
    echo ./vars.bash exists
    source ./vars.bash
fi

export PATH=/home/z44377r/arm_gcc/bin/:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/z44377r/arm_gcc/lib/:/home/z44377r/arm_gcc/lib64/

export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
#./a.out

export PATH=/home/z44377r/ARM/gcc_latest/bin:$PATH
export INCLUDE=/home/z44377r/ARM/gcc_latest/include:$INCLUDE
export LD_LIBRARY_PATH=/home/z44377r/ARM/gcc_latest/lib64:$LD_LIBRARY_PATH


export OMP_NUM_THREADS=48
export OMP_PROC_BIND=close
export OMP_PLACES=cores

g++ -O3 -std=c++14 -fopenmp -D CMG_NUM=1 -fpermissive saxpy.cpp
./a.out

g++ -O3 -std=c++14 -fopenmp -D CMG_NUM=2 -fpermissive saxpy.cpp
./a.out

g++ -O3 -std=c++14 -fopenmp -D CMG_NUM=4 -fpermissive saxpy.cpp
./a.out

g++ -O3 -std=c++14 -fopenmp -D CMG_NUM=8 -fpermissive saxpy.cpp
./a.out

g++ -O3 -std=c++14 -fopenmp -D CMG_NUM=16 -fpermissive saxpy.cpp
./a.out

g++ -O3 -std=c++14 -fopenmp -D CMG_NUM=32 -fpermissive saxpy.cpp
./a.out

g++ -O3 -std=c++14 -fopenmp -D CMG_NUM=64 -fpermissive saxpy.cpp
./a.out

#./spmv -s $1 -e 32 -type $2 -format $3

#./spmv -s 21 -e 32 -type RU -format CSR
#./spmv -s 21 -e 32 -type RMAT -format CSR
#./spmv -graph RW ./lj.mtx -format CSR

#./spmv -s 20 -e 32 -type RU -format CSR
#./spmv -graph RW ./lj.mtx -format CSR
#./spmv -graph RW ./lj.mtx -format CSR_SEG
#./spmv -graph RW ./lj.mtx -format SIGMA

#./spmv -s 20 -e 32 -type RU -format CSR_SEG

#./spmv -s 21 -e 27 -type RU -format CSR_SEG