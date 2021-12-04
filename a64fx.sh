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

g++ --version
which g++

cmake -D CMAKE_C_COMPILER=/home/z44377r/ARM/gcc_latest/bin/gcc -D CMAKE_CXX_COMPILER=/home/z44377r/ARM/gcc_latest/bin/g++ .
make clean
make

# FCCpx -Kfast,openmp -Nfjomplib -mcmodel=large -Koptmsg=2 -D_A64FX -Kzfill saxpy.cpp -o saxpy.bin

export OMP_NUM_THREADS=48
export OMP_PROC_BIND=close
export OMP_PLACES=cores

#./saxpy.bin

./spmv -s 20 -e 32 -type RU -format CSR_SEG
#echo "######################################################"
#./spmv -s 20 -e 32 -type RU -format CSR_SEG
#echo "######################################################"
#./spmv -s 20 -e 32 -type RU -format SIGMA
