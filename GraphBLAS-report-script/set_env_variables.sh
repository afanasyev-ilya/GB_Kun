export OMP_NUM_THREADS=96
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PATH=../build/:$PATH # for all GB_Kun binaries
export LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH # for TBB
