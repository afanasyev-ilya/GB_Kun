cmake_minimum_required(VERSION 3.17)
project(GB_Kun)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp -fpermissive")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Ofast -fopenmp -fno-tree-vectorize -fpermissive -mcpu=native -fopenmp-simd -funroll-loops -fno-builtin -msve-vector-bits=512 -march=armv8.2-a+sve -mcpu=a64fx")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Kfast,openmp -Nfjomplib -mcmodel=large -Koptmsg=2 -D_A64FX -Kzfill")

#backend root dir can be changed easily
set(kun_backend_dir /Users/slava/CLionProjects/GB_Kun/src/backend)


#set(desc_dir ${kun_backend_dir}/descriptor/)
#set(matrix_dir ${kun_backend_dir}/matrix/)
#set(operations_dir ${kun_backend_dir}/operations/)
#set(vector_dir ${kun_backend_dir}/vector/)
#set(spmv_dir ${kun_backend_dir}/spmv/)
set(spmspv_dir ${kun_backend_dir}/spmspv/)
#
#set(interface_dir src/cpp_graphblas)
#
#add_executable(spmv spmspv.cpp ${spmspv_dir}/spmspv.h ${vector_dir}/vector.h ${matrix_dir}/matrix.h ${desc_dir}/descriptor.h ${operations_dir}/operations.h spmspv.cpp src/cpp_graphblas/dimensions.hpp src/backend/operations/operations.h src/backend/operations/assign_sparse.h src/backend/operations/scatter.h src/backend/operations/reduce.h algorithms/bfs/bfs.hpp src/backend/operations/apply_sparse.h src/backend/operations/apply_dense.h src/backend/operations/ewise_mult.h)
#target_include_directories(spmspv PRIVATE ${desc_dir} ${matrix_dir} ${operations_dir} ${vector_dir} ${spmspv_dir})

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

add_executable(spmv ${spmspv_dir}/spmspv.cpp)
