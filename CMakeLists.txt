cmake_minimum_required(VERSION 3.17)
project(GB_Kun)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp -fpermissive")


#backend root dir can be changed easily
set(kun_backend_dir src/backend)


set(desc_dir ${kun_backend_dir}/descriptor/)
set(matrix_dir ${kun_backend_dir}/matrix/)
set(operations_dir ${kun_backend_dir}/operations/)
set(vector_dir ${kun_backend_dir}/vector/)
set(spmv_dir ${kun_backend_dir}/spmv/)

set(interface_dir src/cpp_graphblas)

add_executable(bfs bfs.cpp ${spmv_dir}/spmv.h ${vector_dir}/vector.h ${matrix_dir}/matrix.h ${desc_dir}/descriptor.h ${operations_dir}/operations.h)
target_include_directories(bfs PRIVATE ${desc_dir} ${matrix_dir} ${operations_dir} ${vector_dir} ${spmv_dir})


add_executable(spmv spmv.cpp ${spmv_dir}/spmv.h ${vector_dir}/vector.h ${matrix_dir}/matrix.h ${desc_dir}/descriptor.h ${operations_dir}/operations.h spmv.cpp)
target_include_directories(spmv PRIVATE ${desc_dir} ${matrix_dir} ${operations_dir} ${vector_dir} ${spmv_dir})