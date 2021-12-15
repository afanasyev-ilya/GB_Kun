cmake_minimum_required(VERSION 3.11)
project(GB_Kun)
include(ProcessorCount)
ProcessorCount(N)

message(STATUS "Number of cores: ${N}")
if(N EQUAL 48 OR N EQUAL 64 OR N EQUAL 96 OR N EQUAL 128)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14 -fno-trapping-math -fopenmp-simd -fopenmp  -ffreestanding -ffast-math -march=armv8.2-a -mtune=tsv110 -fpermissive")
else ()
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp -fpermissive")
endif()
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14 -fno-trapping-math -fopenmp-simd -fopenmp  -ffreestanding -ffast-math -march=armv8.2-a -mtune=tsv110")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Ofast -fopenmp -fno-tree-vectorize -fpermissive -mcpu=native -fopenmp-simd -funroll-loops -fno-builtin -msve-vector-bits=512 -march=armv8.2-a+sve -mcpu=a64fx")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Kfast,openmp -Nfjomplib -mcmodel=large -Koptmsg=2 -D_A64FX -Kzfill")

#backend root dir can be changed easily
set(kun_backend_dir src/backend)


set(desc_dir ${kun_backend_dir}/descriptor/)
set(matrix_dir ${kun_backend_dir}/matrix/)
set(operations_dir ${kun_backend_dir}/operations/)
set(vector_dir ${kun_backend_dir}/vector/)
set(spmv_dir ${kun_backend_dir}/spmv/)

set(interface_dir src/cpp_graphblas)

# scan all dirs recuresivly

MACRO(HEADER_DIRECTORIES return_list)
    FILE(GLOB_RECURSE new_list *.h*)
    SET(dir_list "")
    FOREACH(file_path ${new_list})
        GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
        SET(dir_list ${dir_list} ${dir_path})
    ENDFOREACH()
    LIST(REMOVE_DUPLICATES dir_list)
    SET(${return_list} ${dir_list})
ENDMACRO()

MACRO(HEADER_FILES return_list)
    FILE(GLOB_RECURSE new_list *.h*)
    SET(dir_list "")
    FOREACH(file_path ${new_list})
        SET(dir_list ${dir_list} ${file_path})
    ENDFOREACH()
    LIST(REMOVE_DUPLICATES dir_list)
    SET(${return_list} ${dir_list})
ENDMACRO()

HEADER_DIRECTORIES(header_dir_list)
HEADER_FILES(headers_list)

list(LENGTH header_dir_list header_dir_list_count)
message(STATUS "[INFO] Found ${header_dir_list_count} header directories.")
message(STATUS "[INFO] header dir list: ${header_dir_list}")

list(LENGTH headers_list headers_list_count)
message(STATUS "[INFO] Found ${headers_list_count} headers.")
message(STATUS "[INFO] header siles list: ${headers_list}")
#

file(MAKE_DIRECTORY scripts)

add_executable(bfs bfs.cpp ${spmv_dir}/spmv.h ${vector_dir}/vector.h ${matrix_dir}/matrix.h ${desc_dir}/descriptor.h ${operations_dir}/operations.h src/cpp_graphblas/dimensions.hpp src/backend/operations/operations.h src/backend/operations/assign_sparse.h src/backend/operations/assign_dense.h src/backend/operations/scatter.h src/backend/operations/reduce.h algorithms/bfs/bfs.hpp src/backend/operations/apply_sparse.h src/backend/operations/apply_dense.h src/backend/operations/ewise_mult.h)
target_include_directories(bfs PRIVATE ${desc_dir} ${matrix_dir} ${operations_dir} ${vector_dir} ${spmv_dir})

add_executable(spmv spmv.cpp ${headers_list})
target_include_directories(spmv PRIVATE ${header_dir_list})

add_executable(gen gen.cpp ${vector_dir}/vector.h ${matrix_dir}/matrix.h ${desc_dir}/descriptor.h ${operations_dir}/operations.h src/cpp_graphblas/dimensions.hpp src/backend/operations/operations.h src/backend/operations/assign_sparse.h src/backend/operations/scatter.h src/backend/operations/reduce.h algorithms/bfs/bfs.hpp src/backend/operations/apply_sparse.h src/backend/operations/apply_dense.h src/backend/operations/ewise_mult.h)
target_include_directories(gen PRIVATE ${desc_dir} ${matrix_dir} ${operations_dir} ${vector_dir})