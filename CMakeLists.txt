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

message(Working in ${PROJECT_SOURCE_DIR})
list(LENGTH header_dir_list header_dir_list_count)
message(STATUS "[INFO] Found ${header_dir_list_count} header directories.")
STRING(REPLACE ";" " " header_dir_list "${header_dir_list}")
STRING(REPLACE "${PROJECT_SOURCE_DIR}/" "" header_dir_list "${header_dir_list}")
message(STATUS "[INFO] header dir list: ${header_dir_list}")

list(LENGTH headers_list headers_list_count)
message(STATUS "[INFO] Found ${headers_list_count} headers.")
STRING(REPLACE ";" " " headers_list "${headers_list}")
STRING(REPLACE "${PROJECT_SOURCE_DIR}/" "" headers_list "${headers_list}")
message(STATUS "[INFO] header files list: ${headers_list}")
#

file(MAKE_DIRECTORY scripts)

add_executable(bfs bfs.cpp)
target_include_directories(bfs PRIVATE ${header_dir_list})

add_executable(spmv spmv.cpp)
target_include_directories(spmv PRIVATE ${header_dir_list})

add_executable(locality locality_test.cpp)
target_include_directories(locality PRIVATE ${header_dir_list})

add_executable(gen gen.cpp)
target_include_directories(gen PRIVATE ${header_dir_list})

add_executable(sparse_test sparse_test.cpp)
target_include_directories(sparse_test PRIVATE ${header_dir_list})

add_executable(saxpy saxpy.cpp)
target_include_directories(saxpy PRIVATE ${header_dir_list})
