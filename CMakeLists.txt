cmake_minimum_required(VERSION 3.11)
project(GB_Kun)
include(CheckIncludeFileCXX)
CHECK_INCLUDE_FILE_CXX("tbb/info.h" TBB_EXISTS)
include(ProcessorCount)
ProcessorCount(N)

find_package(GTest)
if(GTest_FOUND)
    include_directories(${GTEST_INCLUDE_DIRS})
endif()
if (GTest_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D __USE_GTEST__")
endif()

message("${CMAKE_CXX_COMPILER_ID}")
message("${CMAKE_CXX_COMPILER}")
message("${CMAKE_C_COMPILER}")

if(CMAKE_C_COMPILER MATCHES "ncc")
    message("COMPILING FOR NEC SX-Aurora TSUBASA") # this is needed to test Sell-C and evaluate vectorization efficiency on Kunpeng
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D __USE_NEC_SX_AURORA__ -std=c++14 -fopenmp -O3 -fopenmp -mparallel -mretain-list-vector")
else ()
    message(STATUS "Number of cores: ${N}")
    if(N EQUAL 48 OR N EQUAL 64 OR N EQUAL 96 OR N EQUAL 128)
        message("COMPILING FOR KUNPENG 920")
        set(CMAKE_CXX_STANDARD 17)
        if(TBB_EXISTS)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D __USE_KUNPENG__ -D MAX_CORES=${N} -O3 -std=c++17 -fno-trapping-math -fopenmp-simd -fopenmp -ffreestanding -ffast-math -march=armv8.2-a -fpermissive -fprefetch-loop-arrays -ltbb -D __USE_TBB__")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D __USE_KUNPENG__ -D MAX_CORES=${N} -O3 -std=c++17 -fno-trapping-math -fopenmp-simd -fopenmp -ffreestanding -ffast-math -march=armv8.2-a -fpermissive -fprefetch-loop-arrays")
        endif()
    else ()
        message("CLION compilation") # this is for local testing
        set(CMAKE_CXX_STANDARD 17)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D MAX_CORES=${N} -std=c++17 -fopenmp -fpermissive")
    endif()
endif()

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
#message(STATUS "[INFO] header dir list: ${header_dir_list}")

list(LENGTH headers_list headers_list_count)
message(STATUS "[INFO] Found ${headers_list_count} headers.")
STRING(REPLACE ";" " " headers_list "${headers_list}")
STRING(REPLACE "${PROJECT_SOURCE_DIR}/" "" headers_list "${headers_list}")
#message(STATUS "[INFO] header files list: ${headers_list}")
#

file(MAKE_DIRECTORY benchmark/integration_scripts)

add_executable(cc cc.cpp)
target_include_directories(cc PRIVATE)

add_executable(bfs bfs.cpp)
target_include_directories(bfs PRIVATE)

add_executable(spmv spmv.cpp)
target_include_directories(spmv PRIVATE)

add_executable(pr pr.cpp)
target_include_directories(pr PRIVATE)

add_executable(sssp sssp.cpp)
target_include_directories(sssp PRIVATE)

add_executable(gen gen.cpp)
target_include_directories(gen PRIVATE)

add_executable(gemm gemm.cpp)
target_include_directories(gemm PRIVATE)

add_executable(tc tc.cpp tc.cpp)
target_include_directories(tc PRIVATE)

add_executable(transpose transpose.cpp)
target_include_directories(transpose PRIVATE)

add_executable(switch_test switch_test.cpp)
target_include_directories(switch_test PRIVATE)

add_executable(vectorization_test vectorization_test.cpp)
target_include_directories(vectorization_test PRIVATE)

add_executable(unit_tests unit_tests.cpp)
target_include_directories(unit_tests PRIVATE)

add_executable(graph_modification graph_modification.cpp)
target_include_directories(graph_modification PRIVATE)

#if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#add_executable(test_matrix tests/GB_objects/test_matrix_ops.cpp)
#target_link_libraries(test_matrix ${GTEST_LIBRARIES})

#add_executable(test_primitives tests/primitives/test_primitives.cpp)
#target_link_libraries(test_primitives ${GTEST_LIBRARIES})
#endif()

#comment
