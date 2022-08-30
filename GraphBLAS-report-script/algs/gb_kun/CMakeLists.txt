cmake_minimum_required(VERSION 3.11)
project(GB_Kun_algs)
include(ProcessorCount)
ProcessorCount(N)

message(STATUS "${CMAKE_CXX_COMPILER_ID}")
message(STATUS "${CMAKE_CXX_COMPILER}")
message(STATUS "${CMAKE_C_COMPILER}")

message(STATUS "Number of cores: ${N}")
if(N EQUAL 48 OR N EQUAL 96) # 48-core Kunpeng is armv8.2-a
    message("COMPILING FOR KUNPENG 920")
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D __USE_KUNPENG__ -D MAX_CORES=${N} -O3 -std=c++17 -fno-trapping-math -fopenmp-simd -fopenmp -ffreestanding -ffast-math -march=armv8.2-a -fpermissive -fprefetch-loop-arrays -ltbb")
elseif (N EQUAL 64 OR N EQUAL 128)
    message("COMPILING FOR KUNPENG 920")
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D __USE_KUNPENG__ -D MAX_CORES=${N} -O3 -std=c++17 -fno-trapping-math -fopenmp-simd -fopenmp -ffreestanding -ffast-math -march=armv8.2-a+crypto+dotprod+fp16fml -fpermissive -fprefetch-loop-arrays -ltbb")
else ()
    message("CLION compilation") # this is for local testing
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D MAX_CORES=${N} -std=c++17 -fopenmp -fpermissive")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release)
endif()

if ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
    add_compile_options(-Wall -Wextra -Wpedantic -fsanitize=address -fsanitize=leak)
    add_link_options(-fsanitize=address -fsanitize=leak)
endif()

set(GB_KUN_SRC $ENV{GB_KUN_PATH})
include_directories(${GB_KUN_SRC} ${GB_KUN_SRC}/src ${GB_KUN_SRC}/algorithms)

add_subdirectory(algorithms)
add_subdirectory(operations)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/build)