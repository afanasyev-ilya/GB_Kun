cmake_minimum_required(VERSION 3.11)
project(SuiteSpareGraphBlas_algs)

message(STATUS "${CMAKE_CXX_COMPILER_ID}")
message(STATUS "${CMAKE_CXX_COMPILER}")
message(STATUS "${CMAKE_C_COMPILER}")

if(NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release)
endif()

if ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
    add_compile_options(-Wall -Wextra -Wpedantic -fsanitize=address -fsanitize=leak)
    add_link_options(-fsanitize=address -fsanitize=leak)
endif()

add_subdirectory(algorithms)
add_subdirectory(operations)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/build)