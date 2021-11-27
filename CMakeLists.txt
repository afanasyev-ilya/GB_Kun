cmake_minimum_required(VERSION 3.17)
project(GB_Kun)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp -fpermissive")

include_directories(src/common)

add_executable(GB_Kun main.cpp src/common/matrix.hpp src/common/types.hpp src/helpers/utils.hpp src/common/vector.hpp src/backend/vector/vector.h src/common/descriptor.hpp src/common/types.hpp src/backend/descriptor/descriptor.h src/backend/la_backend.h src/common/operations.hpp src/backend/operations/operations.h)