cmake_minimum_required(VERSION 3.17)
project(GB_Kun)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp -fpermissive")

add_executable(GB_Kun main.cpp)