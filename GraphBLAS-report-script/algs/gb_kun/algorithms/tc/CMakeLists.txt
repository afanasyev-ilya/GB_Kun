list(APPEND
    METHODS
    LAGraph_TriangleCount_Burkhardt
    LAGraph_TriangleCount_Cohen
    LAGraph_TriangleCount_Sandia
    LAGraph_TriangleCount_Sandia2
    LAGraph_TriangleCount_SandiaDot
    LAGraph_TriangleCount_SandiaDot2
)

set(
    SOURCES
    ${GB_KUN_SRC}/src/gb_kun.h
    ${GB_KUN_SRC}/algorithms/tc/tc.hpp
)

if(DEFINED FULL)
    foreach(METHOD IN LISTS METHODS)
        add_executable(tc_${METHOD} tc.cpp ${SOURCES})
        target_compile_definitions(tc_${METHOD} PUBLIC METHOD=${METHOD})
        target_include_directories(tc_${METHOD} PUBLIC ${GB_KUN_SRC}/algorithms/tc)
    endforeach()
endif(DEFINED FULL)

add_executable(tc_default tc.cpp)
target_include_directories(tc_default PUBLIC ${GB_KUN_SRC}/algorithms/tc)