list(APPEND
    METHODS
    LAGraph_TriangleCount_Burkhardt
    LAGraph_TriangleCount_Cohen
    LAGraph_TriangleCount_Sandia
    LAGraph_TriangleCount_Sandia2
    LAGraph_TriangleCount_SandiaDot
    LAGraph_TriangleCount_SandiaDot2
    )

if(DEFINED FULL)
    foreach(METHOD IN LISTS METHODS)
        add_executable(tc_${METHOD} run_LAGr_TriangleCount.c)
        target_compile_definitions(tc_${METHOD} PUBLIC METHOD=${METHOD})
        target_link_libraries(tc_${METHOD} lagraph graphblas)
    endforeach()
endif(DEFINED FULL)

add_executable(tc_default run_LAGr_TriangleCount.c)
target_link_libraries(tc_default lagraph graphblas)