add_executable(cc_default run_LAGr_ConnectedComponents.c)
target_link_libraries(cc_default lagraph lagraphx graphblas)

#add_executable(cc_lacc run_LAGraph_cc_lacc.c)
#target_compile_definitions(cc_lacc PUBLIC SANITIZE="false")
#target_link_libraries(cc_lacc lagraph lagraphx graphblas)
