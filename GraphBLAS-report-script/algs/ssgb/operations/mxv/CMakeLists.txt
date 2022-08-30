add_library(lagraph STATIC IMPORTED)
set_property(TARGET lagraph PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/third_party/liblagraph.so)

add_library(graphblas STATIC IMPORTED)
set_property(TARGET graphblas PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/third_party/libgraphblas.so)

add_executable(mxv_SpMSpV mxv_unmasked.c)
target_compile_definitions(mxv_SpMSpV PUBLIC VEC_SPARSITY=0.01)
target_link_libraries(mxv_SpMSpV lagraph graphblas)

add_executable(mxv_SpMV mxv_unmasked.c)
target_compile_definitions(mxv_SpMV PUBLIC VEC_SPARSITY=0.99)
target_link_libraries(mxv_SpMV lagraph graphblas)