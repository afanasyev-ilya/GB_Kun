add_library(lagraph STATIC IMPORTED)
set_property(TARGET lagraph PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/third_party/liblagraph.so)

add_library(graphblas STATIC IMPORTED)
set_property(TARGET graphblas PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/third_party/libgraphblas.so)

add_executable(vxm_SpVSpM vxm_unmasked.c)
target_compile_definitions(vxm_SpVSpM PUBLIC VEC_SPARSITY=0.01)
target_link_libraries(vxm_SpVSpM lagraph graphblas)

add_executable(vxm_VSpM vxm_unmasked.c)
target_compile_definitions(vxm_VSpM PUBLIC VEC_SPARSITY=0.99)
target_link_libraries(vxm_VSpM lagraph graphblas)