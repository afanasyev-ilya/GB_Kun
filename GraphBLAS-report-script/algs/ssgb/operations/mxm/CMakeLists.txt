add_library(lagraph STATIC IMPORTED)
set_property(TARGET lagraph PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/third_party/liblagraph.so)

add_library(graphblas STATIC IMPORTED)
set_property(TARGET graphblas PROPERTY IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/third_party/libgraphblas.so)

add_executable(mxm mxm.c)
target_link_libraries(mxm lagraph graphblas)
