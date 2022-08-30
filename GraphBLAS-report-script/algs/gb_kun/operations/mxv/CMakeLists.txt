list(APPEND FORMATS CSR CSR_SEG SELL_C)

set(SOURCES mxv.cpp ${GB_KUN_SRC}/src/gb_kun.h)
foreach(FORMAT IN LISTS FORMATS)
    add_executable(mxv_SpMSpV_${FORMAT} ${SOURCES})
    target_compile_definitions(mxv_SpMSpV_${FORMAT} PUBLIC FORMAT=${FORMAT} VEC_SPARSITY=0.01)

    add_executable(mxv_SpMV_${FORMAT} ${SOURCES})
    target_compile_definitions(mxv_SpMV_${FORMAT} PUBLIC FORMAT=${FORMAT} VEC_SPARSITY=0.99)
endforeach()