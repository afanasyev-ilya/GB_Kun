list(APPEND FORMATS CSR CSR_SEG SELL_C)

set(SOURCES vxm.cpp ${GB_KUN_SRC}/src/gb_kun.h)
foreach(FORMAT IN LISTS FORMATS)
    add_executable(vxm_SpVSpM_${FORMAT} ${SOURCES})
    target_compile_definitions(vxm_SpVSpM_${FORMAT} PUBLIC FORMAT=${FORMAT} VEC_SPARSITY=0.01)

    add_executable(vxm_VSpM_${FORMAT} ${SOURCES})
    target_compile_definitions(vxm_VSpM_${FORMAT} PUBLIC FORMAT=${FORMAT} VEC_SPARSITY=0.99)
endforeach()