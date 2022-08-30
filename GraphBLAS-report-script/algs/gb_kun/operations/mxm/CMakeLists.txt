list(APPEND FORMATS CSR CSR_SEG SELL_C)

set(SOURCES mxm.cpp ${GB_KUN_SRC}/src/gb_kun.h)
foreach(FORMAT IN LISTS FORMATS)
    add_executable(mxm_${FORMAT} ${SOURCES})
    target_compile_definitions(mxm_${FORMAT} PUBLIC FORMAT=${FORMAT})
endforeach()