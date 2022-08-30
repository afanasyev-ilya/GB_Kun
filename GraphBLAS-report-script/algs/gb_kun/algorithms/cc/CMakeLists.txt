set(
    SOURCES
    ${GB_KUN_SRC}/src/gb_kun.h
    ${GB_KUN_SRC}/algorithms/cc/cc.hpp
    ${GB_KUN_SRC}/algorithms/cc/cc_traditional.hpp
)
add_executable(cc cc.cpp ${SOURCES})
target_include_directories(cc PUBLIC ${GB_KUN_SRC}/algorithms/cc)

#add_executable(cc_bfs_based cc_bfs_based.cpp ${SOURCES})
#target_include_directories(cc_bfs_based PUBLIC ${GB_KUN_SRC}/algorithms/cc)