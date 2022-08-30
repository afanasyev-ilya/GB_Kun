set(
    SOURCES
    ${GB_KUN_SRC}/src/gb_kun.h
    ${GB_KUN_SRC}/algorithms/bfs/bfs.hpp
    ${GB_KUN_SRC}/algorithms/bfs/bfs_traditional.hpp
)
add_executable(bfs_blast bfs_blast.cpp ${SOURCES})
target_include_directories(bfs_blast PUBLIC ${GB_KUN_SRC}/algorithms/bfs)

#add_executable(bfs_traditional bfs_LG.cpp ${SOURCES})
#target_include_directories(bfs_traditional PUBLIC ${GB_KUN_SRC}/algorithms/bfs)