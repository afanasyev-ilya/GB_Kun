set(
    SOURCES
    ${GB_KUN_SRC}/src/gb_kun.h
    ${GB_KUN_SRC}/algorithms/sssp/sssp.hpp
    ${GB_KUN_SRC}/algorithms/sssp/sssp_blast.hpp
    ${GB_KUN_SRC}/algorithms/sssp/sssp_traditional.hpp
)
add_executable(sssp_bellman_ford_blast sssp_bellman_ford_blast.cpp ${SOURCES})
target_include_directories(sssp_bellman_ford_blast PUBLIC ${GB_KUN_SRC}/algorithms/sssp)
# also taking a lot of time (or looping?)
# add_executable(sssp_bellman_ford_gbkun sssp_bellman_ford_gbkun.cpp ${SOURCES})
# target_include_directories(sssp_bellman_ford_gbkun PUBLIC ${GB_KUN_SRC}/algorithms/sssp)
# GBTL taking a lot of time even on small graphs
#add_executable(sssp_bellman_ford_GBTL sssp_bellman_ford_GBTL.cpp ${SOURCES})
#target_include_directories(sssp_bellman_ford_GBTL PUBLIC ${GB_KUN_SRC}/algorithms/sssp)

#add_executable(sssp_traditional_dijkstra sssp_traditional_dijkstra.cpp ${SOURCES})
#target_include_directories(sssp_traditional_dijkstra PUBLIC ${GB_KUN_SRC}/algorithms/sssp)