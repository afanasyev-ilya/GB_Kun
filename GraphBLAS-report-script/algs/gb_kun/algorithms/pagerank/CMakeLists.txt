set(
    SOURCES
    ${GB_KUN_SRC}/src/gb_kun.h
    ${GB_KUN_SRC}/algorithms/pr/pr.hpp
    ${GB_KUN_SRC}/algorithms/pr/pr_traditional.hpp
)
add_executable(pagerank_lagr pagerank_lagr.cpp ${SOURCES})
target_include_directories(pagerank_lagr PUBLIC ${GB_KUN_SRC}/algorithms/pr)

#add_executable(pagerank_seq pagerank_seq.cpp ${SOURCES})
#target_include_directories(pagerank_seq PUBLIC ${GB_KUN_SRC}/algorithms/pr)