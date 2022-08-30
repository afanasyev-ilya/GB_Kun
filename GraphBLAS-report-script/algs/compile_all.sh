#!/bin/bash
backends=(ssgb gb_kun)

for backend in ${backends[@]}; do
    cd ${backend}
    rm -r build
    mkdir build && cd build
    cmake "$@" .. && cmake --build . --parallel $(nproc)
    cd ../../
done