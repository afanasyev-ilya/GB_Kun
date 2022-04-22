#include "src/gb_kun.h"

#include "algorithms/sssp/sssp_lagraph.hpp"

int main() {

    lablas::Vector<float>* v;
    lablas::Matrix<float> m;
    LAGraph_Graph<float> G(m);

    lablas::algorithm::LAGr_SingleSourceShortestPath(
        &v,
        &G,
        0.f,
        0.f,
        "aaa"
    );

}