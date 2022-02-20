#include "src/gb_kun.h"

#include "algorithms/cc/cc.hpp"

int main(int argc, char** argv)
{
    std::vector<Index> row_indices;
    std::vector<Index> col_indices;
    std::vector<int> values;
    Index nrows, ncols, nvals;

    Parser parser;
    parser.parse_args(argc, argv);
    VNT scale = parser.get_scale();
    VNT avg_deg = parser.get_avg_degree();


//    // Descriptor desc
//    graphblas::Descriptor desc;
//    CHECK(desc.loadArgs(vm));
//    if (transpose)
//        CHECK(desc.toggle(graphblas::GrB_INP1));

    // Matrix A
    lablas::Matrix<int> A;
    A.set_preferred_matrix_format(parser.get_storage_format());
    init_matrix(A,parser);

    nrows = A.nrows();
    ncols = A.ncols();
    nvals = A.get_nvals(&nvals);

    bool debug = true;
    if (debug) {
        A.print_graphviz("mtx");
    }

    // Vector v
    lablas::Vector<int> v(nrows);

    lablas::Descriptor desc;

    for (int i = 0; i < 1; i++) {
        lablas::algorithm::cc(&v, &A, 0, &desc);
    }

    if (debug) {
        A.print_graphviz("mtx_answer");
    }

    return 0;
}