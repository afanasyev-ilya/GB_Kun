#include "gb_kun.h"
#include "tc.hpp"
#include <chrono>

int compute_tc(const std::string& inFileName, const std::string& outFileName) {
    lablas::Descriptor desc;

    lablas::Matrix<int> matrix;
    matrix.init_from_mtx(inFileName);

    if (!matrix.is_symmetric()) {
        matrix.to_symmetric();
    }    

    LAGraph_Graph<int> graph(matrix);

    lablas::algorithm::LAGraph_TriangleCount_Method tc_algorithm = lablas::algorithm::LAGraph_TriangleCount_Method::LAGraph_TriangleCount_Default;
    uint64_t ntriangles;

    lablas::algorithm::LAGr_TriangleCount(&ntriangles, &graph, tc_algorithm,
                                     lablas::algorithm::LAGraph_TriangleCount_Presort::LAGraph_TriangleCount_NoSort,
                                     &lablas::GrB_DESC_IKJ_MASKED);

    std::ofstream ofile(outFileName, std::ios_base::out);
    ofile << ntriangles << '\n';
    ofile.close();

    return 0;       
}

int main(int argc, char **argv) {
    if (argc <= 2) {
        printf("Usage: ./compute_algorithm input.mtx output.txt\n");
        exit(1);
    }

    return compute_tc(argv[1], argv[2]);
}