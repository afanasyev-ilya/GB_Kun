#include "gb_kun.h"
#include "cc.hpp"
#include <chrono>
#include <iomanip>

int compute_cc(const std::string& inFileName, const std::string& outFileName){
    lablas::Descriptor desc;
    desc.set(GrB_MXVMODE, SPMV_GENERAL);

    lablas::Matrix<int> matrix;
    matrix.init_from_mtx(inFileName);

    if (!matrix.is_symmetric()) {
        matrix.to_symmetric();
    }

    Index nrows;
    matrix.get_nrows(&nrows);  
    lablas::Vector<int> components(nrows);

    lablas::algorithm::cc(&components, &matrix, 0, &desc);

    auto Val = components.get_vector()->getDense()->get_vals();
    auto nvals = components.get_vector()->getDense()->get_size();

    std::ofstream ofile(outFileName, std::ios_base::out);
    ofile << std::fixed << std::setprecision(17);
    for (int i = 0; i < nvals; ++i){
        ofile << i+1 << ' ' << Val[i] << '\n';
    }
    ofile.close();

    return 0;
}

int main(int argc, char **argv) {
    if (argc <= 2) {
        printf("Usage: ./compute_algorithm input.mtx output.txt\n");
        exit(1);
    }

    return compute_cc(argv[1], argv[2]);
}