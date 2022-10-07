#include "gb_kun.h"
#include "bfs.hpp"
#include <chrono>
#include <iomanip>

int compute_bfs(const std::string& inFileName, const std::string& outFileName, Index src = 0){
    auto start = std::chrono::steady_clock::now();

    lablas::Descriptor desc;
    lablas::Matrix<signed short> matrix;

    matrix.init_from_mtx(inFileName);

    //matrix.print();

    Index nrows;
    matrix.get_nrows(&nrows);

    lablas::Vector<signed short> levels(nrows);

    lablas::algorithm::bfs_blast(&levels, &matrix, src, &desc);

    auto Idx = levels.get_vector()->getSparse()->get_ids();
    auto Val = levels.get_vector()->getSparse()->get_vals();
    auto nvals = levels.get_vector()->get_nvals();

    std::ofstream ofile(outFileName, std::ios_base::out);
    ofile << std::fixed << std::setprecision(17);
    for (int i = 0; i < nvals; ++i){
        ofile << Idx[i]+1 << ' ' << Val[i] << '\n';
    }
    ofile.close();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = end - start;
    auto t_end_to_end = dt.count();
    std::cout << "End to end time: " << t_end_to_end <<" sec\n";

    return 0;
}

int main(int argc, char **argv) {
    if (argc <= 2) {
        printf("Usage: ./compute_algorithm input.mtx output.txt\n");
        exit(1);
    }
    if (argc == 4){
        int i = strtol(argv[3], NULL, 10);
        return compute_bfs(argv[1], argv[2], i-1);
    }
    return compute_bfs(argv[1], argv[2]);
}