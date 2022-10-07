#include "gb_kun.h"
#include "pr.hpp"
#include <chrono>
#include <iomanip>

#define MASK_NULL static_cast<const lablas::Vector<double> *>(NULL)

template <typename T>
struct inv_div {
    inline T operator()(const T lhs, const T rhs) const {
        return rhs / lhs;
    }
};

int compute_pr(const std::string& inFileName, const std::string& outFileName){
    lablas::Descriptor desc;

    lablas::Matrix<float> matrix;
    matrix.init_from_mtx(inFileName);

    Index nrows;
    matrix.get_nrows(&nrows);
    lablas::Vector<float> ranks(nrows);        

    LAGraph_Graph<float> graph(matrix);   

    int itermax = 100;
    double damping = 0.85;
    double tol = 1e-9;
    int iters_taken = 0;    

    lablas::algorithm::LAGraph_page_rank_sinks(&ranks, &graph, &iters_taken);

    double centrality_sum = 0;
    lablas::reduce(&centrality_sum, NULL, lablas::PlusMonoid<double>(), &ranks, &desc);
    std::cout << centrality_sum << '\n';
    lablas::apply(&ranks, MASK_NULL, lablas::second<double>(), inv_div<double>(), centrality_sum, &ranks, &desc);

    auto Val = ranks.get_vector()->getDense()->get_vals();
    auto nvals = ranks.get_vector()->getDense()->get_size();

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

    return compute_pr(argv[1], argv[2]);
}