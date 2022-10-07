#include "gb_kun.h"
#include "sssp_blast.hpp"
#include <chrono>
#include <iomanip>

int compute_sssp(const std::string& inFileName, const std::string& outFileName, Index src = 0){
    lablas::Descriptor desc;

    lablas::Matrix<float> matrix;
    matrix.init_from_mtx(inFileName);    

    GrB_Index size;
    matrix.get_nrows(&size);
    LAGraph_Graph<float> graph(matrix);

    lablas::Vector<float> distances(size);

    lablas::algorithm::sssp_bellman_ford_blast(&distances, &matrix, src, &desc);

    auto Val = distances.get_vector()->getDense()->get_vals();
    auto nvals = distances.get_vector()->getDense()->get_size();

    std::ofstream ofile(outFileName, std::ios_base::out);
    ofile << std::fixed << std::setprecision(17);
    for (int i = 0; i < nvals; ++i){
        if(Val[i] == std::numeric_limits<float>::max()){
            ofile << i+1 << ' ' << "inf" << '\n';
        }
        else{
            ofile << i+1 << ' ' << Val[i]<< '\n';
        }
    }
    ofile.close();

    return 0;   
}

int main(int argc, char **argv) {
    if (argc <= 2) {
        printf("Usage: ./compute_algorithm input.mtx output.txt\n");
        exit(1);
    }
    if (argc == 4){
        int i = strtol(argv[3], NULL, 10);
        return compute_sssp(argv[1], argv[2], i-1);
    }
    return compute_sssp(argv[1], argv[2]);
}

