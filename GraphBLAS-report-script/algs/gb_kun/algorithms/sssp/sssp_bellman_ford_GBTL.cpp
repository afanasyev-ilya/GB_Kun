#include "gb_kun.h"

#include "sssp.hpp"
#include "sssp_blast.hpp"
#include "sssp_traditional.hpp"

int main(int argc, char **argv) {
  lablas::Descriptor desc;

  lablas::Matrix<float> matrix;

  auto start = std::chrono::steady_clock::now();
  // matrix.set_preferred_matrix_format(FORMAT);
  matrix.init_from_mtx(argv[1]);
  if (!matrix.is_symmetric()) {
    matrix.to_symmetric();
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> dt = end - start;
  auto read_t = dt.count();

  std::cout << "read time: " << read_t << endl;

  GrB_Index size;
  matrix.get_nrows(&size);
  LAGraph_Graph<float> graph(matrix);

  lablas::Vector<float> distances(size);

  Index source_vertex = 1;

  ofstream myfile("algo.txt");

  int N = strtol(argv[2], NULL, 10);
  for (int i = 0; i < N; ++i) {
    start = std::chrono::steady_clock::now();
    lablas::algorithm::sssp_bellman_ford_GBTL(&distances, &matrix,
                                              source_vertex);
    end = std::chrono::steady_clock::now();
    dt = end - start;
    auto run_t = dt.count();

    std::cout << "run time: " << run_t << endl;

    std::string output = "{'Read time': " + std::to_string(read_t) +
                         ", 'Run time': " + std::to_string(run_t) + "}\n";

    if (myfile.is_open()) {
      myfile << output;
    }
    GrB_Vector_clear(&distances);
  }

  myfile.close();

  return 0;
}