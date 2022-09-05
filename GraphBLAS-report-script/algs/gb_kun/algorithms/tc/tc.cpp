#include "gb_kun.h"

#include "tc.hpp"

#ifndef METHOD
#define METHOD LAGraph_TriangleCount_Default
#endif

int main(int argc, char **argv) {
  lablas::Descriptor desc;

  lablas::Matrix<int> matrix;

  auto start = std::chrono::steady_clock::now();
  // matrix.set_preferred_matrix_format(FORMAT);
  matrix.init_from_mtx(argv[1]);

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> dt = end - start;
  auto read_t = dt.count();

  std::cout << "read time: " << read_t << endl;

  /*
  // As in LAGraph implementation the Parser is the one responsible for calling this function
  if (!matrix.is_symmetric()) {
      matrix.to_symmetric();
  }
  */

  LAGraph_Graph<int> graph(matrix);

  lablas::algorithm::LAGraph_TriangleCount_Method tc_algorithm = lablas::algorithm::LAGraph_TriangleCount_Method::METHOD;

  uint64_t ntriangles;
  ofstream myfile("algo.txt");

  int N = strtol(argv[2], NULL, 10);
  for (int i = 0; i < N; ++i) {
    start = std::chrono::steady_clock::now();
    lablas::algorithm::LAGr_TriangleCount(&ntriangles, &graph, tc_algorithm,
                                     lablas::algorithm::LAGraph_TriangleCount_Presort::LAGraph_TriangleCount_NoSort,
                                     &lablas::GrB_DESC_IKJ_MASKED);
    end = std::chrono::steady_clock::now();
    dt = end - start;
    auto run_t = dt.count();

    std::cout << "run time: " << run_t << endl;

    std::string output = "{'Read time': " + std::to_string(read_t) +
                         ", 'Run time': " + std::to_string(run_t) + "}\n";

    if (myfile.is_open()) {
      myfile << output;
    }
    ntriangles = 0;
  }

  myfile.close();

  return 0;
}