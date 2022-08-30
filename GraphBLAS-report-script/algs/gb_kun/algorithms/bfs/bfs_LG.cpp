#include "bfs.hpp"
#include "bfs_traditional.hpp"
#include "gb_kun.h"
#include <chrono>

int main(int argc, char **argv) {
  lablas::Matrix<float> matrix;

  auto start = std::chrono::steady_clock::now();
  // matrix.set_preferred_matrix_format(FORMAT);
  matrix.init_from_mtx(argv[1]);

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> dt = end - start;
  auto read_t = dt.count();

  std::cout << "read time: " << read_t << endl;

  Index nrows;
  matrix.get_nrows(&nrows);
  Index source_vertex = 0;

  lablas::Vector<float> levels(nrows);

  ofstream myfile("algo.txt");

  int N = strtol(argv[2], NULL, 10);
  for (int i = 0; i < N; ++i) {
    source_vertex = i;
    start = std::chrono::steady_clock::now();
    lablas::algorithm::bfs_traditional(&levels, &matrix, source_vertex);
    end = std::chrono::steady_clock::now();
    dt = end - start;
    auto run_t = dt.count();

    std::cout << "run time: " << run_t << endl;

    std::string output = "{'Read time': " + std::to_string(read_t) +
                         ", 'Run time': " + std::to_string(run_t) + "}\n";

    if (myfile.is_open()) {
      myfile << output;
    }
    GrB_Vector_clear(&levels);
  }

  myfile.close();

  return 0;
}