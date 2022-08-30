#include "gb_kun.h"
#include "pr.hpp"
#include "pr_traditional.hpp"
#include <chrono>

int main(int argc, char **argv) {
  lablas::Descriptor desc;

  lablas::Matrix<float> matrix;
   matrix.init_from_mtx(argv[1]);

  auto start = std::chrono::steady_clock::now();
  // matrix.set_preferred_matrix_format(FORMAT);
  matrix.init_from_mtx(argv[1]);

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> dt = end - start;
  auto read_t = dt.count();

  std::cout << "read time: " << read_t << endl;

  if (!matrix.is_symmetric()) {
    matrix.to_symmetric();
  }

  Index nrows;
  matrix.get_nrows(&nrows);
  lablas::Vector<float> ranks(nrows);

  int max_iter = 100;
  int iters_taken = 0;

  ofstream myfile("algo.txt");

  int N = strtol(argv[2], NULL, 10);
  for (int i = 0; i < N; ++i) {
    start = std::chrono::steady_clock::now();
    lablas::algorithm::seq_page_rank(&ranks, &matrix, &iters_taken);
    end = std::chrono::steady_clock::now();
    dt = end - start;
    auto run_t = dt.count();

    std::cout << "run time: " << run_t << endl;

    std::string output = "{'Read time': " + std::to_string(read_t) +
                         ", 'Run time': " + std::to_string(run_t) + "}\n";

    if (myfile.is_open()) {
      myfile << output;
    }
    GrB_Vector_clear(&ranks);
  }

  myfile.close();

  return 0;
}