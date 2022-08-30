#include "gb_kun.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdlib.h>

#ifndef FORMAT
#define FORMAT CSR
#endif

#ifndef VEC_SPARSITY
#define VEC_SPARSITY 0.01
#endif

int main(int argc, char **argv) {
  srand(0); // set seed

  lablas::Matrix<float> A;

  lablas::Descriptor desc;

  auto start = std::chrono::steady_clock::now();
  A.set_preferred_matrix_format(FORMAT);
  A.init_from_mtx(argv[1]);

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> dt = end - start;
  auto read_t = dt.count();

  std::cout << "read time: " << read_t << endl;

  GrB_Index nr;
  GrB_Index nc;
  A.get_nrows(&nr);
  A.get_ncols(&nc);

  float sparcity = VEC_SPARSITY;
  lablas::Vector<float> w(nc);
  lablas::Vector<float> u(nr);
  auto nvals = (long long)(nr * sparcity);
  printf("# nnz in vector %lld\n", nvals);
  if (sparcity < 0.5) {
    std::vector<GrB_Index> idxs(nvals);
    std::vector<float> vals(nvals);
    for (int i = 0; i < nvals; ++i) {
      idxs[i] = rand() % nr;
      vals[i] = (float)(rand() % nr) / (float)nr;
    }
    std::cout << "# Should be sparce!\n";

    u.build(&idxs, &vals, nvals);
  } else {
    std::vector<float> vals(nr, 0);
    for (int i = 0; i < nr; ++i) {
      GrB_Index idx = rand() % nr;
      float val = (float)(rand() % nr) / (float)nr;

      vals[idx] = val;
    }
    // std::cout << "# Should be dance!\n";

    u.build(&vals, nr);
  }
  // u.print_storage_type();

  ofstream myfile("algo.txt");

  int N = strtol(argv[2], NULL, 10);
  for (int i = 0; i < N; ++i) {
    start = std::chrono::steady_clock::now();
#define MASK_NULL static_cast<const lablas::Vector<float> *>(NULL)
    lablas::vxm(&w, MASK_NULL, NULL, lablas::PlusMultipliesSemiring<float>(),
                &u, &A, &desc);
#undef MASK_NULL
    end = std::chrono::steady_clock::now();
    dt = end - start;
    auto run_t = dt.count();

    std::cout << "run time: " << run_t << endl;

    std::string output = "{'Read time': " + std::to_string(read_t) +
                         ", 'Run time': " + std::to_string(run_t) + "}\n";

    if (myfile.is_open()) {
      myfile << output;
    }
    GrB_Vector_clear(&w);
  }

  myfile.close();

  return 0;
}