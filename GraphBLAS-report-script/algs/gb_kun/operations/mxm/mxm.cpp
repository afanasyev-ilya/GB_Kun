#include "gb_kun.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#define MASK_NULL static_cast<const lablas::Matrix<double> *>(NULL)

lablas::Descriptor GrB_DESC_T1({{GrB_INP1, GrB_TRAN}, {GrB_MXMMODE, GrB_IKJ}});
lablas::Descriptor GrB_DESC_RST1({{GrB_INP1, GrB_TRAN}, {GrB_OUTPUT, GrB_REPLACE}, {GrB_MASK, GrB_STRUCTURE}, {GrB_MXMMODE, GrB_IKJ}});
lablas::Descriptor GrB_DESC_ST1({{GrB_INP1, GrB_TRAN}, {GrB_MASK, GrB_STRUCTURE}, {GrB_MXMMODE, GrB_IKJ}});

template <typename c, typename m, typename a, typename b,
        typename BinaryOpT, typename SemiringT>
void do_test(
  std::string name,
  const int& read_t,
  lablas::Matrix<c>* C,
  const lablas::Matrix<m>* mask,
  BinaryOpT accum,
  SemiringT op,
  const lablas::Matrix<a>* A,
  const lablas::Matrix<b>* B,
  lablas::Descriptor* desc
){
  ofstream myfile("algo.txt", std::ios_base::app);

  auto start = std::chrono::steady_clock::now();
  lablas::mxm(C, mask, accum, op, A, B, desc);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> dt = end - start;
  double run_t = dt.count();

  std::cout << "run time: " << run_t << endl;

  std::string output = "{'Operation': " + name + ", " +
            "'Read time': " + std::to_string(read_t) +
            ", 'Run time': " + std::to_string(run_t) + "}\n";

  if (myfile.is_open()) {
    myfile << output;
  }    

  myfile.close();
}

int main(int argc, char **argv) {
  lablas::Matrix<double> A;
  lablas::Matrix<double> B;
  lablas::Matrix<double> C;
  lablas::Matrix<double> M;

  std::string output = "";

  auto start = std::chrono::steady_clock::now();
  //A.set_preferred_matrix_format(FORMAT);
  M.init_from_mtx(argv[1]);

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> dt = end - start;
  auto read_t = dt.count();

  std::cout << "read time: " << read_t << endl;

  int N = strtol(argv[2], NULL, 10);
  for (int i = 0; i < N; ++i) {

  A = M;
  B = M;
  do_test(
    "GrB_mxm(M1, NULL, NULL, {min, second}, M1, M2, NULL)",
    read_t,
    &A, MASK_NULL, NULL, lablas::MinimumSelectSecondSemiring<double>(), &A, &B, &lablas::GrB_DESC_IKJ
  );

  A = M;
  B = M;
  do_test(
    "GrB_mxm(M2, M1, NULL, {+, *}, M1, M1, GrB_DESC_T1)",
    read_t,
    &B, &A, NULL, lablas::PlusMultipliesSemiring<double>(), &A, &A, &GrB_DESC_T1
  );  

  }

#undef MASK_NULL
  return 0;
}
