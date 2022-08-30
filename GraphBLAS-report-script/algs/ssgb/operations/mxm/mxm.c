#include "GraphBLAS.h"
#include "LAGraph.h"
#include <stdlib.h>

#define FREE_WORKSPACE                                                         \
  {                                                                            \
    GrB_free(&A);                                                              \
    GrB_free(&B);                                                              \
    GrB_free(&C);                                                              \
    GrB_free(&M);                                                              \
  }

#define CLEAR_ALL                                                             \
  {                                                                           \
    GrB_Matrix_clear(A);                                                      \
    GrB_Matrix_clear(B);                                                      \
    GrB_Matrix_clear(C);                                                      \
  }

void do_test(
  const char * name,
  double * t_run,
  const double * t_read,
  GrB_Matrix C,
  const GrB_Matrix Mask,
  const GrB_BinaryOp accum,
  const GrB_Semiring semiring,
  const GrB_Matrix A,
  const GrB_Matrix B,
  const GrB_Descriptor desc
){
  double tic[2];
  FILE * f = fopen("algo.txt", "a");

  LAGraph_Tic(tic, NULL);
  int res = GrB_mxm(C, Mask, accum, semiring, A, B, desc);
  LAGraph_Toc(t_run, tic, NULL);
  printf("Run time : %g sec \n", *t_run);

  if (res != 0) {
    printf("Code : %d\n", res);
  }

  fprintf(f, "{'Operation': '%s', 'Read time': %g, 'Run time': %g}\n",name, *t_read, *t_run);

  GrB_Matrix_clear(C);
  fclose(f);
  f = NULL;
}

int main(int argc, char **argv) {

  // LaGraph and GraphBLAS initialization
  char msg[LAGRAPH_MSG_LEN];
  LAGraph_Init(msg);

  GrB_Matrix M = NULL;
  GrB_Matrix A = NULL;
  GrB_Matrix B = NULL;
  GrB_Matrix C = NULL;

  double t_read = 0;
  double t_preproc = 0;
  double t_run = 0;
  double t_save = 0;

  printf("# input:  %s\n", argv[1]);

  // Read graph
  double tic[2];
  LAGraph_Tic(tic, msg);

  FILE *f = fopen(argv[1], "r");
  if (f == NULL) {
    printf("Matrix file not found: [%s]\n", argv[1]);
    exit(1);
  }
  LAGraph_MMRead(&M, f, msg);
  fclose(f);
  f = NULL;
  LAGraph_Toc(&t_read, tic, msg);
  printf("read time: %g sec\n", t_read);

  uint64_t nr = 0;
  uint64_t nc = 0;
  GrB_Matrix_nrows(&nr, M);
  GrB_Matrix_ncols(&nc, M);

  int N = strtol(argv[2], NULL, 10);

for (int i = 0; i < N; ++i) {

  GrB_Matrix_dup(&A, M);
  GrB_Matrix_dup(&B, M);
  do_test(
    "GrB_mxm(M1, NULL, NULL, {min, second}, M1, M2, NULL)",
    &t_run, &t_read,
    A, NULL, NULL, GrB_MIN_SECOND_SEMIRING_FP64, A, B, GrB_NULL
  );
  CLEAR_ALL;

  GrB_Matrix_dup(&A, M);
  GrB_Matrix_dup(&B, M);
  do_test(
    "GrB_mxm(M2, M1, NULL, {+, *}, M1, M1, GrB_DESC_T1)",
    &t_run, &t_read,
    B, A, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, A, A, GrB_DESC_T1
  );
  CLEAR_ALL;

  GrB_Matrix_dup(&A, M);
  GrB_Matrix_dup(&B, M);
  do_test(
    "GrB_mxm(M2, M1, NULL, {+, *}, M1, M1, NULL)",
    &t_run, &t_read,
    B, A, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, A, A, GrB_NULL
  );
  CLEAR_ALL;

  GrB_Matrix_dup(&A, M);
  GrB_Matrix_dup(&B, M);
  do_test(
    "GrB_mxm(M2, M1, NULL, {+, first}, M1, M1, GrB_DESC_RST1)",
    &t_run, &t_read,
    B, A, NULL, LAGraph_plus_one_fp64, A, A, GrB_DESC_RST1
  );
  CLEAR_ALL;

  GrB_Matrix_dup(&A, M);
  GrB_Matrix_dup(&B, M);
  do_test(
    "GrB_mxm(M2, M1, NULL, {+, first}, M1, M1, GrB_DESC_S)",
    &t_run, &t_read,
    B, A, NULL, LAGraph_plus_one_fp64, A, A, GrB_DESC_S
  );
  CLEAR_ALL;

  GrB_Matrix_dup(&A, M);
  GrB_Matrix_dup(&B, M);
  do_test(
    "GrB_mxm(M2, M1, NULL, {+, first}, M1, M1, GrB_DESC_ST1)",
    &t_run, &t_read,
    B, A, NULL, LAGraph_plus_one_fp64, A, A, GrB_DESC_ST1
  );
  CLEAR_ALL;  

  GrB_Matrix_dup(&A, M);
  GrB_Matrix_dup(&B, M);
  GrB_Matrix_new(&C, GrB_FP64, nr, nc);
  do_test(
    "GrB_mxm(M3, M1, NULL, {+, *}, M1, M2, GrB_DESC_T1)",
    &t_run, &t_read,
    C, A, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, A, B, GrB_DESC_ST1
  );
  CLEAR_ALL;  

  GrB_Matrix_dup(&A, M);
  GrB_Matrix_dup(&B, M);
  GrB_Matrix_new(&C, GrB_FP64, nr, nc);
  do_test(
    "GrB_mxm(M3, M1, NULL, {+, first}, M1, M2, GrB_DESC_S)",
    &t_run, &t_read,
    C, A, NULL, LAGraph_plus_one_fp64, A, B, GrB_DESC_S
  );
  CLEAR_ALL;  
  }
  
  // clean workspace
  FREE_WORKSPACE;

  LAGraph_Finalize(msg);

  printf("#-----------------------\n");

  return (GrB_SUCCESS);
}