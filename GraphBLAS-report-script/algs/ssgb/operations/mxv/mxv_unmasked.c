#include "GraphBLAS.h"
#include "LAGraph.h"
#include <stdlib.h>

#ifndef TYPE
#define TYPE GxB_PLUS_TIMES_FP64
#endif

#ifndef VEC_SPARSITY
#define VEC_SPARSITY 0.01
#endif

#define FREE_WORKSPACE                                                         \
  {                                                                            \
    GrB_free(&A);                                                              \
    GrB_free(&u);                                                              \
    GrB_free(&w);                                                              \
  }

int main(int argc, char **argv) {

  // LaGraph and GraphBLAS initialization
  char msg[LAGRAPH_MSG_LEN];
  LAGraph_Init(msg);

  GrB_Matrix A = NULL;
  GrB_Vector u = NULL;
  GrB_Vector w = NULL;

  double t_read = 0;
  double t_preproc = 0;
  double t_run = 0;
  double t_save = 0;

  srand(0); // set seed

  printf("# input:  %s\n", argv[1]);

  // Read graph
  double tic[2];
  LAGraph_Tic(tic, msg);

  FILE *f = fopen(argv[1], "r");
  if (f == NULL) {
    printf("Matrix file not found: [%s]\n", argv[1]);
    exit(1);
  }
  LAGraph_MMRead(&A, f, msg);
  fclose(f);
  f = NULL;
  LAGraph_Toc(&t_read, tic, msg);
  printf("read time: %g sec\n", t_read);

  uint64_t nr = 0;
  uint64_t nc = 0;
  GrB_Matrix_nrows(&nr, A);
  GrB_Matrix_ncols(&nc, A);

  // generate vector u
  // nnz controlled by sparcity parameter
  float sparcity = VEC_SPARSITY;
  GrB_Vector_new(&u, GrB_FP64, nc);
  GrB_Vector_new(&w, GrB_FP64, nr);
  printf("# nnz in vector %lld\n", (long long)(nc * sparcity));
  for (int i = 0; i < nc * sparcity; ++i) {
    GrB_Index pos = rand() % nc;
    double val = (double)(rand() % nc) / (double)nc;

    GrB_Vector_setElement(u, val, pos);
  }

  char filename[20];
  sprintf(filename, "algo.txt");
  f = fopen(filename, "a");

  int N = strtol(argv[2], NULL, 10);
  for (int i = 0; i < N; ++i) {
    LAGraph_Tic(tic, NULL);
    int res = GrB_mxv(w, NULL, NULL, TYPE, A, u, GrB_DESC_R);
    LAGraph_Toc(&t_run, tic, NULL);
    printf("Run time : %g sec \n", t_run);

    if (res != 0) {
      printf("Code : %d\nMessage : %s\n", res, msg);
    }

    fprintf(f, "{'Read time': %g, 'Run time': %g}\n", t_read, t_run);

    GrB_Vector_clear(w);
  }
  fclose(f);
  f = NULL;

  // clean workspace
  FREE_WORKSPACE;

  LAGraph_Finalize(msg);

  printf("#-----------------------\n");

  return (GrB_SUCCESS);
}