#include "LAGraph.h"
#include "LAGraphX.h"

#ifndef SANITIZE
#define SANITIZE true
#endif

#define FREE_WORKSPACE                                                         \
  {                                                                            \
    GrB_free(&parents);                                                        \
    GrB_free(&A);                                                              \
  }

int main(int argc, char **argv) {

  // LaGraph and GraphBLAS initialization

  char msg[LAGRAPH_MSG_LEN];
  LAGraph_Init(msg);

  GrB_Matrix A = NULL;
  GrB_Vector parents = NULL;

  double t_read = 0;
  double t_preproc = 0;
  double t_run = 0;
  double t_save = 0;

  if (argc <= 1) {
    printf("Usage: demo input.mtx\n");
    exit(1);
  }

  printf("input:  %s\n", argv[1]);

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
  LAGraph_Toc(&t_read, tic, msg);
  printf("read time: %g sec\n", t_read);

  // preprocess graph
  printf("preprocess time: %d sec\n", 0);

  // Do cc_lacc
  char filename[20];
  sprintf(filename, "algo.txt");
  f = fopen(filename, "a");

  int N = strtol(argv[2], NULL, 10);
  for (int i = 0; i < N; ++i) {

    LAGraph_Tic(tic, NULL);
    int res = LAGraph_cc_lacc(&parents, A, SANITIZE, msg);
    LAGraph_Toc(&t_run, tic, NULL);
    printf("Run time : %g sec \n", t_run);

    if (res != 0) {
      printf("Code : %d\nMessage : %s\n", res, msg);
    }

    LAGraph_Tic(tic, NULL);
    // f = fopen("ans.mtx", "w");
    // LAGraph_Vector_Print(parents, LAGraph_COMPLETE_VERBOSE, f, msg);
    // fclose(f);
    LAGraph_Toc(&t_save, tic, NULL);

    printf("Saving time : %g sec \n", t_save);
    printf("End-to-end time : %g sec \n", t_read + t_preproc + t_run + t_save);

    fprintf(f, "{'Read time': %g, 'Run time': %g}\n", t_read, t_run);

    GrB_Vector_clear(parents);
  }
  fclose(f);
  f = NULL;

  // clean workspace
  // remove("ans.mtx");
  FREE_WORKSPACE;

  LAGraph_Finalize(msg);

  return (GrB_SUCCESS);
}