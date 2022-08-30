#include "LAGraph.h"
#include <stdlib.h>

#ifndef DELTA
#define DELTA 2
#endif

#define FREE_WORKSPACE                                                         \
  {                                                                            \
    LAGraph_Delete(&G, NULL);                                                  \
    GrB_free(&path_length);                                                    \
    GrB_free(&A);                                                              \
  }

#define min(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b; })

  GrB_Index select_ntv(LAGraph_Graph G, int seed, GrB_Index _range)
{
    GrB_Index max_val = _range;
    GrB_Index nrows = 0;
    GrB_Index ncols = 0;
    GrB_Vector degree = G->out_degree;
    GrB_Matrix_nrows(&nrows, G->A);
    GrB_Matrix_ncols(&ncols, G->A);
    int idx_degree = 0;
    if(_range == -1) // not provided
    {
        max_val = min(ncols, nrows);
    }
    else
    {
        max_val = min(_range, min(ncols, nrows));
    }
    GrB_Index vertex = 0;
    srand(seed);
    do {
        vertex = rand() %  max_val;
        GrB_Vector_extractElement(&idx_degree, degree, vertex);
    } while(idx_degree == 0);
    return vertex;
}

int main(int argc, char **argv) {
  char msg[LAGRAPH_MSG_LEN];
  LAGraph_Init(msg);

  LAGraph_Graph G = NULL;
  GrB_Matrix A = NULL;
  GrB_Matrix A2 = NULL;
  GrB_Vector path_length = NULL;

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
  LAGraph_Tic(tic, msg);
  // according to LAGr realization we need typecasting // kinda bad
  GrB_Index n;
  GrB_Matrix_nrows(&n, A);
  GrB_Matrix_new(&A2, GrB_FP32, n, n);
  GrB_apply(A2, NULL, NULL, GrB_IDENTITY_FP32, A, NULL);
  if (A2 != NULL) {
    GrB_free(&A);
    A = A2;
    A2 = NULL;
    GrB_wait(A, GrB_MATERIALIZE);
  }
  GrB_free(&A2);
  // end of typecast
  LAGraph_New(&G, &A, LAGraph_ADJACENCY_DIRECTED, msg);
  // out degree needed for generating sources
  LAGraph_Cached_OutDegree(G, msg);
  LAGraph_Cached_EMin(G, msg);

  LAGraph_Toc(&t_preproc, tic, msg);
  printf("preprocess time: %g sec\n", t_preproc);

  // Do sssp
  GrB_Index src = 1; // source always zero?

  GrB_Scalar Delta = NULL;
  GrB_Scalar_new(&Delta, GrB_INT32);
  GrB_Scalar_setElement(Delta, DELTA);

  char filename[20];
  sprintf(filename, "algo.txt");
  f = fopen(filename, "a");

  int N = strtol(argv[2], NULL, 10);
  for (int i = 0; i < N; ++i) {
    src = select_ntv(G, i, -1);
    printf("source : %ld, ineration : %d\n",src, i);
    LAGraph_Tic(tic, NULL);
    int res = LAGr_SingleSourceShortestPath(&path_length, G, src, Delta, msg);
    LAGraph_Toc(&t_run, tic, NULL);

    printf("Run time : %g sec \n", t_run);

    if (res != 0) {
      printf("Code : %d\nMessage : %s\n", res, msg);
    }
    LAGraph_Tic(tic, NULL);

    // one day there will be saveing of path_length to `path_length.mtx`

    LAGraph_Toc(&t_save, tic, NULL);

    printf("Saving time : %g sec \n", t_save);
    printf("End-to-end time : %g sec \n", t_read + t_preproc + t_run + t_save);

    fprintf(f, "{'Read time': %g, 'Run time': %g}\n", t_read, t_run);

    GrB_Vector_clear(path_length);
  }
  fclose(f);
  f = NULL;

  // clean workspace
  // remove("centrality.mtx");
  FREE_WORKSPACE;

  LAGraph_Finalize(msg);

  return (GrB_SUCCESS);
}
