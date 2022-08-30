#include "LAGraph.h"
#include "LAGraphX.h"
#include <stdlib.h>

#define FREE_WORKSPACE                                                         \
  {                                                                            \
    LAGraph_Delete(&G, NULL);                                                  \
    GrB_free(&A);                                                              \
    for (int kk = 3; kk <= kmax; kk = kk + 1) {                                \
      GrB_free(&(Cset[kk]));                                                   \
    }                                                                          \
    LAGraph_Free((void **)&Cset, NULL);                                        \
  }

int main(int argc, char **argv) {
  char msg[LAGRAPH_MSG_LEN];
  LAGraph_Init(msg);

  LAGraph_Graph G = NULL;
  GrB_Matrix A = NULL;

  GrB_Index n = 0;
  int64_t kmax = 0;
  GrB_Matrix *Cset = NULL;
  int64_t *ntris = NULL;
  int64_t *nedges = NULL;
  int64_t *nstepss = NULL;

  double t_read = 0;
  double t_preproc = 0;
  double t_run = 0;
  double t_save = 0;

  if (argc <= 1) {
    printf("Usage: demo input.mtx\n");
    exit(1);
  }

  printf("input:  %s\n", argv[1]);

  // Read matrix
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
  LAGraph_New(&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg);
  LAGraph_Cached_NSelfEdges(G, msg);
  if (G->nself_edges != 0) {
    LAGraph_DeleteSelfEdges(G, msg);
    printf("Diagonal elements were detected and deleted\n");
  }

  // alloc memory

  GrB_Matrix_nrows(&n, A);
  int64_t n4 = (n > 4) ? n : 4;
  LAGraph_Malloc((void **)&Cset, n4, sizeof(GrB_Matrix), msg);
  LAGraph_Malloc((void **)&ntris, n4, sizeof(int64_t), msg);
  LAGraph_Malloc((void **)&nedges, n4, sizeof(int64_t), msg);
  LAGraph_Malloc((void **)&nstepss, n4, sizeof(int64_t), msg);

  LAGraph_Toc(&t_preproc, tic, msg);
  printf("preprocess time: %g sec\n", t_preproc);

  // Do all k-rtuss
  LAGraph_Tic(tic, NULL);
  int res = LAGraph_AllKTruss(Cset, &kmax, ntris, nedges, nstepss, G, msg);
  LAGraph_Delete(&G, NULL);
  LAGraph_Toc(&t_run, tic, NULL);

  printf("Run time : %g sec \n", t_run);

  if (res != 0) {
    printf("Code : %d Message : %s\n", res, msg);
  }

  LAGraph_Tic(tic, NULL);
  // f = fopen("ans.mtx", "w");
  // LAGraph_MMWrite(C, f, NULL, msg) ;
  // fclose(f);
  // f = NULL;
  LAGraph_Toc(&t_save, tic, NULL);

  printf("Saving time : %g sec \n", t_save);

  printf("End-to-end time : %g sec \n", t_read + t_preproc + t_run + t_save);

  // clean workspace of tmp files
  // remove("ans.mtx");

  printf("--------------------------------------------\n");

  printf("%ld", ntris[0]);

  // free memory
  FREE_WORKSPACE;

  LAGraph_Finalize(msg);

  return (GrB_SUCCESS);
}
