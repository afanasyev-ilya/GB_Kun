#include "LAGraph.h"
#include <stdlib.h>

#define FREE_WORKSPACE                                                         \
  {                                                                            \
    LAGraph_Delete(&G, NULL);                                                  \
    GrB_free(&centrality);                                                     \
    GrB_free(&A);                                                              \
  }

int main(int argc, char **argv) {

  // LaGraph and GraphBLAS initialization
  char msg[LAGRAPH_MSG_LEN];
  LAGraph_Init(msg);

  GrB_Matrix A = NULL;
  LAGraph_Graph G = NULL;
  GrB_Vector centrality = NULL;

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
  int res_ = LAGraph_MMRead(&A, f, msg);
  if (res_ != 0) {
    printf("Code : %d\nMessage : %s\n", res_, msg);
  }
  fclose(f);
  LAGraph_Toc(&t_read, tic, msg);
  printf("read time: %g sec\n", t_read);

  // preprocess graph
  LAGraph_Tic(tic, msg);
  LAGraph_New(&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg);
  LAGraph_Cached_AT(G, msg);
  LAGraph_Toc(&t_preproc, tic, msg);
  printf("preprocess time: %g sec\n", t_preproc);
  // Do bc

    GxB_fprint(G->A, GxB_SUMMARY, NULL);

  GrB_Index n;
  GrB_Matrix_nrows (&n, G->A);
  GrB_Index* src = (GrB_Index*)malloc(n*sizeof(GrB_Index));
  for(int i = 0; i < n; ++i){
    src[i]=i;
  }

  LAGraph_Tic(tic, NULL);
  int res = LAGr_Betweenness(&centrality, G, src, n, msg);
  LAGraph_Toc(&t_run, tic, NULL);
  printf("Run time : %g sec \n", t_run);

  if (res != 0) {
    printf("Code : %d\nMessage : %s\n", res, msg);
  }

  LAGraph_Tic(tic, NULL);
  //f = fopen("ans.mtx", "w");
  //LAGraph_Vector_Print(centrality, LAGraph_COMPLETE_VERBOSE, f, msg);
  //fclose(f);
  GxB_fprint(centrality, GxB_SHORT, NULL);
  LAGraph_Toc(&t_save, tic, NULL);

  printf("Saving time : %g sec \n", t_save);
  printf("End-to-end time : %g sec \n", t_read + t_preproc + t_run + t_save);

  // clean workspace
  remove("ans.mtx");
  FREE_WORKSPACE;

  LAGraph_Finalize(msg);
  free((void*)src);
  return (GrB_SUCCESS);
}