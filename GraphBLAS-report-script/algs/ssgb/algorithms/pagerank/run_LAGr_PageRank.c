#include "LAGraph.h"

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

  LAGraph_Graph G = NULL;
  GrB_Matrix A = NULL;
  GrB_Vector centrality = NULL;

  // constants from function definition
  float damping = 0.85;
  float tol = 1e-4;
  int iters = 0, itermax = 100;

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

  LAGraph_New(&G, &A, LAGraph_ADJACENCY_DIRECTED, msg);
  LAGraph_Cached_OutDegree(G, msg);
  LAGraph_Cached_AT(G, msg);

  LAGraph_Toc(&t_preproc, tic, msg);
  printf("preprocess time: %g sec\n", t_preproc);

  // Do scc

  char filename[20];
  sprintf(filename, "algo.txt");
  f = fopen(filename, "a");

  int N = strtol(argv[2], NULL, 10);
  for (int i = 0; i < N; ++i) {

    LAGraph_Tic(tic, NULL);
    int res = LAGr_PageRank(&centrality, &iters, G, damping, tol, itermax, msg);
    LAGraph_Toc(&t_run, tic, NULL);

    printf("Run time : %g sec \n", t_run);

    if (res != 0) {
      printf("Code : %d\nMessage : %s\n", res, msg);
    }

    LAGraph_Tic(tic, NULL);

    // one day there will be saveing of centrality to `centrality.mtx`

    LAGraph_Toc(&t_save, tic, NULL);

    printf("Saving time : %g sec \n", t_save);
    printf("End-to-end time : %g sec \n", t_read + t_preproc + t_run + t_save);

    fprintf(f, "{'Read time': %g, 'Run time': %g}\n", t_read, t_run);

    GrB_Vector_clear(centrality);
    iters = 0;
  }
  fclose(f);
  f = NULL;

  // clean workspace
  // remove("centrality.mtx");
  FREE_WORKSPACE;

  LAGraph_Finalize(msg);

  return (GrB_SUCCESS);
}
