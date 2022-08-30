#include "LAGraph.h"

#define FREE_WORKSPACE                                                         \
  {                                                                            \
    LAGraph_Delete(&G, NULL);                                                  \
    GrB_free(&level);                                                          \
    GrB_free(&parent);                                                         \
    GrB_free(&A);                                                              \
  }

int main(int argc, char **argv) {

  // LaGraph and GraphBLAS initialization
  char msg[LAGRAPH_MSG_LEN];
  LAGraph_Init(msg);

  LAGraph_Graph G = NULL;
  GrB_Matrix A = NULL;
  GrB_Vector level = NULL;
  GrB_Vector parent = NULL;

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
  LAGraph_Cached_IsSymmetricStructure(G, msg);
  LAGraph_Cached_OutDegree(G, msg);
  LAGraph_Cached_InDegree(G, msg);

  LAGraph_Toc(&t_preproc, tic, msg);
  printf("preprocess time: %g sec\n", t_preproc);

  // Do bfs

  GrB_Index src = 0;

  char filename[20];
  sprintf(filename, "algo.txt");
  f = fopen(filename, "a");

  int N = strtol(argv[2], NULL, 10);
  for (int i = 0; i < N; ++i) {
    src = i;
    LAGraph_Tic(tic, NULL);
    // default source is 0

    int res = LAGr_BreadthFirstSearch(&level, &parent, G, src, msg);
    LAGraph_Toc(&t_run, tic, NULL);

    printf("Run time : %g sec \n", t_run);

    if (res != 0) {
      printf("Code : %d\nMessage : %s\n", res, msg);
    }

    LAGraph_Tic(tic, NULL);

    // f = fopen("parent.mtx", "w");
    // LAGraph_Vector_Print(parent, LAGraph_COMPLETE_VERBOSE, f, msg);
    // fclose(f);

    // f = fopen("level.mtx", "w");
    // LAGraph_Vector_Print(level, LAGraph_COMPLETE_VERBOSE, f, msg);
    // fclose(f);

    LAGraph_Toc(&t_save, tic, NULL);

    printf("Saving time : %g sec \n", t_save);
    printf("End-to-end time : %g sec \n", t_read + t_preproc + t_run + t_save);

    // fprintf(f,
    //         "{'Read time': %g, 'Preprocessing time': %g, 'Run time': %g,
    //         'Saving " "time': %g}\n", t_read, t_preproc, t_run, t_save);
    fprintf(f, "{'Read time': %g, 'Run time': %g}\n", t_read, t_run);

    GrB_Vector_clear(parent);
    GrB_Vector_clear(level);
  }
  fclose(f);
  f = NULL;

  // clean workspace
  // remove("parent.mtx");
  // remove("level.mtx");
  FREE_WORKSPACE;

  LAGraph_Finalize(msg);

  return (GrB_SUCCESS);
}