#include "LAGraph.h"
#include "LAGraphX.h"

#ifndef TCC_METHOD
#define TCC_METHOD 3
#endif

int main(int argc, char **argv) {

  // LaGraph and GraphBLAS initialization

  char msg[LAGRAPH_MSG_LEN];
  LAGraph_Init(msg);

  GrB_Matrix A = NULL;
  LAGraph_Graph G = NULL;
  GrB_Vector centrality = NULL;
  uint64_t ntri;

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

  LAGraph_New(&G, &A, LAGraph_ADJACENCY_UNDIRECTED, msg);

  // extra line deletes self edges if they exist
  LAGraph_DeleteSelfEdges(G, msg);

  double t_read;
  LAGraph_Toc(&t_read, tic, msg);
  printf("read time: %g sec\n", t_read);

  // Do bc

  double ttrial;

  LAGraph_Tic(tic, NULL);

  int res =
      LAGraph_VertexCentrality_Triangle(&centrality, &ntri, TCC_METHOD, G, msg);
  LAGraph_Toc(&ttrial, tic, NULL);

  printf("Run time : %g sec \n", ttrial);

  printf("End status : %d \n", res);
  printf("Message : %s\n", msg);

  printf("# of tris : %ld\n", ntri);

  GxB_fprint(centrality, GxB_SUMMARY, NULL);

  LAGraph_Finalize(msg);

  return (GrB_SUCCESS);
}