#include "LAGraph.h"
#include "LAGraphX.h"

int main(int argc, char **argv) {
  char msg[LAGRAPH_MSG_LEN];
  LAGraph_Init(msg);

  GrB_Matrix A = NULL;
  GrB_Vector result = NULL;

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

  double t_read;
  LAGraph_Toc(&t_read, tic, msg);
  printf("read time: %g sec\n", t_read);

  // Do scc

  double ttrial;

  LAGraph_Tic(tic, NULL);
  int res = LAGraph_scc(&result, A, msg);
  LAGraph_Toc(&ttrial, tic, NULL);

  printf("Run time : %g sec \n", ttrial);

  printf("End status : %d \n", res);
  printf("Message : %s\n", msg);

  GxB_fprint(result, GxB_SUMMARY, NULL);

  LAGraph_Finalize(msg);

  return (GrB_SUCCESS);
}
