#include <GraphBLAS.h>
#include <LAGraph.h>

#define TRY(method, free)                       \
{                                               \
    int status = method ;                       \
    if (status < 0)                             \
    {                                           \
        free ;                                  \
        printf("Return code : %d\n", status);   \
        return (status) ;                       \
    }                                           \
}

#define FREE_WORKSPACE                                                         \
  {                                                                            \
    LAGraph_Delete(&G, NULL);                                                  \
    GrB_free(&path_length);                                                    \
    GrB_free(&A);                                                              \
  }

int _compute_sssp(const char * inFileName, const char * outFileName, GrB_Index src){
    LAGraph_Init(NULL);

    double t_end_to_end = 0;
    double tic[2];

    LAGraph_Graph G = NULL;
    GrB_Matrix A = NULL;
    GrB_Matrix A2 = NULL;
    GrB_Vector path_length = NULL;

    TRY(LAGraph_Tic(tic, NULL), FREE_WORKSPACE);

    FILE *f = fopen(inFileName, "r");
    TRY(LAGraph_MMRead(&A, f, NULL), FREE_WORKSPACE);   
    fclose(f); 

    //strange typecast
    GrB_Index n;
    TRY(GrB_Matrix_nrows(&n, A), FREE_WORKSPACE);
    TRY(GrB_Matrix_new(&A2, GrB_FP64, n, n), FREE_WORKSPACE);
    TRY(GrB_apply(A2, NULL, NULL, GrB_IDENTITY_FP64, A, NULL), FREE_WORKSPACE);
    if (A2 != NULL) {
      GrB_free(&A);
      A = A2;
      A2 = NULL;
      GrB_wait(A, GrB_MATERIALIZE);
    }
    GrB_free(&A2);
    //end of typecast

    TRY(LAGraph_New(&G, &A, LAGraph_ADJACENCY_DIRECTED, NULL), FREE_WORKSPACE);
    TRY(LAGraph_Cached_EMin(G, NULL), FREE_WORKSPACE);

    GrB_Scalar Delta = NULL;
    TRY(GrB_Scalar_new(&Delta, GrB_INT64), FREE_WORKSPACE);
    TRY(GrB_Scalar_setElement(Delta, 2), FREE_WORKSPACE);

    TRY(LAGr_SingleSourceShortestPath(&path_length, G, src, Delta, NULL), FREE_WORKSPACE);

    GrB_Index nvals;
    TRY(GrB_Vector_nvals(&nvals, path_length), FREE_WORKSPACE);
    GrB_Index *Idx = (GrB_Index *)malloc(nvals*sizeof(GrB_Index));
    double *Val = (double *)malloc(nvals*sizeof(double));
    TRY(GrB_Vector_extractTuples_FP64(Idx, Val, &nvals, path_length), FREE_WORKSPACE);

    f = fopen(outFileName, "w");
    for(int i = 0; i < nvals; ++i){
        fprintf(f, "%ld %.17f\n", Idx[i]+1, Val[i]);
    }
    fclose(f);

    TRY(LAGraph_Toc(&t_end_to_end, tic, NULL), FREE_WORKSPACE);
    printf("End to end time: %g sec\n", t_end_to_end);

    f = NULL;
    free((void*)Idx);
    free((void*)Val);
    FREE_WORKSPACE;
    LAGraph_Finalize(NULL);

    return (GrB_SUCCESS);
}

int compute_sssp(const char * inFileName, const char * outFileName){
  return _compute_sssp(inFileName, outFileName, 0);
}

int main(int argc, char **argv) {
    if (argc <= 2) {
        printf("Usage: ./compute_algorithm input.mtx output.txt\n");
        exit(1);
    }
    if (argc == 4){
        int i = strtol(argv[3], NULL, 10);
        return _compute_sssp(argv[1], argv[2], i-1);
    }
    return compute_sssp(argv[1], argv[2]);
}