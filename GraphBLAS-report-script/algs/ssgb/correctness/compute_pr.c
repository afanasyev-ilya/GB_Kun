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

#define FREE_WORKSPACE                                                      \
  {                                                                            \
    LAGraph_Delete(&G, NULL);                                                  \
    GrB_free(&centrality);                                                     \
    GrB_free(&A);                                                              \
  }

int compute_pr(const char * inFileName, const char * outFileName){
    LAGraph_Init(NULL);

    double t_end_to_end = 0;
    double tic[2];

    LAGraph_Graph G = NULL;
    GrB_Matrix A = NULL;
    GrB_Vector centrality = NULL;

    TRY(LAGraph_Tic(tic, NULL), FREE_WORKSPACE);

    // constants from function definition
    float damping = 0.85;
    float tol = 1e-9;
    int iters = 0, itermax = 100;

    FILE *f = fopen(inFileName, "r");
    TRY(LAGraph_MMRead(&A, f, NULL), FREE_WORKSPACE);   
    fclose(f); 

    TRY(LAGraph_New(&G, &A, LAGraph_ADJACENCY_DIRECTED, NULL), FREE_WORKSPACE); 
    TRY(LAGraph_Cached_OutDegree(G, NULL), FREE_WORKSPACE);
    TRY(LAGraph_Cached_AT(G, NULL), FREE_WORKSPACE);

    TRY(LAGr_PageRank(&centrality, &iters, G, damping, tol, itermax, NULL), FREE_WORKSPACE);

    //normalize pagerank centrlities
    double centrality_sum = 0;
    TRY(GrB_reduce (&centrality_sum, NULL, GrB_PLUS_MONOID_FP64, centrality, NULL), FREE_WORKSPACE);
    GrB_Vector_apply_BinaryOp2nd_FP64(centrality, NULL, NULL, GrB_DIV_FP64, centrality, centrality_sum, GrB_DESC_R);
  
    GrB_Index nvals;
    TRY(GrB_Vector_nvals(&nvals, centrality), FREE_WORKSPACE);
    GrB_Index *Idx = (GrB_Index *)malloc(nvals*sizeof(GrB_Index));
    double *Val = (double *)malloc(nvals*sizeof(double));
    TRY(GrB_Vector_extractTuples_FP64(Idx, Val, &nvals, centrality), FREE_WORKSPACE);

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

int main(int argc, char **argv) {
    if (argc <= 2) {
        printf("Usage: ./compute_algorithm input.mtx output.txt\n");
        exit(1);
    }

    return compute_pr(argv[1], argv[2]);
}