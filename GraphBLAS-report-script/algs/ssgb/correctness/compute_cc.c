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
    GrB_free(&components);                                                     \
    GrB_free(&A);                                                              \
  }

int compute_cc(const char * inFileName, const char * outFileName){
    LAGraph_Init(NULL);

    double t_end_to_end = 0;
    double tic[2];

    LAGraph_Graph G = NULL;
    GrB_Matrix A = NULL;
    GrB_Vector components = NULL;

    TRY(LAGraph_Tic(tic, NULL), FREE_WORKSPACE);

    FILE *f = fopen(inFileName, "r");
    TRY(LAGraph_MMRead(&A, f, NULL), FREE_WORKSPACE);   
    fclose(f); 

    TRY(LAGraph_New(&G, &A, LAGraph_ADJACENCY_DIRECTED, NULL), FREE_WORKSPACE);
    //TRY(LAGraph_Cached_IsSymmetricStructure(G, NULL), FREE_WORKSPACE);

    TRY(LAGraph_Cached_AT(G, NULL), FREE_WORKSPACE);
    bool sym ;
    LAGraph_Matrix_IsEqual (&sym, G->A, G->AT, NULL);
    if (!sym){
        GrB_eWiseAdd (G->A, NULL, NULL, GrB_PLUS_INT64, G->A, G->AT, NULL);
    }
    GrB_Matrix_free (&(G->AT));
    G->kind = LAGraph_ADJACENCY_UNDIRECTED;
    TRY(LAGraph_Cached_OutDegree(G, NULL), FREE_WORKSPACE);
    G->is_symmetric_structure = LAGraph_TRUE;

    TRY(LAGr_ConnectedComponents(&components, G, NULL), FREE_WORKSPACE);

    GrB_Index nvals;
    TRY(GrB_Vector_nvals(&nvals, components), FREE_WORKSPACE);
    GrB_Index *Idx = (GrB_Index *)malloc(nvals*sizeof(GrB_Index));
    long *Val = (long *)malloc(nvals*sizeof(long));
    TRY(GrB_Vector_extractTuples_INT64(Idx, Val, &nvals, components), FREE_WORKSPACE);

    f = fopen(outFileName, "w");
    for(int i = 0; i < nvals; ++i){
        fprintf(f, "%ld %ld\n", Idx[i]+1, Val[i]);
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

    return compute_cc(argv[1], argv[2]);
}