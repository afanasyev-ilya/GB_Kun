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
    GrB_free(&A);                                                              \
  }

int compute_tc(const char * inFileName, const char * outFileName){
    LAGraph_Init(NULL);

    double t_end_to_end = 0;
    double tic[2];

    LAGraph_Graph G = NULL;
    GrB_Matrix A = NULL;
    uint64_t ntriangles = -1;

    TRY(LAGraph_Tic(tic, NULL), FREE_WORKSPACE);

    FILE *f = fopen(inFileName, "r");
    TRY(LAGraph_MMRead(&A, f, NULL), FREE_WORKSPACE);   
    fclose(f); 

    TRY(LAGraph_New(&G, &A, LAGraph_ADJACENCY_DIRECTED, NULL), FREE_WORKSPACE);
    //TRY(LAGraph_Cached_IsSymmetricStructure(G, NULL), FREE_WORKSPACE);
    TRY(LAGraph_Cached_NSelfEdges(G, NULL), FREE_WORKSPACE);
    if (G->nself_edges != 0) {
        TRY(LAGraph_DeleteSelfEdges(G, NULL), FREE_WORKSPACE);
    }
    TRY(LAGraph_Cached_OutDegree(G, NULL), FREE_WORKSPACE);

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


    int presort = LAGraph_TriangleCount_AutoSort;
    TRY(LAGr_TriangleCount(&ntriangles, G, LAGraph_TriangleCount_Default, &presort, NULL), FREE_WORKSPACE);

    f = fopen(outFileName, "w");
    fprintf(f, "%lu\n", ntriangles);
    fclose(f);

    TRY(LAGraph_Toc(&t_end_to_end, tic, NULL), FREE_WORKSPACE);
    printf("End to end time: %g sec\n", t_end_to_end); 

    f = NULL;   
    FREE_WORKSPACE;
    LAGraph_Finalize(NULL);

    return (GrB_SUCCESS);
}

int main(int argc, char **argv) {
    if (argc <= 2) {
        printf("Usage: ./compute_algorithm input.mtx output.txt\n");
        exit(1);
    }

    return compute_tc(argv[1], argv[2]);
}