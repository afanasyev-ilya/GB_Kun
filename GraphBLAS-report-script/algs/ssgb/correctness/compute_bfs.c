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

#define FREE_WORKSPACE                                                     \
  {                                                                            \
    LAGraph_Delete(&G, NULL);                                                  \
    GrB_free(&level);                                                          \
    GrB_free(&parent);                                                         \
    GrB_free(&A);                                                              \
  }

int _compute_bfs(const char * inFileName, const char * outFileName, GrB_Index src){
    LAGraph_Init(NULL);

    double t_end_to_end = 0;
    double tic[2];

    LAGraph_Graph G = NULL;
    GrB_Matrix A = NULL;
    GrB_Vector level = NULL;
    GrB_Vector parent = NULL;

    TRY(LAGraph_Tic(tic, NULL), FREE_WORKSPACE);

    FILE *f = fopen(inFileName, "r");
    TRY(LAGraph_MMRead(&A, f, NULL), FREE_WORKSPACE);   
    fclose(f); 

    TRY(LAGraph_New(&G, &A, LAGraph_ADJACENCY_DIRECTED, NULL), FREE_WORKSPACE);
    TRY(LAGraph_Cached_IsSymmetricStructure(G, NULL), FREE_WORKSPACE);
    TRY(LAGraph_Cached_OutDegree(G, NULL), FREE_WORKSPACE);
    TRY(LAGraph_Cached_AT(G, NULL), FREE_WORKSPACE);

    TRY(LAGr_BreadthFirstSearch(&level, &parent, G, src, NULL), FREE_WORKSPACE);

    GrB_Index nvals;
    TRY(GrB_Vector_nvals(&nvals, level), FREE_WORKSPACE);
    GrB_Index *Idx = (GrB_Index *)malloc(nvals*sizeof(GrB_Index));
    int64_t *Val = (int64_t *)malloc(nvals*sizeof(int64_t));
    TRY(GrB_Vector_extractTuples_INT64(Idx, Val, &nvals, level), FREE_WORKSPACE);

    f = fopen(outFileName, "w");
    for(int i = 0; i < nvals; ++i){
        fprintf(f, "%ld %ld\n", Idx[i]+1, Val[i]+1);
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

int compute_bfs(const char * inFileName, const char * outFileName){
    return _compute_bfs(inFileName, outFileName, 0);
}

int main(int argc, char **argv) {
    if (argc <= 2) {
        printf("Usage: ./compute_algorithm input.mtx output.txt\n");
        exit(1);
    }
    if (argc == 4){
        int i = strtol(argv[3], NULL, 10);
        return _compute_bfs(argv[1], argv[2], i-1);
    }
    return compute_bfs(argv[1], argv[2]);
}