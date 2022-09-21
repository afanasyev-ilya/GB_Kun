//------------------------------------------------------------------------------
// LG_CC_Boruvka.c:  connected components using GrB* methods only
//------------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// See additional acknowledgments in the LICENSE file,
// or contact permission@sei.cmu.edu for the full terms.

//------------------------------------------------------------------------------

// Code is based on Boruvka's minimum spanning forest algorithm.
// Contributed by Yongzhe Zhang (zyz915@gmail.com).
// revised by Tim Davis (davis@tamu.edu)

// This method relies solely on GrB* methods in the V2.0 C API, but it much
// slower in general than LG_CC_FastSV6, which uses GxB pack/unpack methods
// for faster access to the contents of the matrices and vectors.

#include "LG_internal.h"

//------------------------------------------------------------------------------
// Reduce_assign
//------------------------------------------------------------------------------

// w[Px[i]] = min(w[Px[i]], s[i]) for i in [0..n-1].

static GrB_Info Reduce_assign
(
    GrB_Vector w,       // input/output vector of size n
    GrB_Vector s,       // input vector of size n
    GrB_Index *Px,      // Px: array of size n
    GrB_Index *mem,     // workspace of size 3*n
    GrB_Index n
)
{
    char *msg = NULL ;
    GrB_Index *ind  = mem ;
    GrB_Index *sval = ind + n ;
    GrB_Index *wval = sval + n ;
    GrB_TRY (GrB_Vector_extractTuples (ind, wval, &n, w)) ;
    GrB_TRY (GrB_Vector_extractTuples (ind, sval, &n, s)) ;
    for (GrB_Index j = 0 ; j < n ; j++)
    {
        if (sval [j] < wval [Px [j]])
        {
            wval [Px [j]] = sval [j] ;
        }
    }
    GrB_TRY (GrB_Vector_clear (w)) ;
    GrB_TRY (GrB_Vector_build (w, ind, wval, n, GrB_PLUS_UINT64)) ;
    LAGraph_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// select_func: IndexUnaryOp for pruning entries from S
//------------------------------------------------------------------------------

// Rather than using a global variable to access the Px array, it is passed to
// the select function as a uint64_t value that contains a pointer to Px.  It
// might be better to create a user-defined type for y, but this works fine.

void my_select_func (void *z, const void *x,
                 const GrB_Index i, const GrB_Index j, const void *y)
{
    GrB_Index *Px = (*(GrB_Index **) y) ;
    (*((bool *) z)) = (Px [i] != Px [j]) ;
}

#define SAVE_STATS(call_instruction, op_name, bytes_per_flop, iterations, matrix)    \
GrB_Index my_nvals = 0;                                                                 \
GrB_Matrix_nvals(&my_nvals, matrix);                                                    \
/*printf("matrix has %ld\n edges", nvals);*/                                         \
double my_t1 = omp_get_wtime();                                                         \
call_instruction;                                                                    \
double my_t2 = omp_get_wtime();                                                         \
double my_time = (my_t2 - my_t1)*1000;                                                        \
double my_perf = my_nvals * 2.0 / ((my_t2 - my_t1)*1e9);                                         \
double my_bw = my_nvals * bytes_per_flop/((my_t2 - my_t1)*1e9);                                  \
/*printf("edges: %lf\n", nvals);*/                                                   \
printf("%s time %lf (ms)\n", op_name, (my_t2-my_t1)*1000);                             \
/*printf("%s perf %lf (GFLop/s)\n", op_name, perf);*/                                \
/*printf("%s BW %lf (GB/s)\n", op_name, bw);*/                                       \
FILE *my_f;                                                                             \
my_f = fopen("perf_stats.txt", "a");                                                    \
fprintf(my_f, "%s %lf (ms) %lf (GFLOP/s) %lf (GB/s) %ld\n", op_name, my_time, my_perf, my_bw, my_nvals);\
fclose(my_f); 

//------------------------------------------------------------------------------
// LG_CC_Boruvka
//------------------------------------------------------------------------------

#undef  LAGraph_FREE_ALL
#define LAGraph_FREE_ALL            \
{                                   \
    LAGraph_FREE_WORK ;             \
    GrB_free (&parent) ;            \
}

#undef  LAGraph_FREE_WORK
#define LAGraph_FREE_WORK           \
{                                   \
    LAGraph_Free ((void **) &I) ;   \
    LAGraph_Free ((void **) &Px) ;  \
    LAGraph_Free ((void **) &mem) ; \
    GrB_free (&gp) ;                \
    GrB_free (&mnp) ;               \
    GrB_free (&ccmn) ;              \
    GrB_free (&ramp) ;              \
    GrB_free (&mask) ;              \
    GrB_free (&select_op) ;         \
}

int LG_CC_Boruvka
(
    // output
    GrB_Vector *component,  // output: array of component identifiers
    // inputs
    LAGraph_Graph G,        // input graph, not modified
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Index n, *I = NULL, *Px = NULL, *mem = NULL ;
    GrB_Vector parent = NULL, gp = NULL, mnp = NULL, ccmn = NULL, ramp = NULL,
        mask = NULL ;
    GrB_IndexUnaryOp select_op = NULL ;
    GrB_Matrix S = NULL ;

    LG_CLEAR_MSG ;
    LG_CHECK (LAGraph_CheckGraph (G, msg), GrB_INVALID_OBJECT,
        "graph is invalid") ;
    LG_CHECK (component == NULL, GrB_NULL_POINTER, "input is NULL") ;

    if (G->kind == LAGRAPH_ADJACENCY_UNDIRECTED ||
       (G->kind == LAGRAPH_ADJACENCY_DIRECTED &&
        G->A_structure_is_symmetric == LAGRAPH_TRUE))
    {
        // A must be symmetric
        ;
    }
    else
    {
        // A must not be unsymmetric
        LG_CHECK (false, GrB_INVALID_VALUE, "input must be symmetric") ;
    }

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    // S = structure of G->A
    LAGraph_TRY (LAGraph_Structure (&S, G->A, msg)) ;

    GrB_TRY (GrB_Matrix_nrows (&n, S)) ;
    GrB_TRY (GrB_Vector_new (&parent, GrB_UINT64, n)) ; // final result
    GrB_TRY (GrB_Vector_new (&gp, GrB_UINT64, n)) ;     // grandparents
    GrB_TRY (GrB_Vector_new (&mnp, GrB_UINT64, n)) ;    // min neighbor parent
    GrB_TRY (GrB_Vector_new (&ccmn, GrB_UINT64, n)) ;   // cc's min neighbor
    GrB_TRY (GrB_Vector_new (&mask, GrB_BOOL, n)) ;     // various uses

    mem = (GrB_Index *) LAGraph_Malloc (3*n, sizeof (GrB_Index)) ;
    Px = (GrB_Index *) LAGraph_Malloc (n, sizeof (GrB_Index)) ;
    LG_CHECK (Px == NULL || mem == NULL, GrB_OUT_OF_MEMORY, "out of memory") ;

    #if !LG_SUITESPARSE
    // I is not needed for SuiteSparse and remains NULL
    I = (GrB_Index *) LAGraph_Malloc (n, sizeof (GrB_Index)) ;
    LG_CHECK (I == NULL, GrB_OUT_OF_MEMORY, "out of memory") ;
    #endif

    // parent = 0:n-1, and copy to ramp
    GrB_TRY (GrB_assign (parent, NULL, NULL, 0, GrB_ALL, n, NULL)) ;
    GrB_TRY (GrB_apply  (parent, NULL, NULL, GrB_ROWINDEX_INT64, parent, 0,
        NULL)) ;
    GrB_TRY (GrB_Vector_dup (&ramp, parent)) ;

    // Px is a non-opaque copy of the parent GrB_Vector
    GrB_TRY (GrB_Vector_extractTuples (I, Px, &n, parent)) ;

    GrB_TRY (GrB_IndexUnaryOp_new (&select_op, my_select_func, GrB_BOOL,
        /* aij: ignored */ GrB_BOOL, /* y: pointer to Px */ GrB_UINT64)) ;

    GrB_Index nvals ;
    GrB_TRY (GrB_Matrix_nvals (&nvals, S)) ;

    //--------------------------------------------------------------------------
    // find the connected components
    //--------------------------------------------------------------------------

    while (nvals > 0)
    {

        //----------------------------------------------------------------------
        // mnp[u] = u's minimum neighbor's parent for all nodes u
        //----------------------------------------------------------------------
        GrB_Index loc_nvals = 0;
        GrB_Vector_nvals (&loc_nvals, parent);
        printf("nvals = %lld\n", loc_nvals);

        // every vertex points to a root vertex at the begining
        GrB_TRY (GrB_assign (mnp, NULL, NULL, n, GrB_ALL, n, NULL)) ;
        //GrB_TRY (GrB_mxv (mnp, NULL, GrB_MIN_UINT64,
        //            GrB_MIN_SECOND_SEMIRING_UINT64, S, parent, NULL)) ;
        SAVE_STATS(GrB_TRY (GrB_mxv (mnp, NULL, GrB_MIN_UINT64,
                    GrB_MIN_SECOND_SEMIRING_UINT64, S, parent, NULL)) ;,
                   "borumvka_mxv", (sizeof(float)*2 + sizeof(size_t)), 1, (G->A));
        //----------------------------------------------------------------------
        // find the minimum neighbor
        //----------------------------------------------------------------------

        // ccmn[u] = connect component's minimum neighbor | if u is a root
        //         = n                                    | otherwise
        GrB_TRY (GrB_assign (ccmn, NULL, NULL, n, GrB_ALL, n, NULL)) ;
        GrB_TRY (Reduce_assign (ccmn, mnp, Px, mem, n)) ;

        //----------------------------------------------------------------------
        // parent[u] = ccmn[u] if ccmn[u] != n
        //----------------------------------------------------------------------

        // mask = (ccnm != n)
        GrB_TRY (GrB_apply (mask, NULL, NULL, GrB_NE_UINT64, ccmn, n, NULL)) ;
        // parent<mask> = ccmn
        GrB_TRY (GrB_assign (parent, mask, NULL, ccmn, GrB_ALL, n, NULL)) ;

        //----------------------------------------------------------------------
        // select new roots
        //----------------------------------------------------------------------

        // identify all pairs (u,v) where parent [u] == v and parent [v] == u
        // and then select the minimum of u, v as the new root;
        // if (parent [parent [i]] == i) parent [i] = min (parent [i], i)

        // compute grandparents: gp = parent (parent)
        GrB_TRY (GrB_Vector_extractTuples (I, Px, &n, parent)) ;
        GrB_TRY (GrB_extract (gp, NULL, NULL, parent, Px, n, NULL)) ;

        // mask = (gp == 0:n-1)
        GrB_TRY (GrB_eWiseMult (mask, NULL, NULL, GrB_EQ_UINT64, gp, ramp,
            NULL)) ;
        // parent<mask> = min (parent, ramp)
        GrB_TRY (GrB_assign (parent, mask, GrB_MIN_UINT64, ramp, GrB_ALL, n,
            NULL)) ;

        //----------------------------------------------------------------------
        // shortcutting: parent [i] = parent [parent [i]] until convergence
        //----------------------------------------------------------------------

        bool changing = true ;
        while (true)
        {
            // compute grandparents: gp = parent (parent)
            GrB_TRY (GrB_Vector_extractTuples (I, Px, &n, parent)) ;
            GrB_TRY (GrB_extract (gp, NULL, NULL, parent, Px, n, NULL)) ;

            // changing = or (parent != gp)
            GrB_TRY (GrB_eWiseMult (mask, NULL, NULL, GrB_NE_UINT64, parent, gp,
                NULL)) ;
            GrB_TRY (GrB_reduce (&changing, NULL, GrB_LOR_MONOID_BOOL, mask,
                NULL)) ;
            if (!changing) break ;

            // parent = gp
            GrB_TRY (GrB_assign (parent, NULL, NULL, gp, GrB_ALL, n, NULL)) ;
        }

        //----------------------------------------------------------------------
        // remove the edges inside each connected component
        //----------------------------------------------------------------------

        // This step is the costliest part of this algorithm when dealing with
        // large matrices.
        GrB_TRY (GrB_select (S, NULL, NULL, select_op, S, (uint64_t) Px, NULL));
        GrB_TRY (GrB_Matrix_nvals (&nvals, S)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    (*component) = parent ;
    LAGraph_FREE_WORK ;
    return (GrB_SUCCESS) ;
}
