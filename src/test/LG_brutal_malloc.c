//------------------------------------------------------------------------------
// LG_brutal_malloc: brutal memory debugging
//------------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// See additional acknowledgments in the LICENSE file,
// or contact permission@sei.cmu.edu for the full terms.

//------------------------------------------------------------------------------

// To enable brutal memory debugging, these four functions must be passed to
// LAGraph_Xinit.

#include "LG_internal.h"

//------------------------------------------------------------------------------
// global variables: LG_brutal and LG_nmalloc
//------------------------------------------------------------------------------

// If LG_brutal >= 0, then that value gives the # of malloc/calloc/realloc
// calls that will succeed.  Each time malloc/calloc/realloc is called, the
// LG_brutal count is decremented.  Once it reaches zero, no more memory
// allocations will occur, and LG_brutal_malloc, etc, all pretend to fail
// and return NULL.

// If LG_brutal is negative, the LG_brutal_malloc/calloc/realloc functions act
// like the regular malloc/calloc/realloc functions, with no pretend failures.

// LG_nmalloc is the count of the # of allocated blocks.  It is incremented by
// LG_brutal_malloc/calloc and by LG_brutal_realloc if a new block is allocated.
// It is decremented by LG_brutal_free.  After LAGraph_Finalize is called,
// this value should be zero.  If nonzero, a memory leak has occured.

int64_t LG_brutal = -1 ;
int64_t LG_nmalloc = 0 ;

//------------------------------------------------------------------------------
// LG_brutal_malloc
//------------------------------------------------------------------------------

LAGRAPH_PUBLIC
void *LG_brutal_malloc      // return pointer to allocated block of memory
(
    size_t size             // # of bytes to allocate
)
{
    void *p ;
    if (LG_brutal == 0)
    {
        // pretend to fail
        p = NULL ;
    }
    else
    {
        // malloc a new block
        #pragma omp critical (LG_brutal_malloc_critical)
        {
            // malloc the block of memory (of size at least 1 byte)
            p = malloc (LAGraph_MAX (1, size)) ;
            if (LG_brutal > 0)
            {
                // one step closer to pretending to fail
                LG_brutal-- ;
            }
            if (p != NULL)
            {
                // one more block of memory successfully allocated
                LG_nmalloc++ ;
            }
        }
    }
    return (p) ;
}

//------------------------------------------------------------------------------
// LG_brutal_calloc
//------------------------------------------------------------------------------

LAGRAPH_PUBLIC
void *LG_brutal_calloc      // return pointer to allocated block of memory
(
    size_t nitems,          // # of items to allocate
    size_t itemsize         // # of bytes per item
)
{
    size_t size = LAGraph_MAX (1, nitems * itemsize) ;
    void *p = LG_brutal_malloc (size) ;
    if (p != NULL)
    {
        memset (p, 0, size) ;
    }
    return (p) ;
}

//------------------------------------------------------------------------------
// LG_brutal_free
//------------------------------------------------------------------------------

LAGRAPH_PUBLIC
void LG_brutal_free
(
    void *p                 // block to free
)
{
    if (p != NULL)
    {
        #pragma omp critical (LG_brutal_malloc_critical)
        {
            // free the block
            free (p) ;
            // one less block of memory allocated
            LG_nmalloc-- ;
        }
    }
}

//------------------------------------------------------------------------------
// LG_brutal_realloc
//------------------------------------------------------------------------------

LAGRAPH_PUBLIC
void *LG_brutal_realloc     // return pointer to reallocated memory
(
    void *p,                // block to realloc
    size_t size             // new size of the block
)
{
    if (p == NULL)
    {
        // malloc a new block
        p = LG_brutal_malloc (size) ;
    }
    else
    {
        // realloc an existing block
        #pragma omp critical (LG_brutal_malloc_critical)
        {
            if (LG_brutal == 0)
            {
                // pretend to fail
                p = NULL ;
            }
            else
            {
                // realloc the block
                p = realloc (p, size) ;
                if (LG_brutal > 0)
                {
                    // one step closer to pretending to fail
                    LG_brutal-- ;
                }
            }
        }
    }
    return (p) ;
}

