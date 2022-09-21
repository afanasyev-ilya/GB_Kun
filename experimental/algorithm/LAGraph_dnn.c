//------------------------------------------------------------------------------
// LAGraph_dnn: sparse deep neural network
//------------------------------------------------------------------------------

// LAGraph, (c) 2021 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// See additional acknowledgments in the LICENSE file,
// or contact permission@sei.cmu.edu for the full terms.

//------------------------------------------------------------------------------

// LAGraph_dnn: sparse deep neural network.  Contributed by Tim Davis,
// Texas A&M University.  Based on inferenceReLUvec.m by Jeremy Kepner, MIT.

// Performs ReLU inference using input feature vectors Y0.

// See http://graphchallenge.org/ for a description of the algorithm.

// On input, Y0 is the initial feature vectors, of size nfeatures-by-nneurons.
// This format uses the graph convention that A(i,j) is the edge (i,j).
// Each row of Y0 is a single feature.

// W is an array of size nlayers of sparse matrices.  Each W[layer] matrix has
// the same size: nneurons-by-nneurons.  W[layer] represents the DNN weights
// for that layer.

// The Bias[layer] matrices are diagonal, and the same size as W[layer].

// All matrices should have type GrB_FP32; the method will be very slow
// otherwise.

// On output, Y is the computed result, of the same size as Y0.

#define LAGraph_FREE_ALL    \
{                           \
    GrB_free (&Y) ;         \
}

#include "LG_internal.h"
#include "LAGraphX.h"

//****************************************************************************
GrB_Info LAGraph_dnn    // returns GrB_SUCCESS if successful
(
    // output
    GrB_Matrix *Yhandle,    // Y, created on output
    // input: not modified
    GrB_Matrix *W,      // W [0..nlayers-1], each nneurons-by-nneurons
    GrB_Matrix *Bias,   // Bias [0..nlayers-1], diagonal nneurons-by-nneurons
    int nlayers,        // # of layers
    GrB_Matrix Y0       // input features: nfeatures-by-nneurons
)
{
    GrB_Info info ;
    char *msg = NULL ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (Yhandle == NULL || W == NULL || Bias == NULL || Y0 == NULL)
    {
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // create the output matrix Y
    //--------------------------------------------------------------------------

    GrB_Matrix Y = NULL ;
    (*Yhandle) = NULL ;
    GrB_Index nfeatures, nneurons ;
    GrB_TRY (GrB_Matrix_nrows (&nfeatures, Y0)) ;
    GrB_TRY (GrB_Matrix_ncols (&nneurons,  Y0)) ;
    GrB_TRY (GrB_Matrix_new (&Y, GrB_FP32, nfeatures, nneurons)) ;

    //--------------------------------------------------------------------------
    // propagate the features through the neuron layers
    //--------------------------------------------------------------------------

    for (int layer = 0 ; layer < nlayers ; layer++)
    {
        // Y = Y * W [layer], using the conventional PLUS_TIMES semiring
        GrB_TRY (GrB_mxm (Y, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP32,
            ((layer == 0) ? Y0 : Y), W [layer], NULL)) ;

        // Y = Y * Bias [layer], using the MIN_PLUS semiring.  This computes
        // Y(i,j) += Bias [layer] (j,j) for each entry Y(i,j).  It does not
        // introduce any new entries in Y.  The MIN monoid is not actually used
        // since Bias [layer] is a diagonal matrix.  The prior version used
        // a PLUS_PLUS semiring, which also works but is not a GrB built-in.
        GrB_TRY (GrB_mxm (Y, NULL, NULL, GrB_MIN_PLUS_SEMIRING_FP32, Y,
            Bias [layer], NULL)) ;

        // delete entries from Y: keep only those entries greater than zero
        GrB_TRY (GrB_select (Y, NULL, NULL, GrB_VALUEGT_FP32, Y, (float) 0,
            NULL));

        // threshold maximum values: Y = min (Y, 32)
        GrB_TRY (GrB_apply (Y, NULL, NULL, GrB_MIN_FP32, Y, (float) 32,
            NULL)) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*Yhandle) = Y ;
    return (GrB_SUCCESS) ;
}
