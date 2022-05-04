#pragma once

#include <map>

#include "../matrix/matrix.h"
#include "../vector/vector.h"
#include "../descriptor/descriptor.h"
#include "../la_backend.h"
#include "generic_operations.h"
#include "indexed_operations.h"
#include "../../cpp_graphblas/types.hpp"

namespace lablas{
namespace backend{

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename U, typename T, typename BinaryOpT, typename SelectOpT>
static LA_Info select(DenseVector<W> *w,
                      const DenseVector<M> *mask,
                      BinaryOpT accum,
                      SelectOpT op,
                      const DenseVector<U> *u,
                      const T val,
                      Descriptor *desc)
{
          W* w_vals    = w->get_vals();
    const U* u_vals    = u->get_vals();
    const M* mask_vals = mask->get_vals();

    // Getting mask properties
    Desc_value mask_field, mask_output;
    desc->get(GrB_MASK, &mask_field);
    desc->get(GrB_OUTPUT, &mask_output);

    #pragma omp parallel for
    for (Index i = 0; i < w->get_size(); ++i)
    {   
        if (mask_vals[i] ^ (mask_field == GrB_COMP)) 
        {
            w_vals[i] = accum(w_vals[i], op(u_vals[i], i, 0, val));
        }
        else
        {
            if (mask_output == GrB_REPLACE)
            {
                w_vals[i] = 0;
            }
        }
    }

    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename U, typename T, typename BinaryOpT, typename SelectOpT>
static LA_Info select(DenseVector<W> *w,
                      const DenseVector<M> *mask,
                      BinaryOpT accum,
                      SelectOpT op,
                      const SparseVector<U> *u,
                      const T val,
                      Descriptor *desc)
{
          W* w_vals    = w->get_vals();
    const U* u_vals    = u->get_vals();
    const M* mask_vals = mask->get_vals();

    // Getting mask properties
    Desc_value mask_field, mask_output;
    desc->get(GrB_MASK, &mask_field);
    desc->get(GrB_OUTPUT, &mask_output);

    // Building sparse U vector mapping <index | value> for O(log(n)) indexation instead of O(n)
    const Index* u_indicies = u->get_ids();
    std::map<Index, U> u_sparse_tree;
    for (Index i = 0; i < u->get_nvals(); ++i)
    {
        u_sparse_tree[u_indicies[i]] = u_vals[i];
    }

    #pragma omp parallel for
    for (Index i = 0; i < w->get_size(); ++i)
    {
        if (mask_vals[i] ^ (mask_field == GrB_COMP))
        {
            // Searching for index in U vector
            auto match = u_sparse_tree.find(i);
            if (match != u_sparse_tree.end())
            {
                w_vals[i] = accum(w_vals[i], op(match->second, i, 0, val));
            }
            else
            {
                w_vals[i] = accum(w_vals[i], op(0, i, 0, val));
            }
        }
        else
        {
            if (mask_output == GrB_REPLACE)
            {
                w_vals[i] = 0;
            }
        }
    }

    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename U, typename T, typename BinaryOpT, typename SelectOpT>
static LA_Info select(DenseVector<W> *w,
                      const SparseVector<M> *mask,
                      BinaryOpT accum,
                      SelectOpT op,
                      const DenseVector<U> *u,
                      const T val,
                      Descriptor *desc)
{
          W* w_vals    = w->get_vals();
    const U* u_vals    = u->get_vals();
    const M* mask_vals = mask->get_vals();

    // Getting mask properties
    Desc_value mask_field, mask_output;
    desc->get(GrB_MASK, &mask_field);
    desc->get(GrB_OUTPUT, &mask_output);

    const Index* mask_indicies = mask->get_ids();

    W* temp;
    if (mask_output == GrB_REPLACE)
    {
        temp = new W[w->get_size()];
        std::memset(temp, 0, w->get_size() * sizeof(W));
    }

    #pragma omp parallel for
    for (Index i = 0; i < mask->get_nvals(); ++i)
    {
        Index idx = mask_indicies[i];
        if (mask_vals[i] ^ (mask_field == GrB_COMP))
        {
            if (mask_output == GrB_REPLACE)
            {
                temp[idx] = accum(w_vals[idx], op(u_vals[idx], idx, 0, val));
            }
            else
            {
                w_vals[idx] = accum(w_vals[idx], op(u_vals[idx], idx, 0, val));
            }
        }
    }

    if (mask_output == GrB_REPLACE)
    {
        std::memcpy(w_vals, temp, w->get_size() * sizeof(W));
        delete[] temp;
    }

    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename U, typename T, typename BinaryOpT, typename SelectOpT>
static LA_Info select(DenseVector<W> *w,
                      const SparseVector<M> *mask,
                      BinaryOpT accum,
                      SelectOpT op,
                      const SparseVector<U> *u,
                      const T val,
                      Descriptor *desc)
{
          W* w_vals    = w->get_vals();
    const U* u_vals    = u->get_vals();
    const M* mask_vals = mask->get_vals();

    // Getting mask properties
    Desc_value mask_field, mask_output;
    desc->get(GrB_MASK, &mask_field);
    desc->get(GrB_OUTPUT, &mask_output);

    // Building sparse U vector mapping <index | value> for O(log(n)) indexation instead of O(n)
    const Index* u_indicies = u->get_ids();
    std::map<Index, U> u_sparse_tree;
    for (Index i = 0; i < u->get_nvals(); ++i)
    {
        u_sparse_tree[u_indicies[i]] = u_vals[i];
    }

    // Building sparse mask vector mapping <index | value> for O(log(n)) indexation instead of O(n)
    const Index* mask_indicies = mask->get_ids();
    std::map<Index, M> mask_sparse_tree;
    for (Index i = 0; i < mask->get_nvals(); ++i)
    {
        mask_sparse_tree[mask_indicies[i]] = mask_vals[i];
    }

    #pragma omp parallel for
    for (Index i = 0; i < w->get_size(); ++i)
    {
        // Seaching for index in mask
        auto mask_match = mask_sparse_tree.find(i);
        if (mask_match != mask_sparse_tree.end())
        {
            if (mask_match->second ^ (mask_field == GrB_COMP))
            {
                // Searching for index in U
                auto u_match = u_sparse_tree.find(i);
                if (u_match != u_sparse_tree.end())
                {
                    w_vals[i] = accum(w_vals[i], op(u_match->second, i, 0, val));
                }
                else
                {
                    w_vals[i] = accum(w_vals[i], op(0, i, 0, val));
                }
            }
            else
            {
                if (mask_output == GrB_REPLACE)
                {
                    w_vals[i] = 0;
                }
            }
        }
        else
        {
            if (mask_field == GrB_COMP)
            {
                // Searching for index in U
                auto u_match = u_sparse_tree.find(i);
                if (u_match != u_sparse_tree.end())
                {
                    w_vals[i] = accum(w_vals[i], op(u_match->second, i, 0, val));
                }
                else
                {
                    w_vals[i] = accum(w_vals[i], op(0, i, 0, val));
                }
            }
            else
            {
                if (mask_output == GrB_REPLACE)
                {
                    w_vals[i] = 0;
                }
            }
        }
    }

    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename U, typename T, typename BinaryOpT, typename SelectOpT>
static LA_Info select(DenseVector<W> *w,
                      BinaryOpT accum,
                      SelectOpT op,
                      const DenseVector<U> *u,
                      const T val)
{
          W* w_vals = w->get_vals();
    const U* u_vals = u->get_vals();

    #pragma omp parallel for
    for (Index i = 0; i < w->get_size(); ++i)
    {
        w_vals[i] = accum(w_vals[i], op(u_vals[i], i, 0, val));
    }

    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename U, typename T, typename BinaryOpT, typename SelectOpT>
static LA_Info select(DenseVector<W> *w,
                      BinaryOpT accum,
                      SelectOpT op,
                      const SparseVector<U> *u,
                      const T val)
{
          W* w_vals = w->get_vals();
    const U* u_vals = u->get_vals();

    // Building sparse U vector mapping <index | value> for O(log(n)) indexation instead of O(n)
    const Index* u_indicies = u->get_ids();
    std::map<Index, U> u_sparse_tree;
    for (Index i = 0; i < u->get_nvals(); ++i) 
    {
        u_sparse_tree[u_indicies[i]] = u_vals[i];
    }

    #pragma omp parallel for
    for (Index i = 0; i < w->get_size(); ++i)
    {
        // Searching for index in U
        auto match = u_sparse_tree.find(i);
        if (match != u_sparse_tree.end())
        {
            w_vals[i] = accum(w_vals[i], op(match->second, i, 0, val));
        }
        else
        {
            w_vals[i] = accum(w_vals[i], op(0, i, 0, val));
        }
    }    

    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*!
 * Backend select operation wrapper
 * Chooses the necessary variant of the select function depending on the types of input vectors
 * Inout vector W is always converted to Dense due to the unpredictability of the result of the operator 'op' on zero values
 * Mask and u vectors types can vary
 */
template <typename W, typename M, typename U, typename T, typename BinaryOpT, typename SelectOpT>
LA_Info select(Vector<W> *w,
               const Vector<M> *mask,
               BinaryOpT accum,
               SelectOpT op,
               const Vector<U> *u,
               const T val,
               Descriptor *desc)
{
    if (mask != NULL)
    {
        if (mask->is_dense())
        {
            if (u->is_dense())
                return select(w->getDense(), mask->getDense(), accum, op, u->getDense(), val, desc);
            else
                return select(w->getDense(), mask->getDense(), accum, op, u->getSparse(), val, desc);
        }
        else
        {   
            if (u->is_dense())
                return select(w->getDense(), mask->getSparse(), accum, op, u->getDense(), val, desc);
            else
                return select(w->getDense(), mask->getSparse(), accum, op, u->getSparse(), val, desc);
        }
    }
    else
    {
        if (u->is_dense())
            return select(w->getDense(), accum, op, u->getDense(), val);
        else
            return select(w->getDense(), accum, op, u->getSparse(), val);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename M, typename U, typename T, typename BinaryOpT, typename SelectOpT>
static LA_Info select(Matrix<W> *w,
                      const Matrix<M> *mask,
                      BinaryOpT accum,
                      SelectOpT op,
                      const Matrix<U> *u,
                      const T val,
                      Descriptor *desc)
{
    VNT n = u->get_nrows();
    ENT nnz = u->get_nnz();
    VNT *row_ptr = new VNT[n + 1];
    #pragma omp parallel for
    for (VNT i = 0; i <= n; ++i) {
        row_ptr[i] = u->get_csr()->get_row_ptr()[i];
    }
    ENT *col_ids = new ENT[nnz];
    #pragma omp parallel for
    for (VNT i = 0; i < nnz; ++i) {
        col_ids[i] = u->get_csr()->get_col_ids()[i];
    }
    W* w_vals = new W[nnz];
    #pragma omp parallel for
    for (VNT i = 0; i < nnz; ++i) {
        w_vals[i] = 0;
    }
    const U* u_vals = u->get_csr()->get_vals();

    #pragma omp parallel for
    for (VNT i = 0; i < u->get_nrows(); ++i)
    {
        for (ENT cur_row_id = u->get_csr()->get_row_ptr()[i]; cur_row_id < u->get_csr()->get_row_ptr()[i + 1]; ++cur_row_id) {
            ENT j = u->get_csr()->get_col_ids()[cur_row_id];
            w_vals[cur_row_id] = accum(w_vals[cur_row_id], op(u_vals[cur_row_id], i, j, val));
        }
    }
    SpMSpM_alloc(w);
    w->build_from_csr_arrays(row_ptr, col_ids, w_vals, n, nnz);

    return GrB_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}
