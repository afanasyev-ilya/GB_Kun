/// @file spmspm_esc.h
/// @author Anton Potapov
/// @version Revision 1.2
/// @brief Parallel map-based SpMSpV with omp critical synchronizations
/// @details Implements parallel map-based SpMSpV with omp critical synchronizations
/// @date June 13, 2022

#pragma once

/// @brief Parallel map-based SpMSpV with omp critical synchronizations
///
/// This algorithm implements parallel map-based SpMSpV with omp critical synchronizations which uses the same approach
/// as sequential map-based algorithm but is parallel with omp critical synchronizations to access hash-map.
///
/// @param[in] _matrix Pointer to the input matrix
/// @param[in] _x Pointer to the input vector
/// @param[out] _y Pointer to the DenseVector object that will contain the result vector.
/// @param[in] _desc Pointer to a descriptor object
/// @param[in] _accum BinaryOp accumulator
/// @param[in] _op Semiring operation
/// @param[in] _mask SpMSpV mask pointer
/// @see SpMSpV_map_par
/// @see SpMSpV_map_seq
/// @see SpMSpV_esc
template <typename A, typename X, typename Y, typename M, typename SemiringT, typename BinaryOpTAccum>
void SpMSpV_map_par_critical(const MatrixCSR<A> *_matrix,
                    const SparseVector <X> *_x,
                    SparseVector <Y> *_y,
                    Descriptor *_desc,
                    BinaryOpTAccum _accum,
                    SemiringT _op,
                    const Vector <M> *_mask)
{
    LOG_TRACE("Running SpMSpV_map_par_critical")
    const X *x_vals = _x->get_vals(); // y is guaranteed to be sparse
    const Index *y_ids = _y->get_ids();

    Y *y_vals = _y->get_vals(); // x is guaranteed to be sparse
    const Index *x_ids = _x->get_ids();

    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    VNT x_nvals = _x->get_nvals();

    tsl::hopscotch_map<VNT, Y> map_output;

    #pragma omp parallel for
    for (VNT i = 0; i < x_nvals; i++)
    {
        VNT ind = x_ids[i];
        X x_val = x_vals[i];
        ENT row_start = _matrix->row_ptr[ind]; // this is actually col ptr for mxv operation
        ENT row_end   = _matrix->row_ptr[ind + 1];

        for (ENT j = row_start; j < row_end; j++)
        {
            VNT dest_ind = _matrix->col_ids[j]; // this is row_ids
            A mat_val = _matrix->vals[j];

            const auto mul_op_result = mul_op(mat_val, x_val);

            #pragma omp critical
            {
                if (map_output.find(dest_ind) == map_output.end())
                    map_output[dest_ind] = add_op(identity_val, mul_op_result);
                else
                    map_output[dest_ind] = add_op(map_output[dest_ind], mul_op_result);
            }
        }
    }

    if(_mask != 0) // apply mask and save results
    {
        Desc_value mask_field;
        _desc->get(GrB_MASK, &mask_field);
        if(!_mask->is_dense())
            LOG_DEBUG("warning! costly mask conversion to dense in spmspv seq_map_based");
        const M *mask_vals = _mask->getDense()->get_vals();
        _y->clear();

        if (mask_field == GrB_STR_COMP) // CMP mask
        {
            for (auto [index, val]: map_output)
            {
                if (mask_vals[index] == 0) // since CMP we keep when 0
                    _y->push_back(index, val);
            }
        }
        else
        {
            for (auto [index, val]: map_output)
            {
                if (mask_vals[index] != 0) // since non-CMP we keep when not 0
                    _y->push_back(index, val);
            }
        }
    }
    else // save results in unmasked case
    {
        _y->clear();
        for (auto [index, val]: map_output)
        {
            _y->push_back(index, val);
        }
    }
}