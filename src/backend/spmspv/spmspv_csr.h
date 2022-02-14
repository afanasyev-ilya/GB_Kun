#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
void spmspv_unmasked(const MatrixCSR<A> *_matrix,
                     const DenseVector<X> *_x,
                     DenseVector<Y> *_y,
                     BinaryOpTAccum _accum,
                     SemiringT op,
                     Descriptor *_desc,
                     Workspace *_workspace)
{
    const X *x_vals = _x->get_vals();
    Y *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    VNT vec_size = _x->get_size();

    #pragma omp parallel for
    for (VNT row = 0; row < vec_size; row++)
    {
        VNT ind = row;
        X x_val = x_vals[row];
        ENT row_start   = _matrix->row_ptr[ind];
        ENT row_end     = _matrix->row_ptr[ind + 1];

        for (ENT j = row_start; j < row_end; j++)
        {
            VNT dest_ind = _matrix->col_ids[j];
            A dest_val = _matrix->vals[j];

            #pragma omp atomic
            y_vals[dest_ind] += mul_op(x_val, dest_val);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template <typename A, typename X, typename Y, typename M, typename SemiringT, typename BinaryOpTAccum>
void SpMSpV(const Matrix<A> *_matrix,
            const DenseVector <X> *_x,
            DenseVector <Y> *_y,
            Descriptor *_desc,
            BinaryOpTAccum _accum,
            SemiringT _op,
            const Vector <M> *_mask)
{
    if(_mask == NULL) // all active case
    {
        spmspv_unmasked(_matrix->get_csr(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
    }
    else
    {
        throw "Masked SpMSpV not supported yet";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
