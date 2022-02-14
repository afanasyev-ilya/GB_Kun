#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename X, typename Y, typename SemiringT, typename BinaryOpTAccum>
void spmspv_unmasked_add(const MatrixCSR<A> *_matrix,
                         const DenseVector<X> *_x,
                         DenseVector<Y> *_y,
                         BinaryOpTAccum _accum,
                         SemiringT _op,
                         Descriptor *_desc,
                         Workspace *_workspace)
{
    const X *x_vals = _x->get_vals();
    Y *y_vals = _y->get_vals();
    auto add_op = extractAdd(_op);
    auto mul_op = extractMul(_op);
    auto identity_val = _op.identity();

    #pragma omp parallel
    {
        #pragma omp for
        for (VNT row = 0; row < _y->get_size(); row++)
        {
            y_vals[row] = identity_val;
        }

        #pragma omp for
        for (VNT row = 0; row < _y->get_size(); row++)
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
        auto add_op = extractAdd(_op);
        /*!
          * /brief atomicAdd() 3+5  = 8
          *        atomicSub() 3-5  =-2
          *        atomicMin() 3,5  = 3
          *        atomicMax() 3,5  = 5
          *        atomicOr()  3||5 = 1
          *        atomicXor() 3^^5 = 0
        */
        int functor = add_op(3, 5);
        if (functor == 8)
        {
            spmspv_unmasked_add(_matrix->get_csc(), _x, _y, _accum, _op, _desc, _matrix->get_workspace());
        }
        else
        {
            throw "Error in SpMSpV : unsupported additive operation in semiring";
        }
    }
    else
    {
        throw "Error in SpMSpV : Masked SpMSpV not supported yet";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
