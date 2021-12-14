#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

template <typename T, typename SemiringT>
void SpMV(MatrixCSR<T> *_matrix,
          MatrixCSR<T> *_matrix_socket_dub,
          const DenseVector<T> *_x,
          DenseVector<T> *_y,
          SemiringT op)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);

    #pragma omp parallel
    {
        int total_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int socket = tid/(total_threads/2);
        int tid_s = omp_get_thread_num() % (total_threads/2);

        T *local_x_vals;
        MatrixCSR<T> *local_matrix;
        if(socket == 0)
        {
            local_matrix = _matrix;
            local_x_vals = x_vals;
        }
        else if(socket == 1)
        {
            local_matrix = _matrix_socket_dub;
            local_x_vals = (T*)_matrix->tmp_buffer;
        }

        if(socket == 1)
            for(VNT row = tid_s; row < local_matrix->size; row += total_threads/2)
            {
                local_x_vals[row] = x_vals[row];
            }

        #pragma omp for schedule(guided, 1024)
        for(VNT row = 0; row < local_matrix->size; row++)
        {
            for(ENT j = local_matrix->row_ptr[row]; j < local_matrix->row_ptr[row + 1]; j++)
            {
                VNT col = local_matrix->col_ids[j];
                T val = local_matrix->vals[j];
                y_vals[row] = add_op(y_vals[row], mul_op(val, local_x_vals[col])) ;
            }
        }
    };
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool CMP, typename T, typename Y, typename SemiringT>
void SpMV_dense(const MatrixCSR<T> *_matrix,
          const DenseVector<T> *_x,
          DenseVector<T> *_y, SemiringT op, const Vector<Y> *_mask)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for(VNT i = 0; i < _matrix->size; i++)
        {
            bool mask_val = (bool)_mask->getDense()->get_vals()[i];
            if (mask_val && !CMP) {
                for(ENT j = _matrix->row_ptr[i]; j < _matrix->row_ptr[i + 1]; j++)
                {
                    y_vals[i] = add_op(y_vals[i], mul_op(_matrix->vals[j], x_vals[_matrix->col_ids[j]])) ;
                }
            }
        }
    }
}

template <typename T, typename SemiringT>
void SpMV(const MatrixCSR<T> *_matrix,
          const DenseVector<T> *_x,
          DenseVector<T> *_y, SemiringT op)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    /*#pragma omp parallel
    {
        //ENT cnt = 0;
        #pragma omp for schedule(guided, 1024)
        for(VNT row = 0; row < _matrix->size; row++)
        {
            for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
            {
                VNT col = _matrix->col_ids[j];
                T val = _matrix->vals[j];
                y_vals[row] = add_op(y_vals[row], mul_op(val, x_vals[col])) ;
                //cnt++;
            }
        }
        //#pragma omp critical
        //cout << "cnt: " << 100.0*(double)cnt/_matrix->nz << endl;
    }*/

    #pragma omp parallel
    {
        for(int vg = 0; vg < _matrix->vg_num; vg++)
        {
            const VNT *vertices = &(_matrix->vertex_groups[vg].data[0]);
            VNT vertex_group_size = _matrix->vertex_groups[vg].data.size();

            #pragma omp for nowait schedule(guided, 1)
            for(VNT idx = 0; idx < vertex_group_size; idx++)
            {
                VNT row = vertices[idx];
                T res = identity_val;
                for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
                {
                    VNT col = _matrix->col_ids[j];
                    T val = _matrix->vals[j];
                    res = add_op(res, mul_op(val, x_vals[col]));
                }
                y_vals[row] = res;
            }
        }
    }

    /*#pragma omp parallel num_threads(6)
    {
        #pragma omp for schedule (static)
        for(int vg = 0; vg < _matrix->vg_num; vg++)
        {
            const VNT *vertices = &(_matrix->vertex_groups[vg].data[0]);
            VNT vertex_group_size = _matrix->vertex_groups[vg].data.size();

            #pragma omp parallel for schedule(guided, 1) num_threads(8)
            for(VNT idx = 0; idx < vertex_group_size; idx++)
            {
                VNT row = vertices[idx];
                T res = 0;
                for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
                {
                    VNT col = _matrix->col_ids[j];
                    T val = _matrix->vals[j];
                    res = add_op(res, mul_op(val, x_vals[col]));
                }
                y_vals[row] = res;
            }
        }
    }*/
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////