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

    /*#pragma omp parallel // casual version without good load balancing
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
        //cout << "cnt: " << 100.0*(double)cnt/_matrix->nnz << endl;
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

    /*#pragma omp parallel num_threads(6) // dynamic parallelism version (worse perf)
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

template <typename T, typename SemiringT>
void SpMV_test(const MatrixCSR<T> *_matrix,
               const DenseVector<T> *_x,
               DenseVector<T> *_y, SemiringT op)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();
    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    double t1 = omp_get_wtime();
    ENT nnz = _matrix->get_nnz();
    #pragma omp parallel for
    for(VNT i = 0; i < nnz; i++)
    {
        VNT col = _matrix->col_ids[i];
        _matrix->vals[i] = x_vals[col];
    }
    double t2 = omp_get_wtime();
    cout << "xvals size: " << sizeof(T) * _matrix->size / 1e6 << " MB" << endl;
    cout << "inner gather time: " << (t2 - t1) *1000 << " ms" << endl;
    cout << "inner gather bw: " << nnz*(sizeof(T)*2 + sizeof(Index))/((t2 - t1)*1e9) << " GB/s" << endl;

    T*new_cols;
    MemoryAPI::allocate_array(&new_cols, _matrix->nnz);
    #pragma omp for
    for(VNT i = 0; i < nnz; i++)
        new_cols[i] = _matrix->col_ids[i];

    size_t seg_size = 256*1024/sizeof(T);
    std::sort(new_cols, new_cols + nnz,
              [seg_size](int index1, int index2)
              {
                  return index1 / seg_size < index2 / seg_size;
              });

    t1 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp for
        for(VNT i = 0; i < nnz; i++)
        {
            VNT col = new_cols[i];
            _matrix->vals[i] = x_vals[col];
        }
    };
    t2 = omp_get_wtime();
    cout << "opt gather time: " << (t2 - t1) *1000 << " ms" << endl;
    cout << "opt gather bw: " << nnz*(sizeof(T)*2 + sizeof(Index))/((t2 - t1)*1e9) << " GB/s" << endl;

    MemoryAPI::free_array(new_cols);
}


}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////