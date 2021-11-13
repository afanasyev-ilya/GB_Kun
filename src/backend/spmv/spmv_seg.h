#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
double SpMV(MatrixSegmentedCSR<T> &_A, DenseVector<T> &_x, DenseVector<T> &_y)
{
    T *x_vals = _x.get_vals();
    T *y_vals = _y.get_vals();

    int num_segments = _A.get_num_segments();

    double t1 = omp_get_wtime();
    for(int seg_id = 0; seg_id < num_segments; seg_id++)
    {
        SubgraphSegment<T> *segment = _A.get_segment(seg_id);
        T *buffer = (T*)segment->vertex_buffer;

        #pragma omp parallel for schedule(static)
        for(VNT i = 0; i < segment->size; i++)
        {
            buffer[i] = 0;
            for(ENT j = segment->row_ptr[i]; j < segment->row_ptr[i + 1]; j++)
            {
                buffer[i] += segment->vals[j] * x_vals[segment->col_ids[j]];
            }
        }
    }
    double t2 = omp_get_wtime();
    double compute_time = t2 - t1;

    double bw = (3.0*sizeof(VNT)+sizeof(T))*_A.get_nz()/(compute_time*1e9);
    cout << "Inner bw: " << bw << " GB/s" << endl;

    t1 = omp_get_wtime();
    for(int seg_id = 0; seg_id < num_segments; seg_id++)
    {
        SubgraphSegment<T> *segment = _A.get_segment(seg_id);
        T *buffer = (T*)segment->vertex_buffer;
        VNT *conversion_indexes = segment->conversion_to_full;

        #pragma omp parallel for schedule(static)
        for(VNT i = 0; i < segment->size; i++)
        {
            y_vals[conversion_indexes[i]] += buffer[i];
        }
    }
    t2 = omp_get_wtime();
    double merge_time = t2 - t1;
    double wall_time = merge_time + compute_time;

    return bw;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

