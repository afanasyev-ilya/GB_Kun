#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace lablas{
    namespace backend {


template <typename T>
void SpMV(const MatrixSegmentedCSR<T> *_matrix, const DenseVector<T> *_x, DenseVector<T> *_y)
{
    int cores_num = omp_get_max_threads();
    int num_segments;
    _matrix->get_segments(&num_segments);
    #pragma omp parallel
    {
        for(int seg_id = 0; seg_id < num_segments; seg_id++)
        {
            const SubgraphSegment<T> *segment = &(_matrix->get_segment()[seg_id]);
            T *buffer = (T*)segment->get_vbuffer();
            VNT seg_size;
            segment->get_size(&seg_size);

            #pragma omp for schedule(static)
            for(VNT i = 0; i < seg_size; i++)
                buffer[i] = 0;

            #pragma omp for schedule(guided, 1024)
            for(VNT i = 0; i < seg_size; i++)
            {
                for(ENT j = segment->get_row()[i]; j < segment->get_row()[i + 1]; j++)
                {
                    buffer[i] += segment->get_vals()[j] * _x->get_vals()[segment->get_col()[j]];
                }
            }
        }
    }
    int matrix_size;
    _matrix->get_size(&matrix_size);

    if(matrix_size > pow(2.0, 22)) // cache aware merge
    {
        int matrix_blocks;
        _matrix->get_blocks(&matrix_blocks);
        int outer_threads = min(matrix_blocks, cores_num);
        int inner_threads = cores_num/outer_threads;
        #pragma omp parallel num_threads(outer_threads)
        {
            #pragma omp for schedule(static)
            for(VNT cur_block = 0; cur_block < matrix_blocks; cur_block++)
            {
                for(int seg_id = 0; seg_id < num_segments; seg_id++)
                {
                    const SubgraphSegment<T> *segment = &(_matrix->get_segment()[seg_id]);
                    T *buffer = (T*)segment->get_vbuffer();
                    const VNT *conversion_indexes = segment->get_conversion();

                    VNT block_start = segment->get_block_start()[cur_block];
                    VNT block_end = segment->get_block_end()[cur_block];

                    #pragma omp parallel for num_threads(inner_threads)
                    for(VNT i = block_start; i < block_end; i++)
                    {
                        _y->get_vals()[conversion_indexes[i]] += buffer[i];
                    }
                }
            }
        };
        double t4 = omp_get_wtime();
    }
    else
    {
        for(int seg_id = 0; seg_id < num_segments; seg_id++)
        {
            const SubgraphSegment<T> *segment = &(_matrix->get_segment()[seg_id]);
            T *buffer = (T*)segment->get_vbuffer();
            const VNT *conversion_indexes = segment->get_conversion();
            VNT seg_size;
            segment->get_size(&seg_size);

            #pragma omp parallel for schedule(static)
            for(VNT i = 0; i < seg_size; i++)
            {
                _y->get_vals()[conversion_indexes[i]] += buffer[i];
            }
        }
    }

    //cout << "compare: " << (t2 - t1)*1000 << "(edge proc) vs " << (t4 - t3)*1000 << "(cache-aware) vs " << (t6 - t5)*1000 << "(usual merge)" << endl;
}
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

