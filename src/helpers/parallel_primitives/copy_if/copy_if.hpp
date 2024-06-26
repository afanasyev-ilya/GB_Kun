#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __USE_GPU__
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#endif

#define MAX_SX_AURORA_THREADS 8

template <typename T>
void scan(T* input_data, T* output_data, T init_num, size_t input_size) {
    T sum = init_num;
    for (int i = 0; i < input_size + 1; i++) {
        T old_sum = sum;
        if (i != input_size) {
            sum += input_data[i];
        }
        output_data[i] = old_sum;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename CopyCondition>
inline int ParallelPrimitives::vector_copy_if_indexes(CopyCondition &&_cond,
                                                      int *_out_data,
                                                      size_t _size,
                                                      int *_buffer,
                                                      const int _buffer_size,
                                                      const int _index_offset)
{
    int _threads_count = MAX_SX_AURORA_THREADS;
    int elements_per_thread = (_buffer_size - 1)/_threads_count + VECTOR_LENGTH;
    int elements_per_vector = (elements_per_thread - 1)/VECTOR_LENGTH + 1;
    int shifts_array[MAX_SX_AURORA_THREADS];

    int elements_count = 0;
    #pragma omp parallel num_threads(_threads_count) shared(elements_count)
    {
        int tid = omp_get_thread_num();
        int start_pointers_reg[VECTOR_LENGTH];
        int current_pointers_reg[VECTOR_LENGTH];
        int last_pointers_reg[VECTOR_LENGTH];

        #pragma _NEC vreg(start_pointers_reg)
        #pragma _NEC vreg(current_pointers_reg)
        #pragma _NEC vreg(last_pointers_reg)

        #pragma _NEC vector
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            start_pointers_reg[i] = tid * elements_per_thread + i * elements_per_vector;
            current_pointers_reg[i] = tid * elements_per_thread + i * elements_per_vector;
            last_pointers_reg[i] = tid * elements_per_thread + i * elements_per_vector;
        }

        #pragma omp for schedule(static)
        for(int vec_start = 0; vec_start < _size; vec_start += VECTOR_LENGTH)
        {
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int src_id = vec_start + i;
                int global_id = src_id + _index_offset;
                if((src_id < _size) && (_cond(global_id) > 0))
                {
                    _buffer[current_pointers_reg[i]] = global_id;
                    current_pointers_reg[i]++;
                }
            }
        }

        int max_difference = 0;
        int save_values_per_thread = 0;
        for(int i = 0; i < VECTOR_LENGTH; i++)
        {
            int difference = current_pointers_reg[i] - start_pointers_reg[i];
            save_values_per_thread += difference;
            if(difference > max_difference)
                max_difference = difference;
        }

        shifts_array[tid] = save_values_per_thread;
        int loc_s = shifts_array[tid];
        #pragma omp barrier

        #pragma omp master
        {
            int cur_shift = 0;
            for(int i = 1; i < _threads_count; i++)
            {
                shifts_array[i] += shifts_array[i - 1];
            }

            elements_count = shifts_array[_threads_count - 1];

            for(int i = (_threads_count - 1); i >= 1; i--)
            {
                shifts_array[i] = shifts_array[i - 1];
            }
            shifts_array[0] = 0;
        }

        #pragma omp barrier

        int tid_shift = shifts_array[tid];
        int *private_ptr = &(_out_data[tid_shift]);

        int loc_st = shifts_array[tid];

        int local_pos = 0;
        #pragma _NEC novector
        for(int pos = 0; pos < max_difference; pos++)
        {
            #pragma _NEC vovertake
            #pragma _NEC novob
            #pragma _NEC vector
            for(int i = 0; i < VECTOR_LENGTH; i++)
            {
                int loc_size = current_pointers_reg[i] - start_pointers_reg[i];

                if(pos < loc_size)
                {
                    private_ptr[local_pos] = _buffer[last_pointers_reg[i]];
                    last_pointers_reg[i]++;
                    local_pos++;
                }
            }
        }
    }

    return elements_count;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename CopyCondition>
inline int ParallelPrimitives::omp_copy_if_indexes(CopyCondition &&_cond,
                                                   int *_out_data,
                                                   size_t _size,
                                                   int *_buffer,
                                                   const int _buffer_size,
                                                   const int _index_offset)
{
    int omp_work_group_size = omp_get_max_threads();

    const int max_threads = 400;
    int sum_array[max_threads];
    if(omp_work_group_size > max_threads)
        throw " Error in omp_copy_if_indexes : max_threads = 400 is too small for this architecture, please increase";

    #pragma omp parallel
    {
        const int ithread = omp_get_thread_num();
        const int nthreads = omp_work_group_size;

        int local_pos = 0;
        int buffer_max_size = (_buffer_size - 1)/nthreads + 1;
        int *local_buffer = &_buffer[ithread*buffer_max_size];

        #pragma omp single
        {
            sum_array[0] = 0;
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < _size; i++)
        {
            int global_src_id = i + _index_offset;
            if(_cond(global_src_id) > 0)
            {
                local_buffer[local_pos] = global_src_id;
                local_pos++;
            }
        }

        int local_size = local_pos;
        sum_array[ithread+1] = local_pos;

        #pragma omp barrier
        int offset = 0;
        for(int i=0; i<(ithread+1); i++)
        {
            offset += sum_array[i];
        }

        int *dst_ptr = &_out_data[offset];
        for (int i = 0; i < local_size; i++)
        {
            dst_ptr[i] = local_buffer[i];
        }
    }

    int output_size = 0;
    for(int i = 0; i < (omp_work_group_size + 1); i++)
    {
        output_size += sum_array[i];
    }

    return output_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename CopyCondition, typename _T>
inline int ParallelPrimitives::omp_copy_if_data(CopyCondition &&_cond,
                                                _T *_in_data,
                                                _T *_out_data,
                                                size_t _size,
                                                _T *_buffer,
                                                const int _buffer_size)
{
    int omp_work_group_size = omp_get_max_threads();

    const int max_threads = 400;
    int sum_array[max_threads];
    if(omp_work_group_size > max_threads)
        throw " Error in omp_copy_if_indexes : max_threads = 400 is too small for this architecture, please increase";

    #pragma omp parallel
    {
        const int ithread = omp_get_thread_num();
        const int nthreads = omp_work_group_size;

        int local_pos = 0;
        int buffer_max_size = (_buffer_size - 1)/nthreads + 1;
        int *local_buffer = &_buffer[ithread*buffer_max_size];

        #pragma omp single
        {
            sum_array[0] = 0;
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < _size; i++)
        {
            int old_data = _in_data[i];
            if(_cond(old_data))
            {
                local_buffer[local_pos] = old_data;
                local_pos++;
            }
        }

        int local_size = local_pos;
        sum_array[ithread+1] = local_pos;

        #pragma omp barrier
        int offset = 0;
        for(int i=0; i<(ithread+1); i++)
        {
            offset += sum_array[i];
        }

        _T *dst_ptr = &_out_data[offset];
        for (int i = 0; i < local_size; i++)
        {
            dst_ptr[i] = local_buffer[i];
        }
    }

    int output_size = 0;
    for(int i = 0; i < (omp_work_group_size + 1); i++)
    {
        output_size += sum_array[i];
    }
    return output_size;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




#ifdef __USE_GPU__
template <typename _T>
void __global__ init_indexes(_T *_data, int _size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _size)
    {
        _data[idx] = idx;
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename CopyCondition>
inline int ParallelPrimitives::copy_if_indexes(CopyCondition &&_cond,
                                               int *_out_data,
                                               size_t _size,
                                               int *_buffer,
                                               const int _buffer_size,
                                               const int _index_offset)
{
    int num_elements = 0;
    #ifdef __USE_NEC_SX_AURORA__
    num_elements = vector_copy_if_indexes(_cond, _out_data, _size, _buffer, _buffer_size, _index_offset);
    #elif defined(__USE_MULTICORE__)
    num_elements = omp_copy_if_indexes(_cond, _out_data, _size, _buffer, _buffer_size, _index_offset);
    #else
    /*int *indexes;
    MemoryAPI::allocate_array(&indexes, _size);
    for(int i = 0; i < _size; i++)
    {
        indexes[i] = i;
    }
    num_elements = thrust::copy_if(thrust::device, indexes, indexes + _size, _out_data, _cond) - _out_data;
    MemoryAPI::free_array(indexes);*/
    num_elements = omp_copy_if_indexes(_cond, _out_data, _size, _buffer, _buffer_size, _index_offset);
    #endif
    return num_elements;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename CopyCondition, typename _T>
inline int ParallelPrimitives::copy_if_data(CopyCondition &&_cond,
                                            _T *_in_data,
                                            _T *_out_data,
                                            size_t _size,
                                            _T *_buffer,
                                            const int _buffer_size)
{
    int num_elements = 0;
    #ifdef __USE_NEC_SX_AURORA__
    num_elements = omp_copy_if_data(_cond, _in_data, _out_data, _size, _buffer, _buffer_size); // TODO vector version
    #elif defined(__USE_MULTICORE__)
    num_elements = omp_copy_if_data(_cond, _in_data, _out_data, _size, _buffer, _buffer_size);
    #elif defined(__USE_GPU__)
    num_elements = thrust::copy_if(thrust::device, _in_data, _in_data + _size, _out_data, _cond) - _out_data;
    #else
    throw "Error in copy_if_indexes : unsupported architecture";
    #endif
    return num_elements;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

