#ifndef GB_KUN_EXCLUSIVE_SCAN_HPP
#define GB_KUN_EXCLUSIVE_SCAN_HPP

template <typename _T>
inline void ParallelPrimitives::exclusive_scan(_T *_in_data,
                                               _T *_out_data,
                                               size_t _size) {
    int omp_work_group_size = omp_get_max_threads();
    const int max_threads = 400;
    _T sum_array[max_threads];
    if (omp_work_group_size > max_threads)
        throw " Error in omp_copy_if_indexes : max_threads = 400 is too small for this architecture, please increase";

#pragma omp parallel num_threads(omp_work_group_size)
    {
        const int ithread = omp_get_thread_num();
        const int nthreads = omp_work_group_size;
        long long int local_size;
        long long int offset;
        if (ithread < _size % nthreads) {
            local_size = _size / nthreads + 1;
            offset = local_size * ithread;
        } else {
            local_size = _size / nthreads;
            offset = (_size % nthreads) * (local_size + 1) + (ithread - _size % nthreads) * local_size;
        }
        _T* temp = new _T [local_size + 1]();
        scan(_in_data + offset, temp,static_cast<_T>(0), local_size);
        _T local_max = temp[local_size];

        sum_array[ithread] = local_max;

#pragma omp barrier
#pragma omp single
        {
            scan(sum_array, sum_array, static_cast<_T>(0), omp_work_group_size);
        }

        _T local_additive = sum_array[ithread];

        for (int i = 0; i < local_size; i++) {
            _out_data[i + offset] = temp[i] + local_additive;
        }
#pragma omp single
        {
            _out_data[_size] = sum_array[omp_work_group_size];
        }

        delete[] temp;

    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif //GB_KUN_EXCLUSIVE_SCAN_HPP
