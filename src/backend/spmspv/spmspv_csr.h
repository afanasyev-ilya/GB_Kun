#pragma once

#define VNT Index
#define ENT Index

#define INF 1000000000

// nt - number of threads
// nb - number of buskets
template <typename T>
void estimate_buckets(const MatrixCSR<T> *matrix, const SparseVector<T> *x, vector<vector<int>> &Boffset, int nb, int nt)
{

//    omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(nt); // Use 1 threads for all consecutive parallel regions

    // This function is essential in implementing synchronization free insertion
    VNT nz, matrix_size;

    const ENT *col_ptr = matrix->get_row_ptr(); // we assume csr of AT is equal to csc of A
    const VNT *row_ids = matrix->get_col_ids(); // we assume csr of AT is equal to csc of A

    x->get_nnz(&nz);
    matrix->get_size(&matrix_size);
    //vector<vector<int>> Boffset(nt, vector<int>(nb)); // Boffset is described in the SpmSpv function

    const Index *x_ids = x->get_ids();
    double t3 = omp_get_wtime();
    int cnt = 0;

    #pragma omp parallel
    {
    int local_cnt = 0;
        #pragma omp for schedule(static)
        for (int t = 0; t < nt; t++) {
            int offset_ = t * nz / nt; // Every thread has it's own piece of vector x
            for (int j = offset_; j < nz; j++) {
                Index vector_index = x_ids[j]; // index of j-th non-zero element in vector x                  // 1
                Index iter_count = col_ptr[vector_index + 1] - col_ptr[vector_index];
                local_cnt += sizeof(Index) * 5 + iter_count * (sizeof(int) + sizeof(Index)); // REMOVE!!!
                for (int i = col_ptr[vector_index]; i < col_ptr[vector_index + 1]; i++) {                   // 2, 3
                    // bucket index depends on the row of an element
                    int bucket_index = row_ids[i] * nb / matrix_size;                                       // 4
                    Boffset[t][bucket_index] += 1;                                                          // 5
                }
            }
        }
        #pragma omp atomic
        cnt += local_cnt;
    }
    double t4 = omp_get_wtime();
    double time = t4 - t3;

    double bw = cnt / (time * 1e9);
    cout << "Estimating buckets: \n";
    cout << "\t-BW: " << bw << " GB/s" << endl;
    cout << "\t-Time: " << time * 1e3 << " ms  " << endl;
}

template <typename T>
void SpMSpV_csr(const MatrixCSR<T> *_matrix_csc,
                const SparseVector<T> *_x,
                DenseVector<T> *_y,
                int _number_of_buckets,
                Workspace *_workspace)
{
    double merging_entries, estimating_buckets, filling_buckets, overall_time, matrix_prop, preparing_for_filling_buckets, alloc_time;

    double t1 = omp_get_wtime();

    VNT nz, matrix_size;
    _x->get_nnz(&nz);
    _matrix_csc->get_size(&matrix_size);

    const ENT *col_ptr = _matrix_csc->get_row_ptr(); // we assume csr of AT is equal to csc of A
    const VNT *row_ids = _matrix_csc->get_col_ids(); // we assume csr of AT is equal to csc of A

    int number_of_threads = 64;

//    omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(number_of_threads); // Use 1 threads for all consecutive parallel regions

    vector<vector<int>> Boffset(number_of_threads, vector<int>(_number_of_buckets));
    estimate_buckets(_matrix_csc, _x, Boffset, _number_of_buckets, number_of_threads);
    //vector<vector<int>> Boffset = estimate_buckets_old(_matrix_csc, _x, _number_of_buckets, number_of_threads);
    // The point of function estimate_buckets is to fill the matrix Boffset in which
    // Boffset[i][j] means how many insertions the i-th thread will make in the j-th bucket

    double sub_time1, sub_time2;

    // We need to fill the matrix Boffset in order to make synchronization free insertions in step 1
    double a1 = omp_get_wtime();

    vector<int> bucket_amount(_number_of_buckets); // Stores how many insertions will be made in i-th bucket
    vector<vector<int>> offset_(_number_of_buckets, vector<int>(number_of_threads));

    double a2 = omp_get_wtime();

    alloc_time += a2 - a1;

    for (int bucket_number = 0; bucket_number < _number_of_buckets; bucket_number++) {
        for (int thread_number = 0; thread_number < number_of_threads; thread_number++) {
            if (thread_number)
                offset_[bucket_number][thread_number] = offset_[bucket_number][thread_number - 1] + Boffset[thread_number - 1][bucket_number];
            bucket_amount[bucket_number] += Boffset[thread_number][bucket_number];
        }
    }

    a1 = omp_get_wtime();

    vector<vector<bucket<T>>> buckets(_number_of_buckets);
    for (int i = 0; i < _number_of_buckets; i++) {
        buckets[i] = vector<bucket<T>>(bucket_amount[i]);
    }

    a2 = omp_get_wtime();
    alloc_time += a2 - a1;

    const Index *x_ids = _x->get_ids();
    const T *vals = _matrix_csc->get_vals();
    const T *x_vals = _x->get_vals();
    int cnt = 0;

    // Step 1. Filling buckets.
    // Every bucket gets it's own set of rows
    double t3 = omp_get_wtime();

    #pragma omp parallel
    {
        int local_cnt = 0;
        int cur_thread_number = omp_get_thread_num();
        vector<int> insertions(_number_of_buckets);
        #pragma omp for schedule(static)
        for (int i = 0; i < nz; i++) { // Going through all non-zero elements of vector x
            // We only need matrix's columns which numbers are equal to indices of non-zero elements of the vector x
            Index vector_index = x_ids[i]; // Index of non-zero element in vector x

            int iter_count = col_ptr[vector_index + 1] - col_ptr[vector_index];
            local_cnt += sizeof(Index) * 4 + sizeof(Index) + iter_count * (sizeof(T) * 2 + sizeof(Index) * 2 + sizeof(int) * 2 + sizeof(bucket<T>)); // REMOVE!!!

            for (int j = col_ptr[vector_index]; j < col_ptr[vector_index + 1]; j++) {
                // Going through all the elements of the vector_index-th column
                // Essentially we are doing multiplication here
                float mul = vals[j] * x_vals[i]; // mul - is a product of matrix and vector non-zero elements.
                int bucket_index = (row_ids[j] * _number_of_buckets) / matrix_size;
                // bucket's index depends on the row number
                // Implementing synchronization free insertion below
                // Boffset[i][j] - amount, i - thread, j - bucket
                // offset - how many elements have been inserted in the bucket by other threads
                buckets[bucket_index][offset_[bucket_index][cur_thread_number] + (insertions[bucket_index])++] = {row_ids[j], mul}; // insertion
            }
        }
        #pragma omp atomic
        cnt += local_cnt;
    }

    double t4 = omp_get_wtime();
    double time = t4 - t3;

    double bw = cnt / (time * 1e9);
    cout << "Filling buckets: \n";
    cout << "\t-BW: " << bw << " GB/s" << endl;
    cout << "\t-Time: " << time * 1e3 << " ms" << endl;

    a1 = omp_get_wtime();

    //T *SPA = (T*)_workspace->get_first_socket_vector(); // SPA -  a sparse accumulator

    // The SPA vector is essentially a final vector-answer, but it is dense and we will have to transform it in sparse vector
    vector<int> offset(_number_of_buckets);

    a2 = omp_get_wtime();
    alloc_time += a2 - a1;

    t3 = omp_get_wtime();
    cnt = 0;
    #pragma omp parallel
    {
        map<VNT, T>SPA;
        int local_cnt = 0;
        #pragma omp for schedule(dynamic, 1)
        for (int number_of_bucket = 0;
            number_of_bucket < _number_of_buckets; number_of_bucket++) { // Going through all buckets
            vector<int> uind; // uing - unique indices in the number_of_bucket-th bucket

            // Step 2. Merging entries in each bucket.
            local_cnt += bucket_amount[number_of_bucket] * (sizeof(T) + sizeof(Index)) + 2 * sizeof(int);  // REMOVE!!!
            for (int i = 0; i < buckets[number_of_bucket].size(); i++) {
                int row = buckets[number_of_bucket][i].row;
                SPA[row] = INF; // Initializing with a value of INF
                // in order to know whether a row have been to the uind vector or not
            }
            // Accumulating values in a row and getting unique indices
            local_cnt += bucket_amount[number_of_bucket] * (sizeof(T) + sizeof(Index) + sizeof(T) * 2) + 2 * sizeof(int);  // REMOVE!!!
            for (int i = 0; i < buckets[number_of_bucket].size(); i++) {
                int row = buckets[number_of_bucket][i].row;
                float val = buckets[number_of_bucket][i].val;
                if (SPA[row] == INF) {
                    uind.push_back(row);
                    SPA[row] = val;
                } else {
                    SPA[row] += val;
                }
            }
            if (number_of_bucket) // if not the first bucket
            {
                offset[number_of_bucket] += offset[number_of_bucket - 1] + bucket_amount[number_of_bucket - 1];
                local_cnt += sizeof(int) * 3;  // REMOVE!!!

            }
            // offset is needed to properly fill the final vector

            local_cnt += uind.size() * (sizeof(int) * 2 + sizeof(float) * 3);
            for (int i = 0; i < uind.size(); i++) { // filling the final vector
                int ind = uind[i];
                float value = SPA[ind];
                int off = offset[number_of_bucket];
                _y->set_element(SPA[ind], ind);
            }
        }
        #pragma omp atomic
        cnt += local_cnt;
    }
    t4 = omp_get_wtime();
    time = t4 - t3;

    bw = cnt / (time * 1e9);
    cout << "Merging buckets and making final vector: \n";
    cout << "\t-BW: " << bw << " GB/s" << endl;
    cout << "\t-Time: " << time * 1e3 << " ms" << endl;

    cout << "Memory allocation: \n";
    cout << "\t-Time: " << alloc_time * 1e3 << " ms" << endl;

    double t2 = omp_get_wtime();

    overall_time = t2 - t1;

    printf("\033[0;31m");
    printf("SpMSpV time: %lf ms.\n", overall_time * 1e3);
    printf("\033[0m");
}
