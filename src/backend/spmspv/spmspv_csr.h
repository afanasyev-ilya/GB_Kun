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

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < nt; t++) {
        int offset_ = t * nz / nt; // Every thread has it's own piece of vector x
        for (int j = offset_; j < nz; j++) {
            Index vector_index = x_ids[j]; // index of j-th non-zero element in vector x                  // 1
            cnt += sizeof(int) * 2 + sizeof(Index); // REMOVE!!!
            for (int i = col_ptr[vector_index]; i < col_ptr[vector_index + 1]; i++) {                   // 2, 3
                // bucket index depends on the row of an element
                int bucket_index = row_ids[i] * nb / matrix_size;                                       // 4
                Boffset[t][bucket_index] += 1;                                                          // 5
                cnt += sizeof(int) * 2; // REMOVE!!!
            }
        }
    }

    double t4 = omp_get_wtime();
    double time = t4 - t3;

    double bw = cnt / (time * 1e9);
    cout << "BW Estimating buckets: " << bw << " GB/s" << endl;
}


template <typename T>
void SpMSpV_csr(const MatrixCSR<T> *_matrix_csc,
                const SparseVector<T> *_x,
                DenseVector<T> *_y,
                int _number_of_buckets, int _number_of_threads,
                int *bucket_amount, int *offset_, lablas::backend::bucket<T> *buckets, float *SPA, int *offset)
{
    double overall_time;
    
    double t1 = omp_get_wtime();

    VNT nz, matrix_size, matrix_nz;
    _x->get_nnz(&nz);
    _matrix_csc->get_size(&matrix_size);
    matrix_nz = _matrix_csc->get_nnz();
    long long max_number_of_insertions = nz * _matrix_csc->get_max_degree();

    const ENT *col_ptr = _matrix_csc->get_row_ptr(); // we assume csr of AT is equal to csc of A
    const VNT *row_ids = _matrix_csc->get_col_ids(); // we assume csr of AT is equal to csc of A


//    omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(_number_of_threads); // Use _number_of_threads threads for all consecutive parallel regions

    vector<vector<int>> Boffset(_number_of_threads, vector<int>(_number_of_buckets));
    estimate_buckets(_matrix_csc, _x, Boffset, _number_of_buckets, _number_of_threads);

    // The point of function estimate_buckets is to fill the matrix Boffset in which
    // Boffset[i][j] means how many insertions the i-th thread will make in the j-th bucket

    // We need to fill the matrix Boffset in order to make synchronization free insertions in step 1

    for (int bucket_number = 0; bucket_number < _number_of_buckets; bucket_number++) {
        for (int thread_number = 0; thread_number < _number_of_threads; thread_number++) {
            if (thread_number)
                *(offset_ + bucket_number * _number_of_threads + thread_number) = *(offset_ + bucket_number * _number_of_threads + thread_number - 1) + Boffset[thread_number - 1][bucket_number];
            bucket_amount[bucket_number] += Boffset[thread_number][bucket_number];
        }
    }

    const Index *x_ids = _x->get_ids();
    const T *vals = _matrix_csc->get_vals();
    const T *x_vals = _x->get_vals();
    int cnt = 0;

    // Step 1. Filling buckets.
    // Every bucket gets it's own set of rows
    double t3 = omp_get_wtime();

    #pragma omp parallel
    {
        int cur_thread_number = omp_get_thread_num();
        vector<int> insertions(_number_of_buckets);
        #pragma omp for schedule(static)
        for (int i = 0; i < nz; i++) { // Going through all non-zero elements of vector x
            // We only need matrix's columns which numbers are equal to indices of non-zero elements of the vector x
            Index vector_index = x_ids[i]; // Index of non-zero element in vector x                                       // 1
            cnt += sizeof(int) * 2 + sizeof(Index); // REMOVE!!!
            for (int j = col_ptr[vector_index]; j < col_ptr[vector_index + 1]; j++) {                                   // 2, 3
                // Going through all the elements of the vector_index-th column
                // Essentially we are doing multiplication here
                float mul = vals[j] * x_vals[i]; // mul - is a product of matrix and vector non-zero elements.          // 4, 5
                int bucket_index = (row_ids[j] * _number_of_buckets) / matrix_size;                                     // 6
                // bucket's index depends on the row number
                // Implementing synchronization free insertion below
                // Boffset[i][j] - amount, i - thread, j - bucket
                // offset - how many elements have been inserted in the bucket by other threads
                int off = *(offset_ + bucket_index * _number_of_threads + cur_thread_number);                           // 7
                *(buckets + bucket_index * max_number_of_insertions + off + insertions[bucket_index]++) = {row_ids[j], mul}; // insertion           // 8, 9, 10
                cnt += sizeof(float) * 2 + sizeof(int) * 4 + sizeof(bucket<T>); // REMOVE!!!
            }
        }
    }

    double t4 = omp_get_wtime();
    double time = t4 - t3;

    double bw = cnt / (time * 1e9);
    cout << "BW Filling buckets: " << bw << " GB/s" << endl;
    // The SPA vector is essentially a final vector-answer, but it is dense and we will have to transform it in sparse vector
    //vector<int> offset(_number_of_buckets);

    T *y_vals = (T *)malloc(sizeof(T) * matrix_size);
    VNT *y_ids = (VNT *)malloc(sizeof(VNT) * matrix_size);

    t3 = omp_get_wtime();
    cnt = 0;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int number_of_bucket = 0; number_of_bucket < _number_of_buckets; number_of_bucket++) { // Going through all buckets
        vector<int> uind; // uing - unique indices in the number_of_bucket-th bucket

        cnt += sizeof(int);  // REMOVE!!!
        // Step 2. Merging entries in each bucket.
        for (int i = 0; i < bucket_amount[number_of_bucket]; i++) {                                         // 1
            int row = (buckets + number_of_bucket * max_number_of_insertions + i)->row;                     // 2
            SPA[row] = INF; // Initializing with a value of INF                                             // 3
            cnt += sizeof(float) + sizeof(Index);  // REMOVE!!!
            // in order to know whether a row have been to the uind vector or not
        }
        // Accumulating values in a row and getting unique indices
        cnt += sizeof(int);  // REMOVE!!!
        for (int i = 0; i < bucket_amount[number_of_bucket]; i++) {                                         // 4
            int row = (buckets + number_of_bucket * max_number_of_insertions + i)->row;                     // 5
            float val = (buckets + number_of_bucket * max_number_of_insertions + i)->val;                   // 6
            cnt += sizeof(int) + sizeof(float); // REMOVE !!!
            if (SPA[row] == INF) {                                                                          // 7
                uind.push_back(row);
                SPA[row] = val;                                                                             // 8
            } else {
                SPA[row] += val;                                                                            // 9
            }
            cnt += sizeof(float) * 2;  // REMOVE!!!
        }
        if (number_of_bucket) // if not the first bucket
        {
            offset[number_of_bucket] += offset[number_of_bucket - 1] + bucket_amount[number_of_bucket - 1]; // 10, 11, 12
            cnt += sizeof(int) * 3;  // REMOVE!!!
        }
        // offset is needed to properly fill the final vector

        for (int i = 0; i < uind.size(); i++) { // filling the final vector
            int ind = uind[i];                               // 13
            float value = SPA[ind];                         // 14
            int off = offset[number_of_bucket];            // 15
            _y->set_element(SPA[ind], ind);               // 16, 17
            cnt += sizeof(int) * 2 + sizeof(float) * 3;  // REMOVE!!!
        }
    }

    t4 = omp_get_wtime();
    time = t4 - t3;

    bw = cnt / (time * 1e9);
    cout << "BW Merging buckets and making final vector: " << bw << " GB/s" << endl;

    double t2 = omp_get_wtime();

    overall_time = t2 - t1;

    printf("\033[0;31m");
    printf("SpMSpV time: %lf seconds.\n", overall_time);
    printf("\033[0m");
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// nt - number of threads
// nb - number of buskets
template <typename T>
vector<vector<int>> estimate_buckets_old(const MatrixCSR<T> *matrix, const SparseVector<T> *x, int nb, int nt)
{

//    omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(nt); // Use 1 threads for all consecutive parallel regions

    // This function is essential in implementing synchronization free insertion
    VNT nz, matrix_size;

    const ENT *col_ptr = matrix->get_row_ptr(); // we assume csr of AT is equal to csc of A
    const VNT *row_ids = matrix->get_col_ids(); // we assume csr of AT is equal to csc of A

    x->get_nnz(&nz);
    matrix->get_size(&matrix_size);
    vector<vector<int>> Boffset(nt, vector<int>(nb)); // Boffset is described in the SpmSpv function
    //vector<vector<int>> Boffset(nt, vector<int>(nb)); // Boffset is described in the SpmSpv function

    const Index *x_ids = x->get_ids();
    double t3 = omp_get_wtime();
    int cnt = 0;

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < nt; t++) {
        int offset_ = t * nz / nt; // Every thread has it's own piece of vector x
        for (int j = offset_; j < nz; j++) {
            Index vector_index = x_ids[j]; // index of j-th non-zero element in vector x                  // 1
            cnt += sizeof(int) * 2 + sizeof(Index); // REMOVE!!!
            for (int i = col_ptr[vector_index]; i < col_ptr[vector_index + 1]; i++) {                   // 2, 3
                // bucket index depends on the row of an element
                int bucket_index = row_ids[i] * nb / matrix_size;                                       // 4
                Boffset[t][bucket_index] += 1;                                                          // 5
                cnt += sizeof(int) * 2; // REMOVE!!!
            }
        }
    }

    double t4 = omp_get_wtime();
    double time = t4 - t3;

    double bw = cnt / (time * 1e9);
    cout << "BW Estimating buckets: " << bw << " GB/s" << endl;

    return Boffset;
}


template <typename T>
void SpMSpV_csr_old(const MatrixCSR<T> *_matrix_csc,
                const SparseVector<T> *_x,
                DenseVector<T> *_y,
                int _number_of_buckets)
{
    double merging_entries, estimating_buckets, filling_buckets, overall_time, matrix_prop, preparing_for_filling_buckets;

    double t1 = omp_get_wtime();

    VNT nz, matrix_size;
    _x->get_nnz(&nz);
    _matrix_csc->get_size(&matrix_size);

    const ENT *col_ptr = _matrix_csc->get_row_ptr(); // we assume csr of AT is equal to csc of A
    const VNT *row_ids = _matrix_csc->get_col_ids(); // we assume csr of AT is equal to csc of A

    int number_of_threads = 64;

    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(number_of_threads); // Use 1 threads for all consecutive parallel regions

    vector<vector<int>> Boffset = estimate_buckets_old(_matrix_csc, _x, _number_of_buckets, number_of_threads);
    // The point of function estimate_buckets is to fill the matrix Boffset in which
    // Boffset[i][j] means how many insertions the i-th thread will make in the j-th bucket

    double sub_time1, sub_time2;

    // We need to fill the matrix Boffset in order to make synchronization free insertions in step 1
    vector<int> bucket_amount(_number_of_buckets); // Stores how many insertions will be made in i-th bucket
    vector<vector<int>> offset_(_number_of_buckets, vector<int>(number_of_threads));

    for (int bucket_number = 0; bucket_number < _number_of_buckets; bucket_number++) {
        for (int thread_number = 0; thread_number < number_of_threads; thread_number++) {
            if (thread_number)
                offset_[bucket_number][thread_number] = offset_[bucket_number][thread_number - 1] + Boffset[thread_number - 1][bucket_number];
            bucket_amount[bucket_number] += Boffset[thread_number][bucket_number];
        }
    }

    vector<vector<bucket<T>>> buckets(_number_of_buckets);
    for (int i = 0; i < _number_of_buckets; i++) {
        buckets[i] = vector<bucket<T>>(bucket_amount[i]);
    }

    const Index *x_ids = _x->get_ids();
    const T *vals = _matrix_csc->get_vals();
    const T *x_vals = _x->get_vals();
    int cnt = 0;

    // Step 1. Filling buckets.
    // Every bucket gets it's own set of rows
    double t3 = omp_get_wtime();

    #pragma omp parallel
    {
        int cur_thread_number = omp_get_thread_num();
        vector<int> insertions(_number_of_buckets);
        #pragma omp for schedule(static)
        for (int i = 0; i < nz; i++) { // Going through all non-zero elements of vector x
            // We only need matrix's columns which numbers are equal to indices of non-zero elements of the vector x
            Index vector_index = x_ids[i]; // Index of non-zero element in vector x
            cnt += sizeof(int) * 2 + sizeof(Index);
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
                cnt += sizeof(float) * 2 + sizeof(int) * 4 + sizeof(bucket<T>); // REMOVE!!!
            }
        }
    }

    double t4 = omp_get_wtime();
    double time = t4 - t3;

    double bw = cnt / (time * 1e9);
    cout << "BW Filling buckets: " << bw << " GB/s" << endl;

    vector<float> SPA(matrix_size); // SPA -  a sparse accumulator
    // The SPA vector is essentially a final vector-answer, but it is dense and we will have to transform it in sparse vector
    vector<int> offset(_number_of_buckets);

    T *y_vals = (T *)malloc(sizeof(T) * matrix_size);
    VNT *y_ids = (VNT *)malloc(sizeof(VNT) * matrix_size);

    t3 = omp_get_wtime();
    cnt = 0;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int number_of_bucket = 0; number_of_bucket < _number_of_buckets; number_of_bucket++) { // Going through all buckets
        vector<int> uind; // uing - unique indices in the number_of_bucket-th bucket

        // Step 2. Merging entries in each bucket.
        cnt += sizeof(int);
        for (int i = 0; i < buckets[number_of_bucket].size(); i++) {
            int row = buckets[number_of_bucket][i].row;
            SPA[row] = INF; // Initializing with a value of INF
            cnt += sizeof(int) + sizeof(float);
            // in order to know whether a row have been to the uind vector or not
        }
        // Accumulating values in a row and getting unique indices
        cnt += sizeof(bucket<T>);
        for (int i = 0; i < buckets[number_of_bucket].size(); i++) {
            int row = buckets[number_of_bucket][i].row;
            float val = buckets[number_of_bucket][i].val;
            cnt += sizeof(int) + sizeof(float) * 2;
            if (SPA[row] == INF) {
                uind.push_back(row);
                SPA[row] = val;
            } else {
                SPA[row] += val;
            }
            cnt += sizeof(float);
        }
        if (number_of_bucket) // if not the first bucket
        {
            offset[number_of_bucket] += offset[number_of_bucket - 1] + bucket_amount[number_of_bucket - 1];
            cnt += sizeof(int) * 3;
        }
        // offset is needed to properly fill the final vector

        for (int i = 0; i < uind.size(); i++) { // filling the final vector
            int ind = uind[i];
            float value = SPA[ind];
            int off = offset[number_of_bucket];
            _y->set_element(SPA[ind], ind);
            cnt += sizeof(int) * 2 + sizeof(float) * 3;
        }
    }

    t4 = omp_get_wtime();
    time = t4 - t3;

    bw = cnt / (time * 1e9);
    cout << "BW Merging buckets and making final vector: " << bw << " GB/s" << endl;

    double t2 = omp_get_wtime();

    overall_time = t2 - t1;

    printf("\033[0;31m");
    printf("SpMSpV_alloc time: %lf seconds.\n", overall_time);
    printf("\033[0m");
    /*
    printf("\t- Getting matrix properties and allocating memory: %.1lf %%\n", matrix_prop / overall_time * 100.0);
    printf("\t- Estimating buckets: %.1lf %%\n", estimating_buckets / overall_time * 100.0);
    printf("\t- Preparing for filling buckets: %.1lf %%\n", preparing_for_filling_buckets / overall_time * 100.0);
    printf("\t\t- Allocating memory: %.1lf %%\n", sub_time1 / preparing_for_filling_buckets * 100.0);
    printf("\t\t- double for: %.1lf %%\n", sub_time2 / preparing_for_filling_buckets * 100.0);
    printf("\t- Filling buckets: %.1lf %%\n", filling_buckets / overall_time * 100.0);
    printf("\t- Merging entries: %.1lf %%\n", merging_entries / overall_time * 100.0);
    */
}
