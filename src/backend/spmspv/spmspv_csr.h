#pragma once

#define VNT Index
#define ENT Index

template <typename T>
struct bucket {
    VNT row;
    T val;
};


#define INF 1000000000

/*
vector<int> bucket_amount(_number_of_buckets);
vector<vector<int>> offset_(_number_of_buckets, vector<int>(number_of_threads));
vector<vector<bucket>> buckets(_number_of_buckets);
for (int i = 0; i < _number_of_buckets; i++) {
    buckets[i] = vector<bucket>(bucket_amount[i]);
}
vector<float> SPA(matrix_size);
vector<int> offset(_number_of_buckets);

 */

// nt - number of threads
// nb - number of buskets
template <typename T>
vector<vector<int>> estimate_buckets(const MatrixCSR<T> *matrix, const SparseVector<T> *x, int nb, int nt)
{

    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(nt); // Use 1 threads for all consecutive parallel regions

    // This function is essential in implementing synchronization free insertion
    VNT nz, matrix_size;

    const ENT *col_ptr = matrix->get_row_ptr(); // we assume csr of AT is equal to csc of A
    const VNT *row_ids = matrix->get_col_ids(); // we assume csr of AT is equal to csc of A

    x->get_nnz(&nz);
    matrix->get_size(&matrix_size);
    vector<vector<int>> Boffset(nt, vector<int>(nb)); // Boffset is described in the SpmSpv function
    
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < nt; t++) {
        int offset_ = t * nz / nt; // Every thread has it's own piece of vector x
        for (int j = offset_; j < nz; j++) {
            int vector_index = x->get_ids()[j]; // index of j-th non-zero element in vector x
            for (int i = col_ptr[vector_index]; i < col_ptr[vector_index + 1]; i++) {
                // bucket index depends on the row of an element
                int bucket_index = row_ids[i] * nb / matrix_size;
                Boffset[t][bucket_index] += 1;
            }
        }
    }
    return Boffset;
}


template <typename T>
void SpMSpV_csr(const MatrixCSR<T> *_matrix_csc,
                const SparseVector<T> *_x,
                DenseVector<T> *_y,
                int _number_of_buckets, int number_of_threads,
                int *bucket_amount, int **offset_, bucket<T> **buckets, float *SPA, int *offset)
{
    double merging_entries, estimating_buckets, filling_buckets, overall_time, matrix_prop, preparing_for_filling_buckets;

    double t1 = omp_get_wtime();

    double t3 = omp_get_wtime();

    VNT nz, matrix_size;
    _x->get_nnz(&nz);
    _matrix_csc->get_size(&matrix_size);

    const ENT *col_ptr = _matrix_csc->get_row_ptr(); // we assume csr of AT is equal to csc of A
    const VNT *row_ids = _matrix_csc->get_col_ids(); // we assume csr of AT is equal to csc of A


    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(number_of_threads); // Use 1 threads for all consecutive parallel regions

    double t4 = omp_get_wtime();

    matrix_prop = t4 - t3;


    t3 = omp_get_wtime();
    vector<vector<int>> Boffset = estimate_buckets(_matrix_csc, _x, _number_of_buckets, number_of_threads);
    t4 = omp_get_wtime();

    estimating_buckets = t4 - t3;
    // The point of function estimate_buckets is to fill the matrix Boffset in which
    // Boffset[i][j] means how many insertions the i-th thread will make in the j-th bucket

    double sub_time1, sub_time2;

    t3 = omp_get_wtime();

    double t5 = omp_get_wtime();
    // We need to fill the matrix Boffset in order to make synchronization free insertions in step 1

//    vector<int> bucket_amount(_number_of_buckets); // Stores how many insertions will be made in i-th bucket
//    vector<vector<int>> offset_(_number_of_buckets, vector<int>(number_of_threads));

    double t6 = omp_get_wtime();

    sub_time1 = t6 - t5;

    t5 = omp_get_wtime();

    for (int bucket_number = 0; bucket_number < _number_of_buckets; bucket_number++) {
        for (int thread_number = 0; thread_number < number_of_threads; thread_number++) {
            if (thread_number)
                offset_[bucket_number][thread_number] = offset_[bucket_number][thread_number - 1] + Boffset[thread_number - 1][bucket_number];
            bucket_amount[bucket_number] += Boffset[thread_number][bucket_number];
        }
    }

    t6 = omp_get_wtime();

    sub_time2 = t6 - t5;

    // Step 1. Filling buckets.
    // Every bucket gets it's own set of rows


    t5 = omp_get_wtime();

    /*
    vector<vector<bucket>> buckets(_number_of_buckets);
    for (int i = 0; i < _number_of_buckets; i++) {
        buckets[i] = vector<bucket>(bucket_amount[i]);
    }
    */
    t6 = omp_get_wtime();

    t4 = omp_get_wtime();
    preparing_for_filling_buckets = t4 - t3;

    t3 = omp_get_wtime();

    #pragma omp parallel
    {
        int cur_thread_number = omp_get_thread_num();
        vector<int> insertions(_number_of_buckets);
        #pragma omp for schedule(static)
        for (int i = 0; i < nz; i++) { // Going through all non-zero elements of vector x
            // We only need matrix's columns which numbers are equal to indices of non-zero elements of the vector x
            int vector_index = _x->get_ids()[i]; // Index of non-zero element in vector x
            for (int j = col_ptr[vector_index]; j < col_ptr[vector_index + 1]; j++) {
                // Going through all the elements of the vector_index-th column
                // Essentially we are doing multiplication here
                float mul = _matrix_csc->get_vals()[j] * _x->get_vals()[i]; // mul - is a product of matrix and vector non-zero elements.
                int bucket_index = (row_ids[j] * _number_of_buckets) / matrix_size;
                // bucket's index depends on the row number
                // Implementing synchronization free insertion below
                // Boffset[i][j] - amount, i - thread, j - bucket
                // offset - how many elements have been inserted in the bucket by other threads
                buckets[bucket_index][offset_[bucket_index][cur_thread_number] + (insertions[bucket_index])++] = {row_ids[j], mul}; // insertion
            }
        }
    }

    t4 = omp_get_wtime();

    filling_buckets = t4 - t3;

    t3 = omp_get_wtime();

    //vector<float> SPA(matrix_size); // SPA -  a sparse accumulator
    // The SPA vector is essentially a final vector-answer, but it is dense and we will have to transform it in sparse vector
    //vector<int> offset(_number_of_buckets);

    T *y_vals = (T *)malloc(sizeof(T) * matrix_size);
    VNT *y_ids = (VNT *)malloc(sizeof(VNT) * matrix_size);

    t4 = omp_get_wtime();

    matrix_prop += t4 - t3;

    t3 = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int number_of_bucket = 0; number_of_bucket < _number_of_buckets; number_of_bucket++) { // Going through all buckets
        vector<int> uind; // uing - unique indices in the number_of_bucket-th bucket

        // Step 2. Merging entries in each bucket.
        for (int i = 0; i < buckets[number_of_bucket].size(); i++) {
            int row = buckets[number_of_bucket][i].row;
            SPA[row] = INF; // Initializing with a value of INF
            // in order to know whether a row have been to the uind vector or not
        }
        // Accumulating values in a row and getting unique indices
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
            offset[number_of_bucket] += offset[number_of_bucket - 1] + bucket_amount[number_of_bucket - 1];
        // offset is needed to properly fill the final vector

        // !CHECK!
        for (int i = 0; i < uind.size(); i++) { // filling the final vector
            int ind = uind[i];
            float value = SPA[ind];
            int off = offset[number_of_bucket];
            _y->set_element(SPA[ind], ind);
        }
        // !CHECK!
    }

    t4 = omp_get_wtime();

    merging_entries = t4 - t3;

    double t2 = omp_get_wtime();

    overall_time = t2 - t1;

    printf("\033[0;31m");
    printf("SpMSpV time: %lf seconds.\n", overall_time);
    printf("\033[0m");
    printf("\t- Getting matrix properties and allocating memory: %.1lf %%\n", matrix_prop / overall_time * 100.0);
    printf("\t- Estimating buckets: %.1lf %%\n", estimating_buckets / overall_time * 100.0);
    printf("\t- Preparing for filling buckets: %.1lf %%\n", preparing_for_filling_buckets / overall_time * 100.0);
    printf("\t\t- Allocating memory: %.1lf %%\n", sub_time1 / preparing_for_filling_buckets * 100.0);
    printf("\t\t- double for: %.1lf %%\n", sub_time2 / preparing_for_filling_buckets * 100.0);
    printf("\t- Filling buckets: %.1lf %%\n", filling_buckets / overall_time * 100.0);
    printf("\t- Merging entries: %.1lf %%\n", merging_entries / overall_time * 100.0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////