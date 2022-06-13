#pragma once

#define VNT Index
#define ENT Index

#define INF 1000000000

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct bucket {
    VNT row;
    T val;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// nt - number of threads
// nb - number of buskets
template <typename T>
void estimate_buckets(const MatrixCSR<T> *matrix, const SparseVector<T> *x, vector<vector<int>> &Boffset, int nb, int nt)
{
    LOG_TRACE("Running estimate_buckets")

    // This function is essential in implementing synchronization free insertion
    VNT nz = x->get_nvals();
    VNT matrix_size = matrix->get_num_rows();

    const ENT *col_ptr = matrix->get_row_ptr(); // we assume csr of AT is equal to csc of A
    const VNT *row_ids = matrix->get_col_ids(); // we assume csr of AT is equal to csc of A

    const Index *x_ids = x->get_ids();
    int cnt = 0;

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int t = 0; t < nt; t++) {
            int offset_ = t * nz / nt; // Every thread has it's own piece of vector x
            for (int j = offset_; j < nz; j++) {
                Index vector_index = x_ids[j]; // index of j-th non-zero element in vector x
                Index iter_count = col_ptr[vector_index + 1] - col_ptr[vector_index];
                for (int i = col_ptr[vector_index]; i < col_ptr[vector_index + 1]; i++) {
                    // bucket index depends on the row of an element
                    int bucket_index = row_ids[i] * nb / matrix_size;
                    Boffset[t][bucket_index] += 1;
                }
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename A, typename Y, typename X, typename SemiringT, typename BinaryOpTAccum>
void spmspv_buckets(const MatrixCSR<A> *_matrix_csc,
                    const SparseVector <X> *_x,
                    DenseVector<Y> *_y,
                    int _number_of_buckets,
                    Workspace *_workspace,
                    BinaryOpTAccum _accum,
                    SemiringT op)
{
    LOG_TRACE("Running spmspv_buckets")

    auto add_op = extractAdd(op);
    auto mul_op = extractMul(op);
    auto identity_val = op.identity();

    double merging_entries, estimating_buckets, filling_buckets, overall_time, matrix_prop, preparing_for_filling_buckets, alloc_time;

    VNT nz = _x->get_nvals();
    VNT matrix_size = _matrix_csc->get_num_rows();

    const ENT *col_ptr = _matrix_csc->get_row_ptr(); // we assume csr of AT is equal to csc of A
    const VNT *row_ids = _matrix_csc->get_col_ids(); // we assume csr of AT is equal to csc of A

    int number_of_threads = 64;

    vector<vector<int>> Boffset(number_of_threads, vector<int>(_number_of_buckets));
    estimate_buckets(_matrix_csc, _x, Boffset, _number_of_buckets, number_of_threads);
    // The point of function estimate_buckets is to fill the matrix Boffset in which
    // Boffset[i][j] means how many insertions the i-th thread will make in the j-th bucket

    double sub_time1, sub_time2;

    // We need to fill the matrix Boffset in order to make synchronization free insertions in step 1
    double a1 = omp_get_wtime();

    vector<int> bucket_amount(_number_of_buckets); // Stores how many insertions will be made in i-th bucket
    vector<vector<int>> offset_(_number_of_buckets, vector<int>(number_of_threads));

    for (int bucket_number = 0; bucket_number < _number_of_buckets; bucket_number++) {
        for (int thread_number = 0; thread_number < number_of_threads; thread_number++) {
            if (thread_number)
                offset_[bucket_number][thread_number] = offset_[bucket_number][thread_number - 1] + Boffset[thread_number - 1][bucket_number];
            bucket_amount[bucket_number] += Boffset[thread_number][bucket_number];
        }
    }

    vector<vector<bucket<Y>>> buckets(_number_of_buckets);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < _number_of_buckets; i++) {
        buckets[i] = vector<bucket<Y>>(bucket_amount[i]);
        buckets[i].resize(bucket_amount[i]);
    }

    const Index *x_ids = _x->get_ids();
    const A *vals = _matrix_csc->get_vals();
    const X *x_vals = _x->get_vals();

    // Step 1. Filling buckets.
    // Every bucket gets it's own set of rows

    #pragma omp parallel
    {
        int cur_thread_number = omp_get_thread_num();
        vector<int> insertions(_number_of_buckets);
        #pragma omp for schedule(static)
        for (int i = 0; i < nz; i++) { // Going through all non-zero elements of vector x
            // We only need matrix's columns which numbers are equal to indices of non-zero elements of the vector x
            Index vector_index = x_ids[i]; // Index of non-zero element in vector x

            for (int j = col_ptr[vector_index]; j < col_ptr[vector_index + 1]; j++) {
                // Going through all the elements of the vector_index-th column
                // Essentially we are doing multiplication here
                Y mul = mul_op(vals[j], x_vals[i]); // mul - is a product of matrix and vector non-zero elements.

                int bucket_index = (row_ids[j] * _number_of_buckets) / matrix_size;
                // bucket's index depends on the row number
                // Implementing synchronization free insertion below
                // Boffset[i][j] - amount, i - thread, j - bucket
                // offset - how many elements have been inserted in the bucket by other threads
                buckets[bucket_index][offset_[bucket_index][cur_thread_number] + (insertions[bucket_index])++] = {row_ids[j], mul}; // insertion
            }
        }
    }

    vector<int> offset(_number_of_buckets);

    #pragma omp parallel
    {
        map<VNT, A>SPA;
        #pragma omp for schedule(dynamic, 1)
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
                    SPA[row] = add_op(SPA[row], val);
                }
            }
            if (number_of_bucket) // if not the first bucket
            {
                offset[number_of_bucket] += offset[number_of_bucket - 1] + bucket_amount[number_of_bucket - 1];
            }
            // offset is needed to properly fill the final vector

            for (int i = 0; i < uind.size(); i++) { // filling the final vector
                int ind = uind[i];
                Y value = SPA[ind];
                int off = offset[number_of_bucket];
                _y->set_element(SPA[ind], ind);
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
