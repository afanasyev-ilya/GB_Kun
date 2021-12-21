#pragma once

struct bucket {
    int row;
    int val;
};

#define INF 1000000000

template <typename T>
bool check(MatrixCSR<T> &matrix_csr, SparseVector<T> &x, SparseVector<T> &y)
{

}

// nt - number of threads
// nb - number of buskets
template <typename T>
void estimate_buckets(MatrixCSR<T> &matrix, SparseVector<T> &x, int nb, int nt)
{
    // This function is essential in implementing synchronization free insertion
    int nz = x.nz;
    vector<vector<int>> Boffset(nt, vector<int>(nb)); // Boffset is described in the SpmSpv function
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < nt; t++) {
        int offset = t * nz / nt; // Every thread has it's own piece of vector x
        for (int j = offset; j < offset + nt && j < nz; j++) {
            int vector_index = x.ids[j]; // index of j-th non-zero element in vector x
            for (int i = matrix.col_indx[vector_index]; i < matrix.col_indx[vector_index + 1]; i++) {
                // bucket index depends on the row of an element
                int bucket_index = matrix.rows[i] * nb / matrix.size;
                Boffset[t][bucket_index] += 1;
            }
        }
    }
}


template <typename T>
void SpMSpV_csr(const MatrixCSR<T> *_matrix_csr,
            const SparseVector<T> *_x,
            SparseVector<T> *_y,
            int _number_of_buckets)
{
    _x->print();
    /* ????????????????? */ int number_of_threads = 64; /* ????????????????? */

    /*vector<vector<int>> Boffset = estimate_buckets(matrix_csr, x, number_of_buckets, number_of_threads);
    // The point of function estimate_buckets is to fill the matrix Boffset in which
    // Boffset[i][j] means how many insertions the i-th thread will make in the j-th bucket

    // We need to fill the matrix Boffset in order to make synchronization free insertions in step 1

    vector<int> bucket_amount(number_of_buckets); // Stores how many insertions will be made in i-th bucket
    for (int thread_number = 0; thread_number < number_of_threads; thread_number++) {
        for (int bucket_number = 0; bucket_number < number_of_buckets; bucket_number++) {
            bucket_amount[bucket_number] += Boffset[thread_number][bucket_number];
        }
    }

    MatrixCSR<T> matrix = matrix_csr.transposed_data;

    // Step 1. Filling buckets.
    // Every bucket gets it's own set of rows

    vector<vector<bucket>> buckets(number_of_buckets);

    for (int i = 0; i < number_of_buckets; i++) {
        bucket[i] = new vector<bucket>(bucket_amount[i]);
    }

    #pragma omp parallel
    {
        int number_of_insertions = 0;
        #pragma omp for schedule(static)
        for (int i = 0; i < x.nz; i++) { // Going through all non-zero elements of vector x
            // We only need matrix's columns which numbers are equal to indices of non-zero elements of the vector x
            int vector_index = x.ids[i]; // Index of non-zero element in vector x
            for (int j = matrix.col_indx[vector_index]; j < matrix.col_indx[vector_index + 1]; j++) {
                // Going through all the elements of the vector_index-th column
                int mul = matrix.vals[j] *
                          x.vals[vector_index]; // mul - is a product of matrix and vector non-zero elements.
                // Essentially we are doing multiplication here
                int bucket_index = (matrix.rows[j] * number_of_buckets) / matrix.size;
                // bucket's index depends on the row number

                // Implementing synchronization free insertion below
                int cur_thread_number = omp_get_thread_num();
                // Boffset[i][j] - amount, i - thread, j - bucket
                int offset = 0;
                for (int thread_number = 0; thread_number < i; thread_number++) {
                    offset += Boffset[thread_number][bucket_index];
                }
                // offset - how many elements have been inserted in the bucket by other threads
                buckets[offset + (number_of_insertions++)] = {matrix.rows[j], mul}; // insertion
            }
        }
    }


    vector<int> SPA(matrix.size); // SPA -  a sparse accumulator
    // The SPA vector is essentially a final vector-answer, but it is dense and we will have to transform it in sparse vector
    vector<int> offset(number_of_buckets);

    #pragma omp parallel for schedule(static)
    for (int number_of_bucket = 0; number_of_bucket < number_of_buckets; number_of_bucket++) { // Going through all buckets
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
            int val = buckets[number_of_bucket][i].val;
            if (SPA[row] == INF) {
                uind.push_back(row);
                SPA[row] = val;
            } else {
                SPA[row] += val;
            }
        }
        if (number_of_bucket) // if not the first bucket
            offset[number_of_bucket] += offset[number_of_bucket - 1];
        // offset is needed to properly fill the final vector

        // !CHECK!
        for (int i = 0; i < uind.size(); i++) { // filling the final vector
            int ind = uind[i];
            y.vals[offset[k] + i] = SPA[ind];
            y.ids[offset[k] + i] = ind;
        }
        // !CHECK!
    }*/
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////