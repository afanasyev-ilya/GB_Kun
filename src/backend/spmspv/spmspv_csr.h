#pragma once

struct bucket {
    int row;
    int val;
};

#define INF 1000000000

#include <vector>


// nt - number of threads
// nb - number of buskets
template <typename T>
vector<vector<int>> estimate_buckets(MatrixCSR<T> &matrix, SparseVector<T> &x, int nb, int nt)
{
    int nz = x.nz;
    vector<vector<int>> Boffset(nt, vector<int>(nb));
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < nt; t++) {
        int offset = t * nz / nt; // Every thread has it's own piece of vector x
        for (int j = offset; j < offset + nt && j < nz; j++) {
            int vector_index = x.ids[j]; // index of j-th non-zero element in vector x
            for (int i = matrix.col_indx[vector_index]; i < matrix.col_indx[vector_index] < matrix.nx && matrix.col_indx[vector_index + 1]; i++) {
                int bucket_index = i * nb / matrix.size;
                Boffset[t][bucket_index] += 1;
            }
        }
    }
}



template <typename T>
void SpMSpV(MatrixCSR<T> &matrix_csr,
          SparseVector<T> &x,
          SparseVector<T> &y,
          int number_of_buckets)
{
    vector<vector<int>> Boffset = estimate_buckets(matrix, x, number_of_buckets);

    MatrixCSR<T> matrix = matrix_csr.transposed_data;

    // Step 1. Filling buckets.
    // Every bucket gets it's own set of rows
    #pragma omp parallel for schedule(static)
    vector<vector<bucket>> buckets(number_of_buckets);
    for (int i = 0; i < x.nz; i++) { // Going through all non-zero elements of vector x
        // We only need matrix's columns which numbers are equal to indexes of non-zero elements of the vector
        int vector_index = x.ids[i]; // Index of non-zero element
        for (int j = matrix.col_indx[vector_index]; j < matrix.col_indx[vector_index]; j++) {
            // Going through the column with index equal to vector_index
            int mul = matrix.vals[j] * x.vals[vector_index]; // mul - is a product of matrix and vector non-zero elements.
            // Essentially we are doing multiplication here
            int bucket_index = (matrix.rows[j] * number_of_buckets) / matrix.size;
            // bucket's index depends on the row number
            buckets[bucket_index].push_back({matrix.rows[j], mul}); // Adding row number and product of elements in the becket
        }
    }


    vector<int> SPA(matrix.size);
    vector<int> offset(number_of_buckets);

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < number_of_buckets; k++) {
        vector<int> uind;
        // Step 2. Merging entries in each bucket
        for (int i = 0; i < buckets[k].size(); i++) {
            int row = buckets[k][i].row;
            SPA[row] = INF;
        }
        for (int i = 0; i < buckets[k].size(); i++) {
            int row = buckets[k][i].row;
            int val = buckets[k][i].val;
            if (SPA[row] == INF) {
                uind.push_back(row);
                SPA[row] = val;
            } else {
                SPA[row] += val;
            }
        }
        if (k)
            offset[k] += offset[k - 1];
        for (int i = 0; i < uind.size(); i++) {
            int ind = uind[i];
            y.vals[offset[k] + i] = SPA[ind];
            y.ids[offset[k] + i] = ind;
        }
    }

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////