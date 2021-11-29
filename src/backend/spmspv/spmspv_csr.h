#pragma once

struct bucket {
    int row;
    int val;
};

#define INF 1000000000

template <typename T>
vector<vector<int>> estimate_buckets(MatrixCST<T> &matrix, SparseVector<T> &x, int nb, int nt)
{
    int nz = x.nz;
    vector<vector<int>> Boffset(nt, vector<int>(nb));
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < nt; t++) {
        int offset = t * nz / nt;
        for (int j = offset; j < offset + nt && j < nz; j++) {
            int vector_index = x.ids[j];
            for (int i = matrix.col_indx[vector_index]; i < matrix.col_indx[vector_index]; i++) {
                int bucket_index = i * nb / matrix.size;
                Boffset[t][bucket_index] += 1;
            }
        }
    }
}



template <typename T>
void SpMSpV(MatrixCSR<T> &matrix,
          SparseVector<T> &x,
          SparseVector<T> &y,
          number_of_buckets)
{
    vector<vector<int>> Boffset = estimate_buckets(matrix, x, number_of_buckets);

    // Step 1. Filling buckets.
    #pragma omp parallel for schedule(static)
    vector<vector<bucket>> buckets(number_of_buckets);
    for (int i = 0; i < nz; i++) {
        int vector_index = x.ids[i];
        for (int j = matrix.col_indx[vector_index]; j < matrix.col_indx[vector_index]; j++) {
            int mul = matrix.vals[j] * x.vals[vector_index];
            int bucket_index = (i * number_of_buckets) / matrix.size;
            buckets[bucket_index].push_back({matrix.rows[j], mul});
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