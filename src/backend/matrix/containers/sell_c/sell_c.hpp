#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> void sort_perm(T *arr, ENT *perm, ENT len, bool rev=false)
{
    if(rev == false)
    {
        std::stable_sort(perm+0, perm+len, [&](const ENT& a, const ENT& b) {return (arr[a] < arr[b]);});
    }
    else
    {
        std::stable_sort(perm+0, perm+len, [&](const ENT& a, const ENT& b) {return (arr[a] > arr[b]); });
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
MatrixSellC<T>::MatrixSellC():size(0), nnz(0), vals(NULL), row_ptr(NULL), col_ids(NULL), chunkLen(NULL), chunkPtr(NULL), colSellC(NULL), valSellC(NULL), unrollFac(1), C(1)
{
    #pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    rcmPerm = NULL;
    rcmInvPerm = NULL;
    nnz_per_row = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
MatrixSellC<T>::~MatrixSellC()
{
    if(vals)
        delete[] vals;

    if(row_ptr)
        delete[] row_ptr;

    if(col_ids)
        delete[] col_ids;

    if(chunkLen)
        delete[] chunkLen;

    if(chunkPtr)
        delete[] chunkPtr;

    if(colSellC)
        delete[] colSellC;

    if(valSellC)
        delete[] valSellC;

    if(nnz_per_row)
    {
        delete[] nnz_per_row;
    }

    MemoryAPI::free_array(sigmaPerm);
    MemoryAPI::free_array(sigmaInvPerm);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
T MatrixSellC<T>::get(VNT _row, VNT _col) const
{
    for(ENT i = row_ptr[_row]; i < row_ptr[_row + 1]; i++)
    {
        if(col_ids[i] == _col)
            return vals[i];
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixSellC<T>::print() const
{
    for(VNT row = 0; row < size; row++)
    {
        cout << row + 1 << ") ";
        for(VNT col = 0; col < size; col++)
        {
            cout << get(row, col) << " ";
        }
        cout << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixSellC<T>::print_stats()
{
    ENT cell_c_nnz = 0;
    for(VNT chunk=0; chunk<nchunks; ++chunk)
    {
        cell_c_nnz += chunkLen[chunk]*C;
    }

    cout << endl << " -------------------- " << endl;
    cout << "SellC matrix stats" << endl;
    cout << "Num rows: " << size << " (vertices)" << endl;
    cout << "nnz: " << nnz << " (edges)" << endl;
    cout << "SellC nnz: " << cell_c_nnz << ", growing factor - " << (double)cell_c_nnz/nnz << endl;
    cout << "Cache size: " << size * sizeof(T) / 1e6 << " MB indirectly array" << endl;
    cout << " -------------------- " << endl << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}
