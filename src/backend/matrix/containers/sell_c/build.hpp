#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixSellC<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz, int _socket)
{
    cout << "in build" << endl;

    int *row;
    int *col_unsorted;
    T *val_unsorted;

    //permute the col and val according to row
    ENT* perm = new ENT[nz];
    for(ENT idx = 0; idx < nz; ++idx)
    {
        perm[idx] = idx;
    }

    sort_perm(_row_ids, perm, nz);

    col = new VNT[nz];
    val = new T[nz];

    for(ENT idx = 0; idx < nz; ++idx)
    {
        col[idx] = _col_ids[perm[idx]];
        val[idx] = _vals[perm[idx]];
    }

    rowPtr = new VNT[size+1];

    nnzPerRow = new VNT[size];
    for(VNT i = 0; i < size; ++i)
    {
        nnzPerRow[i] = 0;
    }

    //count nnz per row
    for(ENT i = 0; i < nz; ++i)
    {
        ++nnzPerRow[row[i]];
    }

    rowPtr[0] = 0;
    for(VNT i = 0; i < size; ++i)
    {
        rowPtr[i+1] = rowPtr[i]+nnzPerRow[i];
    }

    delete[] row;
    delete[] perm;

    NUMA_init();

    cout << "build done" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixSellC<T>::NUMA_init()
{
    T* newVal = new T[nz];
    VNT* newCol = new VNT[nz];
    ENT* newRowPtr = new ENT[size+1];

    //NUMA init
    #pragma omp parallel for schedule(static)
    for(int row=0; row<size+1; ++row)
    {
        newRowPtr[row] = rowPtr[row];
    }
    #pragma omp parallel for schedule(static)
    for(int row=0; row<size; ++row)
    {
        for(int idx=newRowPtr[row]; idx<newRowPtr[row+1]; ++idx)
        {
            //newVal[idx] = val[idx];
            newCol[idx] = col[idx];
            newVal[idx] = val[idx];
        }
    }

    //free old _perm_utations
    delete[] val;
    delete[] rowPtr;
    delete[] col;

    val = newVal;
    rowPtr = newRowPtr;
    col = newCol;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
