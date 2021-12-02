#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixSellC<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz, int _socket)
{
    /*nz = _nz;
    size = _size;

    int *col_unsorted;
    T *vals_unsorted;

    //permute the col and val according to row
    ENT* perm = new ENT[nz];
    for(ENT idx = 0; idx < nz; ++idx)
    {
        perm[idx] = idx;
    }

    sort_perm(_row_ids, perm, nz);

    col_ids = new VNT[nz];
    vals = new T[nz];

    for(ENT idx = 0; idx < nz; ++idx)
    {
        col_ids[idx] = _col_ids[perm[idx]];
        vals[idx] = _vals[perm[idx]];
    }

    row_ptr = new VNT[size+1];

    nz_per_row = new VNT[size];
    for(VNT i = 0; i < size; ++i)
    {
        nz_per_row[i] = 0;
    }

    //count nnz per row
    for(ENT i = 0; i < nz; ++i)
    {
        ++nz_per_row[_row_ids[i]];
    }

    row_ptr[0] = 0;
    for(VNT i = 0; i < size; ++i)
    {
        row_ptr[i+1] = row_ptr[i]+nz_per_row[i];
    }

    delete[] perm;*/

    generateHPCG(128, 128, 128);

    NUMA_init();

    construct_sell_c_sigma(VECTOR_LENGTH, 1);

    cout << "matrix stats: " << size << " vertices" << endl;
    cout << "nz: " << nz << " edges" << endl;
    cout << "cache: " << size * sizeof(float) / 1e6 << " MB indirectly array" << endl;
    ENT cell_c_nz = 0;
    for(VNT chunk=0; chunk<nchunks; ++chunk)
    {
        cell_c_nz += chunkLen[chunk]*C;
    }
    cout << "cellc nz: " << cell_c_nz << " " << (double)cell_c_nz/nz << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixSellC<T>::NUMA_init()
{
    T* new_vals = new T[nz];
    VNT* new_col_is = new VNT[nz];
    ENT* new_row_ptrs = new ENT[size+1];

    //NUMA init
    #pragma omp parallel for schedule(static)
    for(VNT row = 0; row< size + 1; ++row)
    {
        new_row_ptrs[row] = row_ptr[row];
    }
    #pragma omp parallel for schedule(static)
    for(int row=0; row<size; ++row)
    {
        for(int idx=new_row_ptrs[row]; idx<new_row_ptrs[row+1]; ++idx)
        {
            new_col_is[idx] = col_ids[idx];
            new_vals[idx] = vals[idx];
        }
    }

    //free old _perm_utations
    delete[] vals;
    delete[] row_ptr;
    delete[] col_ids;

    vals = new_vals;
    row_ptr = new_row_ptrs;
    col_ids = new_col_is;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixSellC<T>::construct_sell_c_sigma(VNT chunkHeight, VNT sigma, VNT pad)
{
    C = chunkHeight;
    P = pad;

    VNT nSigmaChunks = (VNT)(size/(double)sigma);
    if(sigma > 1)
    {
        VNT *sigmaPerm = new VNT[size];
        for(VNT i=0; i<size; ++i)
        {
            sigmaPerm[i] = i;
        }

        for(VNT sigmaChunk=0; sigmaChunk<nSigmaChunks; ++sigmaChunk)
        {
            VNT *perm_begin = &(sigmaPerm[sigmaChunk*sigma]);
            sort_perm(nz_per_row, perm_begin, sigma);
        }

        VNT restSigmaChunk = size%sigma;
        if(restSigmaChunk > C)
        {
            VNT *perm_begin = &(sigmaPerm[nSigmaChunks*sigma]);
            sort_perm(nz_per_row, perm_begin, restSigmaChunk);
        }

        VNT *sigmaInvPerm = new VNT[size];

        for(VNT i=0; i<size; ++i)
        {
            sigmaInvPerm[sigmaPerm[i]] = i;
        }

        permute(sigmaPerm, sigmaInvPerm);

        delete[] sigmaPerm;
        delete[] sigmaInvPerm;
    }

    nchunks = (VNT)(size/(double)C);
    if(size%C > 0)
    {
        nchunks += 1;
    }

    chunkLen = new VNT[nchunks];
    chunkPtr = new ENT[nchunks+1];

    #pragma omp parallel for schedule(static)
    for(VNT i=0; i < nchunks; ++i)
    {
        chunkLen[i] = 0;
        chunkPtr[i] = 0;
    }

    nnzSellC = 0;
    //find chunkLen
    for(VNT chunk = 0; chunk < nchunks; ++chunk)
    {
        int maxRowLen = 0;
        for(VNT rowInChunk = 0; rowInChunk < C; ++rowInChunk)
        {
            VNT row = chunk*C + rowInChunk;
            if(row<size)
            {
                maxRowLen = std::max(maxRowLen, row_ptr[row+1]-row_ptr[row]);
            }
        }
        //pad it to be multiple of P
        if((maxRowLen%P) != 0)
        {
            maxRowLen = ((VNT)(maxRowLen/(double)P)+1)*P;
        }
        chunkLen[chunk] = maxRowLen;
        nnzSellC += maxRowLen*C;
    }

    colSellC = new VNT[nnzSellC];
    valSellC = new T[nnzSellC];

    #pragma omp parallel for schedule(static)
    for(VNT i = 0; i <= (nchunks); ++i)
    {
        chunkPtr[i] = 0;
    }

    for(VNT i = 0; i< (nchunks); ++i)
    {
        chunkPtr[i+1] = chunkPtr[i] + C*chunkLen[i];
    }


    #pragma omp parallel for schedule(static)
    for(VNT chunk=0; chunk<nchunks; ++chunk)
    {
        for(VNT rowInChunk=0; rowInChunk<C; ++rowInChunk)
        {
            for(ENT idx=0; idx<chunkLen[chunk]; ++idx)
            {
                if(C == size)
                {
                    colSellC[chunkPtr[chunk]+idx*C+rowInChunk] = chunk*C + rowInChunk;//ELLPACK of Fujitsu needs it this way (the rowIndex)
                }
                else
                {
                    colSellC[chunkPtr[chunk]+idx*C+rowInChunk] = 0;
                }
                valSellC[chunkPtr[chunk]+idx*C+rowInChunk] = 0;
            }
        }
    }


    for(VNT chunk=0; chunk<nchunks; ++chunk)
    {
        for(VNT rowInChunk=0; rowInChunk<C; ++rowInChunk)
        {
            VNT row = chunk*C + rowInChunk;
            if(row < size)
            {
                for(ENT idx=row_ptr[row],j=0; idx<row_ptr[row+1]; ++idx,++j)
                {
                    valSellC[chunkPtr[chunk]+j*C+rowInChunk] = vals[idx];
                    colSellC[chunkPtr[chunk]+j*C+rowInChunk] = col_ids[idx];
                }
            }
        }
    }

    std::vector<double> strideAvg(nchunks*C, 0);
    double strideAvg_total = 0;
    for(int chunk=0; chunk<nchunks; ++chunk)
    {
        for(int rowInChunk=0; rowInChunk<C; ++rowInChunk)
        {
            for(int idx=1; idx<chunkLen[chunk]; ++idx)
            {
                strideAvg[chunk*C+rowInChunk] += std::abs(colSellC[chunkPtr[chunk]+idx*C+rowInChunk] - colSellC[chunkPtr[chunk]+(idx-1)*C+rowInChunk]);
            }

            strideAvg[chunk*C+rowInChunk] = strideAvg[chunk*C+rowInChunk]/(double)chunkLen[chunk];
            strideAvg_total += strideAvg[chunk*C+rowInChunk];
        }
    }
    strideAvg_total = strideAvg_total/((double)nchunks*C);

    printf("Average stride length = %lf\n", strideAvg_total);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixSellC<T>::permute(VNT *perm, VNT*  invPerm)
{
    T* newVal = new T[nz];
    ENT* newRowPtr = new ENT[size+1];
    VNT* newCol = new VNT[nz];

    newRowPtr[0] = 0;

    //NUMA init
    #pragma omp parallel for schedule(static)
    for(VNT row=0; row<size; ++row)
    {
        newRowPtr[row+1] = 0;
    }

    //first find newRowPtr; therefore we can do proper NUMA init
    VNT permIdx=0;
    printf("size = %d\n", size);
    for(VNT row=0; row<size; ++row)
    {
        //row permutation
        VNT permRow = perm[row];
        nz_per_row[row] = (row_ptr[permRow+1]-row_ptr[permRow]);
        for(ENT idx=row_ptr[permRow]; idx<row_ptr[permRow+1]; ++idx)
        {
            ++permIdx;
        }
        newRowPtr[row+1] = permIdx;
    }

    //with NUMA init
    #pragma omp parallel for schedule(static)
    for(VNT row=0; row<size; ++row)
    {
        //row permutation
        VNT permRow = perm[row];
        for(ENT permIdx=newRowPtr[row],idx=row_ptr[permRow]; permIdx<newRowPtr[row+1]; ++idx,++permIdx)
        {
            //permute column-wise also
            newVal[permIdx] = vals[idx];
            newCol[permIdx] = invPerm[col_ids[idx]];
        }
    }

    //free old permutations
    delete[] vals;
    delete[] row_ptr;
    delete[] col_ids;

    vals = newVal;
    row_ptr = newRowPtr;
    col_ids = newCol;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}
