#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace backend {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixSellC<T>::print_connections(VNT _row)
{
    VNT max_dif = 0;
    cout << "vert " << _row << " is connected to : ";
    for(ENT i = row_ptr[_row]; i < row_ptr[_row + 1]; i++)
    {
        if(std::abs(i - col_ids[i]) > max_dif)
            max_dif = std::abs(i - col_ids[i]);
        cout << col_ids[i] << " ";
    }
    cout << endl;
    cout << "max diff: " << max_dif << ", " << (double)max_dif/size << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixSellC<T>::build(VNT _nrows,
                           VNT _ncols,
                           ENT _nnz,
                           const ENT *_row_ptr,
                           const VNT *_col_ids,
                           const T *_vals,
                           int _target_socket)
{
    nnz = _nnz;
    size = _nrows;

    MemoryAPI::allocate_array(&row_ptr, size + 1);
    MemoryAPI::allocate_array(&nnz_per_row, size);
    MemoryAPI::allocate_array(&col_ids, nnz);
    MemoryAPI::allocate_array(&vals, nnz);

    MemoryAPI::copy(row_ptr, _row_ptr, size + 1);
    MemoryAPI::copy(col_ids, _col_ids, nnz);
    MemoryAPI::copy(vals, _vals, nnz);

    #pragma omp parallel for
    for(int i = 0; i < size; i++)
    {
        nnz_per_row[i] = row_ptr[i + 1] - row_ptr[i];
    }

    VNT max_reordered_vertices = (LLC_CACHE_SIZE/4) / sizeof(T);
    sigma = size;
    while(sigma >= max_reordered_vertices)
    {
        sigma /= 2;
    }

    construct_sell_c_sigma(VECTOR_LENGTH, sigma);

    print_stats();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixSellC<T>::construct_sell_c_sigma(VNT chunkHeight, VNT _sigma, VNT pad)
{
    C = chunkHeight;
    P = pad;

    MemoryAPI::allocate_array(&sigmaPerm, size);
    MemoryAPI::allocate_array(&sigmaInvPerm, size);
    sigma = sigma;

    VNT nSigmaChunks = (VNT)(size/(double)sigma); // number of sigma chinks // TODO - + 1
    if(sigma > 1) // if sigma == 1 - no rows reordering is performed
    {
        for(VNT i=0; i<size; ++i)
        {
            sigmaPerm[i] = i;
        }

        for(VNT sigmaChunk=0; sigmaChunk<nSigmaChunks; ++sigmaChunk)
        {
            VNT *perm_begin = &(sigmaPerm[sigmaChunk*sigma]);
            sort_perm(nnz_per_row, perm_begin, sigma, true); // do sorting of rows in each sigma chunk
        }

        VNT restSigmaChunk = size%sigma; // process remider (if size % sigma != 0)
        if(restSigmaChunk > C)
        {
            VNT *perm_begin = &(sigmaPerm[nSigmaChunks*sigma]);
            sort_perm(nnz_per_row, perm_begin, restSigmaChunk, true); // do sorting of last chunk
        }

        for(VNT i=0; i<size; ++i)
        {
            sigmaInvPerm[sigmaPerm[i]] = i;
        }

        permute(sigmaPerm, sigmaInvPerm);
    }

    nchunks = (VNT)(size/(double)C);
    if(size%C > 0)
    {
        nchunks += 1;
    }

    chunkLen = new VNT[nchunks];
    problematic_chunk = new int[nchunks];
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
        ENT csr_nnz_in_chunk = 0;
        ENT maxRowLen = 0;
        for(VNT rowInChunk = 0; rowInChunk < C; ++rowInChunk)
        {
            VNT row = chunk*C + rowInChunk;
            if(row<size)
            {
                maxRowLen = std::max(maxRowLen, row_ptr[row+1]-row_ptr[row]);
                csr_nnz_in_chunk += row_ptr[row+1]-row_ptr[row];
            }
        }
        //pad it to be multiple of P
        if((maxRowLen%P) != 0)
        {
            maxRowLen = ((VNT)(maxRowLen/(double)P)+1)*P;
        }

        if((maxRowLen*C >= 2*csr_nnz_in_chunk) && (csr_nnz_in_chunk > 1024))
        {
            cout << "problems is chunk " << chunk << " / " << nchunks << " : " << maxRowLen*C << " made from "
                 << csr_nnz_in_chunk << " nnz, " << ((double)maxRowLen)*C/csr_nnz_in_chunk << " times larger" << endl;

            chunkLen[chunk] = 0;
            nnzSellC += 0;
            problematic_chunk[chunk] = 1;
        }
        else
        {
            chunkLen[chunk] = maxRowLen;
            nnzSellC += maxRowLen*C;
            problematic_chunk[chunk] = 0;
        }
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
                    colSellC[chunkPtr[chunk]+idx*C+rowInChunk] = chunk*C + rowInChunk;
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
        if(!problematic_chunk[chunk])
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
void MatrixSellC<T>::permute(VNT *perm, VNT* invPerm)
{
    T* newVal = new T[nnz];
    ENT* newRowPtr = new ENT[size+1];
    VNT* newCol = new VNT[nnz];

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
        nnz_per_row[row] = (row_ptr[permRow+1]-row_ptr[permRow]);
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
            newCol[permIdx] = col_ids[idx];//invPerm[col_ids[idx]];
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
