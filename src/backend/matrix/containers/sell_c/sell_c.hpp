/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> void sort_perm(T *arr, int *perm, int len, bool rev=false)
{
    if(rev == false) {
        std::stable_sort(perm+0, perm+len, [&](const int& a, const int& b) {return (arr[a] < arr[b]);});
    } else {
        std::stable_sort(perm+0, perm+len, [&](const int& a, const int& b) {return (arr[a] > arr[b]); });
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//for debugging
template <typename T> void sort_perm_v(T *arr, int *perm, int len, bool rev=false)
{
    if(rev == false) {
        std::stable_sort(perm+0, perm+len, [&](const int& a, const int& b) {printf("comparing arr[%d] = %d, arr[%d] = %d\n", a, arr[a], b, arr[b]); return (arr[a] < arr[b]);});
    } else {
        std::stable_sort(perm+0, perm+len, [&](const int& a, const int& b) {printf("comparing arr[%d] = %d, arr[%d] = %d\n", a, arr[a], b, arr[b]); return (arr[a] > arr[b]); });
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
MatrixSellC<T>::MatrixSellC():nrows(0), nnz(0), val(NULL), rowPtr(NULL), col(NULL), chunkLen(NULL), chunkPtr(NULL), colSellC(NULL), valSellC(NULL), unrollFac(1), C(1)
{
    #pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    rcmPerm = NULL;
    rcmInvPerm = NULL;
    nnzPerRow = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
MatrixSellC<T>::~MatrixSellC()
{
    if(val)
        delete[] val;

    if(rowPtr)
        delete[] rowPtr;

    if(col)
        delete[] col;

    if(chunkLen)
        delete[] chunkLen;

    if(chunkPtr)
        delete[] chunkPtr;

    if(colSellC)
        delete[] colSellC;

    if(valSellC)
        delete[] valSellC;

    if(nnzPerRow)
    {
        delete[] nnzPerRow;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixSellC<T>::build(const VNT *_row_ids, const VNT *_col_ids, const T *_vals, VNT _size, ENT _nz, int _socket)
{
    cout << "in build" << endl;

    int ncols;
    int *row;
    int *col_unsorted;
    double *val_unsorted;

    //permute the col and val according to row
    int* perm = new int[nnz];
    for(int idx=0; idx<nnz; ++idx)
    {
        perm[idx] = idx;
    }

    sort_perm(row, perm, nnz);

    col = new int[nnz];
    val = new double[nnz];

    for(int idx=0; idx<nnz; ++idx)
    {
        col[idx] = col_unsorted[perm[idx]];
        val[idx] = val_unsorted[perm[idx]];
    }

    delete[] col_unsorted;
    delete[] val_unsorted;


    rowPtr = new int[nrows+1];

    nnzPerRow = new int[nrows];
    for(int i=0; i<nrows; ++i)
    {
        nnzPerRow[i] = 0;
    }

    //count nnz per row
    for(int i=0; i<nnz; ++i)
    {
        ++nnzPerRow[row[i]];
    }

    rowPtr[0] = 0;
    for(int i=0; i<nrows; ++i)
    {
        rowPtr[i+1] = rowPtr[i]+nnzPerRow[i];
    }

    delete[] row;
    delete[] perm;

    NUMA_init();
}

template<typename T>
void MatrixSellC<T>::constructSellCSigma(int chunkHeight, int sigma, int pad)
{
    C = chunkHeight;
    P = pad;

    int nSigmaChunks = (int)(nrows/(double)sigma);
    if(sigma > 1)
    {
        int *sigmaPerm = new int[nrows];
        for(int i=0; i<nrows; ++i)
        {
            sigmaPerm[i] = i;
        }

        for(int sigmaChunk=0; sigmaChunk<nSigmaChunks; ++sigmaChunk)
        {
            int *perm_begin = &(sigmaPerm[sigmaChunk*sigma]);
            sort_perm(nnzPerRow, perm_begin, sigma);
        }

        int restSigmaChunk = nrows%sigma;
        if(restSigmaChunk > C)
        {
            int *perm_begin = &(sigmaPerm[nSigmaChunks*sigma]);
            sort_perm(nnzPerRow, perm_begin, restSigmaChunk);
        }

        int *sigmaInvPerm = new int[nrows];

        for(int i=0; i<nrows; ++i)
        {
            sigmaInvPerm[sigmaPerm[i]] = i;
        }

        permute(sigmaPerm, sigmaInvPerm);

        delete[] sigmaPerm;
        delete[] sigmaInvPerm;
    }

    nchunks = (int)(nrows/(double)C);
    if(nrows%C > 0)
    {
        nchunks += 1;
    }

    chunkLen = new int[nchunks];
    chunkPtr = new int[nchunks+1];

    #pragma omp parallel for schedule(static)
    for(int i=0; i<nchunks; ++i)
    {
        chunkLen[i] = 0;
        chunkPtr[i] = 0;
    }

    nnzSellC = 0;
    //find chunkLen
    for(int chunk=0; chunk<nchunks; ++chunk)
    {
        int maxRowLen = 0;
        for(int rowInChunk=0; rowInChunk<C; ++rowInChunk)
        {
            int row = chunk*C + rowInChunk;
            if(row<nrows)
            {
                maxRowLen = std::max(maxRowLen, rowPtr[row+1]-rowPtr[row]);
            }
        }
        //pad it to be multiple of P
        if((maxRowLen%P) != 0)
        {
            maxRowLen = ((int)(maxRowLen/(double)P)+1)*P;
        }
        chunkLen[chunk] = maxRowLen;
        nnzSellC += maxRowLen*C;
    }

    colSellC = new int[nnzSellC];
    valSellC = new double[nnzSellC];


    #pragma omp parallel for schedule(static)
    for(int i=0; i<=(nchunks); ++i)
    {
        chunkPtr[i] = 0;
    }

    for(int i=0; i<(nchunks); ++i)
    {
        chunkPtr[i+1] = chunkPtr[i] + C*chunkLen[i];
    }


    #pragma omp parallel for schedule(static)
    for(int chunk=0; chunk<nchunks; ++chunk)
    {
        for(int rowInChunk=0; rowInChunk<C; ++rowInChunk)
        {
            for(int idx=0; idx<chunkLen[chunk]; ++idx)
            {
                if(C == nrows)
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


    for(int chunk=0; chunk<nchunks; ++chunk)
    {
        for(int rowInChunk=0; rowInChunk<C; ++rowInChunk)
        {
            int row = chunk*C + rowInChunk;
            if(row<nrows)
            {
                for(int idx=rowPtr[row],j=0; idx<rowPtr[row+1]; ++idx,++j)
                {
                    valSellC[chunkPtr[chunk]+j*C+rowInChunk] = val[idx];
                    colSellC[chunkPtr[chunk]+j*C+rowInChunk] = col[idx];
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

    printf("Average stride length = %f\n", strideAvg_total);
}

template<typename T>
void MatrixSellC<T>::permute(int *perm, int*  invPerm)
{
    double* newVal = new double[nnz];
    int* newRowPtr = new int[nrows+1];
    int* newCol = new int[nnz];

    newRowPtr[0] = 0;

    //NUMA init
    #pragma omp parallel for schedule(static)
    for(int row=0; row<nrows; ++row)
    {
        newRowPtr[row+1] = 0;
    }

    //first find newRowPtr; therefore we can do proper NUMA init
    int permIdx=0;
    printf("nrows = %d\n", nrows);
    for(int row=0; row<nrows; ++row)
    {
        //row permutation
        int permRow = perm[row];
        nnzPerRow[row] = (rowPtr[permRow+1]-rowPtr[permRow]);
        for(int idx=rowPtr[permRow]; idx<rowPtr[permRow+1]; ++idx)
        {
            ++permIdx;
        }
        newRowPtr[row+1] = permIdx;
    }

    //with NUMA init
    #pragma omp parallel for schedule(static)
    for(int row=0; row<nrows; ++row)
    {
        //row permutation
        int permRow = perm[row];
        for(int permIdx=newRowPtr[row],idx=rowPtr[permRow]; permIdx<newRowPtr[row+1]; ++idx,++permIdx)
        {
            //permute column-wise also
            newVal[permIdx] = val[idx];
            newCol[permIdx] = invPerm[col[idx]];
        }
    }

    //free old permutations
    delete[] val;
    delete[] rowPtr;
    delete[] col;

    val = newVal;
    rowPtr = newRowPtr;
    col = newCol;
}

template<typename T>
void MatrixSellC<T>::NUMA_init()
{
    double* newVal = new double[nnz];
    int* newCol = new int[nnz];
    int* newRowPtr = new int[nrows+1];

    /*
       double *newVal = (double*) allocate(1024, sizeof(double)*nnz);
       int *newCol = (int*) allocate(1024, sizeof(int)*nnz);
       int *newRowPtr = (int*) allocate(1024, sizeof(int)*(nrows+1));
       */

    //NUMA init
    #pragma omp parallel for schedule(static)
    for(int row=0; row<nrows+1; ++row)
    {
        newRowPtr[row] = rowPtr[row];
    }
    #pragma omp parallel for schedule(static)
    for(int row=0; row<nrows; ++row)
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
