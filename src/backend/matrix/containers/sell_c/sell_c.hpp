#pragma once

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
MatrixSellC<T>::MatrixSellC():size(0), nz(0), val(NULL), rowPtr(NULL), col(NULL), chunkLen(NULL), chunkPtr(NULL), colSellC(NULL), valSellC(NULL), unrollFac(1), C(1)
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

template<typename T>
void MatrixSellC<T>::constructSellCSigma(int chunkHeight, int sigma, int pad)
{
    /*C = chunkHeight;
    P = pad;

    int nSigmaChunks = (int)(size/(T)sigma);
    if(sigma > 1)
    {
        int *sigmaPerm = new int[size];
        for(int i=0; i<size; ++i)
        {
            sigmaPerm[i] = i;
        }

        for(int sigmaChunk=0; sigmaChunk<nSigmaChunks; ++sigmaChunk)
        {
            int *perm_begin = &(sigmaPerm[sigmaChunk*sigma]);
            sort_perm(nnzPerRow, perm_begin, sigma);
        }

        int restSigmaChunk = size%sigma;
        if(restSigmaChunk > C)
        {
            int *perm_begin = &(sigmaPerm[nSigmaChunks*sigma]);
            sort_perm(nnzPerRow, perm_begin, restSigmaChunk);
        }

        int *sigmaInvPerm = new int[size];

        for(int i=0; i<size; ++i)
        {
            sigmaInvPerm[sigmaPerm[i]] = i;
        }

        permute(sigmaPerm, sigmaInvPerm);

        delete[] sigmaPerm;
        delete[] sigmaInvPerm;
    }

    nchunks = (int)(size/(double)C);
    if(size%C > 0)
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
            if(row<size)
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


    for(int chunk=0; chunk<nchunks; ++chunk)
    {
        for(int rowInChunk=0; rowInChunk<C; ++rowInChunk)
        {
            int row = chunk*C + rowInChunk;
            if(row<size)
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

    printf("Average stride length = %f\n", strideAvg_total);*/
}

template<typename T>
void MatrixSellC<T>::permute(int *perm, int*  invPerm)
{
    double* newVal = new double[nz];
    int* newRowPtr = new int[size+1];
    int* newCol = new int[nz];

    newRowPtr[0] = 0;

    //NUMA init
    #pragma omp parallel for schedule(static)
    for(int row=0; row<size; ++row)
    {
        newRowPtr[row+1] = 0;
    }

    //first find newRowPtr; therefore we can do proper NUMA init
    int permIdx=0;
    printf("size = %d\n", size);
    for(int row=0; row<size; ++row)
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
    for(int row=0; row<size; ++row)
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


