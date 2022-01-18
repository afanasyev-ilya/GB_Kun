#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
Matrix<T>::Matrix(): _format(CSR)
{
    csr_data = NULL;
    csc_data = NULL;
    data = NULL;
    transposed_data = NULL;
    rowdegrees = NULL;
    coldegrees = NULL;
    #ifdef __USE_SOCKET_OPTIMIZATIONS__
    data_socket_dub = NULL;
    #endif

    workspace = NULL;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
Matrix<T>::Matrix(Index ncols, Index nrows) : _format(CSR)
{
    throw "Error: Matrix(Index ncols, Index nrows) not implemented yet";
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
Matrix<T>::~Matrix()
{
    if(csr_data != NULL)
        delete csr_data;

    if(csc_data != NULL)
        delete csc_data;

    if(data != NULL)
        delete data;
    if(transposed_data != NULL)
        delete transposed_data;

    MemoryAPI::free_array(rowdegrees);
    MemoryAPI::free_array(coldegrees);

    #ifdef __USE_SOCKET_OPTIMIZATIONS__
    if(data_socket_dub != NULL)
        delete data_socket_dub;
    #endif

    delete workspace;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::build(const VNT *_row_indices,
                      const VNT *_col_indices,
                      const T *_values,
                      const VNT _size, // todo remove
                      const ENT _nnz)
{

    // CSR data creation
    double t1, t2;
    t1 = omp_get_wtime();
    csr_data = new MatrixCSR<T>;
    csc_data = new MatrixCSR<T>;
    csr_data->build(_row_indices, _col_indices, _values, _size, _nnz, 0);
    csc_data->build(_col_indices, _row_indices, _values, _size, _nnz, 0);
    t2 = omp_get_wtime();
    cout << "csr creation time: " << t2 - t1 << " sec" << endl;

    MemoryAPI::allocate_array(&rowdegrees, get_nrows());
    MemoryAPI::allocate_array(&coldegrees, get_ncols());

    #pragma omp parallel for
    for(int i = 0; i < get_nrows(); i++)
    {
        rowdegrees[i] = csr_data->get_degree(i);
    }

    #pragma omp parallel for
    for(int i = 0; i < get_ncols(); i++)
    {
        coldegrees[i] = csc_data->get_degree(i);
    }

    t1 = omp_get_wtime();
    // optimized representation creation
    if (_format == CSR)
    {
        data = NULL;
        transposed_data = NULL;

        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixCSR<T>;;
        ((MatrixCSR<T>*)data_socket_dub)->deep_copy((MatrixCSR<T>*)csr_data, 1);
        #endif
        cout << "Using CSR matrix format as optimized representation" << endl;
    }
    else if (_format == LAV)
    {
        data = new MatrixLAV<T>;
        transposed_data = new MatrixLAV<T>;

        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixLAV<T>;
        #endif

        data->build(_row_indices, _col_indices, _values, _size, _nnz, 0);
        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        if(_format == CSR)
            data_socket_dub->build(_row_indices, _col_indices, _values, _size, _nnz, 1);
        #endif
        transposed_data->build(_col_indices, _row_indices, _values, _size, _nnz, 0);
        cout << "Using LAV matrix format as optimized representation" << endl;
    }
    else if (_format == COO)
    {
        transposed_data = new MatrixCOO<T>;
        data = new MatrixCOO<T>;

        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixCOO<T>;
        #endif

        data->build(_row_indices, _col_indices, _values, _size, _nnz, 0);
        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        if(_format == CSR)
            data_socket_dub->build(_row_indices, _col_indices, _values, _size, _nnz, 1);
        #endif
        transposed_data->build(_col_indices, _row_indices, _values, _size, _nnz, 0);
        cout << "Using COO matrix format as optimized representation" << endl;
    }
    else if (_format == CSR_SEG) {
        data = new MatrixSegmentedCSR<T>;
        transposed_data = new MatrixSegmentedCSR<T>;

        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixSegmentedCSR<T>;
        #endif

        data->build(_row_indices, _col_indices, _values, _size, _nnz, 0);
        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        if(_format == CSR)
            data_socket_dub->build(_row_indices, _col_indices, _values, _size, _nnz, 1);
        #endif
        transposed_data->build(_col_indices, _row_indices, _values, _size, _nnz, 0);
        cout << "Using CSR_SEG matrix format as optimized representation" << endl;
    }
    else if (_format == VECT_GROUP_CSR) {
        data = new MatrixVectGroupCSR<T>;
        transposed_data = new MatrixVectGroupCSR<T>;

        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixVectGroupCSR<T>;
        #endif

        data->build(_row_indices, _col_indices, _values, _size, _nnz, 0);
        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        if(_format == CSR)
            data_socket_dub->build(_row_indices, _col_indices, _values, _size, _nnz, 1);
        #endif
        transposed_data->build(_col_indices, _row_indices, _values, _size, _nnz, 0);

        cout << "Using MatrixVectGroupCSR matrix format as optimized representation" << endl;
    }
    else if (_format == SELL_C)
    {
        data = new MatrixSellC<T>;
        transposed_data = new MatrixSellC<T>;

        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixSellC<T>;
        #endif

        data->build(_row_indices, _col_indices, _values, _size, _nnz, 0);
        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        if(_format == CSR)
            data_socket_dub->build(_row_indices, _col_indices, _values, _size, _nnz, 1);
        #endif

        transposed_data->build(_col_indices, _row_indices, _values, _size, _nnz, 0);

        cout << "Using SellC matrix format as optimized representation" << endl;
    }
    else if(_format == SORTED_CSR)
    {
        data = new MatrixSortCSR<T>;
        transposed_data = new MatrixSortCSR<T>;

        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        data_socket_dub = new MatrixSortCSR<T>;
        #endif

        data->build(_row_indices, _col_indices, _values, _size, _nnz, 0);
        #ifdef __USE_SOCKET_OPTIMIZATIONS__
        if(_format == CSR)
            data_socket_dub->build(_row_indices, _col_indices, _values, _size, _nnz, 1);
        #endif
        transposed_data->build(_col_indices, _row_indices, _values, _size, _nnz, 0);

        cout << "Using SortedCSR matrix format as optimized representation" << endl;
    }
    else
    {
        throw "Error: unsupported format in Matrix<T>::build";
    }
    t2 = omp_get_wtime();
    cout << "creating optimized representation time: " << t2 - t1 << " sec" << endl;

    workspace = new Workspace(get_nrows(), get_ncols(), csc_data->get_max_degree());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
