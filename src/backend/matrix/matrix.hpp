#pragma once
#include <atomic>
#include <array>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
Matrix<T>::Matrix(): _format(CSR)
{
    csr_data = NULL;
    csc_data = NULL;
    data = NULL;
    transposed_data = NULL;
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

    delete workspace;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void Matrix<T>::build(vector<vector<pair<VNT, T>>>& csr_tmp_matrix, vector<vector<pair<VNT, T>>>& csc_tmp_matrix)
{
    // read mtx file and get tmp representations of csr and csc matrix
    VNT tmp_nrows = csr_tmp_matrix.size(), tmp_ncols = csc_tmp_matrix.size();

    double t1 = omp_get_wtime();
    csr_data = new MatrixCSR<T>;
    csc_data = new MatrixCSR<T>;
    csr_data->build(csr_tmp_matrix, tmp_nrows, tmp_ncols, 0);
    csc_data->build(csc_tmp_matrix, tmp_ncols, tmp_nrows, 0);
    double t2 = omp_get_wtime();
    cout << "csr (from mtx) creation time: " << t2 - t1 << " sec" << endl;

    init_optimized_structures();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void Matrix<T>::sort_csr_columns(const string& mode)
{
    double t1 = omp_get_wtime();
    if (mode == "COUNTING_SORT") {
        const VNT *col_ids = get_csc()->get_col_ids();
        VNT *row_ids = new VNT[get_csc()->get_nnz()];
        const T *vals = get_csc()->get_vals();

        VNT csc_num_rows = get_csc()->get_num_rows();
        VNT csr_num_rows = get_csr()->get_num_rows();

        for (VNT row_id = 0; row_id < csc_num_rows; ++row_id) {
            for (ENT j = get_csc()->get_row_ptr()[row_id]; j < get_csc()->get_row_ptr()[row_id + 1]; ++j) {
                row_ids[j] = row_id;
            }
        }

        std::vector<std::vector<pair<VNT, T>>> _result(csr_num_rows);

        for(ENT i = 0; i < get_csc()->get_nnz(); i++)
        {
            VNT row = col_ids[i];
            VNT col = row_ids[i];
            T val = vals[i];
            _result[row].push_back(make_pair(col, val));
        }

        ENT cur_pos = 0;

        ENT* result_row_ptrs = const_cast<ENT *>(get_csr()->get_row_ptr());
        VNT* result_col_ids = const_cast<VNT *>(get_csr()->get_col_ids());
        T* result_vals = const_cast<T *>(get_csr()->get_vals());

        for(VNT i = 0; i < csr_num_rows; i++)
        {
            result_row_ptrs[i] = cur_pos;
            result_row_ptrs[i + 1] = cur_pos + _result[i].size();
            cur_pos += _result[i].size();
        }
        #pragma omp parallel for
        for(VNT i = 0; i < _result.size(); i++)
        {
            for(ENT j = get_csr()->get_row_ptr()[i]; j < get_csr()->get_row_ptr()[i + 1]; j++)
            {
                result_col_ids[j] = _result[i][j - get_csr()->get_row_ptr()[i]].first;
                result_vals[j] = _result[i][j - get_csr()->get_row_ptr()[i]].second;
            }
        }
    } else if (mode == "STL_SORT") {
        #pragma omp parallel for
        for (int i = 0; i < get_csr()->get_num_rows(); i++) {
            Index* begin_ptr = csr_data->get_col_ids() + csr_data->get_row_ptr()[i];
            Index* end_ptr = csr_data->get_col_ids() + csr_data->get_row_ptr()[i + 1];
            std::sort(begin_ptr, end_ptr);
        }
    } else {
        throw mode;
    }
    double t2 = omp_get_wtime();
    GLOBAL_SORT_TIME += t2 - t1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void Matrix<T>::sort_csc_rows(const string& mode)
{
    double t1 = omp_get_wtime();
    if (mode == "COUNTING_SORT") {
        /*
        const VNT *col_ids = get_csc()->get_col_ids();
        VNT *row_ids = new VNT[get_csc()->get_nnz()];
        const T *vals = get_csc()->get_vals();

        VNT csc_num_rows = get_csc()->get_num_rows();
        VNT csr_num_rows = get_csr()->get_num_rows();

        for (VNT row_id = 0; row_id < csc_num_rows; ++row_id) {
            for (ENT j = get_csc()->get_row_ptr()[row_id]; j < get_csc()->get_row_ptr()[row_id + 1]; ++j) {
                row_ids[j] = row_id;
            }
        }

        std::vector<std::vector<pair<VNT, T>>> _result(csr_num_rows);

        for(ENT i = 0; i < get_csc()->get_nnz(); i++)
        {
            VNT row = col_ids[i];
            VNT col = row_ids[i];
            T val = vals[i];
            _result[row].push_back(make_pair(col, val));
        }

        ENT cur_pos = 0;

        ENT* result_row_ptrs = get_csr()->get_row_ptr();
        VNT* result_col_ids = get_csr()->get_col_ids();
        T* result_vals = get_csr()->get_vals();

        for(VNT i = 0; i < csr_num_rows; i++)
        {
            result_row_ptrs[i] = cur_pos;
            result_row_ptrs[i + 1] = cur_pos + _result[i].size();
            cur_pos += _result[i].size();
        }
        #pragma omp parallel for
        for(VNT i = 0; i < _result.size(); i++)
        {
            for(ENT j = get_csr()->get_row_ptr()[i]; j < get_csr()->get_row_ptr()[i + 1]; j++)
            {
                result_col_ids[j] = _result[i][j - get_csr()->get_row_ptr()[i]].first;
                result_vals[j] = _result[i][j - get_csr()->get_row_ptr()[i]].second;
            }
        }
        */
    } else if (mode == "STL_SORT") {
        #pragma omp parallel for
        for (int i = 0; i < get_csc()->get_num_rows(); i++) {
            VNT* begin_ptr = csc_data->get_col_ids() + csc_data->get_row_ptr()[i];
            VNT* end_ptr = csc_data->get_col_ids() + csc_data->get_row_ptr()[i + 1];
            std::sort(begin_ptr, end_ptr);
        }
    } else {
        throw mode;
    }
    double t2 = omp_get_wtime();
    GLOBAL_SORT_TIME += t2 - t1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
