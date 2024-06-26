#define __REQUIRES_UNDIRECTED_GRAPHS__
#define __DISABLE_SPMSPV__

#include "src/gb_kun.h"

#include "algorithms/cc/cc.hpp"
#include "algorithms/cc/cc_traditional.hpp"

template <typename T>
Index estimate_num_components(T *_components, Index _size)
{
    std::map<T, Index> cmp_data;
    for(Index i = 0; i < _size; i++)
    {
        T component = _components[i];
        cmp_data[component]++;
    }
    return cmp_data.size();
}

template <typename T>
bool equal_components(lablas::Vector<T> &_first,
                      lablas::Vector<T> &_second)
{
    // check if sizes are the same
    _first.get_vector()->force_to_dense();
    _second.get_vector()->force_to_dense();

    // construct equality maps
    map<int, int> f_s_equality;
    map<int, int> s_f_equality;
    int vertices_count = _first.size();
    auto first_ptr  = _first.get_vector()->getDense()->get_vals();
    auto second_ptr = _second.get_vector()->getDense()->get_vals();

    for (int i = 0; i < vertices_count; i++)
    {
        f_s_equality[first_ptr[i]] = second_ptr[i];
        s_f_equality[second_ptr[i]] = first_ptr[i];
    }

    // check if components are equal using maps
    bool result = true;
    int error_count = 0;
    for (int i = 0; i < vertices_count; i++)
    {
        if (f_s_equality[first_ptr[i]] != second_ptr[i])
        {
            result = false;
            error_count++;
        }
        if (s_f_equality[second_ptr[i]] != first_ptr[i])
        {
            result = false;
            error_count++;
        }
    }
    cout << "error_count: " << error_count << " / " << vertices_count << endl;
    if(error_count == 0)
        cout << "Results are equal" << endl;
    else
        cout << "Results are NOT equal, error_count = " << error_count << endl;

    std::cout << "num components in first vector: " << estimate_num_components(first_ptr, vertices_count) << std::endl;
    std::cout << "num components in second vector: " << estimate_num_components(second_ptr, vertices_count) << std::endl;

    return result;
}

int main(int argc, char** argv)
{
    std::vector<Index> row_indices;
    std::vector<Index> col_indices;
    std::vector<int> values;
    Index nrows, ncols, nvals;

    Parser parser;
    parser.parse_args(argc, argv);
    VNT scale = parser.get_scale();
    VNT avg_deg = parser.get_avg_degree();

    // Matrix A
    lablas::Matrix<int> matrix;
    matrix.set_preferred_matrix_format(parser.get_storage_format());
    init_matrix(matrix,parser);

    if(!matrix.is_symmetric())
    {
        matrix.to_symmetric();
    }

    nrows = matrix.nrows();
    ncols = matrix.ncols();
    nvals = matrix.get_nvals(&nvals);

    lablas::Vector<int> components(nrows);

    lablas::Descriptor desc;

    for (int i = 0; i < 3; i++) {
        double cc_time_ms = 0;
        {
            Timer tm("сс");
            lablas::algorithm::cc(&components, &matrix, 0, &desc);
            cc_time_ms = tm.get_time_ms();
        }
        save_teps("cc_chrono", cc_time_ms, matrix.get_nnz(), 1);
    }

    if(parser.check())
    {
        lablas::Vector<int> check_components(nrows);

        lablas::algorithm::cc_bfs_based_sequential(&check_components, &matrix);

        equal_components(components, check_components);
    }

    return 0;
}