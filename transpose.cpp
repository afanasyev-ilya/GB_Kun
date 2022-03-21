#include "src/gb_kun.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void test_spmv(int argc, char **argv)
{
    Parser parser;
    parser.parse_args(argc, argv);

    lablas::Descriptor desc;

    lablas::Matrix<T> matrix;
    matrix.set_preferred_matrix_format(parser.get_storage_format());
    init_matrix(matrix, parser);

    const Index *row_ptr = matrix.get_matrix()->get_csr()->get_row_ptr();
    const Index *col_ids = matrix.get_matrix()->get_csr()->get_col_ids();
    const float *vals = matrix.get_matrix()->get_csr()->get_vals();
    lablas::Matrix<float> matrix_small;

    double t1 = omp_get_wtime();
    matrix_small.build_from_csr_arrays(reinterpret_cast<const Index*>(row_ptr), reinterpret_cast<const Index*>(col_ids),
                                       reinterpret_cast<const float*>(vals), matrix.get_matrix()->get_csr()->get_num_rows(), matrix.get_matrix()->get_csr()->get_nnz());
    double t2 = omp_get_wtime();
    save_teps("build_from_csr", t2 - t1, matrix.get_nnz());
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        print_omp_stats();
        test_spmv<float>(argc, argv);
    }
    catch (string error)
    {
        cout << error << endl;
    }
    catch (const char * error)
    {
        cout << error << endl;
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

