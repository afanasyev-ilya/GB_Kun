#include "../../src/gb_kun.h"
#include "gtest/gtest.h"

int my_argc;
char** my_argv;

float square_root(float a) {
    return 0.0;
}

/*TEST (TransposeTest, SmallTest) {
    lablas::Matrix<int> matrix;
    const std::vector<Index> row_ids = {0, 0, 1, 1, 1, 2, 3, 4, 5};
    const std::vector<Index> col_ids = {2, 4, 0, 3, 5, 1, 4, 2, 0};
    const std::vector<int> csr_val = {10, 10, 10, 10, 10, 10, 10, 10, 10};
    matrix.build(&row_ids,&col_ids,&csr_val, 9, nullptr, nullptr);

    matrix.get_matrix()->transpose_parallel();

    auto *row_result = matrix.get_matrix()->get_csc()->get_row_ptr();
    auto *col_result = matrix.get_matrix()->get_csc()->get_col_ids();

    Index num_cols;
    matrix.get_ncols(&num_cols);

    std::vector<std::set<Index>> correct;

    std::set<Index> col0;
    std::set<Index> col1;
    std::set<Index> col2;
    std::set<Index> col3;
    std::set<Index> col4;
    std::set<Index> col5;
    col0.insert(1); col0.insert(5);
    col1.insert(2);
    col2.insert(0); col2.insert(4);
    col3.insert(1);
    col4.insert(0); col4.insert(3);
    col5.insert(1);
    correct.push_back(col0);
    correct.push_back(col1);
    correct.push_back(col2);
    correct.push_back(col3);
    correct.push_back(col4);
    correct.push_back(col5);

    for (int i = 0; i < num_cols; i++) {
        std::vector<Index> sss;
        for (Index j = row_result[i]; j < row_result[i+1]; j++) {
            sss.push_back(col_result[j]);
        }
        std::set<Index> cols (sss.data(), sss.data() + sss.size());
        ASSERT_EQ(cols, correct[i]);
    }
//    for (int i = 0; i < num_cols + 1; i++) {
//        std::cout << row_result[i] << " ";
//    }
//    std::cout << std::endl;
//
//    for (int i = 0; i < csr_val.size(); i++) {
//        std::cout << col_result[i] << " ";
//    }

}*/

TEST (TransposeTest, RealTest) {

    std::vector<Index> row_indices;
    std::vector<Index> col_indices;
    std::vector<int> values;
    Index nrows, ncols, nvals;

    Parser parser;
    parser.parse_args(my_argc, my_argv);
    VNT scale = parser.get_scale();
    VNT avg_deg = parser.get_avg_degree();

    // Matrix A
    lablas::Matrix<int> A;
    A.set_preferred_matrix_format(parser.get_storage_format());
    init_matrix(A,parser);

    lablas::Matrix<int> B;
    B.set_preferred_matrix_format(parser.get_storage_format());
    init_matrix(B,parser);

    nrows = A.nrows();
    ncols = A.ncols();
    nvals = A.get_nvals(&nvals);

    double seq_a = omp_get_wtime();
    A.get_matrix()->transpose();
    double seq_b = omp_get_wtime();
    double par_a = omp_get_wtime();
    B.get_matrix()->transpose_parallel();
    double par_b = omp_get_wtime();

    std::cout << "Time for sequential " << seq_b - seq_a << " seconds" << std::endl;
    std::cout << "Time for parallel " << par_b - par_a << "seconds" << std::endl;

    auto a_col_ptr = A.get_matrix()->get_csc()->get_row_ptr();
    auto b_col_ptr = B.get_matrix()->get_csc()->get_row_ptr();

    for (Index i = 0; i < ncols; i++) {
        ASSERT_EQ(a_col_ptr[i], b_col_ptr[i]);
    }

    auto a_row_ids = A.get_matrix()->get_csc()->get_col_ids();
    auto b_row_ids = A.get_matrix()->get_csc()->get_col_ids();

    for (int i = 0; i < ncols; i++) {
        std::vector<Index> res_a;
        std::vector<Index> res_b;

        for (Index j = a_col_ptr[i]; j < a_col_ptr[i+1]; j++) {
            res_a.push_back(a_row_ids[j]);
        }
        for (Index j = b_col_ptr[i]; j < b_col_ptr[i+1]; j++) {
            res_b.push_back(b_row_ids[j]);
        }
        std::set<Index> cols_a (res_a.data(), res_a.data() + res_a.size());
        std::set<Index> cols_b (res_b.data(), res_b.data() + res_b.size());
        ASSERT_EQ(cols_a, cols_b);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    my_argc = argc;
    my_argv = argv;
    return RUN_ALL_TESTS();
}
