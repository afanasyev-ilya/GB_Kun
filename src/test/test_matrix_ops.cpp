#include "../gb_kun.h"
#include "gtest/gtest.h"

float square_root(float a) {
    return 0.0;
}

TEST (TransposeTest, SmallTest) {
    lablas::Matrix<int> matrix;
    const std::vector<Index> row_ids = {0, 0, 1, 1, 1, 2, 3, 4, 5};
    const std::vector<Index> col_ids = {2, 4, 0, 3, 5, 1, 4, 2, 0};
    const std::vector<int> csr_val = {10, 10, 10, 10, 10, 10, 10, 10, 10};
    matrix.build(&row_ids,&col_ids,&csr_val, 9, nullptr, nullptr);

    matrix.get_matrix()->transpose();

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
}

TEST (TransposeTest, BigTest) {
    ASSERT_EQ (0.0, 1.0);
    ASSERT_EQ (0.0, 0.0);
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}