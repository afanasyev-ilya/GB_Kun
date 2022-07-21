#include "../../src/gb_kun.h"
#include "../../src/cpp_graphblas/types.hpp"
#include <iostream>
#include <chrono>
#include <random>
#include <gtest/gtest.h>

#define SIZE 5

#define MASK_NULL static_cast<const lablas::Vector<float>*>(NULL)

// w += Au
MXV_TEST(Test, test1)
{
    lablas::Vector<int> u(SIZE);
    lablas::Vector<int> w(SIZE);
    lablas::Vector<int> ans(SIZE);

    int* ans_vals = ans.get_vector()->getDense()->get_vals();
    ans_vals[0] = 9;
    ans_vals[1] = 14;
    ans_vals[2] = 12;
    ans_vals[3] = 6;
    ans_vals[4] = 1;

    int* u_vals = u.get_vector()->getDense()->get_vals();
    for (int i = 0; i < 5; i++) {
        u_vals[i] = i + 1;
    }

    lablas::Matrix<int> A;
    A.init_from_mtx("test_matrix.mtx");
    A.print();

    lablas::Descriptor desc;
    mxv(&w, MASK_NULL, GrB_NULL, lablas::PlusMultipliesSemiring<int>(), &A, &u, &desc);

    ASSERT_EQ(w, ans);
}

int main() {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
