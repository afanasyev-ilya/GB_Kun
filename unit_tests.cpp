#ifdef __USE_GTEST__
#include <gtest/gtest.h>
#endif

#include "src/gb_kun.h"

#define SIZE 5
#define MASK_NULL static_cast<const lablas::Vector<float>*>(NULL)

TEST(MxvTest, test1)
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

    lablas::Descriptor desc;
    mxv(&w, MASK_NULL, GrB_NULL, lablas::PlusMultipliesSemiring<int>(), &A, &u, &desc);

    ASSERT_EQ(w, ans);
}

int main(int argc, char **argv) {
    try
    {
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
    catch (string& error)
    {
        cout << error << endl;
    }
    catch (const char * error)
    {
        cout << error << endl;
    }
    return 0;
}

