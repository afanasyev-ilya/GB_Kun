#include "../../src/gb_kun.h"
#include "gtest/gtest.h"


TEST (ScanTest, SomeTest) {
    int input[] = {3, 5, 2, 1, 6, 7, 3, 2, 5, 3, 4, 7, 1, 4, 1, 5, 6, 1, 2, 6, 5};
    int *output  = new int [sizeof(input) / sizeof(int) + 1];
    ParallelPrimitives::exclusive_scan(input,output, sizeof(input) / sizeof(int), output, 0);

    int local_sum = 0;
    for (int i = 0; i < sizeof(input) / sizeof(int) + 1; i++) {
        ASSERT_EQ(output[i], local_sum);
        local_sum += input[i];
        std::cout << output[i] << " ";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
