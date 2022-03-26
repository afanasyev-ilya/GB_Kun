#include "../../src/gb_kun.h"
#include "gtest/gtest.h"

template <typename T>
void initialize_input(T* ptr, size_t size) {
    for (int i = 0; i < size; i++) {
        ptr[i] = rand()%10;
    }
}

TEST (ScanTest, SomeTest) {
    size_t size = 10000;
    int *input = new int[size];
    initialize_input(input, size);

    int *output  = new int [size + 1];
    ParallelPrimitives::exclusive_scan(input,output, size, output, 0);

    int local_sum = 0;
    for (int i = 0; i < size + 1; i++) {
        ASSERT_EQ(output[i], local_sum);
        local_sum += input[i];
        //std::cout << output[i] << " ";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
