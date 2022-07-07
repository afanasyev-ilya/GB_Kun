#include "./src/gb_kun.h"

#include <iostream>
#include <chrono>
#include <random>

#define SIZE 1000
#define CHANCE_STEP 0.1
#define CHANCE_RESOLUTION 100000
#define TEST_COUNT 1000

int main() {

    lablas::Vector<int> w(SIZE);
    lablas::Vector<int> u(SIZE);

    lablas::Vector<bool> mask(SIZE);
    lablas::Descriptor desc;
    // desc.set(GrB_OUTPUT, GrB_REPLACE);

    int* w_vals = w.get_vector()->getDense()->get_vals();
    for (int i = 0; i < SIZE; ++i) {
        w_vals[i] = 0;
    }

    int* u_vals = u.get_vector()->getDense()->get_vals();
    for (int i = 0; i < SIZE; ++i) {
        u_vals[i] = i;
    }

    auto select_op = [](int x, Index i, Index j, int val){
        return x + val + i;
    };

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist6(1, CHANCE_RESOLUTION);

    for (float zero_chance = 0; zero_chance <= 1.0; zero_chance += CHANCE_STEP) {
        bool* mask_vals = mask.get_vector()->getDense()->get_vals();
        for (int i = 0; i < SIZE; ++i) {
            auto rand = dist6(dev);
            if (rand < CHANCE_RESOLUTION * zero_chance) {
                mask_vals[i] =  0;
            } else {
                mask_vals[i] =  1;
            }
        }
        mask.get_vector()->getSparse();

        int64_t average = 0;
        for (int i = 0; i < TEST_COUNT; ++i) {
            auto begin = std::chrono::system_clock::now();

            GrB_select(&w, &mask, NULL, select_op, &u, 1, &desc);

            auto end = std::chrono::system_clock::now();
            average += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        }
        average /= TEST_COUNT;

        std::cout << "On zero_chane = " << zero_chance << " average time = " << average / 1000.f << std::endl;;
    }

    return 0;
}