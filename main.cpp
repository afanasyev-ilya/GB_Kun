#include "src/gb_kun.h"

int main() {
    cout << "Hello, World!" << endl;

    #pragma omp parallel
    {
        printf("Hello World... from thread = %d\n",
               omp_get_thread_num());
    }

    MatrixCSR<float> matrix;

    VNT rows[] = {0, 4, 4, 2, 1};
    VNT cols[] = {1, 3, 2, 0, 4};
    float vals[] = {1, 1, 1, 1, 1};

    matrix.import(rows, cols, vals, 5, 5);
    matrix.print();

    return 0;
}
