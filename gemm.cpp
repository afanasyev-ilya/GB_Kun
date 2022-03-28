#define NEED_GEMM

#include "src/gb_kun.h"


int main(int argc, char **argv) {
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);
        VNT scale = parser.get_scale();
        VNT avg_deg = parser.get_avg_degree();

        lablas::Matrix<float> A;
        A.set_preferred_matrix_format(CSR);
        init_matrix(A, parser);
        // A.print();

        lablas::Matrix<float> B;
        B.set_preferred_matrix_format(CSR);
        init_matrix(B, parser);
        // B.sort_csc_rows("STL_SORT");
        // B.print();

        lablas::Matrix<float> C;

        #define MASK_NULL static_cast<const lablas::Matrix<float>*>(NULL)
        lablas::mxm(&C, &A, lablas::second<float>(),
                    lablas::PlusMultipliesSemiring<float>(), &A, &A, &lablas::GrB_DESC_IKJ);
        #undef MASK_NULL

        // C.print();

        if (parser.check()) {
            int error_cnt = 0;
            for (int i = 0; i < A.get_matrix()->get_csr()->get_num_rows(); ++i) {
                for (int j = 0; j < A.get_matrix()->get_csr()->get_num_rows(); ++j) {
                    if (!A.get_matrix()->get_csr()->get(i, j)) {
                        continue;
                    }
                    float accumulator = 0;
                    for (int k = 0; k < A.get_matrix()->get_csr()->get_num_rows(); ++k) {
                        accumulator += A.get_matrix()->get_csr()->get(i, k) * A.get_matrix()->get_csr()->get(k, j);
                    }
                    if (C.get_matrix()->get_csr()->get(i, j) != accumulator) {
                        std::cout << i << ' ' << j << " " << accumulator << " " << C.get_matrix()->get_csr()->get(i, j)
                                  << std::endl;
                        ++error_cnt;
                    }
                }
            }
            std::cout << "Matrix multiplication errors cnt: " << error_cnt << std::endl;
        }

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
