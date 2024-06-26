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

        lablas::Matrix<float> B;
        B.set_preferred_matrix_format(CSR);
        init_matrix(B, parser);

        lablas::Matrix<float> C;

        #define MASK_NULL static_cast<const lablas::Matrix<float>*>(NULL)
        lablas::mxm(&C, MASK_NULL, lablas::second<float>(),
                    lablas::PlusMultipliesSemiring<float>(), &A, &B, &lablas::GrB_DESC_IKJ);
        #undef MASK_NULL

        if (parser.check()) {
            int error_cnt = 0;
            for (int i = 0; i < A.get_matrix()->get_csr()->get_num_rows(); ++i) {
                for (int j = 0; j < B.get_matrix()->get_csr()->get_num_rows(); ++j) {
                    float accumulator = 0;
                    for (int k = 0; k < A.get_matrix()->get_csr()->get_num_rows(); ++k) {
                        accumulator += A.get_matrix()->get_csr()->get(i, k) *
                                B.get_matrix()->get_csr()->get(k, j);
                    }
                    if (C.get_matrix()->get_csr()->get(i, j) != accumulator) {
                        std::cout << i << ' ' << j << " " << accumulator << " "
                                  << C.get_matrix()->get_csr()->get(i, j)
                                  << std::endl;
                        ++error_cnt;
                    }
                }
            }
            cout << "error_count: " << error_cnt << "/"
                 << A.get_matrix()->get_nrows() * A.get_matrix()->get_nrows() << endl;
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
