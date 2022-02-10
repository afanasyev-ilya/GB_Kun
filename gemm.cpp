#include "src/gb_kun.h"


int main(int argc, char **argv) {
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);
        VNT scale = parser.get_scale();
        VNT avg_deg = parser.get_avg_degree();

        lablas::Descriptor desc;

        lablas::Matrix<float> A;
        A.set_preferred_matrix_format(CSR);
        init_matrix(A, parser);
        A.print();

        lablas::Matrix<float> B;
        B.set_preferred_matrix_format(CSR);
        init_matrix(B, parser);
        B.sort_csc_rows("STL_SORT");
        B.print();
        /*
        const lablas::backend::MatrixCSR<float> *csr_data = A.get_matrix()->get_csr();
        Index num_rows = csr_data->get_num_rows();
        const Index *row_ptr = csr_data->get_row_ptr();
        const Index *col_ids = csr_data->get_col_ids();
        const float *vals = csr_data->get_vals();
        */

        lablas::Matrix<float> C;

        lablas::mxm(&A, &B, &C);

        C.print();
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
