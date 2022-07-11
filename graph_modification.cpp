#include "src/gb_kun.h"

enum class ModificationTypes { VERTEX_INSERTION_OP = 1, VERTEX_DELETION_OP = 2, EDGE_INSERTION_OP = 3, EDGE_DELETION_OP = 4 };


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

        if (parser.check()) {
            const auto cur_nrows = A.nrows();
            const auto cur_ncols = A.ncols();

            std::vector<std::vector<float> > A_clone(cur_nrows, std::vector<float>(cur_ncols));
            int error_cnt = 0;
            for (int i = 0; i < cur_nrows; ++i) {
                for (int j = 0; j < cur_ncols; ++j) {
                    A_clone[i][j] = A.get_matrix()->get_csr()->get(i, j);
                }
            }

            const int op_blocks_cnt = 10;
            const int op_cnt = 10;
            for (int cur_block_op = 0; cur_block_op < op_blocks_cnt; ++cur_block_op) {
                for (int op_id = 0; op_id < op_cnt; ++op_id) {
                    // generate modifications and update matrices here
                }

                A.get_matrix()->apply_modifications();

                for (int i = 0; i < cur_nrows; ++i) {
                    for (int j = 0; j < cur_ncols; ++j) {
                        if (A.get_matrix()->get_csr()->get(i, j) != A_clone[i][j]) {
                            ++error_cnt;
                        }
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

