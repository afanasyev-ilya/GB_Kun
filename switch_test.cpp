#include "src/gb_kun.h"


int main(int argc, char** argv)
{
    std::vector<Index> row_indices;
    std::vector<Index> col_indices;
    std::vector<int> values;
    Index nrows, ncols, nvals;

    Parser parser;
    parser.parse_args(argc, argv);
    VNT scale = parser.get_scale();
    VNT avg_deg = parser.get_avg_degree();

    // Matrix A
    lablas::Matrix<int> matrix;
    matrix.set_preferred_matrix_format(parser.get_storage_format());
    init_matrix(matrix,parser);

    nrows = matrix.nrows();
    ncols = matrix.ncols();
    nvals = matrix.get_nvals(&nvals);

    /* Mask type (dense or sparse */
    for (int mask_type = 0; mask_type < 2; mask_type++) {
        lablas::Vector<int> mask(nrows);
        mask_type == 0 ? mask.get_vector()->force_to_dense() : mask.get_vector()->force_to_sparse();

        /* Mask sparsity iterations */
        for (int mask_iter = 10 * nrows / 100; mask_iter < nrows; mask_iter += 10 * nrows / 100) {

            std::set<VNT> idx_set;
            size_t mask_nvals = mask_iter;
            for (size_t i = 0; i < mask_nvals; i++) {
                VNT idx = rand() % nrows;
                while (idx_set.find(idx) != idx_set.end()) {
                    idx = rand() % nrows;
                }
                idx_set.insert(idx);
            }
            idx_set.clear();

            /* Vector sparsity iterations */
            for (int iter = 2 * nrows / 100; iter < nrows; iter += 2 * nrows / 100) {
                size_t vec_nvals = iter;
                lablas::Vector<int> components(nrows);

                for (size_t i = 0; i < vec_nvals; i++) {
                    VNT idx = rand() % nrows;
                    while (idx_set.find(idx) != idx_set.end()) {
                        idx = rand() % nrows;
                    }
                    idx_set.insert(idx);
                }


                std::cout << "Matrix dim size: " << nrows << std::endl;
                for (auto &a: idx_set) {
                    std::cout << a << " ";
                }
                std::cout << endl;

                lablas::Descriptor desc;
                desc.set(GrB_MXVMODE, SPMSPV_BUCKET);

                double start_time = omp_get_wtime();
                lablas::mxv(&components, &mask, lablas::second<int>(), lablas::MinimumSelectSecondSemiring<int>(),
                            &matrix, &components, &desc);
                double end_time = omp_get_wtime();
                std::cout << mask_type << " " << mask_iter << " " << iter << " " << end_time - start_time << std::endl;

            }
        }
    }

    return 0;
}
