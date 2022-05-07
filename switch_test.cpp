#include "src/gb_kun.h"

double filling_time = 0.0;
double working_time = 0.0;
double mask_conv = 0.0;

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


    //matrix.get_matrix()->set_mxv_algo(mxv_algorithm::SPMSPV_bucket);

    nrows = matrix.nrows();
    ncols = matrix.ncols();
    nvals = matrix.get_nvals(&nvals);

    /* Mask type (dense or sparse) */
    for (int mask_type = 0; mask_type < 1; mask_type++) {


        /* Mask sparsity iterations */
        for (int mask_iter = 50 * nrows / 100; mask_iter < 51 * nrows / 100; mask_iter += 10 * nrows / 100) {



            /* Vector sparsity iterations - Innermost cycle */
            for (int iter = 2 * nrows / 100; iter < nrows; iter += 2 * nrows / 100) {
                lablas::Vector<int> mask(nrows);
                mask_type == 0 ? mask.get_vector()->force_to_sparse() : mask.get_vector()->force_to_dense();
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
                //std::cout << "Doing iter with " << (float)mask_iter/(float)nrows << " mask sparsity and " << (float)iter/(float)nrows << " vec sparsity" << std::endl;
                size_t vec_nvals = iter;
                lablas::Vector<int> components(nrows);
                auto *vec_values = new int[vec_nvals];

                for (size_t i = 0; i < vec_nvals; i++) {
                    VNT idx = rand() % nrows;
                    while (idx_set.find(idx) != idx_set.end()) {
                        idx = rand() % nrows;
                    }
                    idx_set.insert(idx);
                    vec_values[i] = idx;
                }
                idx_set.clear();

                components.build(vec_values, vec_nvals);
                lablas::Descriptor desc;
                desc.set(GrB_MXVMODE, SPMSPV_MAP_PAR);

                double start_time = omp_get_wtime();
                for (int j = 0; j < 2; j++) {
                    lablas::mxv(&components, &mask, lablas::second<int>(), lablas::LogicalOrAndSemiring<int>(),
                                &matrix,
                                &components, &desc);
                }
                double end_time = omp_get_wtime();
                //std::cout << "" << (float)iter/(float)nrows << " " << end_time - start_time << std::endl;
                delete[] vec_values;
            }
            std::cout << "Filling:" << filling_time <<", Working: " << working_time <<" , Mask conversion: " << mask_conv << std::endl;

        }
    }

    return 0;
}
