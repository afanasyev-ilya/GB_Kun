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

    size_t IN_ITER = 500;

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

                lablas::Vector<int> res_1(nrows);
                lablas::Vector<int> res_2(nrows);
                lablas::Vector<int> res_3(nrows);
                lablas::Vector<int> res_4(nrows);


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
                std::cout << mask_type << " " << mask_iter << " " << iter << std::endl;

                lablas::Descriptor desc;
                double t_general, t_map_seq, t_map_par, t_for;


                desc.set(GrB_MXVMODE, SPMSPV_MAP_SEQ);
                double start_time = omp_get_wtime();
                for (size_t in_iter = 0; in_iter < IN_ITER; in_iter++) {
                    lablas::mxv(&res_1, &mask, lablas::second<int>(), lablas::MinimumSelectSecondSemiring<int>(),
                                &matrix, &components, &desc);
                }
                double end_time = omp_get_wtime();
                t_map_seq = end_time - start_time;
                std::cout << t_map_seq << "\t";

                desc.set(GrB_MXVMODE, SPMV_GENERAL);
                start_time = omp_get_wtime();
                for (size_t in_iter = 0; in_iter < IN_ITER; in_iter++) {
                    lablas::mxv(&res_2, &mask, lablas::second<int>(), lablas::MinimumSelectSecondSemiring<int>(),
                                &matrix, &components, &desc);
                }
                end_time = omp_get_wtime();
                t_general = end_time - start_time;
                std::cout << t_general << std::endl;

                desc.set(GrB_MXVMODE, SPMSPV_MAP_TBB);
                start_time = omp_get_wtime();
                for (size_t in_iter = 0; in_iter < IN_ITER; in_iter++) {
                    lablas::mxv(&res_3, &mask, lablas::second<int>(), lablas::MinimumSelectSecondSemiring<int>(),
                                &matrix, &components, &desc);
                }
                end_time = omp_get_wtime();
                t_map_par = end_time - start_time;
                std::cout << t_map_par << "\t";

                desc.set(GrB_MXVMODE, SPMSPV_FOR);
                start_time = omp_get_wtime();
                for (size_t in_iter = 0; in_iter < IN_ITER; in_iter++) {
                    lablas::mxv(&res_4, &mask, lablas::second<int>(), lablas::MinimumSelectSecondSemiring<int>(),
                                &matrix, &components, &desc);
                }
                end_time = omp_get_wtime();
                t_for = end_time - start_time;
                std::cout << t_for << std::endl;

                if (res_1 == res_2 and res_3 == res_4 and res_1 == res_3) {
                    std::cout << " Correct" << std::endl;
                }

                /*
                auto min_val = min(t_bucket, t_for);
                min_val = min(min_val, t_map_par);
                min_val = min(min_val, t_map_seq);
                if (min_val == t_bucket) {
                    std::cout << "bucket" << std::endl;
                }
                if (min_val == t_map_seq) {
                    std::cout << "map_seq" << std::endl;
                }
                if (min_val == t_map_par) {
                    std::cout << "map_par" << std::endl;
                }
                if (min_val == t_for) {
                    std::cout << "for" << std::endl;
                }
                 */
            }
        }
    }

    return 0;
}
