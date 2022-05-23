#include "src/gb_kun.h"

std::string is_correct(lablas::Vector<int>& a1, lablas::Vector<int>& a2) {
    if (a1 == a2) {
        std::string ans = "correct";
        return ans;
    } else {
        std::string ans = "ERROR";
        return ans;
    }

}


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
                std::vector<VNT> vec_indices(vec_nvals);

                lablas::Vector<int> res_1(nrows);
                lablas::Vector<int> res_2(nrows);
                lablas::Vector<int> res_3(nrows);
                lablas::Vector<int> res_4(nrows);

                std::cout << "Entering generation zone" << std::endl;

                std::vector<int> indices(nrows);
                std::iota (std::begin(indices), std::end(indices), 0);

                for (size_t i = 0; i < vec_nvals; i++) {
                    VNT idx = rand() % (nrows - i);
                    auto it = indices.begin() + idx;
                    VNT elem = *it;
                    vec_indices[i] = elem;
                    indices.erase(it);
                }

                std::vector<int> vec_vals(vec_nvals);
                for (int i = 0; i < vec_nvals; i++) {
                    vec_vals[i] = rand() % INT_MAX;
                }

                components.build(&vec_indices, &vec_vals, vec_nvals);

                std::cout << "Matrix dim size: " << nrows << std::endl;
                std::cout << "(" << mask_type << " " << mask_iter << " " << (double)iter/(double)nrows  << ")" << "\t";

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
                std::cout << t_general << "\t";

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


                std::cout << "correct" << " " << is_correct(res_1, res_2) << " " << is_correct(res_1, res_3) << " " << is_correct(res_1, res_4) << " " << std::endl;
                std::cout << is_correct(res_1, res_2) << " " << "correct" << " " << is_correct(res_2, res_3) << " " << is_correct(res_2, res_4) << " " << std::endl;
                std::cout << is_correct(res_1, res_3) << " "<< is_correct(res_3, res_2) << " "<< "correct" << " "<< is_correct(res_3, res_4) << std::endl;
                std::cout << is_correct(res_1, res_4) << " "<< is_correct(res_4, res_2) << " "<< is_correct(res_4, res_3) << " "<< "correct" << std::endl;

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
