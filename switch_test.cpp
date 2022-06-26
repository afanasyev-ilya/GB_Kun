#include "src/gb_kun.h"
#include <iomanip>

std::string is_correct(lablas::Vector<int>& a1, lablas::Vector<int>& a2) {
    if (a1 == a2) {
        std::string ans = "correct";
        return ans;
    } else {
        std::string ans = "_ERROR_";
        return ans;
    }
}

int to_int(const std::string &str)
{
    if(str == "correct")
        return 1;
    return 0;
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
    int in_iters = 20;
    if(parser.check())
        in_iters = 1;

    // Matrix A
    lablas::Matrix<int> matrix;
    matrix.set_preferred_matrix_format(parser.get_storage_format());
    init_matrix(matrix,parser);

    nrows = matrix.nrows();
    ncols = matrix.ncols();
    nvals = matrix.get_nvals(&nvals);

    std::vector<int> indices(nrows);
    for(int i = 0; i < nrows; i++)
        indices[i] = i;
    random_shuffle(indices.begin(), indices.end());

    int experiments_count = 0;
    int error_count = 0;

    ofstream switch_stats;
    switch_stats.open("switch_stats.txt", std::ios_base::app);

    const int algo_count = 6;
    Desc_value tested_algos[algo_count] = {SPMV_GENERAL, SPMSPV_FOR, SPMSPV_MAP_TBB, SPMSPV_MAP_SEQ, SPMSPV_MAP_PAR, SPMSPV_ESC};
    std::string tested_algo_names[algo_count] = {"GENERIC", "FOR    ", "TBB_MAP", "SEQ_MAP", "MAP_PAR", "ESC    "};

    /* Mask type (dense or sparse */
    for (int mask_type = 0; mask_type < 2; mask_type++)
    {
        lablas::Vector<int> mask(nrows);
        mask_type == 0 ? mask.get_vector()->force_to_dense() : mask.get_vector()->force_to_sparse();

        /* Mask sparsity iterations */
        //for (int mask_iter = 10 * nrows / 100; mask_iter < nrows; mask_iter += 10 * nrows / 100)
        for (int mask_iter = (int)(40 * nrows / 100); mask_iter <= (int)(40 * nrows / 100); mask_iter += 10 * nrows / 100) // 40% sparse mask for now
        {
            std::set<VNT> idx_set;
            size_t mask_nvals = mask_iter;
            for (size_t i = 0; i < mask_nvals; i++) {
                VNT idx = rand() % nrows;
                while (idx_set.find(idx) != idx_set.end()) {
                    idx = rand() % nrows;
                }
                idx_set.insert(idx);
            }
            vector<VNT> mask_vec_ids;
            vector<int> mask_vec_vals;
            for(auto it: idx_set)
                mask_vec_ids.push_back(it), mask_vec_vals.push_back(1);
            mask.build(&mask_vec_ids, &mask_vec_vals, mask_vec_ids.size());

            std::string prev_best_name = "";
            /* Vector sparsity iterations */
            double vec_sparse_step = 1; // initially one element in vector
            for (int iter = vec_sparse_step; iter < nrows; iter += vec_sparse_step) {
                size_t vec_nvals = iter;
                lablas::Vector<int> data_vector(nrows);
                std::vector<VNT> vec_indices(vec_nvals);

                lablas::Vector<int> results[algo_count] = { lablas::Vector<int>(nrows), lablas::Vector<int>(nrows),
                                                            lablas::Vector<int>(nrows), lablas::Vector<int>(nrows),
                                                            lablas::Vector<int>(nrows), lablas::Vector<int>(nrows)};

                LOG_TRACE("Entering generation zone")

                for(int i = 0; i < vec_nvals; i++)
                    vec_indices[i] = indices[i];

                std::vector<int> vec_vals(vec_nvals);
                for (int i = 0; i < vec_nvals; i++) {
                    vec_vals[i] = rand() % INT_MAX;
                }

                data_vector.build(&vec_indices, &vec_vals, vec_nvals);

                LOG_TRACE("Generation and build done, Matrix dim size: ")

                std::cout << std::fixed << std::setprecision(5) << "(mask type: " << mask_type << ", mask sparsity: " <<
                    100.0*(((double)mask_iter)/(double)nrows) << "%, vector sparsity: " <<
                    100.0*(((double)iter)/(double)nrows)  << "%)" << "\t";

                double algo_times[algo_count];
                double best_time = std::numeric_limits<double>::max();
                std::string best_name;
                for(int algo_id = 0; algo_id < algo_count; algo_id++)
                {
                    lablas::Descriptor desc;

                    desc.set(GrB_MXVMODE, tested_algos[algo_id]);
                    double start_time = omp_get_wtime();
                    for (size_t in_iter = 0; in_iter < in_iters; in_iter++) {
                        lablas::mxv(&(results[algo_id]), &mask, lablas::second<int>(), lablas::LogicalOrAndSemiring<bool>(),
                                    &matrix, &data_vector, &desc);
                    }
                    double end_time = omp_get_wtime();
                    double algo_time = end_time - start_time;

                    if(algo_time < best_time)
                    {
                        best_time = algo_time;
                        best_name = tested_algo_names[algo_id];
                    }

                    algo_times[algo_id] = algo_time;
                }

                std::cout << "fastest algo: " << best_name << "  | times: ";
                for(int i = 0; i < algo_count; i++)
                    std::cout << algo_times[i] << "(s) ";
                std::cout << endl;

                if(prev_best_name != best_name && prev_best_name != "")
                {
                    switch_stats << parser.get_file_name() << ") " << prev_best_name << " -> " << best_name << " at : " <<
                         100.0*(((double)iter)/(double)nrows) << "%, vector sparsity" << ", mask_type = " <<
                         mask_type << std::endl;
                }

                prev_best_name = best_name;

                if (parser.check())
                {
                    int cur_experiment_correct_count = 0;
                    std::cout << "  /   " << "   GEN   " << "   FOR   " << "   TBB   " << "   SEQ   " << "   PAR   " << "   ESC   " << endl;
                    for(int i = 0; i < algo_count; i++)
                    {
                        std::cout << tested_algo_names[i] << ": ";
                        for(int j = 0; j < algo_count; j++)
                        {
                            std::cout << is_correct(results[i], results[j]) << " ";
                            cur_experiment_correct_count += to_int(is_correct(results[i], results[j]));
                        }
                        std::cout << std::endl;
                    }
                    experiments_count += algo_count*algo_count;
                    std::cout << "Errors in current experiment: " << (algo_count*algo_count - cur_experiment_correct_count) << std::endl;
                    error_count += (algo_count*algo_count - cur_experiment_correct_count);
                }

                if((vec_sparse_step / (double)nrows) > 0.05)
                    vec_sparse_step = nrows * 0.05;
                else
                    vec_sparse_step *= 2;
            }
            switch_stats << " -------------------------------------------- " << std::endl;
        }
    }

    if (parser.check())
    {
        cout << "error_count: " << error_count << " / " << experiments_count << endl;
    }
    return 0;
}
