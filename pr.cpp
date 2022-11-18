#include "src/gb_kun.h"

#include "algorithms/pr/pr.hpp"
#include "algorithms/pr/pr_traditional.hpp"

int main(int argc, char **argv)
{
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);
        VNT scale = parser.get_scale();
        VNT avg_deg = parser.get_avg_degree();

        lablas::Descriptor desc;
        desc.set(GrB_INV, GrB_INV_READ);

        lablas::Matrix<float> matrix;
        matrix.set_preferred_matrix_format(parser.get_storage_format());
        init_matrix(matrix, parser, desc);

        GrB_Index size;
        matrix.get_nrows(&size);
        lablas::Vector<float> levels(size);

        LAGraph_Graph<float> graph(matrix);
        int max_iter = parser.get_iterations();

        int iters_taken = 1000;
        lablas::Vector<float> ranks(size);
        if(parser.get_algo_name() == "lagraph")
        {
            double pr_time_ms = 0;
            {
                Timer tm("pr");
                lablas::algorithm::LAGraph_page_rank_sinks(&ranks, &graph, &iters_taken, max_iter);
                pr_time_ms = tm.get_time_ms();
            }
            save_teps("PR", pr_time_ms, matrix.get_nnz(), max_iter);
        }
        else
        {
            cout << "Unknown algorithm name in PR" << endl;
        }

        if(parser.check())
        {
            lablas::Vector<float> check_ranks(size);
            lablas::algorithm::seq_page_rank(&check_ranks, &matrix, &iters_taken, max_iter);

            if(ranks == check_ranks)
            {
                print_diff(ranks, check_ranks);
                cout << "page ranks are equal" << endl;
            }
            else
            {
                print_diff(ranks, check_ranks);
                cout << "page ranks are NOT equal" << endl;
            }
        }


    }
    catch (string error)
    {
        cout << error << endl;
    }
    catch (const char * error)
    {
        cout << error << endl;
    }
    return 0;
}
