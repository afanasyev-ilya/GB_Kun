#include "src/gb_kun.h"

#include "algorithms/bfs/bfs.hpp"
#include "algorithms/bfs/bfs_traditional.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);
        VNT scale = parser.get_scale();
        VNT avg_deg = parser.get_avg_degree();

        lablas::Descriptor desc;

        lablas::Matrix<float> matrix;
        matrix.set_preferred_matrix_format(parser.get_storage_format());
        init_matrix(matrix, parser);

        Index nrows;
        matrix.get_nrows(&nrows);
        Index source_vertex = 0;

        lablas::Vector<float> levels(nrows);

        std::ofstream levels_file("bfs_depth.txt", std::ios_base::app);

        double wall_bfs_time = 0;
        Index max_level = 0;
        for(int run = 0; run < parser.get_iterations(); run++)
        {
            source_vertex = select_non_trivial_vertex(matrix);
            double bfs_time_ms = 0;
            {
                Timer tm("bfs");
                Index cur_level = 0;
                lablas::algorithm::bfs_blast(&levels, &matrix, source_vertex, &desc, cur_level);
                bfs_time_ms = tm.get_time_ms();
                max_level = std::max(max_level, cur_level);
            }
            save_teps("BFS_chrono", bfs_time_ms, matrix.get_nnz(), 1);
            wall_bfs_time += bfs_time_ms / 1000;
        }
        levels_file << max_level << std::endl;
        #if(__DEBUG_PERF_STATS_ENABLED__)
        std::cout << "BFS wall time: " << wall_bfs_time << " ms" << std::endl;
        std::cout << "BFS conversion time: " << GLOBAL_CONVERSION_TIME << " ms" << std::endl;
        std::cout << "BFS spmv time: " << GLOBAL_SPMV_TIME << " ms" << std::endl;
        std::cout << "BFS spmspv time: " << GLOBAL_SPMSPV_TIME << " ms" << std::endl;
        std::cout << "check times: " << wall_bfs_time << " vs " << (GLOBAL_CONVERSION_TIME + GLOBAL_SPMV_TIME + GLOBAL_SPMSPV_TIME) << std::endl;
        #endif

        if(parser.check())
        {
            lablas::Vector<float> check_levels(nrows);

            lablas::algorithm::bfs_traditional(&check_levels, &matrix, source_vertex);

            if(levels == check_levels)
            {
                print_diff(levels, check_levels);
                cout << "BFS levels are equal" << endl;
            }
            else
            {
                print_diff(levels, check_levels);
                cout << "BFS levels are NOT equal" << endl;
            }
        }
    }
    catch (const char * error)
    {
        cout << error << endl;
        return 0;
    }
    return 0;
}

// test

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
