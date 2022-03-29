#include "src/gb_kun.h"

#include "algorithms/bfs/bfs.hpp"
#include "algorithms/bfs/bfs_traditional.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
Index select_non_trivial_vertex(lablas::Matrix<T> &_matrix)
{
    Index max_val = min(_matrix.ncols(), _matrix.nrows());
    Index vertex = 0;
    srand(time(NULL));
    do {
        vertex = rand() %  1000/*max_val*/;
    } while((_matrix.get_rowdegrees()[vertex] == 0) || (_matrix.get_coldegrees()[vertex] == 0));
    return vertex;
}

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

        for(int run = 0; run < parser.get_iterations(); run++)
        {
            source_vertex = select_non_trivial_vertex(matrix);
            double bfs_time_ms = 0;
            {
                Timer tm("bfs");
                lablas::algorithm::bfs_blast(&levels, &matrix, source_vertex, &desc);
                bfs_time_ms = tm.get_time_ms();
            }
            save_teps("BFS_chrono", bfs_time_ms, matrix.get_nnz(), 1);
        }

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
