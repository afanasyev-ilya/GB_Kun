/**
  @file pr_traditional.hpp
  @author Lastname:Firstname:A00123456:cscxxxxx
  @version Revision 1.1
  @brief Traditional PR algorithm.
  @details Detailed description.
  @date May 12, 2022
*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//! Lablas namespace

namespace lablas{

    //! Algorithm namespace


    namespace algorithm {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * seq_page_rank function.
 * @brief The function does...
 * @param _ranks _ranks
 * @param _graph _graph
 * @param iters output: number of iterations taken
 * @param itermax maximum number of iterations (typically 100)
 * @param damping damping factor (typically 0.85)
 * @param tol stopping tolerance (typically 1e-4)
*/

template <typename T>
void seq_page_rank(Vector <T> *_ranks,
                   const Matrix <T> *_graph,
                   int *iters,                     // output: number of iterations taken
                   int itermax = 100,              // maximum number of iterations (typically 100)
                   double damping = 0.85,          // damping factor (typically 0.85)
                   double tol = 1e-4)               // stopping tolerance (typically 1e-4) ;
{
    Index vertices_count = _ranks->get_vector()->getDense()->get_size();
    T* ranks = _ranks->get_vector()->getDense()->get_vals();
    backend::MatrixCSR<T> *graph = ((backend::MatrixCSR<T> *) _graph->get_matrix()->get_csr());

    // set PR parameters
    T d = 0.85;
    T k = (1.0 - d) / ((T)vertices_count);

    std::vector<int> number_of_loops(vertices_count, 0);
    std::vector<int> incoming_degrees(vertices_count, 0);
    std::vector<int> incoming_degrees_without_loops(vertices_count, 0);
    std::vector<T> old_page_ranks(vertices_count, 0);

    // init ranks and other data
    for(int i = 0; i < vertices_count; i++)
    {
        ranks[i] = 1.0/((T)vertices_count);
        number_of_loops[i] = 0;
    }

    // calculate number of loops
    for(int src_id = 0; src_id < vertices_count; src_id++)
    {
        Index shift = graph->get_row_ptr()[src_id];
        Index connections_count = graph->get_row_ptr()[src_id + 1] - graph->get_row_ptr()[src_id];
        for(Index edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            Index dst_id = graph->get_col_ids()[shift + edge_pos];
            if(src_id == dst_id)
                number_of_loops[src_id]++;
        }
    }

    // calculate incoming degrees without loops
    for(int i = 0; i < vertices_count; i++)
    {
        incoming_degrees[i] = _graph->get_coldegrees()[i]; // TODO rowdegrees?
    }

    // calculate incoming degrees without loops
    for(int i = 0; i < vertices_count; i++)
    {
        incoming_degrees_without_loops[i] = incoming_degrees[i] - number_of_loops[i];
    }

    for(int iterations_count = 0; iterations_count < itermax; iterations_count++)
    {
        // copy ranks from prev iteration to temporary array
        for(int i = 0; i < vertices_count; i++)
        {
            old_page_ranks[i] = ranks[i];
            ranks[i] = 0;
        }

        // calculate dangling input
        T dangling_input = 0;
        for(int i = 0; i < vertices_count; i++)
        {
            if(incoming_degrees_without_loops[i] <= 0)
            {
                dangling_input += old_page_ranks[i] / vertices_count;
            }
        }

        // traverse graph and calculate page ranks
        for(int src_id = 0; src_id < vertices_count; src_id++)
        {
            Index shift = graph->get_row_ptr()[src_id];
            Index connections_count = graph->get_row_ptr()[src_id + 1] - graph->get_row_ptr()[src_id];
            for(Index edge_pos = 0; edge_pos < connections_count; edge_pos++)
            {
                Index dst_id = graph->get_col_ids()[shift + edge_pos];
                T dst_rank = old_page_ranks[dst_id];

                T dst_links_num = 1.0 / incoming_degrees_without_loops[dst_id];
                if(incoming_degrees_without_loops[dst_id] == 0)
                    dst_links_num = 0;

                if(src_id != dst_id)
                    ranks[src_id] += dst_rank * dst_links_num;
            }

            ranks[src_id] = k + d * (ranks[src_id] + dangling_input);
        }

        // calculate new ranks sum
        double ranks_sum = 0;
        for(int i = 0; i < vertices_count; i++)
        {
            ranks_sum += ranks[i];
        }
        #ifdef __DEBUG_INFO__
        cout << "ranks sum: " << ranks_sum << endl;
        #endif
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}
