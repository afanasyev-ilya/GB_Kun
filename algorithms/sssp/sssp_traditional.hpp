/**
  @file sssp_traditional.hpp
  @author S.krymskiy
  @version Revision 1.1
  @brief Traditional SSSP algorithm.
  @details Detailed description.
  @date May 12, 2022
*/

#pragma once


//! Lablas namespace

namespace lablas {

    //! Algorithm namespace

    namespace algorithm {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * sssp_traditional_dijkstra function.
 * @brief The function does...
 * @param _distances _distances
 * @param _graph _graph
 * @param _source_vertex _source_vertex
*/

template <typename T>
void sssp_traditional_dijkstra(Vector <T> *_distances,
                               const Matrix <T> *_graph,
                               Index _source_vertex)
{
    Index vertices_count = _distances->get_vector()->getDense()->get_size();
    T* distance_vals = _distances->get_vector()->getDense()->get_vals();
    backend::MatrixCSR<T> *graph = ((backend::MatrixCSR<T> *) _graph->get_matrix()->get_csr());

    // Create a priority queue to store vertices that
    // are being preprocessed.
    typedef pair<T, Index> iPair;
    std::priority_queue< iPair, vector <iPair> , greater<iPair> > pq;

    // Create a vector for distances and initialize all
    // distances as infinite (INF)
    T inf_val = std::numeric_limits<T>::max();
    for(int i = 0; i < vertices_count; i++)
        distance_vals[i] = inf_val;

    // Insert source itself in priority queue and initialize
    // its distance as 0.
    pq.push(make_pair(0, _source_vertex));
    distance_vals[_source_vertex] = 0;
    ENT vertices_visited = 0;

    /*cout << "row ptrs: ";
    for(int i = 0; i < 20; i++)
        cout << graph->get_row_ptr()[i] << " ";
    cout << endl;

    cout << "col ptrs: ";
    for(int i = 0; i < 20; i++)
        cout << graph->get_col_ids()[i] << " ";
    cout << endl;*/

    /* Looping till priority queue becomes empty (or all
      distances are not finalized) */
    while (!pq.empty())
    {
        // The first vertex in pair is the minimum distance
        // vertex, extract it from priority queue.
        // vertex label is stored in second of pair (it
        // has to be done this way to keep the vertices
        // sorted distance (distance must be first item
        // in pair)
        int s = pq.top().second;
        pq.pop();
        vertices_visited++;

        Index shift = graph->get_row_ptr()[s];
        Index connections_count = graph->get_row_ptr()[s + 1] - graph->get_row_ptr()[s];

        //cout << "source " << _source_vertex << " vs " << s << endl;
        //cout << " s" << s << " ccnt = " << connections_count << " | " << graph->get_row_ptr()[s + 1] << " " << graph->get_row_ptr()[s] << endl;

        for(Index edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            Index v = graph->get_col_ids()[shift + edge_pos];
            T weight = graph->get_vals()[shift + edge_pos];

            if (distance_vals[v] > distance_vals[s] + weight)
            {
                // Updating distance of dst_id
                distance_vals[v] = distance_vals[s] + weight;

                pq.push(make_pair(distance_vals[v], v));
            }
        }
    }
    cout << "SSSP check visited " << vertices_visited << " vertices" << endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}
