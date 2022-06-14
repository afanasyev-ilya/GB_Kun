/**
  @file cc_traditional.hpp
  @author Lastname:Firstname:A00123456:cscxxxxx
  @version Revision 1.1
  @brief Traditional CC algorithm.
  @details Detailed description.
  @date May 12, 2022
*/

#ifndef GB_KUN_CC_TRADITIONAL_HPP
#define GB_KUN_CC_TRADITIONAL_HPP
#define COMPONENT_UNSET -1
#define FIRST_COMPONENT 1

//! Lablas namespace
namespace lablas {
//! Algorithm namespace
    namespace algorithm {

/**
 * Sequential implementation of Connected Components algorithm based on simple graph primitives
 * @brief The function does implement sequential CC
 * @param _components Vector to store component labels
 * @param _graph Target matrix representing an input graph
*/

template <typename T>
void cc_bfs_based_sequential(Vector <T> *_components,
                             const Matrix <T> *_graph)
{
    Index vertices_count = _components->get_vector()->getDense()->get_size();
    T* components = _components->get_vector()->getDense()->get_vals();
    backend::MatrixCSR<T> *graph = ((backend::MatrixCSR<T> *) _graph->get_matrix()->get_csr());

    for(int v = 0; v < vertices_count; v++)
    {
        components[v] = COMPONENT_UNSET;
    }

    int current_component_num = FIRST_COMPONENT;
    for(int current_vertex = 0; current_vertex < vertices_count; current_vertex++)
    {
        if(components[current_vertex] == COMPONENT_UNSET)
        {
            int source_vertex = current_vertex;
            list<int> queue;
            components[source_vertex] = current_component_num;
            queue.push_back(source_vertex);

            while(!queue.empty())
            {
                int s = queue.front();
                queue.pop_front();

                Index shift = graph->get_row_ptr()[s];
                Index connections_count = graph->get_row_ptr()[s + 1] - graph->get_row_ptr()[s];
                for(Index edge_pos = 0; edge_pos < connections_count; edge_pos++)
                {
                    Index dst_id = graph->get_col_ids()[shift + edge_pos];
                    if(components[dst_id] == COMPONENT_UNSET)
                    {
                        components[dst_id] = current_component_num;
                        queue.push_back(dst_id);
                    }
                }
            }

            current_component_num++;
        }
    }

    //_components->print();
}

}
}


#endif //GB_KUN_CC_TRADITIONAL_HPP
