#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace lablas {
namespace algorithm {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void bfs_traditional(Vector <T> *_levels,
                     const Matrix <T> *_graph,
                     Index _source_vertex)
{
    Index vertices_count = _levels->get_vector()->getDense()->get_size();
    T* levels = _levels->get_vector()->getDense()->get_vals();
    backend::MatrixCSR<T> *graph = ((backend::MatrixCSR<T> *) _graph->get_matrix()->get_csr());

    // Mark all the vertices as not visited
    for(int i = 0; i < vertices_count; i++)
        levels[i] = 0;

    // Create a queue for BFS
    list<Index> queue;

    // Mark the current node as visited and enqueue it
    levels[_source_vertex] = 1;
    queue.push_back(_source_vertex);

    while(!queue.empty())
    {
        // Dequeue a vertex from queue and print it
        Index s = queue.front();
        queue.pop_front();

        Index shift = graph->get_row_ptr()[s];
        Index connections_count = graph->get_row_ptr()[s + 1] - graph->get_row_ptr()[s];
        for(Index edge_pos = 0; edge_pos < connections_count; edge_pos++)
        {
            Index v = graph->get_col_ids()[shift + edge_pos];
            if (levels[v] == 0)
            {
                levels[v] = levels[s] + 1;
                queue.push_back(v);
            }
        }
    }
}

template <typename T>
bool equal_components(Vector<T> &_first,
                      Vector<T> &_second)
{
    // check if sizes are the same
    if(_first.size() != _second.size())
    {
        cout << "Results are NOT equal, incorrect sizes";
        return false;
    }

    // construct equality maps
    map<int, int> f_s_equality;
    map<int, int> s_f_equality;
    int vertices_count = _first.size();
    for (int i = 0; i < vertices_count; i++)
    {
        f_s_equality[_first[i]] = _second[i];
        s_f_equality[_second[i]] = _first[i];
    }

    // check if components are equal using maps
    bool result = true;
    int error_count = 0;
    for (int i = 0; i < vertices_count; i++)
    {
        if (f_s_equality[_first[i]] != _second[i])
        {
            result = false;
            error_count++;
        }
        if (s_f_equality[_second[i]] != _first[i])
        {
            result = false;
            error_count++;
        }
    }
    cout << "error count: " << error_count << endl;
    if(error_count == 0)
        cout << "Results are equal" << endl;
    else
        cout << "Results are NOT equal, error_count = " << error_count << endl;

    return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
}