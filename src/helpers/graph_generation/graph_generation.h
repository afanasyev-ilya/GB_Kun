#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum ConvertDirectionType
{
    DirectedToDirected = 0,
    DirectedToUndirected = 1,
    UndirectedToDirected = 2,
    UndirectedToUndirected = 3
};

enum DirectionType
{
    UNDIRECTED_GRAPH = 0,
    DIRECTED_GRAPH = 1
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct EdgeListContainer
{
    vector<VNT> src_ids;
    vector<VNT> dst_ids;
    vector<T> edge_vals;
    VNT vertices_count;
    ENT edges_count;

    void save_as_mtx(string _file_name);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class GraphGenerationAPI
{
public:
    template <typename T>
    static void random_uniform(EdgeListContainer<T> &_edges_container,
                               VNT _vertices_count,
                               ENT _edges_count,
                               DirectionType _direction_type = DIRECTED_GRAPH);

    template <typename T>
    static void RMAT(EdgeListContainer<T> &_edges_container,
                     VNT _vertices_count,
                     ENT _edges_count,
                     int _a_prob,
                     int _b_prob,
                     int _c_prob,
                     int _d_prob,
                     DirectionType _direction_type = DIRECTED_GRAPH);

    template <typename T>
    static void HPCG(EdgeListContainer<T> &_edges_container, VNT _nx, VNT _ny, VNT _nnz, ENT _edge_factor);

    template <typename T>
    static void init_from_txt_file(EdgeListContainer<T> &_edges_container, string _txt_file_name,
                                   DirectionType _direction_type = DIRECTED_GRAPH);

    template <typename T>
    static void init_from_mtx_file(EdgeListContainer<T> &_edges_container, string _mtx_file_name);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "edges_container.hpp"
#include "graph_generation.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
