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
};

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
    static void HPCG(EdgeListContainer<T> &_edges_container, VNT _nx, VNT _ny, VNT _nz, ENT _edge_factor);

    template <typename T>
    static void init_from_txt_file(EdgeListContainer<T> &_edges_container, string _txt_file_name,
                                   DirectionType _direction_type = DIRECTED_GRAPH);

    template <typename T>
    static void generate_synthetic_graph(EdgeListContainer<T> &_edges_container, Parser &_parser);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "graph_generation.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
