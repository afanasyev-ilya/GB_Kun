#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void init_matrix(lablas::Matrix<T> &_matrix, Parser &_parser)
{
    VNT scale = _parser.get_scale();
    VNT avg_deg = _parser.get_avg_degree();

    if(_parser.get_synthetic_graph_type() == MTX_GRAPH)
    {
        _matrix.init_from_mtx(_parser.get_file_name());
    }
    else
    {
        EdgeListContainer<T> edges_container;

        if(_parser.get_synthetic_graph_type() == RANDOM_UNIFORM_GRAPH)
        {
            GraphGenerationAPI::random_uniform(edges_container,
                                               pow(2.0, scale),
                                               avg_deg * pow(2.0, scale), DIRECTED_GRAPH);
        }
        else if(_parser.get_synthetic_graph_type() == RMAT_GRAPH)
        {
            GraphGenerationAPI::RMAT(edges_container, pow(2.0, scale), avg_deg * pow(2.0, scale), 57, 19, 19, 5, DIRECTED_GRAPH);
        }
        else if(_parser.get_synthetic_graph_type() == HPCG_GRAPH)
        {
            GraphGenerationAPI::HPCG(edges_container, scale, scale, scale, avg_deg);
        }
        else if(_parser.get_synthetic_graph_type() == REAL_WORLD_GRAPH)
        {
            GraphGenerationAPI::init_from_txt_file(edges_container, _parser.get_file_name(), DIRECTED_GRAPH);
        }

        #ifdef NEED_GEMM
        edges_container.remove_duplicated_edges();
        #endif

        const std::vector<VNT> src_ids(edges_container.src_ids);
        const std::vector<VNT> dst_ids(edges_container.dst_ids);
        std::vector<T> edge_vals(edges_container.edge_vals);

        _matrix.build(&src_ids, &dst_ids, &edge_vals, edges_container.edges_count, GrB_NULL_POINTER);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
