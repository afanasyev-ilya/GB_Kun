#include "src/gb_kun.h"

#define NUM_ITERS 3

#define REPORT_STATS( CallInstruction ) { \
    double bw = CallInstruction;          \
    cout << "BW: " << bw << endl;         \
}

int main(int argc, char **argv)
{
    Parser parser;
    parser.parse_args(argc, argv);
    VNT scale = parser.get_scale();
    VNT avg_deg = parser.get_avg_degree();

    EdgeListContainer<float> el;
    if(parser.get_synthetic_graph_type() == RANDOM_UNIFORM)
    {
        GraphGenerationAPI::random_uniform(el,
                                           pow(2.0, scale),
                                           avg_deg * pow(2.0, scale));
        cout << "Using UNIFORM graph" << endl;
    }
    else if(parser.get_synthetic_graph_type() == RMAT)
    {
        GraphGenerationAPI::R_MAT(el, pow(2.0, scale), avg_deg * pow(2.0, scale), 57, 19, 19, 5);
        cout << "Using RMAT graph" << endl;
    }

    VNT vertices_count = pow(2.0, scale);
    DenseVector<float> x(vertices_count);
    DenseVector<float> y(vertices_count);
    DenseVector<float> z(vertices_count);
    x.set_constant(1);
    y.set_constant(0);
    z.set_constant(0);

    if(parser.get_storage_format() == CSR)
    {
        MatrixCSR<float> A;
        A.import(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count);

        SpMV(A, x, y);
        y.set_constant(0);
        REPORT_STATS(SpMV(A, x, y));
    }
    else if(parser.get_storage_format() == CSR_SEG)
    {
        MatrixSegmentedCSR<float> A;

        A.import(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count);
        SpMV(A, x, y);
        y.set_constant(0);
        REPORT_STATS(SpMV(A, x, y));
    }
    else if(parser.get_storage_format() == COO)
    {
        MatrixCOO<float> A;

        A.import(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count, false);
        SpMV(A, x, y);
        y.set_constant(0);
        REPORT_STATS(SpMV(A, x, y));
    }
    else if(parser.get_storage_format() == COO_OPT)
    {
        MatrixCOO<float> A;

        A.import(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count, true);
        SpMV(A, x, y);
        y.set_constant(0);
        REPORT_STATS(SpMV(A, x, y));
    }
    else if(parser.get_storage_format() == LAV)
    {
        MatrixLAV<float> A;

        A.import(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count);
        SpMV(A, x, y);
        y.set_constant(0);
        REPORT_STATS(SpMV(A, x, y));
    }

    if(parser.check())
    {
        MatrixCSR<float> A;
        A.import(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count);

        SpMV(A, x, z);

        if(y == z)
        {
            cout << "Vectors are equal" << endl;
        }
        else
        {
            cout << "Vectors are NOT equal" << endl;
        }
    }

    return 0;
}
