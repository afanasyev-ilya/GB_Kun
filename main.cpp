#include "src/gb_kun.h"

#define NUM_ITERS 3

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
    else if(parser.get_synthetic_graph_type() == RANDOM_UNIFORM)
    {
        GraphGenerationAPI::R_MAT(el, pow(2.0, scale), avg_deg * pow(2.0, scale), 57, 19, 19, 5);
        cout << "Using RMAT graph" << endl;
    }

    MatrixCSR<float> A;
    A.import(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count);

    DenseVector<float> x(A.get_size());
    DenseVector<float> y(A.get_size());
    DenseVector<float> z(A.get_size());
    x.set_constant(1);
    y.set_constant(0);
    z.set_constant(0);

    for(int i = 0; i < NUM_ITERS; i++)
        SpMV(A, x, y);
    cout << endl << endl;

    MatrixSegmentedCSR<float> B;
    B.import(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count);

    /*MatrixCOO<float> B;
    B.import(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count, false);

    for(int i = 0; i < NUM_ITERS; i++)
        SpMV(B, x, z);
    cout << endl << endl;

    B.import(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count, true);

    z.set_constant(0);
    for(int i = 0; i < NUM_ITERS; i++)
        SpMV(B, x, z);
    cout << endl << endl;

    if(y == z)
    {
        cout << "Vectors are equal" << endl;
    }
    else
    {
        cout << "Vectors are NOT equal" << endl;
    }*/

    return 0;
}
