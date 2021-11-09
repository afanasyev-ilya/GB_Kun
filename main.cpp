#include "src/gb_kun.h"

int main()
{
    Parser parser;
    VNT scale = parser.get_scale();
    VNT avg_deg = parser.get_avg_degree();
    EdgeListContainer<float> el;
    GraphGenerationAPI::random_uniform(el,
                                       pow(2.0, scale),
                                       avg_deg * pow(2.0, scale));

    MatrixCSR<float> A;
    A.import(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count);

    DenseVector<float> x(A.get_size());
    DenseVector<float> y(A.get_size());
    DenseVector<float> z(A.get_size());
    x.set_constant(1);
    y.set_constant(0);
    z.set_constant(0);

    SpMV(A, x, y);

    MatrixCOO<float> B;
    B.import(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count, false);

    SpMV(B, x, z);

    B.import(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count, true);

    z.set_constant(0);
    SpMV(B, x, z);

    if(y == z)
    {
        cout << "Vectors are equal" << endl;
    }
    else
    {
        cout << "Vectors are NOT equal" << endl;
    }

    return 0;
}
