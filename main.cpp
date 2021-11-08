#include "src/gb_kun.h"

int main()
{
    VNT scale = 18;
    VNT avg_deg = 16;
    EdgeListContainer<float> el;
    GraphGenerationAPI::random_uniform(el,
                                       pow(2.0, scale),
                                       avg_deg * pow(2.0, scale));

    MatrixCSR<float> A;
    A.import(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count);

    DenseVector<float> x(A.get_size());
    DenseVector<float> y(A.get_size());
    x.set_constant(1);
    y.set_constant(0);

    SpMV(A, x, y);

    MatrixCOO<float> B;
    B.import(el.src_ids.data(), el.dst_ids.data(), el.edge_vals.data(), el.vertices_count, el.edges_count);

    x.set_constant(1);
    y.set_constant(0);

    SpMV(B, x, y);

    return 0;
}
