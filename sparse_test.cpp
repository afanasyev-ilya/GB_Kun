#include "src/gb_kun.h"

template<typename T>
void test_sparse(int argc, char **argv)
{
    Parser parser;
    parser.parse_args(argc, argv);

    EdgeListContainer<T> el;
    GraphGenerationAPI::generate_synthetic_graph(el, parser);

    lablas::Descriptor desc;

    lablas::Matrix<T> matrix;
    /* TODO clearance of ELC vectors in order to free storage */
    const std::vector<VNT> src_ids(el.src_ids);
    const std::vector<VNT> dst_ids(el.dst_ids);
    std::vector<T> edge_vals(el.edge_vals);
    matrix.set_preferred_matrix_format(CSR);
    LA_Info info = matrix.build(&src_ids, &dst_ids, &edge_vals, el.vertices_count, GrB_NULL_POINTER);

    lablas::Vector<T> w(el.vertices_count);
    lablas::Vector<T> u(el.vertices_count);

    for(int run = 0; run < 1; run++)
    {
        w.fill(0.0);

        /*VNT matrix_size = el.vertices_count;
        VNT num_non_zeroes = 100;
        u.fill(0.0);
        for(VNT i = 0; i < num_non_zeroes; i++)
        {
            u.set_element(1.0, rand()%matrix_size);
        }*/
        u.fill(1.0);

        lablas::mxv<T, T, T, T>(&w, NULL, nullptr, lablas::PlusMultipliesSemiring<T>(), &matrix, &u, &desc);
    }
}

int main(int argc, char **argv) {
    try
    {
        test_sparse<float>(argc, argv);
    }
    catch (string error)
    {
        cout << error << endl;
    }
    catch (const char * error)
    {
        cout << error << endl;
    }
    return 0;
}

