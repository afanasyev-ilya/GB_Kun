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

    #define MASK_NULL static_cast<const lablas::Vector<T>*>(NULL)

    for(int run = 0; run < 1; run++)
    {
        int SPARSITY_K = 100;
        cout << "[ SPARSITY_K = " << SPARSITY_K << " ]" << endl;

        for (GrB_Index indexes_num = 1; indexes_num <= el.vertices_count / 100; indexes_num += 100) {
            lablas::Vector<T> w(el.vertices_count);
            lablas::Vector<T> u(el.vertices_count);

            vector<GrB_Index> sparse_indexes(indexes_num);
            for (GrB_Index i = 0; i < indexes_num; i++)
                sparse_indexes[i] = rand() % el.vertices_count;
            GrB_TRY(GrB_assign(&u, MASK_NULL, NULL, 1, &sparse_indexes[0], indexes_num, NULL));

            GrB_mxv(&w, MASK_NULL, NULL, lablas::PlusMultipliesSemiring<T>(), &matrix, &u, &desc);
        }
    }

    #undef MASK_NULL
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

