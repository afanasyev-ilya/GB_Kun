#include "src/gb_kun.h"

template<typename T>
void test_sparse(int argc, char **argv)
{
    Parser parser;
    parser.parse_args(argc, argv);
    VNT scale = parser.get_scale();
    VNT avg_deg = parser.get_avg_degree();

    lablas::Matrix<float> matrix;
    matrix.set_preferred_matrix_format(parser.get_storage_format());
    init_matrix(matrix, parser);

    GrB_Index size;
    matrix.get_nrows(&size);
    LAGraph_Graph<float> graph(matrix);

    lablas::Vector<T> w(size);
    lablas::Vector<T> u(size);

    #define MASK_NULL static_cast<const lablas::Vector<T>*>(NULL)

    lablas::Descriptor desc;
    for(int run = 0; run < 1; run++)
    {
        w.fill(0.0);

        int SPARSITY_K = 100;
        const GrB_Index indexes_num = ceil(size / SPARSITY_K);
        vector<GrB_Index> sparse_indexes(indexes_num);
        for(GrB_Index i = 0; i < indexes_num; i++)
            sparse_indexes[i] = rand() % size;
        GrB_TRY( GrB_assign(&u, MASK_NULL, NULL, 1, &sparse_indexes[0], indexes_num, NULL));

        GrB_mxv(&w, MASK_NULL, NULL, lablas::PlusMultipliesSemiring<T>(), &matrix, &u, &desc);
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

