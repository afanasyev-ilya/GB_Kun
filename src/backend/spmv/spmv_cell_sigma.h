/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif /* __ARM_FEATURE_SVE */

namespace lablas{
namespace backend{

template <typename T>
void SpMV_load_balanced(const MatrixCellSigmaC<T> *_matrix,
                        const DenseVector<T> *_x,
                        DenseVector<T> *_y)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();

    #pragma omp parallel
    {
        for(int vg = 0; vg < _matrix->vertex_groups_num; vg++)
        {
            VNT group_size = _matrix->vertex_groups[vg].size;
            #pragma omp for schedule(static)
            for(VNT i = 0; i < group_size; i++)
            {
                VNT row = _matrix->vertex_groups[vg].ids[i];
                for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
                {
                    y_vals[row] += _matrix->vals[j] * x_vals[_matrix->col_ids[j]];
                }
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __ARM_FEATURE_SVE
template <typename T>
void SpMV_vector(const MatrixCellSigmaC<T> *_matrix,
                 const DenseVector<T> *_x,
                 DenseVector<T> *_y)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();

    #pragma omp parallel
    {
        for(int vg = 0; vg < _matrix->cell_c_start_group; vg++)
        {
            #pragma omp for schedule(static)
            for(VNT i = 0; i < _matrix->vertex_groups[vg].size; i++)
            {
                VNT row = _matrix->vertex_groups[vg].ids[i];
                for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
                {
                    y_vals[row] += _matrix->vals[j] * x_vals[_matrix->col_ids[j]];
                }
            }
        }

        for(int vg = 0; vg < _matrix->cell_c_vertex_groups_num; vg++)
        {
            VNT vector_segments_count = _matrix->cell_c_vertex_groups[vg].vector_segments_count;

            #pragma omp for schedule(static)
            for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
            {
                VNT segment_first_vertex = cur_vector_segment * VECTOR_LENGTH;

                ENT segment_edges_start = _matrix->cell_c_vertex_groups[vg].vector_group_ptrs[cur_vector_segment];
                VNT segment_connections_count = _matrix->cell_c_vertex_groups[vg].vector_group_sizes[cur_vector_segment];

                size_t idx = segment_edges_start;
                svbool_t pg = svwhilelt_b64(idx, segment_edges_start + segment_connections_count*VECTOR_LENGTH);

                svfloat64_t tmp0; tmp0 = svadd_z(svpfalse(),tmp0, tmp0);
                svfloat64_t tmp1; tmp1 = svadd_z(svpfalse(),tmp1, tmp1);
                svfloat64_t tmp2; tmp2 = svadd_z(svpfalse(),tmp2, tmp2);
                svfloat64_t tmp3; tmp3 = svadd_z(svpfalse(),tmp3, tmp3);
                double *base = _matrix->cell_c_vertex_groups[vg].vector_group_vals;

                double *base_val0 = &(base[segment_edges_start + 0*svcntd()]);
                double *base_val1 = &(base[segment_edges_start + 1*svcntd()]);
                double *base_val2 = &(base[segment_edges_start + 2*svcntd()]);
                double *base_val3 = &(base[segment_edges_start + 3*svcntd()]);

                for(ENT edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
                {
                    /*svfloat64_t mat_val0 = svld1(pg, base_val0+idx);
                    svfloat64_t mat_val1 = svld1(pg, base_val1+idx);
                    svfloat64_t mat_val2 = svld1(pg, base_val2+idx);
                    svfloat64_t mat_val3 = svld1(pg, base_val3+idx);*/

                    idx += 4*svcntd();
                }
            }
        }
    }
}
#else
template <typename T>
void SpMV_vector(const MatrixCellSigmaC<T> *_matrix,
                 const DenseVector<T> *_x,
                 DenseVector<T> *_y)
{
    const T *x_vals = _x->get_vals();
    T *y_vals = _y->get_vals();

    #pragma omp parallel
    {
        for(int vg = 0; vg < _matrix->cell_c_start_group; vg++)
        {
            #pragma omp for schedule(static)
            for(VNT i = 0; i < _matrix->vertex_groups[vg].size; i++)
            {
                VNT row = _matrix->vertex_groups[vg].ids[i];
                for(ENT j = _matrix->row_ptr[row]; j < _matrix->row_ptr[row + 1]; j++)
                {
                    y_vals[row] += _matrix->vals[j] * x_vals[_matrix->col_ids[j]];
                }
            }
        }

        for(int vg = 0; vg < _matrix->cell_c_vertex_groups_num; vg++)
        {
            VNT vector_segments_count = _matrix->cell_c_vertex_groups[vg].vector_segments_count;

            #pragma omp for schedule(static)
            for(int cur_vector_segment = 0; cur_vector_segment < vector_segments_count; cur_vector_segment++)
            {
                VNT segment_first_vertex = cur_vector_segment * VECTOR_LENGTH;

                ENT segment_edges_start = _matrix->cell_c_vertex_groups[vg].vector_group_ptrs[cur_vector_segment];
                VNT segment_connections_count = _matrix->cell_c_vertex_groups[vg].vector_group_sizes[cur_vector_segment];

                for(ENT edge_pos = 0; edge_pos < segment_connections_count; edge_pos++)
                {
                    for (VNT i = 0; i < VECTOR_LENGTH; i++)
                    {
                        VNT pos = segment_first_vertex + i;
                        VNT row_id = 0;
                        VNT size = _matrix->cell_c_vertex_groups[vg].size;
                        if(pos < size)
                        {
                            row_id = _matrix->cell_c_vertex_groups[vg].row_ids[pos];
                        }

                        if(pos < size)
                        {
                            const VNT vector_index = i;
                            const ENT internal_edge_pos = segment_edges_start + edge_pos * VECTOR_LENGTH + i;
                            const VNT local_edge_pos = edge_pos;
                            const VNT col_id = _matrix->cell_c_vertex_groups[vg].vector_group_col_ids[internal_edge_pos];
                            if(col_id != -1)
                            {
                                const T val = _matrix->cell_c_vertex_groups[vg].vector_group_vals[internal_edge_pos];
                                y_vals[row_id] += val * x_vals[col_id];
                            }
                        }
                    }
                }
            }
        }
    }
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void SpMV(const MatrixCellSigmaC<T> *_matrix,
          const DenseVector<T> *_x,
          DenseVector<T> *_y)
{
    //SpMV_load_balanced(_matrix, _x, _y);
    SpMV_vector(_matrix, _x, _y);
}

}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
