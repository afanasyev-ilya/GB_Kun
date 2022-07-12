#include "src/gb_kun.h"

enum ModificationTypes { VERTEX_INSERTION_OP, VERTEX_DELETION_OP, EDGE_INSERTION_OP, EDGE_DELETION_OP, MOD_TYPES_LEN };

VNT get_random_vertex_id_to_insert(VNT nrows, VNT ncols) {
    return (rand() % ((nrows * 2)));
}

VNT get_random_vertex_id_to_delete(VNT nrows, VNT ncols) {
    return (rand() % ((int)((double) nrows * 1.2)));
}

std::pair<VNT, VNT> get_random_edge_pair(VNT nrows, VNT ncols) {
    return std::make_pair(rand() % ((int)((double) nrows * 1.2)), rand() % ((int)((double) ncols * 1.2)));
}

float get_random_edge_weight_value() {
    return rand() % 5 + 1;
}

ModificationTypes get_random_modification() {
    const int mod_types_len_value = MOD_TYPES_LEN;
    return static_cast<ModificationTypes>(rand() % mod_types_len_value);
}

void apply_modification(ModificationTypes modification_type, lablas::Matrix<float> &A, std::vector<std::vector<float> > &A_clone, VNT &nrows, VNT &ncols) {
    if (modification_type == VERTEX_INSERTION_OP) {
        const auto vertex_id = get_random_vertex_id_to_insert(nrows, ncols);
        cout << "Applying vertex insertion of <" << vertex_id << ">" << endl;

        A.add_vertex(vertex_id);

        if (vertex_id >= nrows) {
            for (VNT i = 0; i < nrows; ++i) {
                for (VNT j = nrows; j <= vertex_id; ++j) {
                    A_clone[i].push_back(0);
                }
            }
            for (VNT i = nrows; i <= vertex_id; ++i) {
                A_clone.emplace_back(std::vector<float>(vertex_id + 1, 0));
            }
        }
        nrows = std::max(nrows, vertex_id + 1);
        ncols = std::max(ncols, vertex_id + 1);
    } else if (modification_type == VERTEX_DELETION_OP) {
        const auto vertex_id = get_random_vertex_id_to_delete(nrows, ncols);
        cout << "Applying vertex deletion of <" << vertex_id << ">" << endl;

        A.remove_vertex(vertex_id);

        if (vertex_id < nrows) {
            if (vertex_id + 1 == nrows) {
                A_clone.pop_back();
                for (VNT i = 0; i < nrows; ++i) {
                    A_clone[i].pop_back();
                }
                nrows = nrows - 1;
                ncols = ncols - 1;
            } else {
                for (VNT i = 0; i < nrows; ++i) {
                    A_clone[i][vertex_id] = 0;
                }
                for (VNT i = 0; i < ncols; ++i) {
                    A_clone[vertex_id][i] = 0;
                }
            }
        }
    } else if (modification_type == EDGE_INSERTION_OP) {
        const auto [src_id, dst_id] = get_random_edge_pair(nrows, ncols);
        const auto edge_weight = get_random_edge_weight_value();
        cout << "Applying edge insertion of (" << src_id << ", " << dst_id << ", " << edge_weight << ")" << endl;

        A.add_edge(src_id, dst_id, edge_weight);

        if (src_id < nrows and dst_id < ncols) {
            A_clone[src_id][dst_id] = edge_weight;
        }
    } else if (modification_type == EDGE_DELETION_OP) {
        const auto [src_id, dst_id] = get_random_edge_pair(nrows, ncols);
        cout << "Applying edge deletion of (" << src_id << ", " << dst_id <<  ")" << endl;

        A.remove_edge(src_id, dst_id);

        if (src_id < nrows and dst_id < ncols) {
            A_clone[src_id][dst_id] = 0;
        }
    } else {
        throw "unknown modification type";
    }
}

int main(int argc, char **argv) {
    try
    {
        Parser parser;
        parser.parse_args(argc, argv);
        VNT scale = parser.get_scale();
        VNT avg_deg = parser.get_avg_degree();

        lablas::Matrix<float> A;
        A.set_preferred_matrix_format(CSR);
        init_matrix(A, parser);

        if (parser.check()) {
            auto cur_nrows = A.nrows();
            auto cur_ncols = A.ncols();

            std::vector<std::vector<float> > A_clone(cur_nrows, std::vector<float>(cur_ncols));
            int error_cnt = 0;
            for (int i = 0; i < cur_nrows; ++i) {
                for (int j = 0; j < cur_ncols; ++j) {
                    A_clone[i][j] = A.get_matrix()->get_csr()->get(i, j);
                }
            }

            const int op_blocks_cnt = 10;
            const int op_cnt = 10;
            for (int cur_block_op = 0; cur_block_op < op_blocks_cnt; ++cur_block_op) {
                for (int op_id = 0; op_id < op_cnt; ++op_id) {
                    apply_modification(get_random_modification(), A, A_clone, cur_nrows, cur_ncols);
                }

                A.get_matrix()->apply_modifications();

                for (int i = 0; i < cur_nrows; ++i) {
                    for (int j = 0; j < cur_ncols; ++j) {
                        if (A.get_matrix()->get_csr()->get(i, j) != A_clone[i][j]) {
                            ++error_cnt;
                        }
                    }
                }
            }

            cout << "error_count: " << error_cnt << "/" << cur_nrows * cur_ncols << endl;
        }
    }
    catch (string& error)
    {
        cout << error << endl;
    }
    catch (const char * error)
    {
        cout << error << endl;
    }
    return 0;
}

