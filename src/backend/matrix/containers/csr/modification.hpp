#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

template<typename T>
bool added_edge_is_valid(VNT asking_row, const std::pair<VNT, ENT>& edge_pair, T edge_weight,
                         const std::map<VNT, std::map<std::pair<VNT, ENT>, T> >& added_edges) {
    const auto &edge_src_id = edge_pair.first;
    const auto &edge_dst_id = edge_pair.second;
    bool edge_is_valid = true;
    if (asking_row != edge_src_id) {
        if (added_edges.find(edge_src_id) == added_edges.end() or
            added_edges[edge_src_id].find(edge_pair) == added_edges[edge_src_id].end() or
            added_edges[edge_src_id][edge_pair] != edge_weight) {
            edge_is_valid = false;
        }
    }
    if (asking_row != edge_dst_id) {
        if (added_edges.find(edge_dst_id) == added_edges.end() or
            added_edges[edge_dst_id].find(edge_pair) == added_edges[edge_dst_id].end() or
            added_edges[edge_dst_id][edge_pair] != edge_weight) {
            edge_is_valid = false;
        }
    }
    return edge_is_valid;
}

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixCSR<T>::add_row(VNT _row)
{
    ++num_changes;
    if (_row < nrows) {
        if (removed_rows.find(_row) != removed_rows.end()) {
            restored_rows.insert(_row);
        } // otherwise, graph already has the vertex
    } else {
        ongoing_modifications = true;
        added_rows.insert(_row);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixCSR<T>::remove_row(VNT _row)
{
    ++num_changes;
    if (added_edges.find(_row) != added_edges.end()) {
        added_edges.erase(_row);
    }
    if (_row < nrows) {
        if (restored_rows.find(_row) != restored_rows.end()) {
            restored_rows.erase(_row);
        } else if (removed_rows.find(_row) == removed_rows.end()) {
            ongoing_modifications = true;
            removed_rows.insert(_row);
        }
    } else {
        if (added_rows.find(_row) != added_rows.end()) {
            added_rows.erase(_row);
        } else {
            LOG_ERROR("Deleting vertices which does not exist...");
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixCSR<T>::add_val(VNT _row, VNT _col, T _val)
{
    ++num_changes;
    ongoing_modifications = true;
    added_edges[_row][std::make_pair(_row, _col)] = _val;
    added_edges[_col][std::make_pair(_row, _col)] = _val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixCSR<T>::remove_val(VNT _row, VNT _col)
{
    num_changes++;
    ongoing_modifications = true;
    if (added_edges.find(_row) != added_edges.end()) {
        if (added_edges[_row].find(std::make_pair(_row, _col))) {
            added_edges[_row].erase(std::make_pair(_row, _col));
        }
    }
    if (added_edges.find(_col) != added_edges.end()) {
        if (added_edges[_col].find(std::make_pair(_row, _col))) {
            added_edges[_col].erase(std::make_pair(_row, _col));
        }
    }
    removed_edges.insert(std::make_pair(_row, _col));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixCSR<T>::apply_modifications()
{
    VNT new_nrows = nrows;
    VNT new_ncols = 0;
    std::vector<VNT> new_row_ptr;
    new_row_ptr.push_back(0);
    std::vector<ENT> new_col_ids;
    std::vector<T> new_vals;

    for (const auto row: added_rows) {
        new_nrows = std::max(row, new_nrows);
    }

    for (VNT row = 0; row < new_nrows; ++row) {
        VNT cur_row_nnz = 0;
        if (row < nrows) {
            // add old vertices
            if (removed_rows.find(row) != removed_rows.end()) {
                if (restored_rows.find(row) != restored_rows.end()) {
                    for (const auto &[edge_pair, edge_weight] : added_edges[row]) {
                        if (edge_pair.first == row and added_edge_is_valid(row, edge_pair, edge_weight, added_edges)) {
                            new_col_ids.push_back(edge_pair.second);
                            new_ncols = std::max(new_ncols, edge_pair.second);
                            new_col_ids.push_back(edge_weight);
                            ++cur_row_nnz;
                        }
                    }
                }
            } else {
                // TODO: implement this case
            }
        } else {
            // add new vertices
            for (const auto &[edge_pair, edge_weight] : added_edges[row]) {
                if (edge_pair.first == row and added_edge_is_valid(row, edge_pair, edge_weight, added_edges)) {
                    new_col_ids.push_back(edge_pair.second);
                    new_ncols = std::max(new_ncols, edge_pair.second);
                    new_col_ids.push_back(edge_weight);
                    ++cur_row_nnz;
                }
            }
        }
        new_row_ptr.push_back(new_row_ptr.back() + cur_row_nnz);
    }

    // clear datastructures, which store updates
    removed_rows.clear();
    restored_rows.clear();
    added_rows.clear();
    removed_edges.clear();
    added_edges.clear();

    nrows = new_nrows;
    ncols = new_ncols;
    nnz = new_col_ids.size();
    resize(nrows, ncols, nnz);
    #pragma omp parallel
    for (VNT i = 0; i < nrows; ++i) {
        row_ptr[i] = new_row_ptr[i];
    }
    #pragma omp parallel
    for (VNT i = 0; i < nnz; ++i) {
        col_ids[i] = new_col_ids[i];
    }
    #pragma omp parallel
    for (VNT i = 0; i < nnz; ++i) {
        vals[i] = new_vals[i];
    }

    calculate_degrees();

    ongoing_modifications = false;
    num_changes = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
