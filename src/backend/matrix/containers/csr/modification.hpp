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
            added_edges.at(edge_src_id).find(edge_pair) == added_edges.at(edge_src_id).end() or
            added_edges.at(edge_src_id).at(edge_pair) != edge_weight) {
            edge_is_valid = false;
        }
    }
    if (asking_row != edge_dst_id) {
        if (added_edges.find(edge_dst_id) == added_edges.end() or
            added_edges.at(edge_dst_id).find(edge_pair) == added_edges.at(edge_dst_id).end() or
            added_edges.at(edge_dst_id).at(edge_pair) != edge_weight) {
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

    if (removed_vertices.find(_row) != removed_vertices.end()) {
        removed_vertices.erase(_row);
    }

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

    removed_vertices.insert(_row);

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
    auto cur_max_rows = nrows;
    if (!added_rows.empty()) {
        cur_max_rows = max(cur_max_rows, *(added_rows.rbegin()) + 1);
    }
    if (_row >= cur_max_rows or _col >= cur_max_rows or
        removed_vertices.find(_row) != removed_vertices.end() or removed_vertices.find(_col) != removed_vertices.end() or
        (_row >= nrows and added_rows.find(_row) == added_rows.end()) or (_col >= nrows and added_rows.find(_col) == added_rows.end())) {
        LOG_ERROR("Adding edge between non-existent vertices...");
        return;
    }
    ongoing_modifications = true;
    added_edges[_row][std::make_pair(_row, _col)] = _val;
    added_edges[_col][std::make_pair(_row, _col)] = _val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixCSR<T>::remove_val(VNT _row, VNT _col)
{
    ++num_changes;
    if (removed_vertices.find(_row) != removed_vertices.end() or removed_vertices.find(_col) != removed_vertices.end()) {
        LOG_ERROR("Removing edge between non-existent vertices...");
        return;
    }
    ongoing_modifications = true;
    if (added_edges.find(_row) != added_edges.end()) {
        if (added_edges[_row].find(std::make_pair(_row, _col)) != added_edges[_row].end()) {
            added_edges[_row].erase(std::make_pair(_row, _col));
        }
    }
    if (added_edges.find(_col) != added_edges.end()) {
        if (added_edges[_col].find(std::make_pair(_row, _col)) != added_edges[_col].end()) {
            added_edges[_col].erase(std::make_pair(_row, _col));
        }
    }
    removed_edges.insert(std::make_pair(_row, _col));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixCSR<T>::soft_apply_modifications() {
    if (ongoing_modifications and num_changes > 1000) {
        apply_modifications();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixCSR<T>::apply_modifications()
{
    if (!ongoing_modifications) {
        return;
    }

    VNT new_nrows = nrows;
    VNT new_ncols = 0;
    std::vector<VNT> new_row_ptr;
    new_row_ptr.push_back(0);
    std::vector<ENT> new_col_ids;
    std::vector<T> new_vals;

    for (const auto row: added_rows) {
        new_nrows = std::max(row + 1, new_nrows);
    }

    for (VNT row = 0; row < new_nrows; ++row) {
        VNT cur_row_nnz = 0;
        if (removed_vertices.find(row) == removed_vertices.end()) {
            if (row < nrows) {
                // add old vertices
                if (removed_rows.find(row) != removed_rows.end()) {
                    if (restored_rows.find(row) != restored_rows.end()) {
                        // old but deleted and then restored vertices
                        for (const auto &[edge_pair, edge_weight] : added_edges[row]) {
                            if (edge_pair.first == row
                                and removed_vertices.find(edge_pair.second) == removed_vertices.end()
                                and added_edge_is_valid(row, edge_pair, edge_weight, added_edges)) {
                                new_col_ids.push_back(edge_pair.second);
                                new_ncols = std::max(new_ncols, edge_pair.second);
                                new_vals.push_back(edge_weight);
                                ++cur_row_nnz;
                            }
                        }
                    }
                } else {
                    // old and not deleted vertices
                    std::set<std::pair<VNT, ENT> > just_added_edges;
                    for (const auto &[edge_pair, edge_weight] : added_edges[row]) {
                        if (edge_pair.first == row and removed_vertices.find(edge_pair.second) == removed_vertices.end()
                            and added_edge_is_valid(row, edge_pair, edge_weight, added_edges)) {
                            new_col_ids.push_back(edge_pair.second);
                            new_ncols = std::max(new_ncols, edge_pair.second);
                            new_vals.push_back(edge_weight);
                            just_added_edges.insert(edge_pair);
                            ++cur_row_nnz;
                        }
                    }
                    for (ENT i = row_ptr[row]; i < row_ptr[row + 1]; ++i) {
                        VNT col = col_ids[i];
                        T val = vals[i];
                        const auto cur_edge_pair = std::make_pair(row, col);
                        if (removed_rows.find(col) == removed_rows.end()
                            and just_added_edges.find(cur_edge_pair) == just_added_edges.end()
                            and removed_edges.find(cur_edge_pair) == removed_edges.end()
                            and removed_vertices.find(col) == removed_vertices.end()) {
                            new_col_ids.push_back(col);
                            new_ncols = std::max(new_ncols, col);
                            new_vals.push_back(val);
                            ++cur_row_nnz;
                        }
                    }
                }
            } else {
                // add new vertices
                if (added_rows.find(row) == added_rows.end()) {
                    removed_vertices.insert(row);
                } else {
                    for (const auto &[edge_pair, edge_weight] : added_edges[row]) {
                        if (edge_pair.first == row and removed_vertices.find(edge_pair.second) == removed_vertices.end()
                            and added_edge_is_valid(row, edge_pair, edge_weight, added_edges)) {
                            new_col_ids.push_back(edge_pair.second);
                            new_ncols = std::max(new_ncols, edge_pair.second);
                            new_vals.push_back(edge_weight);
                            ++cur_row_nnz;
                        }
                    }
                }
            }
        }
        new_row_ptr.push_back(new_row_ptr.back() + cur_row_nnz);
    }

    std::set<VNT> new_removed_vertices;

    for (const auto el : removed_vertices) {
        if (el < new_nrows) {
            new_removed_vertices.insert(el);
        }
    }

    removed_vertices = new_removed_vertices;

    // clear datastructures, which store updates
    removed_rows.clear();
    restored_rows.clear();
    added_rows.clear();
    removed_edges.clear();
    added_edges.clear();

    nrows = new_nrows;
    ncols = new_nrows;
    nnz = new_col_ids.size();
    resize(nrows, ncols, nnz);
    #pragma omp parallel
    for (VNT i = 0; i <= nrows; ++i) {
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
