#pragma once

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
    num_changes++;
    ongoing_modifications = true;
    new_matrix_rows[_row][_col] = _val;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixCSR<T>::remove_val(VNT _row, VNT _col)
{
    throw "MatrixCSR<T>::remove_edge : not implemented yet";
    num_changes++;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void MatrixCSR<T>::apply_modifications()
{
    if(num_changes < 1000) // TODO improve this criteria
        return;

    std::unordered_map<VNT, std::unordered_map<VNT, T>> new_graph;

    VNT new_nrows = nrows;
    VNT new_ncols = 0;
    ENT new_nnz = 0;
    for(const auto &[row, row_data]: new_matrix_rows)
        new_nrows = std::max(row, new_nrows);

    // remove rows marked for deletion and save others into temporary graph
    for(VNT row = 0; row < nrows; row++)
    {
        if(!row_marked_for_removal(row))
        {
            for(ENT i = row_ptr[row]; i < row_ptr[row + 1]; i++)
            {
                VNT col = col_ids[i];
                VNT val = vals[i];
                if(!val_marked_for_removal(i)) {
                    new_graph[row][col] = val;
                    new_ncols = std::max(col, new_ncols);
                    new_nnz++;
                }
            }
        }
    }

    // add new vertices
    for(const auto &[row, row_data]: new_matrix_rows)
    {
        for(const auto &[col, val]: row_data)
        {
            new_graph[row][col] = val;
            new_ncols = std::max(col, new_ncols);
            new_nnz++;
        }
    }

    // clear datastructures, which store updates
    new_matrix_rows.clear();
    removed_rows.clear();
    removed_edges.clear();

    nrows = new_nrows;
    ncols = new_ncols;
    nnz = new_nnz;

    resize(nrows, ncols, nnz);

    ENT cur_pos = 0;
    for(const auto &[row, row_data]: new_matrix_rows)
    {
        row_ptr[row] = cur_pos;
        row_ptr[row + 1] = cur_pos + row_data.size();
        for(const auto &[col, val]: row_data)
        {
            col_ids[cur_pos] = col;
            vals[cur_pos] = val;
            cur_pos++;
        }
    }

    ongoing_modifications = false;
    num_changes = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////