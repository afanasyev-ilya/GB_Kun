/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void MatrixLAV<T>::import(VNT *_row_ids, VNT *_col_ids, T *_vals, VNT _size, ENT _nz)
{
    // print freqs
    map<int, int> col_freqs;
    map<int, int> row_freqs;

    for(ENT i = 0; i < _nz; i++)
    {
        VNT row_id = _row_ids[i];
        VNT col_id = _col_ids[i];
        col_freqs[col_id]++;
        row_freqs[row_id]++;
    }

    ofstream col_freqs_file;
    col_freqs_file.open ("./output/col_freqs.txt");
    for(auto it = col_freqs.cbegin(); it != col_freqs.cend(); ++it)
    {
        col_freqs_file << it->second << endl;
    }
    col_freqs_file.close();

    ofstream row_freqs_file;
    row_freqs_file.open ("./output/row_freqs.txt");
    for(auto it = row_freqs.cbegin(); it != row_freqs.cend(); ++it)
    {
        row_freqs_file << it->second << endl;
    }
    row_freqs_file.close();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
