import sys
import tqdm
import os
import time


def gen_mtx_graph(in_filename, out_filename):
    try:
        os.remove(out_filename)
    except OSError:
        pass

    edge_freqs = {}

    max_row = 0
    max_col = 0
    starts_from_zero = False

    with tqdm.tqdm(total=os.path.getsize(in_filename), desc='Reading original file progress') as pbar:
        with open(in_filename) as file:
            for line in file:
                if "%" in line: # header line
                    print(line)
                else: # data line
                    pbar.update(len(line))
                    vals = [int(s) for s in line.split() if s.isdigit()]
                    row = int(vals[0])
                    col = int(vals[1])

                    str_key = str(row) + "_" + str(col)
                    if str_key not in edge_freqs:
                        edge_freqs[str_key] = 1
                    else:
                        edge_freqs[str_key] += 1

                    if row > max_row:
                        max_row = row

                    if col > max_col:
                        max_col = col

                    if row == 0 or col == 0:
                        starts_from_zero = True

    nrows = max(max_row + 1, max_col + 1)
    ncols = max(max_row + 1, max_col + 1)
    nnz = int(len(edge_freqs.keys()))
    print("nrows: " + str(nrows))
    print("ncols: " + str(ncols))
    print("nnz: " + str(nnz))
    print("avg degree: " + str(nnz / max(nrows, ncols)))
    print("starts_from_zero: " + str(starts_from_zero))

    if(starts_from_zero):
        nrows += 1
        ncols += 1

    cnt = 0
    with tqdm.tqdm(total=nnz) as pbar:
        with open(out_filename, 'w') as f:
            f.write('%%MatrixMarket matrix coordinate pattern general\n')
            f.write(str(nrows) + " " + str(ncols) + " " + str(nnz) + '\n')
            for key, v in tqdm.tqdm(edge_freqs.items(), desc="Saving MTX file progress"):
                pbar.update(cnt)
                vals = [int(s) for s in key.split("_") if s.isdigit()]

                row = vals[0]
                col = vals[1]
                if starts_from_zero:
                    row += 1
                    col += 1
                f.write(str(row) + " " + str(col) + '\n')
                cnt += 1


def gen_undirected_mtx_graph(in_filename, out_filename):
    try:
        os.remove(out_filename)
    except OSError:
        pass

    edge_freqs = {}

    max_row = 0
    max_col = 0
    starts_from_zero = False

    with tqdm.tqdm(total=os.path.getsize(in_filename), desc='Reading original file progress') as pbar:
        with open(in_filename) as file:
            for line in file:
                if "%" in line: # header line
                    print(line)
                else: # data line
                    pbar.update(len(line))
                    vals = [int(s) for s in line.split() if s.isdigit()]
                    row = int(vals[0])
                    col = int(vals[1])

                    str_key = str(row) + "_" + str(col)
                    if str_key not in edge_freqs:
                        edge_freqs[str_key] = 1
                    else:
                        edge_freqs[str_key] += 1

                    if row > max_row:
                        max_row = row

                    if col > max_col:
                        max_col = col

                    if row == 0 or col == 0:
                        starts_from_zero = True

                    undir_str_key = str(col) + "_" + str(row)
                    if undir_str_key not in edge_freqs:
                        edge_freqs[undir_str_key] = 1
                    else:
                        edge_freqs[undir_str_key] += 1

                    if col > max_row:
                        max_row = col

                    if row > max_col:
                        max_col = row

    nrows = max(max_row + 1, max_col + 1)
    ncols = max(max_row + 1, max_col + 1)
    nnz = int(len(edge_freqs.keys()))
    print("nrows: " + str(nrows))
    print("ncols: " + str(ncols))
    print("nnz: " + str(nnz))
    print("avg degree: " + str(nnz / max(nrows, ncols)))
    print("starts_from_zero: " + str(starts_from_zero))

    if(starts_from_zero):
        nrows += 1
        ncols += 1

    cnt = 0
    with tqdm.tqdm(total=nnz) as pbar:
        with open(out_filename, 'w') as f:
            f.write('%%MatrixMarket matrix coordinate pattern general\n')
            f.write(str(nrows) + " " + str(ncols) + " " + str(nnz) + '\n')
            for key, v in tqdm.tqdm(edge_freqs.items(), desc="Saving MTX file progress"):
                pbar.update(cnt)
                vals = [int(s) for s in key.split("_") if s.isdigit()]

                row = vals[0]
                col = vals[1]
                if starts_from_zero:
                    row += 1
                    col += 1
                f.write(str(row) + " " + str(col) + '\n')
                cnt += 1