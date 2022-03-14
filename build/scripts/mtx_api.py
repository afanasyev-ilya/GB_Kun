import sys
import tqdm
import os
import time


def gen_graph(in_filename, out_filename, undir_out_filename, options):
    edge_freqs = {}
    undir_edge_freqs = {}

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
                        undir_edge_freqs[str_key] = 1
                    else:
                        edge_freqs[str_key] += 1
                        undir_edge_freqs[str_key] += 1

                    if row > max_row:
                        max_row = row

                    if col > max_col:
                        max_col = col

                    undir_str_key = str(col) + "_" + str(row)
                    if undir_str_key not in edge_freqs:
                        undir_edge_freqs[undir_str_key] = 1
                    else:
                        undir_edge_freqs[undir_str_key] += 1

                    if col > max_row:
                        max_row = col

                    if row > max_col:
                        max_col = row

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

    if starts_from_zero:
        nrows += 1
        ncols += 1

    # remove old graphs now
    try:
        os.remove(out_filename)
        os.remove(undir_out_filename)
        os.remove(out_filename + "bin")
        os.remove(undir_out_filename + "bin")
    except OSError:
        pass

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

    cnt = 0
    with tqdm.tqdm(total=nnz) as pbar:
        with open(undir_out_filename, 'w') as f:
            f.write('%%MatrixMarket matrix coordinate pattern general\n')
            f.write(str(nrows) + " " + str(ncols) + " " + str(nnz) + '\n')
            for key, v in tqdm.tqdm(undir_edge_freqs.items(), desc="Saving UNDIRECTED MTX file progress"):
                pbar.update(cnt)
                vals = [int(s) for s in key.split("_") if s.isdigit()]

                row = vals[0]
                col = vals[1]
                if starts_from_zero:
                    row += 1
                    col += 1
                f.write(str(row) + " " + str(col) + '\n')
                cnt += 1

    if options.use_binary_graphs:
        cnt = 0
        with tqdm.tqdm(total=nnz) as pbar:
            with open(out_filename + "bin", 'wb') as f:
                f.write(nrows.to_bytes(8, 'little'))
                f.write(ncols.to_bytes(8, 'little'))
                f.write(nnz.to_bytes(8, 'little'))
                for key, v in tqdm.tqdm(edge_freqs.items(), desc="Saving BINARY file progress"):
                    pbar.update(cnt)
                    vals = [int(s) for s in key.split("_") if s.isdigit()]

                    row = vals[0]
                    col = vals[1]
                    if starts_from_zero:
                        row += 1
                        col += 1
                    f.write(row.to_bytes(8, 'little'))
                    f.write(col.to_bytes(8, 'little'))
                    cnt += 1

        cnt = 0
        with tqdm.tqdm(total=nnz) as pbar:
            with open(undir_out_filename + "bin", 'wb') as f:
                f.write(nrows.to_bytes(8, 'little'))
                f.write(ncols.to_bytes(8, 'little'))
                f.write(nnz.to_bytes(8, 'little'))

                for key, v in tqdm.tqdm(undir_edge_freqs.items(), desc="Saving UNDIRECTED BINARY file progress"):
                    pbar.update(cnt)
                    vals = [int(s) for s in key.split("_") if s.isdigit()]

                    row = vals[0]
                    col = vals[1]
                    if starts_from_zero:
                        row += 1
                        col += 1
                    f.write(row.to_bytes(8, 'little'))
                    f.write(col.to_bytes(8, 'little'))
                    cnt += 1
