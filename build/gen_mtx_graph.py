import sys


def main():
    args = sys.argv[1:]
    in_filename = args[0]
    out_filename = args[1]

    edge_freqs = {}

    max_row = 0
    max_col = 0
    starts_from_zero = False

    with open(in_filename) as file:
        for line in file:
            if "%" in line: # header line
                print(line)
            else: # data line
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

    nrows = max_row + 1
    ncols = max_col + 1
    nnz = int(len(edge_freqs.keys()))
    print("nrows: " + str(nrows))
    print("ncols: " + str(ncols))
    print("nnz: " + str(nnz))
    print("starts_from_zero: " + str(starts_from_zero))

    if(starts_from_zero)
        nrows += 1
        ncols += 1

    with open(out_filename, 'w') as f:
        f.write('%%MatrixMarket matrix coordinate pattern general\n')
        f.write(str(nrows) + " " + str(ncols) + " " + str(nnz) + '\n')
        for key in edge_freqs:
            vals = [int(s) for s in key.split("_") if s.isdigit()]

            row = vals[0]
            col = vals[1]
            if starts_from_zero:
                row += 1
                col += 1
            f.write(str(row) + " " + str(col) + '\n')


if __name__ == "__main__":
    main()