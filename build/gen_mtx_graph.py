import sys

args = sys.argv[1:]
filename = args[0]
out_filename = args[1]
edge_freqs = {}

cnt = 0
with open(filename) as file:
    for line in file:
        if cnt < 10:
            print(line.rstrip())

        if cnt == 1:
            vals = [int(s) for s in line.split() if s.isdigit()]
            nrows = vals[0]
            ncols = vals[1]
            nnz = vals[2]

        if cnt >= 2:
            vals = [int(s) for s in line.split() if s.isdigit()]
            str_key = str(vals[0]) + "_" + str(vals[1])
            if str_key not in edge_freqs:
                edge_freqs[str_key] = 1
            else:
                edge_freqs[str_key] += 1
        cnt += 1

print("nrows: " + str(nrows))
print("ncols: " + str(ncols))
print("nnz: " + str(nnz))

with open(out_filename, 'w') as f:
    new_nnz = len(edge_freqs.keys())
    f.write('%%MatrixMarket matrix coordinate pattern general\n')
    f.write(str(nrows) + " " + str(ncols) + " " + str(new_nnz) + '\n')
    for key in edge_freqs:
        vals = [int(s) for s in key.split("_") if s.isdigit()]
        f.write(str(vals[0]) + " " + str(vals[1]) + '\n')