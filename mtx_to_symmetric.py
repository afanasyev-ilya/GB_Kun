import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

with open(input_path) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

mtx_header = lines[0]
num_rows, num_cols, nnz = lines[1].split()

edges = set(map(lambda x: tuple(x.split()[:2]), lines[2:]))

new_edges = set()
for edge in edges:
    if (edge[1], edge[0]) not in edges:
        new_edges.add((edge[1], edge[0]))

edges = edges.union(new_edges)

with open(output_path, 'w') as file:
    print(mtx_header, file=file)
    print(num_rows, num_cols, len(edges), file=file)
    print("\n".join(map(lambda e: f"{e[0]} {e[1]}", edges)), file=file)
