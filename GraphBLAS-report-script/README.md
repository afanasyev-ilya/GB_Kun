# GraphBLAS reporting sript

# Running

In order to run tests do:
```bash
./run-make-report.py data --operations mxm mxv vxm --algorithms bfs cc pagerank tc sssp
```
Do
```bash
./run-make-report.py --help
```
for more information.

# Building binaries
the folder `algs` contains a script `compile_all.sh` that will compile all the sources of all supported backends. Usage:
```bash
cd algs && ./compile_all.sh
```
Some extra work required before successful compilation. See README in `algs` for deteails.
# Settings

In the file `run-make-report.py` you can replace the paths to the binaries of each backend if necessary.
The required structure of the directory:
```
backend/build
└───algotithms
│   └───sssp
│   │   │   sppp_1
│   │   │   sssp_bellman_ford
│   │   │   ...
│   │   
│   └───bfs
│   │   │   bfs_0
│   │   │   bfs_another
│   │   │   ...
│   │
│   └───...    
│
└───operations
    └───mxm
    │   │   mxm_something
    │   │   mxm_SpMSpM
    │   │   ...
    │   
    └───mxv
    │   │   mxv_sparse
    │   │   mxv_not_sparse
    │   │   ...
    │
    └───...  
```

