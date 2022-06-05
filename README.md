GB_Kun is a high-performance graph-processing library, based on the GraphBLAS standard.
Primary target hardware of GB_Kun is Kunpeng 920 processors (48-core and 64-core),
but it can be also launched on other multicore CPU if all the software requirements are provides.

Software Requirements:

1. CMake >= 3

2. GCC >= 8.7

3. Python >= 3.6

4. Python packages: xlswriter, matlibplot, tqdm, bs4

5. GoogleTest

6. clang-format >= 10

***Important***: for matlibplot installation use

```bash
yum install python3-devel
pip3 install matplotlib
```


On CentOS 8, yum must be fixed according to the following instruction:


***gtest installation***

```bash
cd /usr/src/gtest
sudo mkdir build
cd build
sudo cmake .. -DBUILD_GMOCK=OFF
sudo make install
```

***Compilation***

GB_Kun uses cmake for compilation purposes. In order to compiler GB_Kun, use the following commands:

```bash
cd GB_Kun
mkdir build
cd build
cmake ..
```

All the compiled binaries will be located in GB_kun/build. Currently, the following binaries are compiled:

1. cc - implementation of the connected components problem.
2. bfs - implementation of the breadth-first search algorithms.
3. pr - implementation of the page rank algorithms.
4. tc - implementation of triangle counting problem.
5. sssp - implementation of single source shortest paths problem.
5. spmv - implementation of sparse matrix-vector multiplication.
6. gemm - implementation of sparse matrix-matrix multiplication.

Other binaries are temporary and will be removed in the final version of the project, serving for temporary experiments and performance evaluation.

***Benchmarking***

In order to benchmark implemented graph algorithms, use the following commands:

```bash
cd benchmark
python3 ./run_tests.py --apps=cc,pr,bfs --name=output_file --benchmark --verify --mode=best
```

All the benchmarking and verification results will be saved into output_file.xlsx table (on different pages).

To check all available benchmarking options, type:

```bash
python3 ./run_tests.py --help
```

For example, other options, such as --scaling, are available.

***Logging***

We have implemented logger interface for convenient working process with GB_Kun library

Logging in particular parts of a code are implemented via function-like macros 
LOG_TRACE, LOG_DEBUG or LOG_ERROR

In order to output data from such parts of code, you need to set environment variable LOG_LEVEL like
```bash
LOG_LEVEL=trace ./bfs ...
```

***Documentation***

To generate documentations for all algorithms, go to the directory with algorithms and use doxygen command:
```bash
cd GB_Kun/algorithms
doxygen
```
```/documentation``` directory will be generated with html and latex folders which contain generated documentation in different formats.

***Downloading graphs from Konect***

To download graphs from connect, type:
```bash
cd GB_Kun/benchmark
python3 ./load_feature_maps_from_konnect.py --cnt 100 --file output
```

Parameter ```--cnt``` is used to specify how many graphs to download. 

If ```--cnt``` parameter is not specified all graphs will be downloaded.

Parameter ```--file``` is used to specify the name of the output file.

If ```--file``` parameter is not specified the output file will be named ```dict.pickle```.

***Handling project Code Style***

For this project we applied a Code Style that is specified in `.clang-format` file in the root folder of the repository.

In order to be able to do both safe checking and applying Code Style in a whole project, the has two main options:
- `--apply` to apply the Code Style to all `.cpp`, `.c`, `.hpp`, `.h` files in project
- `--check` to check all these files for a Code Style violations

So for example to check and apply Code Style to a whole project you can use the following command:
```bash
python3 codestyle_formatter.py --apply --check
```

It's important to mention that in order to ignore `#pragma`, `#ifdef` and other preprocessing directives indentations 
in project files, mentioned above options are modifying source code before applying `clang-format` so the indentations
for these preprocessing directives are ignored. In order to avoid this behaviour we added one more option `--safe` which
could be used for a safe check/apply as follows:
```bash
python3 codestyle_formatter.py --apply --safe
```
