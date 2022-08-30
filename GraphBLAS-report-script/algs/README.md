# Required actions
Before compiling using `compile_all.sh` script, you need to perform the following actions:

(1) Go to `gb_kun/CMakeLists.txt` file and change constant `GB_KUN_SRC` to the path where you have the GB_kun project.

(2) Do
```bash
cd ssgb
mkdir third_party && cd third_party
```
Then create links (or copy) there shared libraries of GraphBLAS and LAGraph: `libgraphblas.so`, `liblagraph.so` and `liblagraphx.so`

Now you can run
```bash
./compile_all.sh
```
and fix other issues that cmake will report!