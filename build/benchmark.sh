function quit {
  exit
}

function benchmark_graph {
    prefix="/home/afanasyev/LAGraph/src/benchmark/mtx_graphs"
    app_name="./"$1
    graph_name=$2
    graph_path=$prefix"/"$graph_name

    rm perf_stats.txt

    export OMP_NUM_THREADS=48
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores

    $app_name -graph mtx $graph_path -format CSR_SEG > dump.txt

    python3 ./analize_data.py --graph=$graph_path
}



args=("$@")
number_of_arguments=$#

xls_name="perf_stats.xlsx"

if (( number_of_arguments == 1 )) ; then
  xls_name=${args[0]}
fi

echo "Writing output to "$xls_name" file..."

rm $xls_name
rm perf_dict.pkl

declare -a apps=("pr")
declare -a graphs=("flick.mtx" "lj.mtx" "ork.mtx" "pock.mtx" "rmat_20_16.mtx" "youtube.mtx" "rmat_21_16.mtx" "rmat_22_16.mtx" "ru_21_16.mtx" "ru_22_16.mtx")

for app in "${apps[@]}"
do
  for graph in "${graphs[@]}"
  do
    echo $app $graph
    benchmark_graph $app $graph
  done
done


python3 ./export_to_xls.py --output=$xls_name
rm perf_dict.pkl
rm dump.txt
rm perf_stats.txt