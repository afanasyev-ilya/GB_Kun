function quit {
  exit
}

function benchmark_graph {
    prefix="/home/afanasyev/LAGraph/src/benchmark/"
    app_name="./"$1
    graph_name=$2
    graph_path=$prefix"/"$graph_name

    rm perf_stats.txt
    $app_name -graph mtx $graph_path

    python3 ./analize_data.py --graph=$graph_path
}

rm perf_stats.xlsx
rm perf_dict.pkl

benchmark_graph "pr" "flick.mtx"
benchmark_graph "pr" "lj.mtx"

python3 ./export_to_xls.py
rm perf_dict.pkl
rm perf_stats.txt