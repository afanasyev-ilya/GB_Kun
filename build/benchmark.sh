prefix="/home/afanasyev/LAGraph/src/benchmark/"
graph_name="dbpedia.mtx"
graph_path=$prefix"/"$graph_name

rm perf_stats.txt
./pr -graph mtx $graph_path

python3 ./analize_data.py --graph=$graph_path
python3 ./export_to_xls.py