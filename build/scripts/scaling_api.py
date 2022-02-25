from .helpers import *
from .create_graphs_api import *
from .export import *
from .settings import *
import re
from .export import *
import time
from threading import Timer
from .analize_perf_data import *
import collections


def scale_app(app_name, benchmarking_results, graph_format, run_speed_mode, timeout_length, threads_used):
    list_of_graphs = get_list_of_all_graphs(run_speed_mode)

    create_graphs_if_required(list_of_graphs, run_speed_mode)
    common_args = ["-it", str(common_iterations), "-format", graph_format, "-no-check"]

    arguments = [[""]]
    if app_name in benchmark_args:
        arguments = benchmark_args[app_name]

    for current_args in arguments:
        first_graph = True
        for current_graph in list_of_graphs:
            start = time.time()
            if app_name in apps_and_graphs_ingore and current_graph in apps_and_graphs_ingore[app_name]:
                print("graph " + current_graph + " is set to be ignored for app " + app_name + "!")
                continue

            try:
                os.remove(PERF_DATA_FILE)
            except OSError:
                pass

            cmd = ["bash", "./scripts/scaling_benchmark.sh", str(threads_used), get_binary_path(app_name), "-graph mtx ", get_path_to_graph(current_graph, "mtx")] + current_args + common_args

            proc = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            timer = Timer(int(timeout_length), proc.kill)
            try:
                timer.start()
                stdout, stderr = proc.communicate()
            finally:
                timer.cancel()

            output = stdout.decode("utf-8")
            perf_dict = analyze_perf_file()

            for key in perf_dict.keys():
                data = {"app": str(app_name) + " " + str(current_args) + " " + str(key), "threads": threads_used,
                        "graph": current_graph, "perf": perf_dict[key]["perf"], "time": perf_dict[key]["time"]}
                output_file = open(SCALING_FILE, 'a', encoding='utf-8')
                json.dump(data, output_file)
                output_file.close()
