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


def benchmark_app(app_name, benchmarking_results, graph_format, run_speed_mode, timeout_length, options):
    list_of_graphs = get_list_of_all_graphs(run_speed_mode)

    create_graphs_if_required(list_of_graphs, run_speed_mode, options)
    common_args = ["-it", str(common_iterations), "-format", graph_format, "-no-check"]
    print(common_args)

    algorithms_tested = 0

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

            cmd = ["bash", "./benchmark.sh", get_binary_path(app_name), "-graph mtx ", get_path_to_graph(current_graph, "mtx")] + current_args + common_args
            print(' '.join(cmd))
            proc = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            timer = Timer(int(timeout_length), proc.kill)
            try:
                timer.start()
                stdout, stderr = proc.communicate()
            finally:
                timer.cancel()

            output = stdout.decode("utf-8")
            perf_dict = analyze_perf_file()

            if first_graph:
                if not perf_dict: # in case it timed out
                    benchmarking_results.add_performance_test_name_to_xls_table(app_name, current_args + common_args, 0, 0)
                else:
                    num_part = 0
                    for part_key in perf_dict.keys():
                        part_data = perf_dict[part_key]
                        benchmarking_results.add_performance_test_name_to_xls_table(app_name, current_args + common_args,
                                                                                    part_key, num_part)
                        first_graph = False
                        num_part += 1

            benchmarking_results.add_performance_value_to_xls_table(perf_dict, current_graph)
            end = time.time()
            if print_timings:
                print("TIME: " + str(end-start) + " seconds\n")

        benchmarking_results.add_performance_separator_to_xls_table()
        algorithms_tested += 1

    return algorithms_tested

