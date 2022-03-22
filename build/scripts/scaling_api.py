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
import matplotlib.pyplot as plt


def post_process_scaling_data(arr):
    if os.path.exists(SCALING_FOLDER_NAME):
        shutil.rmtree(SCALING_FOLDER_NAME)
    os.mkdir(SCALING_FOLDER_NAME)

    p = {}
    for d in arr:
        app = d["app"]
        graph = d["graph"]
        threads = d["threads"]
        perf = d["perf"]
        time = d["time"]
        if app not in p:
            p[app] = {}
        if graph not in p[app]:
            p[app][graph] = {"threads": [], "performance": [], "time": []}
        p[app][graph]["threads"].append(threads)
        p[app][graph]["performance"].append(perf)
        p[app][graph]["time"].append(time)

    cnt = 0
    for app in p.keys():
        for graph in p[app].keys():
            cnt += 1
            fig, axs = plt.subplots(2, figsize=(8, 7))
            fig.suptitle("App: " + app + ", Graph: " + graph)

            axs[0].set_xticks(p[app][graph]["threads"], p[app][graph]["threads"])
            axs[0].plot(p[app][graph]["threads"], p[app][graph]["performance"], 'r--')
            axs[0].set(xlabel='threads', ylabel='Performance')

            axs[1].set_xticks(p[app][graph]["threads"], p[app][graph]["threads"])
            axs[1].plot(p[app][graph]["threads"], p[app][graph]["time"], 'b--')
            axs[1].set(xlabel='threads', ylabel='Time')

            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()

            plot_name = app + "_" + graph
            plot_name = plot_name.replace('[', '_')
            plot_name = plot_name.replace(']', '_')
            plot_name = plot_name.replace(' ', '_')
            plot_name = plot_name.replace('\'', '_')
            plot_name = plot_name.replace('-', '_')

            lst = plot_name.split('_')
            while '' in lst:
                lst.remove('')
            plot_name = '_'.join(lst)

            plt.savefig(SCALING_FOLDER_NAME + "/" + plot_name)

    arr = sorted(arr, key=lambda x: (x['app'], x['graph'], x['threads'], x['perf'], x['time']))

    filename = SCALING_FOLDER_NAME + "/" + SCALING_ROW_DATA_NAME
    with open(filename, "w") as f:
        prev = {"graph": ""}
        for item in arr:
            if item["graph"] != prev["graph"]:
                print("", file = f)
            print ("App: " + item["app"] + ", Graph: " + item['graph'] + ", Threads: " + str(item['threads']) + ", Performance: " + str(item['perf']) + ", Time: " + str(item['time']), file = f)
            prev = item
        f.close()


def scale_app(app_name, benchmarking_results, graph_format, run_speed_mode, timeout_length, threads_used, options):
    list_of_graphs = get_list_of_all_graphs(run_speed_mode)

    create_graphs_if_required(list_of_graphs, run_speed_mode, options)
    common_args = ["-it", str(common_iterations), "-format", graph_format, "-no-check"]

    arguments = [[""]]
    if app_name in benchmark_args:
        arguments = benchmark_args[app_name]

    scaling_data = []

    file_extension = "mtx"
    if options.use_binary_graphs:
        file_extension = "mtxbin"
    for current_args in arguments:
        first_graph = True
        for current_graph in list_of_graphs:
            if requires_undir_graphs(app_name):
                current_graph = UNDIRECTED_PREFIX + current_graph

            start = time.time()
            if app_name in apps_and_graphs_ingore and current_graph in apps_and_graphs_ingore[app_name]:
                print("graph " + current_graph + " is set to be ignored for app " + app_name + "!")
                continue

            try:
                os.remove(PERF_DATA_FILE)
            except OSError:
                pass

            cmd = ["bash", "./scripts/scaling_benchmark.sh", str(threads_used), get_binary_path(app_name), "-graph mtx ", get_path_to_graph(current_graph, file_extension)] + current_args + common_args

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
                scaling_data.append(data)
                json.dump(data, output_file)
                output_file.close()
    return scaling_data
