from .helpers import *
from .settings import *
import os.path
from os import path
from .mtx_api import gen_mtx_graph


def get_list_of_synthetic_graphs(run_speed_mode):
    if "rw" in run_speed_mode:
        return []
    if run_speed_mode == "tiny-only":
        return syn_tiny_only
    elif run_speed_mode == "small-only":
        return syn_small_only
    elif run_speed_mode == "medium-only":
        return syn_medium_only
    elif run_speed_mode == "large-only":
        return syn_tiny_small_medium
    elif run_speed_mode == "tiny-small":
        return syn_tiny_small
    elif run_speed_mode == "tiny-small-medium":
        return syn_tiny_small_medium
    elif run_speed_mode == "fastest":
        return syn_fastest
    elif run_speed_mode == "scaling":
        return syn_scaling
    elif run_speed_mode == "best" or run_speed_mode == "bestf" or run_speed_mode == "twenty":
        return []
    else:
        raise ValueError("Unsupported run_speed_mode")


def get_list_of_real_world_graphs(run_speed_mode):
    if run_speed_mode == "tiny-only" or run_speed_mode == "tiny-only-rw":
        return konect_tiny_only
    elif run_speed_mode == "small-only" or run_speed_mode == "small-only-rw":
        return konect_small_only
    elif run_speed_mode == "medium-only" or run_speed_mode == "medium-only-rw":
        return konect_medium_only
    elif run_speed_mode == "large-only" or run_speed_mode == "large-only-rw":
        return konect_large_only
    elif run_speed_mode == "tiny-small" or run_speed_mode == "tiny-small-rw":
        return konect_tiny_small
    elif run_speed_mode == "tiny-small-medium" or run_speed_mode == "tiny-small-medium-rw":
        return konect_tiny_small_medium
    elif run_speed_mode == "fastest":
        return konect_fastest
    elif run_speed_mode == "scaling":
        return []
    elif run_speed_mode == "best":
        return konect_best
    elif run_speed_mode == "bestf":
        return konect_bestf
    elif run_speed_mode == "twenty":
        return konect_twenty_set
    else:
        raise ValueError("Unsupported run_speed_mode")


def get_list_of_all_graphs(run_speed_mode):
    return get_list_of_synthetic_graphs(run_speed_mode) + get_list_of_real_world_graphs(run_speed_mode)


def get_list_of_verification_graphs(run_speed_mode):
    verification_list = []
    i = 0
    for graph in get_list_of_synthetic_graphs(run_speed_mode):
        if i >= 3:
            break
        verification_list.append(graph)
        i += 1
    i = 0
    for graph in get_list_of_real_world_graphs(run_speed_mode):
        if i >= 5:
            break
        verification_list.append(graph)
        i += 1
    return verification_list


def download_all_real_world_graphs(run_speed_mode):
    real_world_graphs = get_list_of_real_world_graphs(run_speed_mode)
    for graph_name in real_world_graphs:
        download_graph(graph_name)


def download_graph(graph_name):
    file_name = SOURCE_GRAPH_DIR + "download.tsv." + all_konect_graphs_data[graph_name]["link"] + ".tar.bz2"
    if not path.exists(file_name):
        link = "http://konect.cc/files/download.tsv." + all_konect_graphs_data[graph_name]["link"] + ".tar.bz2"
        print("Trying to download " + file_name + " using " + link)
        cmd = ["wget", link, "-q", "--no-check-certificate", "--directory", SOURCE_GRAPH_DIR]
        print(' '.join(cmd))
        subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)
        if path.exists(file_name):
            print(file_name + " downloaded!")
        else:
            print("Error! Can not download file " + file_name)
    else:
        print("File " + SOURCE_GRAPH_DIR + "/" + all_konect_graphs_data[graph_name]["link"] + ".tar.bz2" + " exists!")


def get_path_to_graph(short_name, graph_format):
    return GRAPHS_DIR + short_name + "." + graph_format


def verify_graph_existence(graph_file_name):
    if file_exists(graph_file_name):
        print("Success! graph " + graph_file_name + " has been created.")
        return True
    else:
        print("Error! graph " + graph_file_name + " can not be created.")
        return False


def create_real_world_graph(graph_name):
    graph_format = "mtx"
    output_graph_file_name = get_path_to_graph(graph_name, graph_format)
    if not file_exists(output_graph_file_name):
        download_graph(graph_name)

        tar_name = SOURCE_GRAPH_DIR + "download.tsv." + all_konect_graphs_data[graph_name]["link"] + ".tar.bz2"
        cmd = ["tar", "-xjf", tar_name, '-C', SOURCE_GRAPH_DIR]
        subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)

        if "unarch_graph_name" in all_konect_graphs_data[graph_name]:
            source_name = SOURCE_GRAPH_DIR + all_konect_graphs_data[graph_name]["link"] + "/out." + all_konect_graphs_data[graph_name]["unarch_graph_name"]
        else:
            source_name = SOURCE_GRAPH_DIR + all_konect_graphs_data[graph_name]["link"] + "/out." + all_konect_graphs_data[graph_name]["link"]

        gen_mtx_graph(source_name, output_graph_file_name)

        if verify_graph_existence(output_graph_file_name):
            print("Graph " + output_graph_file_name + " has been created\n")
    else:
        print("Warning! Graph " + output_graph_file_name + " already exists!")


def create_synthetic_graph(graph_name):
    graph_format = "mtx"
    dat = graph_name.split("_")
    type = dat[1]
    scale = dat[2]
    edge_factor = dat[3]

    output_graph_file_name = get_path_to_graph(graph_name, graph_format)
    if not file_exists(output_graph_file_name):
        cmd = [get_binary_path(MTX_GENERATOR_BIN_NAME), "-s", scale, "-e", edge_factor, "-type", type,
               "-outfile", output_graph_file_name]
        print(' '.join(cmd))

        subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()

        if verify_graph_existence(output_graph_file_name):
            print("Graph " + output_graph_file_name + " has been created\n")
    else:
        print("Warning! Graph " + output_graph_file_name + " already exists!")


def create_graph(graph_name, run_speed_mode):
    if graph_name in get_list_of_synthetic_graphs(run_speed_mode):
        create_synthetic_graph(graph_name)
    elif graph_name in get_list_of_real_world_graphs(run_speed_mode):
        create_real_world_graph(graph_name)


def create_graphs_if_required(list_of_graphs, run_speed_mode):
    create_dir(GRAPHS_DIR)
    create_dir(SOURCE_GRAPH_DIR)

    if not binary_exists(MTX_GENERATOR_BIN_NAME):
        make_binary(MTX_GENERATOR_BIN_NAME)

    for current_graph in list_of_graphs:
        create_graph(current_graph, run_speed_mode)

