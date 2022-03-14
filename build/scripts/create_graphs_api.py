from .helpers import *
from .settings import *
import os.path
from os import path
from .mtx_api import *
from os import listdir
from os.path import isfile, join
from urllib.request import urlopen
from bs4 import BeautifulSoup


# synthetic
def get_list_of_synthetic_graphs(run_speed_mode):
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
    else:
        raise ValueError("Unsupported run_speed_mode")


def get_list_of_real_world_graphs(run_speed_mode):
    if run_speed_mode == "tiny-only":
        return konect_tiny_only
    elif run_speed_mode == "small-only":
        return konect_small_only
    elif run_speed_mode == "medium-only":
        return konect_medium_only
    elif run_speed_mode == "large-only":
        return konect_large_only
    elif run_speed_mode == "tiny-small":
        return konect_tiny_small
    elif run_speed_mode == "tiny-small-medium":
        return konect_tiny_small_medium
    elif run_speed_mode == "fastest":
        return konect_fastest
    elif run_speed_mode == "scaling":
        return []
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


def download_konect(graph_name):
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


def download_gap(graph_name):
    link = all_konect_graphs_data[graph_name]["link"]
    cmd = ["wget", link, "-q", "--no-check-certificate", "--directory", SOURCE_GRAPH_DIR]
    print(' '.join(cmd))
    subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)


def download_graph(graph_name):
    if 'GAP' in graph_name:
        download_gap(graph_name)
    else:
        download_konect(graph_name)


def get_path_to_graph(short_name, extension):
    return GRAPHS_DIR + short_name + "." + extension


def verify_graph_existence(graph_file_name):
    if file_exists(graph_file_name):
        print("Success! graph " + graph_file_name + " has been created.")
        return True
    else:
        print("Error! graph " + graph_file_name + " can not be created.")
        return False


def clear_dir(dir_name):
    os.system('rm -rf ' + SOURCE_GRAPH_DIR + '/*')


def find_info_on_page(text, pattern):
    for line in text.splitlines():
        if pattern in line:
            return line
    return None


def extract_number(line):
    digits_list = [int(s) for s in line if s.isdigit()]
    return int(''.join(map(str, digits_list)))


def load_page(graph_name):
    graph_link = all_konect_graphs_data[graph_name]["link"]
    url = "http://konect.cc/networks/" + graph_link
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    page_text = '\n'.join(chunk for chunk in chunks if chunk)
    return page_text


def convert_to_mtx_if_no_loops_and_multiple_edges(input_file, output_file, graph_name):
    f_in = open(input_file, 'r')
    f_out = open(output_file, 'w+')

    line = f_in.readline()
    line = f_in.readline()
    vert_and_edges = line[1:].split()
    f_out.write('%%MatrixMarket matrix coordinate pattern general\n')

    if len(vert_and_edges) == 3:
        f_out.write(str(vert_and_edges[1]) + ' ')
        f_out.write(str(vert_and_edges[2]) + ' ')
        f_out.write(str(vert_and_edges[0]) + '\n')
        shutil.copyfileobj(f_in, f_out)
    else:
        page_text = load_page(graph_name)
        vertices_count = extract_number(find_info_on_page(page_text, "Size"))
        edges_count = extract_number(find_info_on_page(page_text, "Volume"))
        f_out.write(str(vertices_count) + ' ')
        f_out.write(str(vertices_count) + ' ')
        f_out.write(str(edges_count) + '\n')
        shutil.copyfileobj(f_in, f_out)

    f_in.close()
    f_out.close()


def check_if_no_loops_and_multiple_edges(graph_name):
    return False

    page_text = load_page(graph_name)

    no_multiple_edges = False
    if "no multiple edges" in page_text:
        no_multiple_edges = True

    no_loops = False
    if "Does not contain loops" in page_text:
        no_loops = True

    if no_loops and no_multiple_edges:
        print("This konect graph does not contain loops or multiple edges, thus optimized generation can be used")
        return True
    else:
        return False


def graph_missing(output_graph_file_name, undir_output_graph_file_name):
    if not file_exists(output_graph_file_name):
        return True
    if not file_exists(undir_output_graph_file_name):
        return True
    if not file_exists(output_graph_file_name + "bin"):
        return True
    if not file_exists(undir_output_graph_file_name + "bin"):
        return True
    return False


def create_real_world_graph(graph_name, options):
    graph_format = "mtx"
    output_graph_file_name = get_path_to_graph(graph_name, graph_format)
    undir_output_graph_file_name = get_path_to_graph(UNDIRECTED_PREFIX + graph_name, graph_format)
    if graph_missing(output_graph_file_name, undir_output_graph_file_name):
        if 'GAP' in graph_name:
            clear_dir(SOURCE_GRAPH_DIR)
            download_graph(graph_name)
            files = [f for f in listdir(SOURCE_GRAPH_DIR)]
            tar_name = files[0]

            cmd = ["tar", "-zxf", SOURCE_GRAPH_DIR + "/" + tar_name, '--directory', SOURCE_GRAPH_DIR]
            print(' '.join(cmd))
            subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)

            old_path = SOURCE_GRAPH_DIR + "/" + graph_name + "/" + graph_name + ".mtx"
            new_path = GRAPHS_DIR + "/" + graph_name + ".mtx"
            shutil.move(old_path, new_path)
        else:
            download_graph(graph_name)
            tar_name = SOURCE_GRAPH_DIR + "download.tsv." + all_konect_graphs_data[graph_name]["link"] + ".tar.bz2"
            cmd = ["tar", "-xjf", tar_name, '-C', SOURCE_GRAPH_DIR]
            subprocess.call(cmd, shell=False, stdout=subprocess.PIPE)

            if "unarch_graph_name" in all_konect_graphs_data[graph_name]:
                source_name = SOURCE_GRAPH_DIR + all_konect_graphs_data[graph_name]["link"] + "/out." + all_konect_graphs_data[graph_name]["unarch_graph_name"]
            else:
                source_name = SOURCE_GRAPH_DIR + all_konect_graphs_data[graph_name]["link"] + "/out." + all_konect_graphs_data[graph_name]["link"]

            # it does not work
            #if check_if_no_loops_and_multiple_edges(graph_name):
            #    convert_to_mtx_if_no_loops_and_multiple_edges(source_name, output_graph_file_name, graph_name)
            #else:
            gen_graph(source_name, output_graph_file_name, undir_output_graph_file_name, options)

            if verify_graph_existence(output_graph_file_name):
                print("Graph " + output_graph_file_name + " has been created\n")

        if verify_graph_existence(output_graph_file_name):
            print("Graph " + output_graph_file_name + " has been created\n")

        if 'GAP' in graph_name:
            clear_dir(SOURCE_GRAPH_DIR)
    else:
        print("Warning! Graph " + output_graph_file_name + " already exists!")


def create_synthetic_graph(graph_name, options):
    graph_format = "mtx"
    dat = graph_name.split("_")
    type = dat[1]
    scale = dat[2]
    edge_factor = dat[3]

    output_graph_file_name = get_path_to_graph(graph_name, graph_format)
    undir_output_graph_file_name = get_path_to_graph(UNDIRECTED_PREFIX + graph_name, graph_format)
    if graph_missing(output_graph_file_name, undir_output_graph_file_name):
        cmd = [get_binary_path(MTX_GENERATOR_BIN_NAME), "-s", scale, "-e", edge_factor, "-type", type,
               "-outfile", output_graph_file_name]
        print(' '.join(cmd))

        subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE).wait()

        gen_graph(output_graph_file_name, output_graph_file_name, undir_output_graph_file_name, options)

        if verify_graph_existence(output_graph_file_name):
            print("Graph " + output_graph_file_name + " has been created\n")
    else:
        print("Warning! Graph " + output_graph_file_name + " already exists!")


def create_graph(graph_name, run_speed_mode, options):
    if graph_name in get_list_of_synthetic_graphs(run_speed_mode):
        create_synthetic_graph(graph_name, options)
    elif graph_name in get_list_of_real_world_graphs(run_speed_mode):
        create_real_world_graph(graph_name, options)


def create_graphs_if_required(list_of_graphs, run_speed_mode, options):
    create_dir(GRAPHS_DIR)
    create_dir(SOURCE_GRAPH_DIR)

    if not binary_exists(MTX_GENERATOR_BIN_NAME):
        make_binary(MTX_GENERATOR_BIN_NAME)

    for current_graph in list_of_graphs:
        create_graph(current_graph, run_speed_mode, options)

