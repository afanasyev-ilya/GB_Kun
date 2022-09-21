import os
import optparse
import argparse
import json
import pickle
from .settings import *


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def analyze_perf_file():
    file = open(PERF_DATA_FILE, 'r')

    cur_graph_dict = {}

    while True:
        line = file.readline()
        if not line:
            break
        line_array = line.split(" ")
        key_name = line_array[0]
        time_val = float(line_array[1])
        perf_val = float(line_array[3])
        bw_val = float(line_array[5])

        if key_name in cur_graph_dict:
            cur_graph_dict[key_name] = {"time": cur_graph_dict[key_name]["time"] + time_val,
                                        "perf": cur_graph_dict[key_name]["perf"] + perf_val,
                                        "bw": cur_graph_dict[key_name]["bw"] + bw_val,
                                        "cnt": cur_graph_dict[key_name]["cnt"] + 1}
        else:
            cur_graph_dict[key_name] = {"time": time_val, "perf": perf_val, "bw": bw_val, "cnt": 1}
    file.close()

    for key in cur_graph_dict.keys():
        cur_graph_dict[key]["time"] /= cur_graph_dict[key]["cnt"]
        cur_graph_dict[key]["perf"] /= cur_graph_dict[key]["cnt"]
        cur_graph_dict[key]["bw"] /= cur_graph_dict[key]["cnt"]

    return cur_graph_dict

