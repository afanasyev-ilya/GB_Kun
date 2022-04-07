import os
import optparse
import argparse
import json
import pickle


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def main():
    file = open('perf_stats.txt', 'r')

    parser = optparse.OptionParser()
    parser.add_option('-g', '--graph',
                      action="store", dest="graph",
                      help="specify name of evaluated graph", default="none.mtx")

    options, args = parser.parse_args()

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

    #print(cur_graph_dict)
    for key in cur_graph_dict.keys():
        cur_graph_dict[key]["time"] /= cur_graph_dict[key]["cnt"]
        cur_graph_dict[key]["perf"] /= cur_graph_dict[key]["cnt"]
        cur_graph_dict[key]["bw"] /= cur_graph_dict[key]["cnt"]

    #print(cur_graph_dict)

    full_perf_data = {}
    try:
        with open('perf_dict.pkl', 'rb') as f:
            full_perf_data = pickle.load(f)
    except:
        full_perf_data = {}

    if options.graph in full_perf_data:
        full_perf_data[options.graph] = merge_two_dicts(full_perf_data[options.graph], cur_graph_dict)
    else:
        full_perf_data[options.graph] = cur_graph_dict

    with open('perf_dict.pkl', 'wb') as f:
        pickle.dump(full_perf_data, f)


if __name__ == "__main__":
    main()
