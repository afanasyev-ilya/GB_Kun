from benchmark.scripts.helpers import *
from benchmark.scripts.create_graphs_api import *
import pickle
import sys
import os
import subprocess
import numpy as np

label_dict = {0: "SEG_CSR", 1: "CSR"}

def get_label(graphname):
    X_data = []
    with open('dict.pickle', 'rb') as f:
        data_new = pickle.load(f)
        for k, v in data_new.items():
            if graphname == k:
                values = []
                for item, value in v.items():
                    if item == "size":
                        values.append(value)
                    if item == "volume":
                        values.append(value)
                    if item == "avg_degree":
                        values.append(value)
                    if item == "percentile":
                        values.append(value)
                    if item == "exponent":
                        values.append(value)
                X_data.append(values)


    # Predict label from downloaded model

    pred_label = loaded_model.predict(X_data)[0]
    return label_dict[pred_label]

if __name__ == "__main__":

    # Download trained model
    filename = 'model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    # Get script parameters

    current_app = sys.argv[1]

    args = sys.argv[2:]
    graphname = " ".join(args)

    # Gather graph data from collected pickle file

    pred_format = get_label(graphname)

    # Running a application

    mult = 10
    file_extension = "mtx"

    common_args = ["-it", str(common_iterations * mult), "-format", pred_format, "-no-check"]

    cmd = ["bash", "./benchmark.sh", get_binary_path(current_app), "-graph mtx ",
           get_path_to_graph(graphname, file_extension)] + common_args

    proc = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.wait()
    stdout, stderr = proc.communicate()

    print(stdout)
