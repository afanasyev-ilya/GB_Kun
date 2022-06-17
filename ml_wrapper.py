import pickle
import sys
import os
import subprocess
import numpy as np


if __name__ == "__main__":

    filename = 'model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    args = sys.argv[1:]
    graphname = " ".join(args)
    print("Looking in the pickle for graph: ", graphname)

    X_data = []
    with open('all_graphs', 'rb') as f:
        data_new = pickle.load(f)
        for k, v in data_new.items():
            if graphname == k:
                print("OKOKOKO")
                values = []
                for item, value in v.items():
                    if item == "size":
                        values.append(value)
                    if item == "volume":
                        values.append(value)
                    if item == "avg_degree":
                        values.append(value)
                X_data.append(values)

    print(X_data)

    pred_label = loaded_model.predict(X_data)[0]

    if pred_label == 0:
        process = subprocess.Popen(['./bfs', '-format', 'CSR_SEG'],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

    if pred_label == 1:
        process = subprocess.Popen(['./bfs', '-format', 'CSR'],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

    print(stdout)

