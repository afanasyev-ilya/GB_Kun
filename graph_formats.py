import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    X_data = []
    with open('all_graphs', 'rb') as f:
        data_new = pickle.load(f)
        for k, v in data_new.items():
            values = []
            for item, value in v.items():
                if item == "size":
                    values.append(value)
                if item == "volume":
                    values.append(value)
                if item == "avg_degree":
                    values.append(value)
                # print(item)
            X_data.append(values)

    # print(X_train)

    seg_perf = []
    csr_perf = []

    with open("seg_csr_perf.txt") as file:
        seg_perf_len = file.readline()
        while True:
            line = file.readline()
            if not line:
                break
            seg_perf.append(int(line))


    with open("csr_perf.txt") as file:
        csr_perf_len = file.readline()
        while True:
            line = file.readline()
            if not line:
                break
            csr_perf.append(int(line))

    if csr_perf_len != seg_perf_len:
        print("Different sizes of training datasets")
        exit(-1)

    #print(len(seg_perf))
    #
    labels = []
    for i in range(len(seg_perf)):
        if seg_perf[i] < csr_perf[i]:
            labels.append(0) #0 is for better segmented csr
        else:
            labels.append(1) #1  is for better simple csr

    X_data = np.array(X_data)
    labels = np.array(labels)
    print("Y length: ", len(labels))
    print("X length: ", len(X_data))

    kf = KFold(n_splits=3, shuffle=True)
    nsplits = kf.get_n_splits(X_data)

    acc_score = 0

    for train_index, test_index in kf.split(X_data):
        #print(train_index, test_index)
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        clf = RandomForestClassifier(n_estimators=100)

        clf = clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc_score += accuracy_score(y_pred, y_test)

    print("Overall accuracy: ", acc_score/nsplits)

    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(X_data, labels)

    filename = 'model.sav'
    pickle.dump(clf, open(filename, 'wb'))
