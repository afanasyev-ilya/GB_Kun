import os
import optparse
import argparse
import json
import pickle
import xlsxwriter


def main():
    full_perf_data = {}
    with open('perf_dict.pkl', 'rb') as f:
        full_perf_data = pickle.load(f)
    print(full_perf_data)

    workbook = xlsxwriter.Workbook('perf_stats.xlsx')
    worksheet_time = workbook.add_worksheet("time")

    cnt_x = 0
    for graph_name in full_perf_data.keys():
        worksheet_time.write(0, cnt_x, graph_name)
        cnt_y = 1
        for func_name in full_perf_data[graph_name].keys():
            worksheet_time.write(cnt_y, cnt_x, full_perf_data[graph_name][func_name]["time"])
            cnt_y += 1
        cnt_x += 1

    workbook.close()


if __name__ == "__main__":
    main()
