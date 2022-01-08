import os
import optparse
import argparse
import json
import pickle
import xlsxwriter


def save_worksheet(workbook, type_key, full_perf_data, col_names_dict):
    worksheet_time = workbook.add_worksheet(type_key)
    cnt_row = 1

    for col_name in col_names_dict.keys():
        worksheet_time.write(0, col_names_dict[col_name], col_name)

    for graph_name in full_perf_data.keys():
        table_graph = graph_name.split("/")[-1]
        worksheet_time.write(cnt_row, 0, table_graph)
        for func_name in full_perf_data[graph_name].keys():
            cnt_col = col_names_dict[func_name]
            worksheet_time.write(cnt_row, cnt_col, full_perf_data[graph_name][func_name][type_key])
        cnt_row += 1


def main():
    full_perf_data = {}
    with open('perf_dict.pkl', 'rb') as f:
        full_perf_data = pickle.load(f)
    print(full_perf_data)

    workbook = xlsxwriter.Workbook('perf_stats.xlsx')

    col_names_dict = {}
    for graph_name in full_perf_data.keys():
        for func_name in full_perf_data[graph_name].keys():
            col_names_dict[func_name] = -1
    cnt = 1
    for key in col_names_dict.keys():
        col_names_dict[key] = cnt
        cnt += 1

    save_worksheet(workbook, "time", full_perf_data, col_names_dict)
    save_worksheet(workbook, "bw", full_perf_data, col_names_dict)
    save_worksheet(workbook, "perf", full_perf_data, col_names_dict)

    workbook.close()


if __name__ == "__main__":
    main()
