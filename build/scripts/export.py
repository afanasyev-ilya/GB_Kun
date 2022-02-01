import xlsxwriter
from .create_graphs_api import *
from random import randrange
import pickle
import shutil


app_name_column_size = 30
graph_name_column_size = 20
data_column_size = 15
colors = ["#CCFFFF", "#CCFFCC", "#FFFF99", "#FF99FF", "#66CCFF", "#FF9966"]


XLSX_DATA_SHIFT = 10


def remove_timed_out(perf_data):
    cleared_list = []
    for item in perf_data:
        if item["perf_val"] != "TIMED OUT":
            cleared_list.append(item)
    return cleared_list


class BenchmarkingResults:
    def __init__(self, name, run_speed_mode):
        self.run_speed_mode = run_speed_mode
        self.current_graph_format = ""

        self.workbook = xlsxwriter.Workbook(name + "_benchmarking_results.xlsx")
        self.worksheet = None # these can be later used for xls output
        self.line_pos = None # these can be later used for xls output
        self.current_format = None # these can be later used for xls output
        self.current_app_name = None # these can be later used for xls output

    def add_performance_header_to_xls_table(self, graph_format):
        self.worksheet = self.workbook.add_worksheet("Perf " + graph_format)
        self.line_pos = 1
        self.current_format = self.workbook.add_format({})
        self.current_app_name = ""
        self.current_graph_format = graph_format

        # make columns wider
        iter = 0
        while iter < 3:
            self.worksheet.set_column(0 + iter * XLSX_DATA_SHIFT, 0 + iter * XLSX_DATA_SHIFT, app_name_column_size)
            self.worksheet.set_column(1 + iter * XLSX_DATA_SHIFT, 1 + iter * XLSX_DATA_SHIFT, graph_name_column_size)
            self.worksheet.set_column(2 + iter * XLSX_DATA_SHIFT, 4 + iter * XLSX_DATA_SHIFT, data_column_size)
            self.worksheet.set_column(5 + iter * XLSX_DATA_SHIFT, 5 + iter * XLSX_DATA_SHIFT, graph_name_column_size)
            self.worksheet.set_column(6 + iter * XLSX_DATA_SHIFT, 8 + iter * XLSX_DATA_SHIFT, data_column_size)
            iter += 1

    def add_performance_test_name_to_xls_table(self, app_name, app_args, part_name, num_part):
        test_name = part_name + '\n' + '(' + ' '.join([app_name] + app_args) + ')'
        self.worksheet.write(self.line_pos, num_part * XLSX_DATA_SHIFT, test_name)
        self.current_app_name = app_name

        if num_part == 0:
            color = colors[randrange(len(colors))]
            self.current_format = self.workbook.add_format({'border': 1,
                                                            'align': 'center',
                                                            'valign': 'vcenter',
                                                            'fg_color': color,
                                                            'text_wrap': 'true'})

        self.worksheet.merge_range(self.line_pos, num_part * XLSX_DATA_SHIFT, self.line_pos + self.lines_in_test() - 1,
                                   num_part * XLSX_DATA_SHIFT, test_name, self.current_format)

    def add_performance_value_to_xls_table(self, perf_dict, graph_name):
        row = int(self.get_row_pos(graph_name))
        col = int(self.get_column_pos(graph_name))

        header_format = self.workbook.add_format({'border': 1,
                                                  'align': 'center',
                                                  'valign': 'vcenter',
                                                  'fg_color': '#cce6ff'})

        if not perf_dict: # in case it timed out
            self.worksheet.write(self.line_pos - 1, col - 2, "Algorithm", header_format)
            self.worksheet.write(self.line_pos - 1, col - 1, "Graph", header_format)
            self.worksheet.write(self.line_pos - 1, col, "Time(ms)", header_format)
            self.worksheet.write(self.line_pos - 1, col + 1, "Perf.", header_format)
            self.worksheet.write(self.line_pos - 1, col + 2, "Band.(GB/s)", header_format)

            self.worksheet.write(self.line_pos + row, col - 1, graph_name, self.current_format)
            self.worksheet.write(self.line_pos + row, col, "TIMED OUT", self.current_format)
            self.worksheet.write(self.line_pos + row, col + 1, "TIMED OUT", self.current_format)
            self.worksheet.write(self.line_pos + row, col + 2, "TIMED OUT", self.current_format)
        else:
            for part_key in perf_dict.keys():
                part_data = perf_dict[part_key]

                time = round(float(part_data["time"]), 2) # 2 points in float after point
                perf = round(float(part_data["perf"]), 2)
                bw = round(float(part_data["bw"]), 2)
                perf_suffix = "MTEPS"
                if "mxv" in part_key or "vxm" in part_key:
                    perf_suffix = "GFlop/s"

                self.worksheet.write(self.line_pos - 1, col - 2, "Algorithm", header_format)
                self.worksheet.write(self.line_pos - 1, col - 1, "Graph", header_format)
                self.worksheet.write(self.line_pos - 1, col, "Time(ms)", header_format)
                self.worksheet.write(self.line_pos - 1, col + 1, "Perf.("+perf_suffix+")", header_format)
                self.worksheet.write(self.line_pos - 1, col + 2, "Band.(GB/s)", header_format)

                self.worksheet.write(self.line_pos + row, col - 1, graph_name, self.current_format)
                self.worksheet.write(self.line_pos + row, col, time, self.current_format)
                self.worksheet.write(self.line_pos + row, col + 1, perf, self.current_format)
                self.worksheet.write(self.line_pos + row, col + 2, bw, self.current_format)

                col += XLSX_DATA_SHIFT

    def add_performance_separator_to_xls_table(self):
        self.line_pos += self.lines_in_test() + 2

    def add_correctness_header_to_xls_table(self, graph_format):
        self.worksheet = self.workbook.add_worksheet("Correctness data " + graph_format)
        self.line_pos = 0
        self.current_format = self.workbook.add_format({})
        self.current_app_name = ""

        # add column names
        for graph_name in get_list_of_verification_graphs(self.run_speed_mode):
            self.worksheet.write(self.line_pos, get_list_of_verification_graphs(self.run_speed_mode).index(graph_name) + 1, graph_name)

        self.worksheet.set_column(self.line_pos, len(get_list_of_verification_graphs(self.run_speed_mode)) + 1, 30)

        self.line_pos = 1

    def add_correctness_test_name_to_xls_table(self, app_name, app_args):
        color = colors[randrange(len(colors))]
        self.current_format = self.workbook.add_format({'border': 1,
                                                       'align': 'center',
                                                       'valign': 'vcenter',
                                                       'fg_color': color,
                                                       'text_wrap': 'true'})

        test_name = ' '.join([app_name] + app_args)
        self.worksheet.write(self.line_pos, 0, test_name, self.current_format)

    def add_correctness_separator_to_xls_table(self):
        self.line_pos += 1

    def add_correctness_value_to_xls_table(self, value, graph_name, app_name):
        self.worksheet.write(self.line_pos, get_list_of_verification_graphs(self.run_speed_mode).index(graph_name) + 1,
                             value, self.current_format)

    def finalize(self):
        self.workbook.close()

    def lines_in_test(self):
        return int(max(len(get_list_of_synthetic_graphs(self.run_speed_mode)), len(get_list_of_real_world_graphs(self.run_speed_mode))))

    def get_column_pos(self, graph_name):
        if graph_name in get_list_of_synthetic_graphs(self.run_speed_mode):
            return 2
        elif graph_name in get_list_of_real_world_graphs(self.run_speed_mode):
            return 6
        else:
            raise ValueError("Incorrect graph name")

    def get_row_pos(self, graph_name):
        if graph_name in get_list_of_synthetic_graphs(self.run_speed_mode):
            return get_list_of_synthetic_graphs(self.run_speed_mode).index(graph_name)
        elif graph_name in get_list_of_real_world_graphs(self.run_speed_mode):
            return get_list_of_real_world_graphs(self.run_speed_mode).index(graph_name)
