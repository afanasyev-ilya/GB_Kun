import os
import optparse
import shutil
import subprocess
import json
import sys
import csv


def get_cores_count():  # returns number of sockets of target architecture
    output = subprocess.check_output(["lscpu"])
    cores = -1
    for item in output.decode().split("\n"):
        if "Core(s) per socket:" in item:
            cores_line = item.strip()
            cores = int(cores_line.split(":")[1])
        if "Ядер на сокет:" in item:
            cores_line = item.strip()
            cores = int(cores_line.split(":")[1])
    if cores == -1:
        raise NameError('Can not detect number of cores of target architecture')
    return cores


def get_sockets_count():  # returns number of sockets of target architecture
    output = subprocess.check_output(["lscpu"])
    cores = -1
    for item in output.decode().split("\n"):
        if "Socket(s)" in item:
            sockets_line = item.strip()
            sockets = int(sockets_line.split(":")[1])
        if "Сокетов:" in item:
            sockets_line = item.strip()
            sockets = int(sockets_line.split(":")[1])
    if sockets == -1:
        raise NameError('Can not detect number of cores of target architecture')
    return sockets


def print_file(file_name):
    f = open(file_name, 'r')
    file_contents = f.read()
    print(file_contents)
    f.close()


def parse_timings(output):  # collect time, perf and BW values
    lines = output.splitlines()
    timings = {"BW": 0, "parser_stats": 0}
    for line in lines:
        for key in timings.keys():
            if key in line:
                timings[key] = line.split(":")[1]
    return timings


def run_and_wait(params, options):
    os.environ['OMP_NUM_THREADS'] = str(get_cores_count() * int(options.sockets))
    os.environ['OMP_PROC_BIND'] = "close"
    print("Running ./spmv " + params)
    subprocess.run(["./spmv"] + params.split(" "), check=True)


if __name__ == "__main__":
    # create .csv files
    # parse arguments
    parser = optparse.OptionParser()
    parser.add_option('--first-scale',
                      action="store", dest="first_scale",
                      help="smallest scale of graph", default=16)
    parser.add_option('--last-scale',
                      action="store", dest="last_scale",
                      help="largest scale of graph", default=22)
    parser.add_option('-g', '--graph',
                      action="store", dest="graph_type",
                      help="type of graph used (rmat, ru)", default="ru")
    parser.add_option('-f', '--format',
                      action="store", dest="graph_format",
                      help="graph storage format used (CSR, COO, CSR_SEG, SIGMA, LAV)", default="CSR")
    parser.add_option('-s', '--sockets',
                      action="store", dest="sockets",
                      help="number of sockets used, default is 1", default=1)

    options, args = parser.parse_args()

    if os.path.exists("benchmark/integration_scripts/perf.txt"):
        os.remove("./output/perf.txt")
    if os.path.exists("benchmark/integration_scripts/bw.txt"):
        os.remove("./output/bw.txt")
    for scale in range(int(options.first_scale), int(options.last_scale) + 1):
        params = "-s " + str(scale) + " -e 16 " + " -format " + options.graph_format + " -no-check" +\
              " -graph " + options.graph_type
        run_and_wait(params, options)

    print("\n")
    print("bandwidth stats (GB/s):")
    print_file("benchmark/integration_scripts/bw.txt")
    print("\n")
    print("performance stats (GFLOP/s):")
    print_file("benchmark/integration_scripts/perf.txt")

