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


def parse_timings(output):  # collect time, perf and BW values
    lines = output.splitlines()
    timings = {"BW": 0}
    for line in lines:
        if "BW" in line:
            timings["BW"] = float(line.split(" ")[1])
    return timings


def run_and_wait(cmd, options):
    os.environ['OMP_NUM_THREADS'] = str(get_cores_count())
    os.environ['OMP_PROC_BIND'] = "close"

    p = subprocess.Popen(cmd, shell=True,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    output, err = p.communicate()
    rc = p.returncode
    p_status = p.wait()
    string_output = output.decode("utf-8")

    return parse_timings(string_output)


if __name__ == "__main__":
    # create .csv files
    # parse arguments
    parser = optparse.OptionParser()
    parser.add_option('-f', '--first-scale',
                      action="store", dest="first_scale",
                      help="smallest scale of graph", default=16)
    parser.add_option('-l', '--last-scale',
                      action="store", dest="last_scale",
                      help="largest scale of graph", default=16)
    parser.add_option('-g', '--graph',
                      action="store", dest="graph_type",
                      help="type of graph used (rmat, ru)", default="ru")
    parser.add_option('-s', '--format',
                      action="store", dest="format",
                      help="graph storage format used (CSR, COO, COO_OPT, CSR_SEG)", default="CSR")

    options, args = parser.parse_args()

    for scale in range(int(options.first_scale), int(options.last_scale) + 1):
        cmd = "./GB_Kun -s " + str(scale) + " -e 16 " + " -format " + options.format + " -no-check" +\
              " -type " + options.graph_type
        timings = run_and_wait(cmd, options)
        print("scale: " + str(scale) + ", BW: " + str(timings["BW"]))


