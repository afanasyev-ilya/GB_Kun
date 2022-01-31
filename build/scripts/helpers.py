from pathlib import Path
import subprocess
import os
import shutil
from .settings import *
import urllib.request


def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def get_binary_path(app_name):
    return app_name


def file_exists(path):
    bin_path = Path(path)
    if bin_path.is_file():
        return True
    return False


def binary_exists(app_name):
    if app_name == "clean":
        return True
    if file_exists(get_binary_path(app_name)):
        return True
    print("Warning! path " + get_binary_path(app_name) + " does not exist")
    return False


def make_binary(app_name):
    cmd = "make " + app_name
    print(cmd)
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    if binary_exists(app_name):
        print("Success! " + app_name + " has been compiled")
    else:
        print("Error! " + app_name + " can not be compiled")


def is_valid(app_name, options):
    if not binary_exists(app_name):
        make_binary(app_name)
        if not binary_exists(app_name):
            return False
    return True


def get_cores_count():  # returns number of sockets of target architecture
    try:
        output = subprocess.check_output(["lscpu"])
        cores = -1
        for item in output.decode().split("\n"):
            if "Core(s) per socket:" in item:
                cores_line = item.strip()
                cores = int(cores_line.split(":")[1])
        if cores == -1:
            raise NameError('Can not detect number of cores of target architecture')
        return cores
    except:
        cores = 8 # SX-Aurora
        return cores


def get_sockets_count():  # returns number of sockets of target architecture
    try:
        output = subprocess.check_output(["lscpu"])
        cores = -1
        sockets = -1
        for item in output.decode().split("\n"):
            if "Socket(s)" in item:
                sockets_line = item.strip()
                sockets = int(sockets_line.split(":")[1])
        if sockets == -1:
            raise NameError('Can not detect number of cores of target architecture')
        return sockets
    except:
        sockets = 1 # SX-Aurora
        return sockets


def get_target_proc_model():  # returns number of sockets of target architecture
    try:
        output = subprocess.check_output(["lscpu"])
        model = "Unknown"
        for item in output.decode().split("\n"):
            if "Model name" in item:
                model_line = item.strip()
                model = int(model_line.split(":")[1])
        return model
    except:
        return "Unknown"


def get_threads_count():
    return get_sockets_count()*get_cores_count()


def set_omp_environments(options):
    threads = get_cores_count()
    if int(options.sockets) > 1:
        threads = int(options.sockets) * threads
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['OMP_PROC_BIND'] = 'true'
    os.environ['OMP_PROC_BIND'] = 'close'


def prepare_list_of_apps(apps_string):
    if apps_string == "all":
        apps_list = []
        for app in benchmark_args:
            apps_list += [app]
        return apps_list
    else:
        return apps_string.split(",")


def internet_on():
    try:
        urllib.request.urlopen('http://216.58.192.142', timeout=1)
        return True
    except Exception as e:
        print(e)
        return False


def get_list_of_formats(formats_str):
    formats_res = []
    if formats_str == "all":
        for current_format in available_formats:
            formats_res.append(current_format)
    elif "," in formats_str:
        formats_res = formats_str.split(",")
    else:
        formats_res = [formats_str]
    return formats_res
