import os
import optparse
import scripts.settings
from scripts.benchmarking_api import *
from scripts.verification_api import *
from scripts.export import BenchmarkingResults
from scripts.helpers import get_list_of_formats


def run_compile(options):
    make_binary("clean")
    list_of_apps = prepare_list_of_apps(options.apps)
    for app_name in list_of_apps:
        make_binary(app_name)
        if not binary_exists(app_name):
            print("Error! Can not compile " + app_name + ", several errors occurred.")


def run_prepare(options):
    create_graphs_if_required(get_list_of_all_graphs(options.mode), options.mode)


def run_benchmarks(options, benchmarking_results):
    list_of_apps = prepare_list_of_apps(options.apps)

    set_omp_environments(options)
    benchmarking_results.add_performance_header_to_xls_table(options.format)

    algorithms_tested = 0
    for app_name in list_of_apps:
        if is_valid(app_name, options):
            algorithms_tested += benchmark_app(app_name, benchmarking_results, options.format, options.mode,
                                               options.timeout)
        else:
            print("Error! Can not benchmark " + app_name + ", several errors occurred.")
    return algorithms_tested


def run_verify(options, benchmarking_results):
    list_of_apps = prepare_list_of_apps(options.apps)

    set_omp_environments(options)
    benchmarking_results.add_correctness_header_to_xls_table(options.format)
    algorithms_verified = 0

    for app_name in list_of_apps:
        if "el" in options.format or app_name == "sswp": # check takes too long to be done
            continue
        if is_valid(app_name, options):
            algorithms_verified += verify_app(app_name, benchmarking_results, options.format, options.mode,
                                              options.timeout)
        else:
            print("Error! Can not compile " + app_name + ", several errors occurred.")
    return algorithms_verified


def benchmark_and_verify(options, benchmarking_results):
    benchmarked_num = 0
    verified_num = 0
    if options.benchmark:
        benchmarked_num = run_benchmarks(options, benchmarking_results)

    if options.verify:
        verified_num = run_verify(options, benchmarking_results)

    print("\n\nEVALUATED PERFORMANCE OF " + str(benchmarked_num) + " GRAPH ALGORITHMS\n")
    print("VERIFIED " + str(verified_num) + " GRAPH ALGORITHMS\n\n")


def run(options, run_info):
    create_dir(DATASETS_DIR)

    benchmarking_results = BenchmarkingResults(options.name, options.mode)

    if options.compile:
        start = time.time()
        run_compile(options)
        end = time.time()
        if print_timings:
            print("compile WALL TIME: " + str(end-start) + " seconds")

    if options.download:
        start = time.time()
        download_all_real_world_graphs(options.mode)
        end = time.time()
        if print_timings:
            print("download WALL TIME: " + str(end-start) + " seconds")

    if options.prepare:
        start = time.time()
        run_prepare(options)
        end = time.time()
        if print_timings:
            print("graph generation WALL TIME: " + str(end-start) + " seconds")

    start = time.time()
    list_of_formats = get_list_of_formats(options.format)
    for format_name in list_of_formats:
        options.format = format_name
        benchmark_and_verify(options, benchmarking_results)

    if run_info != {} and len(list_of_formats) == 1:
        run_info["format"] = options.format
        if benchmarking_results.submit(run_info):
            print("Results sent to server!")
            benchmarking_results.offline_submit(run_info, options.name)
        else:
            print("Can not send results, saving to file...")
            benchmarking_results.offline_submit(run_info, options.name)

    end = time.time()
    if print_timings:
        print("benchmarking WALL TIME: " + str(end-start) + " seconds")

    if options.plot:
        benchmarking_results.plot(list_of_formats)

    benchmarking_results.finalize()


def main():
    # parse arguments
    parser = optparse.OptionParser()
    parser.add_option('-a', '--apps',
                      action="store", dest="apps",
                      help="specify an application to test (default all)", default="all")
    parser.add_option('-f', '--formats',
                      action="store", dest="format",
                      help="specify graph storage format used: "
                           "all, CSR, COO, TODO ... are currently available (default is CSR)", default="CSR")
    parser.add_option('-s', '--sockets',
                      action="store", dest="sockets",
                      help="set number of sockets used (default 1)", default=1)
    parser.add_option('-v', '--verify',
                      action="store_true", dest="verify",
                      help="run verification tests after benchmarking process (default false)", default=False)
    parser.add_option('-c', '--compile',
                      action="store_true", dest="compile",
                      help="preliminary compile all binaries (default false)", default=False)
    parser.add_option('-p', '--prepare',
                      action="store_true", dest="prepare",
                      help="preliminary convert graphs into .mtx format (default false)", default=False)
    parser.add_option('-b', '--benchmark',
                      action="store_true", dest="benchmark",
                      help="run all benchmarking tests (default false)", default=False)
    parser.add_option('-n', '--name',
                      action="store", dest="name",
                      help="specify name prefix of output files (default \"perf_results\")", default="perf_results")
    parser.add_option('-m', '--mode',
                      action="store", dest="mode",
                      help="specify testing mode: tiny-only, small-only, medium-only, large-only"
                           "tiny-small, tiny-small-medium", default="tiny-only")
    parser.add_option('-d', '--download',
                      action="store_true", dest="download",
                      help="download all real-world graphs from internet collections (default false)", default=False)
    parser.add_option('-t', '--timeout',
                      action="store", dest="timeout",
                      help="execution time (in seconds), after which tested app is automatically aborted. "
                           "(default is 1 hour)", default=3600)
    parser.add_option('-i', '--plot',
                      action="store_true", dest="plot",
                      help="plot performance data (default false)", default=False)

    options, args = parser.parse_args()

    run(options, {})


if __name__ == "__main__":
    main()


