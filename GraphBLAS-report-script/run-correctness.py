import argparse
import os
import subprocess
from tqdm import tqdm
import ast
import networkx as nx
import scipy as sp
import scipy.io
import numpy as np

TIME_PREFIX = ["/usr/bin/time", "-o", "time.txt", "-f", "{'User time': %U, 'System time': %S, 'Elapsed time': %e, 'Max memory': %M, 'CPU usage': %P}"]

def time_prefix(string):
    return ["/usr/bin/time", "-o", f"{string}_time.txt", "-f", "{'User time': %U, 'System time': %S, 'Elapsed time': %e, 'Max memory': %M, 'CPU usage': %P}"]

backends = {
    'SuiteSparce:GraphBLAS' : 'algs/ssgb/correctness',
    'GB_Kun' : 'algs/gb_kun/correctness'
    }

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise argparse.ArgumentTypeError(f"'{string}' is not a valid path")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script for correctness testing of different GraphBLAS backends")
    parser.add_argument('-d','--data', 
                        type=dir_path, 
                        metavar='DIR', 
                        help='= Path to directory containing the .mtx files')
    parser.add_argument('-n', 
                        type=int, 
                        dest='N', 
                        default=1000, 
                        help='= Number of graphs to be generated')
    parser.add_argument('-s','--size', 
                        type=int, 
                        dest='size', 
                        default=1000, 
                        help='= Size of generated graphs')
    parser.add_argument('-a','--algorithms',
                        type=str, 
                        dest='algos', 
                        metavar='ALG', 
                        nargs='+', 
                        help='= algorithms to be tested, non-existent ones will be skipped')
    parser.add_argument('-t','--threads', 
                        type=int, 
                        dest='threads', 
                        default=os.cpu_count(), 
                        metavar='THREADS', 
                        help='= Number for OMP_THREADS to use, default : os.cpu_count()')
    parser.add_argument('-p','--probability', 
                    type=float, 
                    dest='prob', 
                    default=0.1, 
                    metavar='PROB', 
                    help='= Probability for graph generator')
    args = parser.parse_args()
    return args
def do_correctness_test(operation, s, N, threads, p):
    err_log = ""
    for i in tqdm(range(N), desc=f'{operation} loop'):
        G = nx.erdos_renyi_graph(s, p, seed=i, directed=True)
        m = nx.to_scipy_sparse_matrix(G)
        sp.io.mmwrite("tmp.mtx", m)

        src = str(np.random.randint(1, s))

        ssgb_script_args = time_prefix('ssgb') + [backends['SuiteSparce:GraphBLAS'] + '/' + f'{operation}_check', 'tmp.mtx', 'ssgb_ans.txt', src]
        gbkun_script_args = time_prefix('gb_kun') + [backends['GB_Kun'] + '/' + f'{operation}_check', 'tmp.mtx', 'gbkun_ans.txt', src]

        os.environ['OMP_NUM_THREADS']=f'{threads}'
        os.environ['OMP_PROC_BIND']='close'
        os.environ['OMP_PLACES']='cores'
        _ = subprocess.run(ssgb_script_args, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        os.environ['OMP_NUM_THREADS']=f'{threads}'
        os.environ['OMP_PROC_BIND']='close'
        os.environ['OMP_PLACES']='cores'
        _ = subprocess.run(gbkun_script_args, stderr=subprocess.PIPE, stdout=subprocess.PIPE)   

        if operation == 'tc':
            with open('ssgb_ans.txt') as f:
                for line in f.readlines():
                    a = line.strip().split(" ")
                    ssgb_ans= a

            with open('gbkun_ans.txt') as f:
                for line in f.readlines():
                    a = line.strip().split(" ")
                    gbkun_ans = a

            if gbkun_ans != ssgb_ans:
                err_log += str(gbkun_ans) + ' ' + str(ssgb_ans) + " " + str(i) + "\n"
        elif operation == 'pr':
            ssgb_ans = {}
            with open('ssgb_ans.txt') as f:
                for line in f.readlines():
                    a, b = line.strip().split(" ")
                    ssgb_ans[a] = b

            gbkun_ans = {}
            with open('gbkun_ans.txt') as f:
                for line in f.readlines():
                    a, b = line.strip().split(" ")
                    gbkun_ans[a] = b

            ssgb_ans = np.array(list(ssgb_ans.items()), dtype=np.double)
            gbkun_ans = np.array(list(gbkun_ans.items()), dtype=np.double)
            err = np.max(np.abs(gbkun_ans[:,1]-ssgb_ans[:,1]))
            #print(f'{ssgb_ans[:,1].flatten()}\n{gbkun_ans[:,1].flatten()}\n')
            if np.array_equal(ssgb_ans[:,0] ,ssgb_ans[:,0]) or (err > 1e-9):
                err_log += f'{operation} : seed {i} : error {err:.17f}\n' 
        else:
            ssgb_ans = {}
            with open('ssgb_ans.txt') as f:
                for line in f.readlines():
                    a, b = line.strip().split(" ")
                    ssgb_ans[a] = b

            gbkun_ans = {}
            with open('gbkun_ans.txt') as f:
                for line in f.readlines():
                    a, b = line.strip().split(" ")
                    gbkun_ans[a] = b

            if gbkun_ans != ssgb_ans:
                err_log += f'{operation} : seed {i}\n'

    if err_log != "":
        print(err_log)
def main():
    args = parse_arguments()
    print(args)

    if args.algos:
        for op in np.unique(args.algos):
            do_correctness_test(op, args.size, args.N, args.threads, args.prob)
    else:
        print(f'# No algorithms were provided')
if __name__ == "__main__":
    main()