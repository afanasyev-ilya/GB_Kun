#!/usr/bin/env python3

import argparse
import os
import subprocess
import xlsxwriter

import ast

import pandas as pd
import numpy as np

from settings import *

def make_absolute(f, directory):
    if os.path.isabs(f):
        return f
    return os.path.normpath(os.path.join(directory, f))


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise argparse.ArgumentTypeError(f"'{string}' is not a valid path")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Script for testing different GraphBLAS backends")
    parser.add_argument('data_path', type=dir_path, metavar='DIR', help='= Path to directory containing the .mtx files')
    parser.add_argument('-r','--repeat', type=int, dest='N', default=10, help='= Number of iterations of the loop')
    parser.add_argument('-o','--output', type=str, dest='output', default=REPORT_DATE, help='= Output file name (must be *.xlsx)')
    parser.add_argument('-l','--loop', type=str, dest='loop', default='outer', choices=['inner', 'outer'], help='= Loop type, inner (in binary) or outer (in script)') # inner or outer
    parser.add_argument('-A','--algorithms', type=str, dest='algos', metavar='ALG', nargs='+', help='= algorithms at `backend/algorithms` to be tested, non-existent ones will be skipped')
    parser.add_argument('-O','--operations', type=str, dest='ops', metavar='OP', nargs='+', help='= operations at `backend/operations` to be tested, non-existent ones will be skipped')
    return parser.parse_args()

def df_to_xlsx(df_):
    df = df_.copy()
    df = df.reset_index()
    df['Backend_Algo'] = df['Backend'] + '/' + df['Algo']
    df = df.drop(['Algo', 'Backend'], axis = 1)
    tmp = df.loc[:,(df.columns.get_level_values(0), 'mean')].T.droplevel(1).T

    tmp['Dataset'] = df['Dataset']
    tmp['Backend_Algo'] = df['Backend_Algo']
    tmp['CPU usage fixed'] = (tmp['System time'] + tmp['User time'] - tmp['Read time'])/(tmp['Elapsed time'] - tmp['Read time'])
    tmp['Max memory'] = tmp['Max memory']/1e6

    tmp = pd.pivot_table(tmp, index = 'Dataset', columns='Backend_Algo', values = ['Read time', 'Run time', 'CPU usage fixed', 'CPU usage', 'Max memory'])
    tmp = tmp.stack().T.stack()
    tmp.index.names = [None, None]

    df = tmp
    tmp = tmp.iloc[0:0]

    return df


def run_tests(args, writer, type):
    if type == 'algorithms':
        variants = args.algos
        postfix = '/algorithms/'
    elif type == 'operations':
        variants = args.ops
        postfix = '/operations/'
    data_dir = os.fsencode(args.data_path)
    N = args.N

    idx = pd.MultiIndex(levels=[[],[],[]], codes=[[],[],[]], names=['Backend', 'Algo', 'Dataset'])
    cols = ['Read time', 'Run time', 'User time', 'System time', 'Elapsed time', 'CPU usage', 'Max memory'] 
    multicols = pd.MultiIndex.from_product([['Read time', 'Run time', 'User time', 'System time', 'Elapsed time', 'CPU usage', 'Max memory'],['mean','std']])

    df = pd.DataFrame(index = idx, columns=multicols)
    df_tmp = pd.DataFrame(columns=['Backend', 'Algo', 'Dataset'] + cols)

    if variants:
        for op in np.unique(variants):
            for backend in backends.keys():
                backend_path = backends[backend]
                bins_dir = backend_path + postfix + op
                if os.path.isdir(bins_dir):
                    for binary in os.listdir(os.fsencode(bins_dir)):
                        if os.fsdecode(binary).startswith(f'{op}'):
                            for dataset in os.listdir(data_dir):
                                binaty_str = bins_dir + '/' + os.fsdecode(binary)
                                dataset_str = args.data_path + '/' + os.fsdecode(dataset)

                                script_args = TIME_PREFIX + [binaty_str, dataset_str]

                                print(f'######### Executing {os.fsdecode(binary)} on { os.fsdecode(dataset)}')

                                df_tmp = df_tmp.iloc[0:0]

                                if args.loop == 'outer':
                                    for i in range(N):
                                        out = subprocess.run(script_args + [str(1)])
                                        if out.returncode == 0: # subprocess ended successfully
                                            with open('time.txt') as f:
                                                data = f.read()
                                                data = data[:-3] + '}'
                                            time_d = ast.literal_eval(data)
                                            with open('algo.txt') as f:
                                                data = f.read()
                                            algo_time_d = ast.literal_eval(data)
                                            time_d.update(algo_time_d)
                                            time_d.update({'Backend' : f'{backend}', 'Algo' : f'{os.fsdecode(binary)}', 'Dataset' : f'{os.fsdecode(dataset)}'})
                                            
                                            df_tmp = pd.concat([df_tmp, pd.DataFrame(time_d, index = [0])], axis = 0)
                                        else:
                                            print(f"######### Error executing {os.fsdecode(binary)} on { os.fsdecode(dataset)}!")
                                        os.remove("time.txt")
                                        os.remove('algo.txt')
                                        #time.sleep(5)
                                    if not df_tmp.empty: # if at least 1 subproccess in loop ended successfully
                                        df_tmp = df_tmp.astype({"CPU usage":"int","Max memory":"int"})
                                        df_pivot = pd.pivot_table(df_tmp, index = ['Backend', 'Algo', 'Dataset'], values = cols,  aggfunc={col : [np.mean, np.std] for col in cols})
                                        df = pd.concat([df, df_pivot])

                                    print(f'######### End of execution')
                                else: # if args.loop == 'inner':
                                    out = subprocess.run(script_args + [str(N)])

                                    if out.returncode == 0: # subprocess ended successfully
                                        with open('time.txt') as f:
                                            data = f.read()
                                            data = data[:-3] + '}'
                                        time_d = ast.literal_eval(data)
                                        with open('algo.txt') as f:
                                            data = f.readlines()
                                            data = [line.rstrip() for line in data]
                                        data = [ast.literal_eval(d) for d in data]
                                        algo_time_d = [d.update(time_d) for d in data]
                                        algo_time_d = [d.update({'Backend' : f'{backend}', 'Algo' : f'{os.fsdecode(binary)}', 'Dataset' : f'{os.fsdecode(dataset)}'}) for d in data]
                                        for d in data:
                                            df_tmp = pd.concat([df_tmp, pd.DataFrame(d, index = [0])], axis = 0)

                                        df_tmp = df_tmp.astype({"CPU usage":"int","Max memory":"int"})
                                        df_pivot = pd.pivot_table(df_tmp, index = ['Backend', 'Algo', 'Dataset'], values = cols,  aggfunc={col : [np.mean, np.std] for col in cols})
                                        df = pd.concat([df, df_pivot])

                                        os.remove('algo.txt')
                                    else:
                                        print(f"######### Error executing {os.fsdecode(binary)} on { os.fsdecode(dataset)}!")
                                    print(f'######### End of execution')

                                    os.remove("time.txt")
                else:
                    print(f'# Realiszation of {op} doesn\'t exist in {backend}, skipping.')
            if not df.empty:        
                df.reorder_levels(['Dataset','Backend','Algo']).sort_index().to_excel(writer, sheet_name = op, merge_cells=True)
                df_for_draw = df_to_xlsx(df)
                df_for_draw.to_excel(writer, sheet_name = f'{op}_draw', merge_cells=True)
                df = df.iloc[0:0]
        
def main():
    args = parse_arguments()
    print(args)
    if args.output.endswith(".xlsx"):
        writer = pd.ExcelWriter(args.output, engine = 'xlsxwriter')
    else:
        raise NameError('Output filename must end with .xlsx!')

    run_tests(args, writer, 'algorithms')
    run_tests(args, writer, 'operations')

    writer.close()

if __name__ == "__main__":
    main()
