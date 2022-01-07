import os
import optparse
import argparse
import json
import pickle


def main():
    full_perf_data = {}
    with open('perf_dict.pkl', 'rb') as f:
        full_perf_data = pickle.load(f)
    print(full_perf_data)


if __name__ == "__main__":
    main()
