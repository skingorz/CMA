"""
Goal
---
1. Read test results from log.txt files
2. Compute mean and std across different folders (seeds)
3. Save result to excel file

Usage
---
Assume the output files are saved under output/my_experiment,
which contains results of different seeds, e.g.,

my_experiment/
    seed1/
        log.txt
    seed2/
        log.txt
    seed3/
        log.txt

Run the following command from the root directory:

$ python tools/parse_test_res.py output/my_experiment

Add --ci95 to the argument if you wanna get 95% confidence
interval instead of standard deviation:

$ python tools/parse_test_res.py output/my_experiment --ci95

If my_experiment/ has the following structure,

my_experiment/
    exp-1/
        seed1/
            log.txt
            ...
        seed2/
            log.txt
            ...
        seed3/
            log.txt
            ...
    exp-2/
        ...
    exp-3/
        ...

Run

$ python tools/parse_test_res.py output/my_experiment --multi-exp
"""
import os
import re
import numpy as np
import os.path as osp
import argparse
import pandas as pd
from collections import OrderedDict, defaultdict

from dassl.utils import check_isfile, listdir_nohidden

datasets = ["caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101", "imagenet", "oxford_flowers", "oxford_pets", "stanford_cars", "sun397", "ucf101"]


def read_file(fpath, metrics):
    '''
    read a log file and get the results
    '''
    output = OrderedDict()
    good_to_go = False
    with open(fpath, "r") as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()

            if line == end_signal:
                good_to_go = True

            for metric in metrics:
                match = metric["regex"].search(line)
                if match and good_to_go:
                    if "file" not in output:
                        output["file"] = fpath
                    num = float(match.group(1))
                    name = metric["name"]
                    output[name] = num

    return output

def save_res(get_path, metrics, args=None, end_signal=None, save_path=''):
    data = {'1': [0] * len(datasets), '2': [0] * len(datasets), '4': [0] * len(datasets), '8': [0] * len(datasets), '16': [0] * len(datasets)}
    df = pd.DataFrame(data, index=datasets)
    for dataset in datasets:
        # if dataset != "imagenet" and dataset != "sun397":
        for shot in [1, 2,4,8,16]:
            shotname = f"{shot}"
            # if dataset == "imagenet":
            #     shotname = f"ep50_{shot}"
            # else:
            #     if shot == 1:
            #         shotname = f"ep50_{shot}"
            #     elif shot == 2 or shot == 4:
            #         shotname = f"ep100_{shot}"
            #     elif shot == 8 or shot == 16:
            #         shotname = f"{shot}"
            #     else:
            #         AssertionError
            for seed in [1]:
                path = get_path(dataset, shotname, seed)
                try:
                    acc = parse_single_function(metrics, directory=path, args=args, end_signal=end_signal)
                except:
                    acc = 0
                df.at[dataset, str(shot)]=acc
    print(df)
    df.to_csv(save_path)


    

            


def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))

def parse_single_function(*metrics, directory="", args=None, end_signal=None):
    '''
    Given a directory
    return the accuracy of these file
    '''
    print(f"Parsing files in {directory}")

    fpath = osp.join(directory, "log.txt")
    assert check_isfile(fpath)

    output = read_file(fpath, metrics)

    return output['accuracy']


def parse_average_function(*metrics, directory="", args=None, end_signal=None):
    print(f"Parsing files in {directory}")
    subdirs = listdir_nohidden(directory, sort=True)

    outputs = []

    for subdir in subdirs:
        fpath = osp.join(directory, subdir, "log.txt")
        assert check_isfile(fpath)

        output = read_file(fpath, metrics)

        if output:
            outputs.append(output)

    assert len(outputs) > 0, f"Nothing found in {directory}"

    metrics_results = defaultdict(list)

    for output in outputs:
        msg = ""
        for key, value in output.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.2f}%. "
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)
        print(msg)

    output_results = OrderedDict()

    print("===")
    print(f"Summary of directory: {directory}")
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = compute_ci95(values) if args.ci95 else np.std(values)
        print(f"* {key}: {avg:.2f}% +- {std:.2f}%")
        output_results[key] = avg
    print("===")

    return output_results

def collectbase2new(args, end_signal):
    parse_function = parse_single_function
    metric = {
        "name": args.keyword,
        "regex": re.compile(fr"\* {args.keyword}: ([\.\deE+-]+)%"),
    }

    Trainer = "PromptSRCMME_CLASSWISE"

    save_path = f"csvres/{Trainer}/base.csv"
    os.makedirs(f"csvres/{Trainer}", exist_ok=True)


    lambds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    data = {}
    for la in lambds:
        data[la] = [0] * len(datasets)
    df = pd.DataFrame(data, index=datasets)

    for seed in [1]:
        for la in lambds:
            Lambda = str(la)
            for dataset in datasets:
                path = f"output/{Trainer}/base2new/lambda{Lambda}/{dataset}/vit_b16_c2_ep20_batch4_4+4ctx_16shots/seed{seed}/base2new/test_base"
                try:
                    acc = parse_single_function(metric, directory=path, args=args, end_signal=end_signal)
                except:
                    acc = 0
                df.at[dataset, la]=acc
        
    df.to_csv(save_path)
            


    # def savecocoop_res(get_path, metrics, args=None, end_signal=None, save_path=''):
    #     data = {0: [0] * len(datasets), 'new': [0] * len(datasets)}
        
    #     df = pd.DataFrame(data, index=datasets)
    #     # print(df)
    #     for dataset in datasets:
    #         for seed in [1]:
    #             path_root = get_path(dataset, seed)
    #             for set_ in ['new']:
    #                 path = os.path.join(path_root, f"test_{set_}")
    #                 print(path)
    #                 acc = parse_single_function(metrics, directory=path, args=args, end_signal=end_signal)
    #                 df.at[dataset, set_]=acc
    #     # print(df)
    #     df.to_csv(save_path)

    # # Trainer = "CoCoOpMME_CLASSWISE"
    # # # lambds = [0.1, 0.01, 0.5, 0.05, 1, 5, 0.3, 3]
    # # lambds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # # lambds = [0.05, 1, 5]
    # for Lambda in lambds:
    #     Lambda = str(Lambda)
    #     save_path = f"csvres/{Trainer}/{Lambda}.csv"
    #     os.makedirs(f"csvres/{Trainer}", exist_ok=True)
    #     get_path = lambda dataset, seed: f"output/{Trainer}/lambda{Lambda}/{dataset}/vit_b16_c4_ep10_batch1_ctxv1_16shots/seed{seed}/base2new"
        
    #     savecocoop_res(get_path, metric, args, end_signal, save_path)

def main(args, end_signal):
    # parse_function=parse_average_function
    parse_function = parse_single_function
    metric = {
        "name": args.keyword,
        "regex": re.compile(fr"\* {args.keyword}: ([\.\deE+-]+)%"),
    }

    if args.multi_exp:
        final_results = defaultdict(list)

        for directory in listdir_nohidden(args.directory, sort=True):
            directory = osp.join(args.directory, directory)
            results = parse_function(
                metric, directory=directory, args=args, end_signal=end_signal
            )

            for key, value in results.items():
                final_results[key].append(value)

        print("Average performance")
        for key, values in final_results.items():
            avg = np.mean(values)
            print(f"* {key}: {avg:.2f}%")

    else:
        # res = parse_function(
        #     metric, directory=args.directory, args=args, end_signal=end_signal
        # )
        Trainers= ["PromptSRCMME_CLASSWISE"]
        # lambds = [0.1, 0.01, 0.5, 0.05, 1, 5, 0.3, 3]
        # lambds = [0.1, 0.01, 0.5, 1, 5, 0.3, 3]
        lambds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # lambds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        for Trainer in Trainers:
            # lambds = [0.05, 1, 5]
            for Lambda in lambds:
                Lambda = str(Lambda)
                save_path = f"csvres/{Trainer}/{Lambda}.csv"
                # get_path = lambda dataset, shot, seed: f"output/{Trainer}/lambda{Lambda}/{dataset}/rn50_{shot}shots/nctx16_cscFalse_ctpend/seed{seed}"
                get_path = lambda dataset, shot, seed: f"output/{Trainer}/few_shot/lambda{Lambda}/{dataset}/vit_b16_c2_ep50_batch4_4+4ctx_few_shot_{shot}shots/seed{seed}"

                save_res(get_path, metric, args, end_signal, save_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("directory", type=str, help="path to directory")
    parser.add_argument(
        "--ci95", action="store_true", help=r"compute 95\% confidence interval"
    )
    parser.add_argument("--test-log", action="store_true", help="parse test-only logs")
    parser.add_argument(
        "--multi-exp", action="store_true", help="parse multiple experiments"
    )
    parser.add_argument(
        "--keyword", default="accuracy", type=str, help="which keyword to extract"
    )
    args = parser.parse_args()

    end_signal = "Finish training"
    if args.test_log:
        end_signal = "=> result"

    collectbase2new(args, end_signal)
    # main(args, end_signal)
