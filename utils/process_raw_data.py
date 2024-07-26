import os
import sys

import json
import random
random.seed(0)

from collections import Counter

import numpy as np

import glob
import pandas as pd

import re

import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="../data/BLMagrXlang4dev")

    parser.add_argument("--sep", default="\t", help="column separator character in the data files")

    parser.add_argument("--train_test_split", default = 0.9)
    #parser.add_argument("--presplit_info", default="split_info.json", help="If the splitting should use information from another split -- e.g. which verbs to put in the train and which in the test -- provide the file with this info. If empty, it will generate it based on the given column name")
    parser.add_argument("--presplit_info", default="", help="If the splitting should use information from another split -- e.g. which verbs to put in the train and which in the test -- provide the file with this info. If empty, it will generate it based on the given column name")
    parser.add_argument("--split_column", default="Answer_1", help="which column to use as a train/test split info to avoid data leakage. This could be provided as a number or a string.")
    parser.add_argument("--proc_types", default="type_I type_II type_III", help="the data types to process")

    args = parser.parse_args()

    args.proc_types = "type_I_EN type_II_EN type_III_EN type_I_FR type_II_FR type_III_FR type_I_IT type_II_IT type_III_IT type_I_RO type_II_RO type_III_RO"

    data_types = args.proc_types.split(" ")

    if args.presplit_info != "":
        split_info = load_split_info(args.data_path + "/" + args.presplit_info)
    else:
        split_info = make_split_info(args, data_types[0])

    generate_jsons(args, data_types, split_info, templates=True)


def generate_jsons(args, data_types, split_info, templates = True):

    for type_data in data_types:
        # Extract sequences from csv files

        data_frames = []
        for data_file in glob.glob(args.data_path + "/" + type_data + "/*.csv"):
            data_x = pd.read_csv(data_file, sep = args.sep)
            data_x.dropna(inplace=True)
            print("headers in file {}: {}".format(data_file, data_x.columns))
            data_frames.append(data_x)

        data = pd.concat(data_frames, ignore_index=True, sort=False)
        data_headers = get_headers(args, data.columns, templates)

        if len(data_headers["sent"]) != len(data_headers["sent_temp"]):
            templates = False
            print("The number of sentence columns does not match the number of templates! No templates will be extracted (and some analyses during BLM tasks will not be feasible!)")
    
        json_path = args.data_path + "/" + type_data + "/datasets/"

        print("splitting {} instances into train/test {}".format(len(data), args.train_test_split))

        indices = split_indices(data, args.train_test_split, split_info, args.split_column, args)

        max_train = 5000 ## not really necessary more than this
        if sum(indices) < max_train:
            max_train = sum(indices)

        ## because of precomputed split info, the train/test split may not fit the given ratio -- there may be many more test instances. In that case, sample
        test_len = int(len(indices)*(1-args.train_test_split))
        if len(indices) - sum(indices) < test_len:
            test_len = len(indices) - sum(indices)

        print("selecting {} data for training".format(max_train))
        process_data(data, random.sample([i for i,x in enumerate(indices) if x == 1], max_train), json_path + "/train/sentences/", data_headers, templates=templates)

        print("selecting {} data for testing".format(len(indices) - sum(indices)))
        process_data(data, random.sample([i for i,x in enumerate(indices) if x == 0], test_len), json_path + "/test/sentences/", data_headers, templates=templates)



def split_indices(data, train_test_split, split_info, split_column, args):

    nr_train = int(len(data) * train_test_split)
    ## to prevent data leakage, split the data according to their value in the "split_column" and in the split_info dictionary

    indices = [0] * len(data)
    for idx in range(len(data)):
        val = data.iloc[idx][split_column]
        if val in split_info["train"]:
            indices[idx] = 1
            
    ## in case there is no overlap between the instances of different types, split randomly
    
    if sum(indices) < nr_train:
        print("only have {} train instances. selecting more ...".format(sum(indices)))
        values = data[split_column].tolist()
        make_split(data, values, indices, split_info, args)
        print("after adding train instances I have {}".format(sum(indices)))

    return indices
    
        
    
    
## I added skip so we do include some of the frequenct answers (fully) in the test set
def skip(n):
    return random.random() > n
    
    
def get_correct_answer(row, answer_headers, truth_headers):
    
    for i in range(len(answer_headers)):
        if is_true(row[truth_headers[i]]):
            return row[answer_headers[i]]
    
    print("Correct answer not found in row: {}".format(row))
    print("Exiting!")
    sys.exit()


def is_true(value):
    if isinstance(value, bool) or isinstance(value, np.bool_):
        return value
    
    return (value == "True")

'''
def in_data(elements, lists):
    inds_list = []
    for i in range(len(elements)):
        inds = all_indices(elements[i], lists[i])
        if len(inds) > 0:
            inds_list.append(inds)
        else:
            return False

    print("\tcontext/answers found at positions {}".format(inds_list))    
    return bool(set.intersection(*[set(x) for x in inds_list]))
'''    
    
def all_indices(el, m_list):
    inds = set()
    i = get_index(el, m_list, 0)

    while i > -1 and i < len(m_list):
        inds.add(i)
        i = get_index(el, m_list, i+1)
        
    return list(inds)
    
def get_index(el, m_list, offset):
    
    try:
        i = m_list.index(el, offset)
    except ValueError:
        i = -1
    return i
    

def get_headers(args, columns, templates=True):

    patts = {"sent": re.compile(r"^\s*Sent_\d+\s*$"), "sent_temp": re.compile(r"^\s*Sent_template_\d+\s*$"),
             "answ": re.compile(r"^\s*Answer_\d+\s*$"), "answ_temp": re.compile(r"^\s*Answer_template_\d+\s*$"),
             "label": re.compile(r"^\s*Answer_label_\d+\s*$"), "truth": re.compile(r"^\s*Answer_value_\d+\s*$")}
    headers = {"sent": [], "sent_temp": [], 
               "answ": [], "answ_temp": [],
               "label": [], "truth": []}
            
    for col in columns:
        for p in patts:
            if patts[p].match(col):
                if col not in headers[p]:  ## because one of the verb alternation csv files contains a duplicate column
                    headers[p].append(col)
                break
                
    if not templates:
        headers["answ_temp"] = []
                
    return headers
    
    
    

def process_data(data, indices, path, data_headers, templates=True, write_data=True):

    print("\tprocess data: {} indices".format(len(indices)))

    os.makedirs(path, exist_ok = True)
    x = []
    y = []
    truth_bool = []
    labels = []
    x_templates = []
    y_templates = []

    for idx in indices:
        row = data.iloc[idx]

        x.append([row[i] for i in data_headers["sent"]])
        y.append([row[i] for i in data_headers["answ"]])
        truth_bool.append([str(row[i]) for i in data_headers["truth"]])
        labels.append([str(row[i]) for i in data_headers["label"]])
        
        if templates:
            x_templates.append([edit_template(row[i]) for i in data_headers["sent_temp"]])
            y_templates.append([edit_template(row[i]) for i in data_headers["answ_temp"]])

    with open(path + "/x.json", "w", encoding="utf-8") as f:
        json.dump(x, f, ensure_ascii=False)
    with open(path + "/truth_bool.json", "w", encoding="utf-8") as f:
        json.dump(truth_bool, f, ensure_ascii=False)
    with open(path + "/y.json", "w", encoding="utf-8") as f:
        json.dump(y, f, ensure_ascii=False)
    with open(path + "/labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)
        
    if templates:
        with open(path + "/x_templates.json", "w", encoding="utf-8") as f:
            json.dump(x_templates, f, ensure_ascii=False)
            print("written {} sentence templates".format(len(x_templates)))
        with open(path + "/y_templates.json", "w", encoding="utf-8") as f:
            json.dump(y_templates, f, ensure_ascii=False)
            print("written {} answer templates".format(len(y_templates)))

    print("Writing {} instances from data ({})".format(len(indices), len(data)))

    if write_data:
        ## also keep the train/test in tsv files
        partition = pd.DataFrame([data.iloc[i] for i in indices])
        partition.to_csv(path + "/data.tsv", sep="\t", index=False, encoding="utf-8")
    


## remove the "None" elements
def edit_template(temp):

    return temp
    '''
    if "Noun" in temp:  ## the verb alternation data uses Noun in patterns, the agreement does not
        return re.sub(r"\d", "", temp)

    temp = temp.lower()
    temp = temp.replace("_", "-")
    temp = temp.replace("-sg", "-s")
    temp = temp.replace("-pl", "-p")
    temp = temp.replace(" p1", " pp1")
    temp = temp.replace(" p2", " pp2")
    temp = temp.replace("subj", "np")
    temp = temp.replace("v-", "vp-")
    temp_list = temp.split(" ")
    for x in temp_list:
        if "none" in x:
            temp_list.remove(x)
            
    return " ".join(temp_list)
    '''

def load_split_info(presplit_info):
    with open(presplit_info, "r") as f:
        return json.load(f)

def make_split_info(args, data_type):
    data_frames = []
    print("Splitting data from {}/{}/*.csv".format(args.data_path, data_type))
    for data_file in glob.glob(args.data_path + "/" + data_type + "/*.csv"):
        print("\tprocessing file {}".format(data_file))
        data_x = pd.read_csv(data_file, sep=args.sep)
        data_x.dropna(inplace=True)
        print("headers in file {}: {}".format(data_file, data_x.columns))
        data_frames.append(data_x)

    data = pd.concat(data_frames, ignore_index=True, sort=False)
    if args.split_column.isnumeric():
        args.split_column = data.columns[int(args.split_column)]

    split_info = {"train": [], "test": []}
    values = data[args.split_column].tolist()
    indices = [0] * len(values)

    make_split(data, values, indices, split_info, args)
    
    return split_info

    
def make_split(data, values, indices, split_info, args):

    print("Before maxing split: {} train instances ({})".format(sum(indices), len(split_info["train"])))
    
    value_indices = get_indices(values, split_info)
    train_len = int(len(indices) * args.train_test_split)

    print("adding {} training instances".format(train_len - sum(indices)))

    for (val, inds) in value_indices.items():
        if (sum(indices) < train_len) and not skip(args.train_test_split):
            for i in inds:
                indices[i] = 1

    print("Splitting the data {}:{} (had aimed for {} training)".format(sum(indices), len(indices) - sum(indices),
                                                                        train_len))

    print("train/test split: {}/{}".format(sum(indices), len(indices) - sum(indices)))

    split_info["train"].extend([values[i] for i in range(len(indices)) if indices[i]==1])
    split_info["test"].extend([values[i] for i in range(len(indices)) if indices[i]==0])

    with open(args.data_path + "/split_info.json","w") as f:
        json.dump(split_info, f)



## do not gather indices of values already in the split info
def get_indices(values, split_info):

    value_indices = {}
    for i, val in enumerate(values):
        if (val not in split_info["train"]) and (val not in split_info["test"]):
            if (val not in value_indices):
                value_indices[val] = []
            value_indices[val].append(i)

    return value_indices


if __name__ == '__main__':
    main()
