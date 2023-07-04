import os
import json
import random

import glob
import pandas as pd

import re

import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="../data/blm4dev", help="data directory")    
    parser.add_argument("--sep", default=",", help="column separator character in the data files")
        
    parser.add_argument("--train_test_split", default = 0.9)
    
    args = parser.parse_args()
    use_templates = False
    
    for type_data in ["type_I", "type_II", "type_III"]:
        # Extract sequences from csv files
        if type_data == ".DS_Store":
            continue
        for data_file in glob.glob(args.data_path + "/" + type_data + "/*.csv"):
            data = pd.read_csv(data_file, sep = args.sep)
            print("headers in file {}: {}".format(data_file, data.columns))
            data_headers = get_headers(args, data.columns, use_templates)
            
            indices = [0] * len(data)
            train_len = int(len(indices) * args.train_test_split) 
            indices[:train_len] = [1] * train_len
            random.shuffle(indices)

            json_path = args.data_path + "/" + type_data + "/datasets/"
           
            process_data(data, [i for i, x in enumerate(indices) if x == 1], json_path + "/train/sentences/", data_headers, use_templates) 
            process_data(data, [i for i, x in enumerate(indices) if x == 0], json_path + "/test/sentences/", data_headers, use_templates)    
    

#    sent_headers, answer_headers, truth_headers, label_headers, sent_template_headers, answer_template_headers = data_headers

def get_headers(args, columns, templates=False):

    patts = {"sent": re.compile(r"Sent_\d+"), "sent_temp": re.compile(r"Sent_template_\d+"), 
             "answ": re.compile(r"Answer_\d+"), "answ_temp": re.compile(r"Answer_template_\d+"), 
             "label": re.compile(r"Answer_label_\d+"), "truth": re.compile(r"Answer_value_\d+")}   
    headers = {"sent": [], "sent_temp": [], 
               "answ": [], "answ_temp": [],
               "label": [], "truth": []}
            
    for col in columns:
        for p in patts:
            if patts[p].search(col):
                headers[p].append(col)
                break
                
    if not templates:
        headers["answ_temp"] = []
                
    return headers["sent"], headers["answ"], headers["truth"], headers["label"], headers["sent_temp"], headers["answ_temp"]
    
    
    

def process_data(data, indices, path, data_headers, templates=False):

    os.makedirs(path, exist_ok = True)
    x = []
    y = []
    truth_bool = []
    labels = []
    templates = []
    #types = []  ## for the agreement data these are rel, main, compl -- but the templates supersede this, I think, so I don't output this anymore 

    sent_headers, answer_headers, truth_headers, label_headers, sent_template_headers, answer_template_headers = data_headers

    for idx in indices:
        row = data.iloc[idx]
        x.append([row[i] for i in sent_headers])
        y.append([row[i] for i in answer_headers])
        truth_bool.append([str(row[i]) for i in truth_headers])
        labels.append([str(row[i]) for i in label_headers])
        
        if templates:
            templates.append([[row[i] for i in sent_template_headers], [row[i] for i in answer_template_headers]])

    with open(path + "/x.json", "w", encoding="utf-8") as f:
        json.dump(x, f, ensure_ascii=False)
    with open(path + "/truth_bool.json", "w", encoding="utf-8") as f:
        json.dump(truth_bool, f, ensure_ascii=False)
    with open(path + "/y.json", "w", encoding="utf-8") as f:
        json.dump(y, f, ensure_ascii=False)
    with open(path + "/labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)
        
    if templates:
        with open(path + "/templates_file.json", "w", encoding="utf-8") as f:
            json.dump(templates, f, ensure_ascii=False)


if __name__ == '__main__':
    main()
