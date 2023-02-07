
import csv
import os
import json
import random
import re

import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="../data/BLM-AgrF",\
                        help="Path to the data directory")
    
    args = parser.parse_args()
    args.data_path = os.path.abspath(args.data_path)
    
    for type_data in ["type_I", "type_II", "type_III"]:
        # Extract sequences from csv files
        sequences = []
        if type_data == ".DS_Store":
            continue
        first_path = args.data_path + "/" + type_data + "/"   ##"/raw/"
        for path in os.listdir(first_path):
            if "csv" in path:
                print(type_data, path)
                if path == ".DS_Store":
                    continue
                full_dir = first_path + path
                #label = path.split("_")[1][:-4]
                label = path.split(".")[0]
                with open(full_dir, encoding="utf-8") as csv_file:
                    reader = csv.reader(csv_file)
                    for i, line in enumerate(reader):
                        if i == 0:
                            continue
                        if len(line) == 0:
                            continue
                        #line = line[1:]
                        line = [process_sent(sent) for sent in line[1:]]
                        # uncomment it when using flaubert uncased 
                        # line = [el.strip(".").strip(" ").lower() for el in line]
                        line.append(label)
                        sequences.append(line)
    
        random.shuffle(sequences)
        print(len(sequences))
    
        nr_train_insts = int(len(sequences) * 0.9)
        # 90% train-valid (80% train +10%valid), 10% test
        train_data = sequences[:nr_train_insts]
        test_data = sequences[nr_train_insts:]
    
        print(len(train_data), len(test_data))
        
        json_path = args.data_path + "/" + type_data + "/datasets/"
       
        process_data(train_data, json_path + "/train/sentences/", "train")
        process_data(test_data, json_path + "/test/sentences/", "test")    
    

def process_data(data, path, prefix):

    os.makedirs(path, exist_ok = True)
    x = []
    y = []
    truth_bool = []
    labels = []
    type_file = []
        
    for sequence in data:
        # sequence
        x.append(sequence[:7])
        # all the possible answers
        ans = sequence[7:13]
        y.append(ans)
        # booleans indicating the truth between answers
        truth_bool.append(sequence[13:19])
        labels.append(sequence[19:-1])
        
        type_file.append(sequence[-1])

    with open(path + "/x.json", "w", encoding="utf-8") as f:
        json.dump(x, f, ensure_ascii=False)
    with open(path + "/truth_bool.json", "w", encoding="utf-8") as f:
        json.dump(truth_bool, f, ensure_ascii=False)
    with open(path + "/y.json", "w", encoding="utf-8") as f:
        json.dump(y, f, ensure_ascii=False)
    with open(path + "/labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)
    with open(path + "/type_file.json", "w", encoding="utf-8") as f:
        json.dump(type_file, f, ensure_ascii=False)


## not really necessary anymore, the data was cleaned up before
def process_sent(sent):
    sent = sent.strip()
    sent = re.sub(r'( [A-Z])', lambda lowercase: lowercase.group(1).lower(), sent)  ## lower case inner capitalization
    sent = re.sub(r"('|´|`|’) ", r"'", sent)
    sent = re.sub(r" ('|´|`|’)", r"'", sent)
    return ' '.join(re.split(r'\s+', sent))


if __name__ == '__main__':
    main()
