
import pandas as pd
import os
import re

import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="data/blm4dev",\
                        help="Path to the data directory")
    parser.add_argument("--out-path", default="data/blm4dev_corrected",\
                        help="Path to directory for the cleaned up data (lower cased articles in the middle of the sentence, consistent character use for apostrophes, cleaned up spaces)")
    
    args = parser.parse_args()
    
    sep = ","
    
    for type_data in ["type_I", "type_II", "type_III"]:
        # Extract sequences from csv files
        sequences = []
        if type_data == ".DS_Store":
            continue
        first_path = args.data_path + "/" + type_data + "/"   ##"/raw/"
        out_path = args.out_path + "/" + type_data + "/"
        os.makedirs(out_path, exist_ok=True)
        for path in os.listdir(first_path):
            if "csv" in path:
                print(type_data, path)
                if path == ".DS_Store":
                    continue
                in_file = first_path + path
                out_file = out_path + path
                with open(in_file, encoding="utf-8") as csv_file:
                    df = pd.read_csv(csv_file, sep=sep)
                    clean_csv(df)
                    
                    df.to_csv(out_file, sep=sep, index=False)



def clean_csv(df):
    
    for i, row in df.iterrows():
        for col in df.columns:
            df.at[i,col] = process_sent(row[col])
        

def process_sent(sent):
    if isinstance(sent, str):
        sent = sent.strip()
        sent = re.sub(r'( [A-Z])', lambda lowercase: lowercase.group(1).lower(), sent)  ## lower case inner capitalization
        sent = re.sub(r" ('|´|`|’)", r"'", sent)
        sent = re.sub(r"('|´|`|’) ", r"'", sent)
        print("S: {}".format(sent))
        return ' '.join(re.split(r'\s+', sent))
    return sent


if __name__ == '__main__':
    main()
