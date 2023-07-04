
from datetime import datetime
import logging
import json
import re

import pandas as pd

import run_funcs

def load_config(args):    
    if args.config_file is not None:

        logging.info("Reading arguments from configuration file {}".format(args.config_file))
        
        config_info = dict()
        with open(args.config_file) as json_file:
            config_info = json.load(json_file)

        for arg in config_info:
            setattr(args, arg, config_info[arg])

    
def save_config(model_path, args):
    config_file = f'{model_path}/config.json'
    config_info = dict()
    config_info['model_path'] = model_path
    for arg in vars(args):
        if arg != "device":
            config_info[arg] = getattr(args, arg)

    with open(config_file, "w") as json_file:
        json.dump(config_info, json_file)
        
        
## ____________________________________________________________________
## reading the answer labels (for computing error statistics)

def get_answer_labels(data_dir):
    
    labels = []
    with open(data_dir + "/labels.json", "r", encoding="utf-8") as file:
        data = json.load(file)    

    labels = data[0]
    ## remove the label for the correct sentence. In the current datasets it is either "True" or "Correct"
    if "True" in labels:
        labels.remove("True")
    elif "Correct" in labels:
        labels.remove("Correct")

    return labels

## ____________________________________________________________________
## for inspecting the latent layer    

def print_latent_layer(args, d_types, model_file):

    #'''
    (model, _model_path, train_dirs, test_dirs, _train_percs, output_dir) = run_funcs.initialize_and_load_model(args, model_file, d_types)
    model_name = model_file.split("/")[-2]  ## the name of the directory contains the information about the model
    latents_file = args.results_dir + "/" + model_name + "__latents.tsv"

    latents_type = []
    latents_train_test = []
    latents_vectors = []
    disc_size = 0
    
    patt = re.compile(r"latent-disc-size_(\d+)x(\d+)_")
    m = patt.search(model_name)
    if m:
        disc_size = int(m.group(1)) * int(m.group(2))
        
    for d_type in d_types:
        latent_values = {"train": [], "test": []}
        
        run_funcs.get_latent_values(model, train_dirs[d_type], latent_values, "train", args.device, args.batch_size)
        N = len(latent_values["train"])
        latents_type.extend([d_type] * N)
        latents_train_test.extend(["train"] * N)
        latents_vectors.extend(latent_values["train"])
        logging.info("{} latent vectors for {}/train".format(N, d_type))
                
        run_funcs.get_latent_values(model, test_dirs[d_type], latent_values, "test", args.device, args.batch_size)
        N = len(latent_values["test"])
        latents_type.extend([d_type] * N)
        latents_train_test.extend(["test"] * N)
        latents_vectors.extend(latent_values["test"])
        logging.info("{} latent vectors for {}/test".format(N, d_type))
        
    disc_size_info = [disc_size] * len(latents_type)

    L = list(map(list, zip(latents_type, disc_size_info, latents_train_test, latents_vectors)))
    latents_df = pd.DataFrame(L, columns = ["type", "discrete_len", "train/test", "latent"])
    latents_df.to_csv(latents_file, sep="\t")
    logging.info("Latents (with discrete size = {}) written to file {}\n\n".format(disc_size, latents_file))
    
    

## ____________________________________________________________________    
## a string containing main arguments info

def get_run_info(args):
    
    timestamp = datetime.now().strftime("%d-%b-%Y_%H:%M")
    
    model_name = ""
    if args.baseline:
        model_name = args.baseline_sys
    else:
        model_name = args.sys + "_" + args.sampling + "-sampling"
        
    return model_name + "_" + "-".join(args.train_perc.split(" ")) + "-train_perc_" + str(args.epochs) + "-epochs_" + str(args.lr) + "-lr_" + timestamp
