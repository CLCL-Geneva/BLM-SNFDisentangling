'''
Created on Nov 16, 2022

@author: vivi
'''


import argparse
import numpy as np
import random

import torch
from torchinfo import summary

import os
import glob
import re
import json

import pickle

from datetime import datetime

import embeddings
import data_extractor

import train 
import test  #, plot
import plots
import baselines.baseline as baseline

import pandas as pd

results_keys = ["run", "train_on", "test_on", "TP", "FP", "FN", "TN", "P", "R", "F1", "Acc", "agreement_error", "coordination", "mis_num", "N1_alter", "N2_alter"]


def initialize(args, d_types):
    
    timestamp = datetime.now().strftime("%d-%b-%Y_%H:%M")
    
    emb_type = args.transformer + "_sentembs" 

    data_path = args.data_dir
    output_dir = args.data_dir + "/" + args.type + "/output/" + emb_type +  "/"
         
    # setting the seed for reproducibility
    print("Setting seeds...")
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    #torch.use_deterministic_algorithms(True)

    #print("use cuda: {}".format(args.cuda))
    #print("is cuda available: {}".format(torch.cuda.is_available()))

    # enable cuda
    args.cuda = True
    args.device = None
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        print("cuda available...")
    else:
        args.device = torch.device("cpu")
        print("cuda not available...")
    
    train_dirs = {}
    test_dirs = {}
    train_percs = {}
    train_percs_str = args.train_perc.split(" ")
    #for i in range(len(d_types)):
    #    x = d_types[i]
    i = 0
    for x in ["type_I", "type_II", "type_III"]:
        train_dirs[x] = data_path + "/" + x + "/datasets/train/embeddings/" + emb_type + "/"
        test_dirs[x] = data_path + "/" + x + "/datasets/test/embeddings/" + emb_type + "/"
        if len(train_percs_str) > i:
            train_percs[x] = float(train_percs_str[i])
        else:
            train_percs[x] = 0
        i += 1
        
    # vectors for training
    # USE BATCHES TO AVOID OUT OF MEMORY!!!
    args.extract = True
    if args.extract:
        for data_type in d_types:
            
            input_train_x = args.data_dir + "/" + data_type + "/datasets/train/sentences/"
            input_test_x = args.data_dir + "/" + data_type + "/datasets/test/sentences/"

            print("Extracting training embeddings for {} to {} ...".format(data_type, train_dirs[data_type]))
            embeddings.extract_emb(input_train_x, train_dirs[data_type], args, args.sent_emb_dim)
            print("Extracting test embeddings for {} to {} ...".format(data_type, test_dirs[data_type]))
            embeddings.extract_emb(input_test_x, test_dirs[data_type], args, args.sent_emb_dim)

    args.seq_size = 7

    # VAE initialization
    print("Model initialization...")
    
    model = model_baseline(args.sent_emb_dim, args)
    print_model_summary(model, args.sent_emb_dim, args)

    model_name = model.getinfo()
    model_path = output_dir + "/model_" + model_name + "_" + "-".join(args.train_perc.split(" ")) + "-train_perc_" + str(args.epochs) + "-epochs_" + str(args.lr) + "-lr_" + timestamp + "/"

    return (model, model_path, train_dirs, test_dirs, train_percs, output_dir)
    


def model_baseline(dim, args):
    print("\nInitialising baseline model ...\n")
    if "ffnn" in args.baseline_sys.lower():
        model = baseline.BaselineFFNN(dim, int(args.seq_size))
    elif "cnn" in args.baseline_sys.lower():
        model = baseline.BaselineCNN(dim, int(args.seq_size))
    else:
        print("Unknown model. defaulting to FFNN")
        model = baseline.BaselineFFNN(dim, int(args.seq_size))
        
    return model



def get_complete_representations(args):
    
    transformers = ["bert", "flaubert"]
    d_types = ["type_I", "type_II", "type_III"]
    
    for transf in transformers:

        outdir = args.data_dir + "/" + transf
        os.makedirs(outdir, exist_ok = True)      
        
        args.transformer = transf
        (tokenizer, model) = embeddings.load_transformer(args)
        representations = {}
        
        for data_type in d_types:
            
            input_train_x = args.data_dir + "/" + data_type + "/datasets/train/sentences/"
            input_test_x = args.data_dir + "/" + data_type + "/datasets/test/sentences/"
    
            embeddings.extract_complete_emb(input_train_x, tokenizer, model, representations, outdir)
            embeddings.extract_complete_emb(input_test_x, tokenizer, model, representations, outdir)
            
        outfile = args.data_dir + "/sentences_index_" + transf + ".pkl"
        with open(outfile, 'wb') as handle:
            pickle.dump(representations, handle, protocol=pickle.HIGHEST_PROTOCOL)        


def run_experiment(N_exp, d_types, d_type, args, results):

    model = None
    for exp_nr in range(N_exp):
        (model, model_path, train_dirs, test_dirs, train_percs, output_dir) = initialize(args, d_types)  
        # ADAM optimizers
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

        exp_type = "_attention-" + str(args.attention) + "_beta_seq-" + str(args.beta) + "_beta_sent-" + str(args.beta_sent) + "_latent-" + str(args.latent) + "_train_expNr_" + str(exp_nr+1) + "_valid-" + args.valid
        
        print("Training model {}".format(model_path))        
        train_x, train_y, train_truth = embeddings.load_embeddings(train_dirs[d_type], train_percs[d_type])
        
        print("checking validation data settings: {}".format(args.valid))
        if args.valid:
            valid_x, valid_y, valid_truth = get_validation_data(d_type, train_dirs)         
            model_file = train_model(model, model_path, exp_type, args.epochs, optimizer, train_x, train_y, train_truth, args, valid_x=valid_x, valid_y=valid_y, valid_truth=valid_truth)
        else:
            model_file = train_model(model, model_path, exp_type, args.epochs, optimizer, train_x, train_y, train_truth, args)
            
                    
        print("\n\nTesting ...")

        print("\n\nTesting model {}".format(model_file))
        try:        
            model.load_state_dict(torch.load(model_file, map_location=args.device))
        except RuntimeError as e:
            print("ERROR loading model for testing from {}".format(model_file))
                                            
        for data_type in ["type_I", "type_II", "type_III"]:                                                
            res = test_model(model, model_path, data_type, test_dirs, exp_type + "__test_" + data_type, args)
            res["train_on"] = d_type
            res["test_on"] = data_type
            res["run"] = exp_nr  
            
            results = results.append(res, ignore_index=True)

    return results, model.getinfo()



def run_X_experiment(N_exp, d_types, d_type, args, results):

    model = None
    for exp_nr in range(N_exp):
        (model, model_path, train_dirs, test_dirs, train_percs, output_dir) = initialize(args, d_types)
        # ADAM optimizers
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

        exp_type = "_train_" + d_type + "_expNr_" + str(exp_nr+1)
 
        print("Training model {}".format(model_path))        
        train_x, train_y, train_truth = embeddings.load_embeddings(train_dirs[d_type], train_percs[d_type])
        model_file = train_model(model, model_path, exp_type, args.epochs, optimizer, train_x, train_y, train_truth, args)
                    
        print("\n\nTesting ...")

        print("\n\nTesting model {}".format(model_file))
        try:        
            model.load_state_dict(torch.load(model_file, map_location=args.device))
        except RuntimeError as e:
            print("ERROR loading model for testing from {}".format(model_file))

        for data_type in d_types:                                                
            res = test_model(model, model_path, data_type, test_dirs, exp_type + "__test_" + data_type, args)
            res["train_on"] = d_type
            res["test_on"] = data_type
            res["run"] = exp_nr  
            
            results = results.append(res, ignore_index=True)
            
    print("\n\nmodel info -- for results: {}".format(model.getinfo()))

    return results, model.getinfo()


def run_X_test_model(model_file, d_types, args, results):
    
    (model, model_path, _train_dirs, test_dirs, _train_percs, output_dir) = initialize(args, d_types)    
                
    exp_type = get_model_info(model_file)
                
    print("\n\nTesting ...")

    print("\n\nTesting model {}".format(model_file))
    try:        
        model.load_state_dict(torch.load(model_file, map_location=args.device))
    except RuntimeError as e:
        print("ERROR loading model for testing from {}".format(model_file))

    for data_type in d_types:                                                
        new_results = test_model(model, model_path, data_type, test_dirs, exp_type + "__test_" + data_type, args)
        new_results["train_on"] = exp_type
        new_results["test_on"] = data_type
        new_results["run"] = 1  
            
        results = results.append(new_results, ignore_index=True)      

    return results, model.getinfo()



def get_validation_data(d_type, train_dirs):
    valid_x = []
    valid_y = []
    valid_truth = []
    
    '''
    for t in ["type_I", "type_II", "type_III"]:
        if t != d_type: 
            x, y, truth = embeddings.load_embeddings(train_dirs[t], 100)
            valid_x.extend(x)
            valid_y.extend(y)
            valid_truth.extend(truth)
            print("after adding {} data: {}".format(t, len(valid_x)))
    '''

    if d_type == "type_I":
        valid_x, valid_y, valid_truth = embeddings.load_embeddings(train_dirs["type_II"], 100) 
            
    return valid_x, valid_y, valid_truth 




def get_model_info(model_file):
    
    model_type = re.compile(r".*\/model_(.*)\/best_model__train_(.*)_expNr.*\.pth")
    m = model_type.search(model_file) 
    
    if m:
        return m.group(1)
    
    return "model_unmatched"


def print_model_summary(model, dim, args):
    if args.baseline_sys == "ffnn":
        print("\nModel summary:\n{}\n\n".format(summary(model,input_size=(args.batch_size,args.sent_emb_dim,args.seq_size))))
    else:
        print("\nModel summary:\n{}\n\n".format(summary(model,input_size=(args.batch_size,1,args.sent_emb_dim,args.seq_size))))


def get_latent_values(model, data, latent_values, k, device, batch_size):
    
    model.to(device)
    
    data_x, data_y, _truth = embeddings.load_embeddings(data, 1.0)  
    dataloader = data_extractor.data_loader(data_x, data_y, _truth, batch_size, True)  
    
    for _idx, (seq_x, y, truth) in enumerate(dataloader):
            
        seq = seq_x.squeeze(2).transpose(-1, -2)  # from (1, 7, 1, 768) to (1, 768, 7)
        model_output = model(seq.to(device))
        
        latent_values[k].extend(model_output["latent_vec"].tolist())



def initialize_and_load_model(args, model_file, d_types): 
    # enable cuda
    args.cuda = True
    args.device = None
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        print("cuda available...")
    else:
        args.device = torch.device("cpu")
        print("cuda not available...")
            
    (model, model_path, train_dirs, test_dirs, train_percs, output_dir) = initialize(args, d_types) 
    try:        
        model.load_state_dict(torch.load(model_file, map_location=args.device))
    except RuntimeError as e:
        print("ERROR loading model for testing from {}".format(model_file))

    return (model, model_path, train_dirs, test_dirs, train_percs, output_dir)


def train_model(model, model_path, exp_type, epochs, optimizer, train_x, train_y, train_truth, args, valid_x=None, valid_y = None, valid_truth = None):
    #print("Training data: \n\tx => {}\n\ty => {},\n\ttruth => {}".format(train_x.size(), train_y.size(), train_truth.size()))
    print("Training data: \n\tx => {}\n\ty => {},\n\ttruth => {}".format(len(train_x), len(train_y), len(train_truth)))

    nr_train_insts = int(len(train_x) * 0.8)

    if (valid_x == None) or (len(valid_x) == 0):
        # Validation data
        valid_x = train_x[nr_train_insts:]
        valid_y = train_y[nr_train_insts:]
        valid_truth = train_truth[nr_train_insts:]
    else:
        valid_x.extend(train_x[nr_train_insts:])
        valid_y.extend(train_y[nr_train_insts:])
        valid_truth.extend(train_truth[nr_train_insts:])
        

    #training data
    train_x = train_x[:nr_train_insts]
    train_y = train_y[:nr_train_insts]
    train_truth = train_truth[:nr_train_insts]
            
    trainloader = data_extractor.data_loader(train_x, train_y, train_truth, int(args.batch_size), True)
    validloader = data_extractor.data_loader(valid_x, valid_y, valid_truth, int(args.batch_size), True)

    print("\n\nTraining...")
    print("\ttraining data: {} instances ({} batches) \n\tvalidation data: {} instances  ({} batches)".format(len(train_x), len(trainloader), len(valid_x), len(validloader)))

    #(trainloader, validloader, model, exp_type, epochs, optimizer, sent_emb_dim, eval_, lr, beta, latent_dim, latent_sent_dim, device, shuffle, base, batch_size, data_type, model_dir)
    #return train.training(trainloader, validloader, model, exp_type, epochs, optimizer, args.sent_emb_dim, args.test, float(args.lr), [args.beta, args.beta_sent], args.latent, args.latent_sent_dim, args.device, args.shuffle, args.baseline, args.batch_size, args.type, model_path)
    return train.training(trainloader, validloader, model, exp_type, epochs, optimizer, args, model_path)
    
def test_model(model, model_path, data_type, test_dirs, exp_type, args):
    input_test_x = args.data_dir + "/" + data_type + "/datasets/test/sentences/"

    test_x, test_y, test_truth = embeddings.load_embeddings(test_dirs[data_type], 1.0)
    
    print("Test data for type {}: \n\tx => {}\n\ty => {},\n\ttruth => {}".format(data_type, test_x.size(), test_y.size(), test_truth.size()))

    with open(input_test_x + "/labels.json", "r", encoding="utf-8") as file:
        labels = json.load(file)
    with open(input_test_x + "/type_file.json", "r", encoding="utf-8") as file:
        type_file = json.load(file)
                                         
    print("\tDataloader...")
    testloader = data_extractor.data_loader(test_x, test_y, test_truth, 1, False)
    
    print("Testing {} data ...".format(data_type))    
    return test.testing(testloader, labels, model, args.device, exp_type)
