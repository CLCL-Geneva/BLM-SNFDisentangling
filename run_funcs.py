
import argparse

import os
import glob
import re
import json

import logging

import pickle

import numpy as np
import random

import pandas as pd

import torch
from torchinfo import summary



import utils.embeddings as embeddings
import utils.data_extractor as data_extractor
import utils.utils as utils

import vaes.sampling as sampling

import train, test  
import baselines.baseline as baseline


def initialize(args, d_types):
        
    if args.attention >=0:
        emb_type = args.transformer + "_" + str(args.attention) + "_" + args.attn_comb
    else:
        emb_type = args.transformer + "_sentembs" 
        if args.layer >=0: 
            emb_type += str(args.layer)

    if args.attn_filter and args.attention >= 0:
        emb_type += "_filtered"

    data_path = args.data_dir
    output_dir = args.data_dir + "/" + args.type + "/output/" + emb_type +  "/"
         
    # setting the seed for reproducibility
    logging.info("Setting seeds...")
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    # enable cuda
    args.cuda = True
    args.device = None
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        logging.info("cuda available...")
    else:
        args.device = torch.device("cpu")
        logging.info("cuda not available...")
    
    if args.attention >= 0:
        #'''
        if args.transformer == "bert":
            dim = 29
        else:
            dim = 26
        '''
        dim = 0
        for data_type in ["type_I", "type_II", "type_III"]:
        #for data_type in ["type_I"]:
            input_train_x = args.data_dir + "/" + data_type + "/datasets/train/sentences/"
            input_test_x = args.data_dir + "/" + data_type + "/datasets/test/sentences/"
            dim = max(dim, embeddings.getMaxLength([input_train_x, input_test_x], args))
        #'''
        args.sent_emb_dim = dim * dim
    else:
        dim = args.sent_emb_dim

    logging.info("Embedding dimensions: {}".format(dim))

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
     
    answer_labels = []   
    for data_type in d_types:
        
        input_train_x = args.data_dir + "/" + data_type + "/datasets/train/sentences/"
        input_test_x = args.data_dir + "/" + data_type + "/datasets/test/sentences/"

        logging.info("Extracting training embeddings for {} to {} ...".format(data_type, train_dirs[data_type]))
        embeddings.extract_emb(input_train_x, train_dirs[data_type], args, dim)
        logging.info("Extracting test embeddings for {} to {} ...".format(data_type, test_dirs[data_type]))
        embeddings.extract_emb(input_test_x, test_dirs[data_type], args, dim)

        if answer_labels == []:        
            logging.info("Extracting answer labels")
            answer_labels = utils.get_answer_labels(input_train_x)

    ## only make the results keys once, this code is rerun every time a model is initialized
    if not type(args.result_keys) is list:
        args.result_keys = args.result_keys.split(" ")
        args.result_keys.extend(answer_labels)
        logging.info("Result keys: {}".format(args.result_keys))

    logging.info("Initializing sampler...")
    sampler = initialize_sampler(args.latent, args)

    if args.indices != "":
        args.seq_size = len(args.indices.split(" "))
    
    if args.latent_sent_dim > 0:
        logging.info("Initializing sentence sampler in case it is needed (e.g. for the DualVAE)")
        sampler_sent = initialize_sampler(args.latent_sent_dim, args)

    # VAE initialization
    logging.info("Model initialization...")
    
    if args.baseline:
        model = model_baseline(dim, args)
    else:
        model = model_vae([sampler, sampler_sent], args)
    
    #print_model_summary(model, dim, args)

    model_name = model.getinfo()
    model_path = output_dir + "/model_" + model_name + "_" + utils.get_run_info(args) + "/"

    return (model, model_path, train_dirs, test_dirs, train_percs, output_dir)
    

def initialize_sampler(latent_dim, args):

    logging.info("Initializing sampler...")
    sampler = None
    if args.sampling == "gamma":
        sampler = sampling.gammaSampling(latent_dim)
    elif args.sampling == "gumbel":
        sampler = sampling.JointSampling(args.latent, args.categorical_dim, args.categ_N, args.device, is_continuous=False) 
        args.latent=0
    elif args.sampling == "joint":   
        sampler = sampling.JointSampling(args.latent, args.categorical_dim, args.categ_N, args.device)
    else:  ## by default, the simple (continuous) sampling
        sampler = sampling.simpleSampling(args.latent)
    
    return sampler


def model_baseline(dim, args):
    logging.info("\nInitialising baseline model ...\n")
    if "ffnn" in args.baseline_sys.lower():
        model = baseline.BaselineFFNN(dim, int(args.seq_size))
    elif "cnn_seq" in args.baseline_sys.lower():
        model = baseline.BaselineCNN_1DxSeq(dim, int(args.seq_size))
    elif "cnn" in args.baseline_sys.lower():
        model = baseline.BaselineCNN(dim, int(args.seq_size))
    else:
        logging.info("Unknown model. Defaulting to FFNN")
        model = baseline.BaselineFFNN(dim, int(args.seq_size))
        
    return model

    
def model_vae(sampler_list, args):
    logging.info("\nInitializing VAE model ...\n")

    if args.sys == "vae":
        import vaes.VAE as VAE
    elif args.sys == "vae_1d":
        import vaes.VAE_1D as VAE
    elif args.sys == "vae_1dxseq":
        import vaes.VAE_1DxSeq as VAE
    elif args.sys == "dual_vae":
        import vaes.dual_VAE as VAE
    elif args.sys == "dual_vae_1dxseq":
        import vaes.dual_VAE_1DxSeq as VAE
    else:
        logging.info("Unknown system. Defaulting to VAE")
        import vaes.VAE as VAE

    return VAE.VariationalAutoencoder(args.sent_emb_dim, args.latent, args.seq_size, sampler_list[0])

    #import vaes.two_level_dual_VAE_attn_2D as VAE    
    #return VAE.VariationalAutoencoder(args.sent_emb_dim, args.latent_sent_dim, args.latent, args.seq_size, sampler_list)


def get_complete_representations(args):
    
    transformers = ["bert", "roberta", "flaubert"]
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
        
        logging.info("Training model {}".format(model_path))        
        train_x, train_y, train_truth = embeddings.load_embeddings(train_dirs[d_type], train_percs[d_type], args.indices)
        
        logging.info("checking validation data settings: {}".format(args.valid))
        if args.valid:
            valid_x, valid_y, valid_truth = get_validation_data(d_type, train_dirs)         
            model_file = train_model(model, model_path, exp_type, args.epochs, optimizer, train_x, train_y, train_truth, args, valid_x=valid_x, valid_y=valid_y, valid_truth=valid_truth)
        else:
            model_file = train_model(model, model_path, exp_type, args.epochs, optimizer, train_x, train_y, train_truth, args)
            
                    
        logging.info("\n\nTesting ...")

        logging.info("\n\nTesting model {}".format(model_file))
        try:        
            model.load_state_dict(torch.load(model_file, map_location=args.device))
        except RuntimeError as e:
            logging.info("ERROR loading model for testing from {}".format(model_file))
                                            
        for data_type in ["type_I", "type_II", "type_III"]:                                                
            res = test_model(model, model_path, data_type, test_dirs, exp_type + "__test_" + data_type, args)
            res["train_on"] = d_type
            res["test_on"] = data_type
            res["run"] = exp_nr  
            
            results = results.append(res, ignore_index=True)
            
        #logging.info("visualizing filters ...")
        #plots.plot_filters(model)

    return results, model.getinfo()



def run_X_experiment(N_exp, d_types, d_type, args, results):

    model = None
    for exp_nr in range(N_exp):
        (model, model_path, train_dirs, test_dirs, train_percs, output_dir) = initialize(args, d_types)
        
        if results is None:
            results = pd.DataFrame(columns=args.result_keys)
        
        # ADAM optimizers
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

        exp_type = "_train_" + d_type + "_expNr_" + str(exp_nr+1)

        if d_type != "type_I":
            #n = 40
            #epochs = min(100, int(n + 10 * (1-train_percs[d_type]) * (100-n)/9))   ## because it saturates faster because of the larger amount of data for type_II and III
            if train_percs[d_type] > 1:
                args.epochs = 120
            else:
                args.epochs = 50
        else:
            args.epochs = 120
 
        logging.info("Training model {}".format(model_path))        
        train_x, train_y, train_truth = embeddings.load_embeddings(train_dirs[d_type], train_percs[d_type], args.indices)

        logging.info("checking validation data settings: {}".format(args.valid))
        if args.valid:
            valid_x, valid_y, valid_truth = get_validation_data(d_type, train_dirs)         
            model_file = train_model(model, model_path, exp_type, args.epochs, optimizer, train_x, train_y, train_truth, args, valid_x=valid_x, valid_y=valid_y, valid_truth=valid_truth)
        else:
            model_file = train_model(model, model_path, exp_type, args.epochs, optimizer, train_x, train_y, train_truth, args)
                    
        logging.info("\n\nTesting ...")

        logging.info("\n\nTesting model {}".format(model_file))
        try:        
            model.load_state_dict(torch.load(model_file, map_location=args.device))
        except RuntimeError as e:
            logging.info("ERROR loading model for testing from {}".format(model_file))

        for data_type in d_types:                                                
            res = test_model(model, model_path, data_type, test_dirs, exp_type + "__test_" + data_type, args)
            res["train_on"] = d_type
            res["test_on"] = data_type
            res["run"] = exp_nr  
            
            results = results.append(res, ignore_index=True)

        #logging.info("visualizing filters ...")
        #plots.plot_filters(model)

            
    logging.info("\n\nmodel info -- for results: {}".format(model.getinfo()))

    return results, model.getinfo()


def run_X_test_model(model_file, d_types, args, results):
    
    (model, model_path, _train_dirs, test_dirs, _train_percs, output_dir) = initialize(args, d_types)    
                
    exp_type = get_model_info(model_file)
                
    logging.info("\n\nTesting ...")

    logging.info("\n\nTesting model {}".format(model_file))
    try:        
        model.load_state_dict(torch.load(model_file, map_location=args.device))
    except RuntimeError as e:
        logging.info("ERROR loading model for testing from {}".format(model_file))

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
            logging.info("after adding {} data: {}".format(t, len(valid_x)))
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
    if args.baseline and (args.baseline_sys == "ffnn"):
        logging.info("\nModel summary:\n{}\n\n".format(summary(model,input_size=(args.batch_size,args.sent_emb_dim,args.seq_size))))
    else:
        logging.info("\nModel summary:\n{}\n\n".format(summary(model,input_size=(args.batch_size,1,args.sent_emb_dim,args.seq_size))))


def get_latent_values(model, data, latent_values, k, device, batch_size):
    
    model.to(device)
    
    data_x, data_y, _truth = embeddings.load_embeddings(data, 1.0)  
    dataloader = data_extractor.data_loader([data_x, data_y, _truth], batch_size, True)  
    
    for _idx, (seq_x, y, truth) in enumerate(dataloader):
            
        #seq = seq_x.squeeze(2).transpose(-1, -2)  # from (1, 7, 1, 768) to (1, 768, 7)
        seq = seq_x
        model_output = model(seq.to(device))
        
        latent_values[k].extend(model_output["latent_vec"].tolist())



def initialize_and_load_model(args, model_file, d_types): 
    # enable cuda
    args.cuda = True
    args.device = None
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        logging.info("cuda available...")
    else:
        args.device = torch.device("cpu")
        logging.info("cuda not available...")
            
    (model, model_path, train_dirs, test_dirs, train_percs, output_dir) = initialize(args, d_types) 
    try:        
        model.load_state_dict(torch.load(model_file, map_location=args.device))
    except RuntimeError as e:
        logging.info("ERROR loading model for testing from {}".format(model_file))

    return (model, model_path, train_dirs, test_dirs, train_percs, output_dir)



def train_model(model, model_path, exp_type, epochs, optimizer, train_x, train_y, train_truth, args, valid_x=None, valid_y = None, valid_truth = None):
    logging.info("Training data: \n\tx => {}\n\ty => {},\n\ttruth => {}".format(len(train_x), len(train_y), len(train_truth)))

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
            
    trainloader = data_extractor.data_loader([train_x, train_y, train_truth], int(args.batch_size), True)
    validloader = data_extractor.data_loader([valid_x, valid_y, valid_truth], int(args.batch_size), True)

    logging.info("\n\nTraining...")
    logging.info("\ttraining data: {} instances ({} batches) \n\tvalidation data: {} instances  ({} batches)".format(len(train_x), len(trainloader), len(valid_x), len(validloader)))

    return train.training(trainloader, validloader, model, exp_type, epochs, optimizer, [args.beta], args, model_path)
    
    
def test_model(model, model_path, data_type, test_dirs, exp_type, args):
    input_test_x = args.data_dir + "/" + data_type + "/datasets/test/sentences/"

    test_x, test_y, test_truth = embeddings.load_embeddings(test_dirs[data_type], 1.0, args.indices)
    
    logging.info("Test data for type {}: \n\tx => {}\n\ty => {},\n\ttruth => {}".format(data_type, test_x.size(), test_y.size(), test_truth.size()))

    with open(input_test_x + "/labels.json", "r", encoding="utf-8") as file:
        labels = json.load(file)
                                         
    logging.info("\tDataloader...")
    testloader = data_extractor.data_loader([test_x, test_y, test_truth], 1, False)
    
    logging.info("Testing {} data ...".format(data_type)) 
    logging.info("result keys: {}".format(args.result_keys))   
    res = test.testing(testloader, labels, model, args.device, args.result_keys)
                
    if args.probe_latent:
        N = args.categorical_dim * args.categ_N
        format_str = "#0" + str(N+2) + "b"  ## to represent the mask on the correct number of bits
        for x in range(1, 2**N):
            mask = format(x, format_str)[2:]
            probe_res = test.testing(testloader, labels, model, args.device, args.result_keys, mask = [int(x) for x in mask])
            for k in probe_res:
                res[k + "_probe_mask_" + mask] = probe_res[k]

    return res

    
