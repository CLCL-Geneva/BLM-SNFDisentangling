'''
Created on Nov 16, 2022

@author: vivi
'''


import logging

import numpy as np
import random

import pandas as pd

import torch

import utils.embeddings as embeddings
import utils.misc as misc

import vaes.sampling as sampling

import train, test  
import baselines.baseline as baseline


def initialize(args, d_types):
    
    '''
    logfile = utils.get_run_info(args) + ".log"
    logging.basicConfig(filename=logfile, filemode='w', level=logging.DEBUG)
    logging.info("_______\nrun_experiments.py {}\n__________\n".format(args))

    print("Logging run info in file {}".format(logfile))
    '''

    data_path = args.data_dir
    output_dir, emb_type = misc.make_output_dir_name(args) 
         
    logging.info("Setting seeds...")
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    # enable cuda
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        logging.info("cuda available...")
    else:
        args.device = torch.device("cpu")
        logging.info("cuda not available...")
    
    dim = args.sent_emb_dim

    logging.info("Embedding dimensions: {}".format(dim))

    train_dirs = {}
    test_dirs = {}
    train_percs = {}
    train_percs_str = args.train_perc.split(" ")
    #for i in range(len(d_types)):
    #    x = d_types[i]
    i = 0
    #for x in ["type_I", "type_II", "type_III"]:
    for x in d_types:
        train_dirs[x] = data_path + "/" + x + "/datasets/train/embeddings/" + emb_type + "/"
        test_dirs[x] = data_path + "/" + x + "/datasets/test/embeddings/" + emb_type + "/"

        if len(train_percs_str) > i:
            train_percs[x] = float(train_percs_str[i])
        else:
            train_percs[x] = 0
        i += 1
     
    answer_labels = []   
    for x in d_types:
        input_train_x = args.data_dir + "/" + x + "/datasets/train/sentences/"
        input_test_x = args.data_dir + "/" + x + "/datasets/test/sentences/"

        logging.info("Extracting training embeddings for {} to {} ...".format(x, train_dirs[x]))
        embeddings.extract_emb(input_train_x, train_dirs[x], args, dim)

        logging.info("Extracting test embeddings for {} to {} ...".format(x, test_dirs[x]))
        embeddings.extract_emb(input_test_x, test_dirs[x], args, dim)

        if answer_labels == []:
            logging.info("Extracting answer labels")
            answer_labels = misc.get_answer_labels(input_train_x)

    ## only make the results keys once, this code is rerun every time a model is initialized
    if not type(args.result_keys) is list:
        args.result_keys = args.result_keys.split(" ")
        args.result_keys.extend(answer_labels)
        logging.info("Result keys: {}".format(args.result_keys))


    model = initialize_model(args)

    model_name = model.getinfo()
    model_path = output_dir + "/model_" + model_name + "_" + misc.get_run_info(args) + "/"

    return (model, model_path, train_dirs, test_dirs, train_percs, output_dir)



def initialize_model(args):
    
    logging.info("Initializing sampler...")
    sampler = initialize_sampler(args.sampling, args.latent, args.categorical_dim, args.categ_N, args)

    if args.latent_sent_dim_cont > 0:
        logging.info("Initializing sentence sampler in case it is needed (e.g. for the DualVAE)")
        sampler_sent = initialize_sampler(args.sent_sampling, args.latent_sent_dim_cont, args.latent_sent_categ, args.latent_sent_categ_N, args)

    logging.info("Model initialization...")    
    if args.baseline:
        model = model_baseline(args.sent_emb_dim, args)
    else:
        model = model_vae([sampler, sampler_sent], args)

    return model


def save_model(model, model_file):
    print("Saving the model to file {}".format(model_file))
    torch.save(model.state_dict(), model_file)


def load_model(model, model_file, args):
    checkpoint = torch.load(model_file, map_location=args.device)
    model.load_state_dict(checkpoint)


def initialize_and_load_model(args, model_file, d_types): 
    # enable cuda
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        logging.info("cuda available...")
    else:
        args.device = torch.device("cpu")
        logging.info("cuda not available...")
            
    (model, model_path, train_dirs, test_dirs, train_percs, output_dir) = initialize(args, d_types) 
    
    try:
        load_model(model, model_file, args)
    except RuntimeError as e:
        logging.info("ERROR loading model for testing from {}".format(model_file))

    return (model, model_path, train_dirs, test_dirs, train_percs, output_dir)

    

def initialize_sampler(sampling_type, latent_dim, categorical_dim, categ_N, args):

    logging.info("Initializing sampler...")
    if sampling_type == "joint":
        sampler = sampling.JointSampling(latent_dim, categorical_dim, categ_N, args.device)
    else:
        sampler = sampling.simpleSampling(latent_dim)
    
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
    logging.info("\nInitializing *{}* model ...\n".format(args.sys))

    if args.sys == "two_level_vae":
        from vaes.VAE import VariationalAutoencoder_2level as VariationalAutoencoder
        model = VariationalAutoencoder(args.sent_emb_dim, args.seq_size, sampler_list, device=args.device)

    elif args.sys == "vae_1d_2level":
        from vaes.VAE import VariationalAutoencoder_1D_2level as VariationalAutoencoder
        model = VariationalAutoencoder(args.sent_emb_dim, args.seq_size, sampler_list, device=args.device)

    elif args.sys == "two_level_vae_subnets":
        from vaes.VAE import VariationalAutoencoder_2level_subnets as VariationalAutoencoder
        model = VariationalAutoencoder(args.sent_emb_dim, args.seq_size, sampler_list, device=args.device)

    else:
            from vaes.VAE import VariationalAutoencoder as VariationalAutoencoder
            model = VariationalAutoencoder(args.sent_emb_dim, args.seq_size, sampler_list[0], device=args.device)

    return model



def make_run(exp_nr, test_on, train_on, args, results, all_predictions):
    (model, model_path, train_dirs, test_dirs, train_percs, _output_dir) = initialize(args, test_on)  

    dataloaders = {}
    model_file, exp_type, dataloaders["train"] = train.train_run(exp_nr, model, model_path, train_dirs, train_percs, train_on, args)
    dataloaders["test"], results = test.test_run(exp_nr, model, model_file, model_path, exp_type, train_on, test_dirs, results, all_predictions, args)

    return model, model_file, dataloaders, results


def run_experiment(N_exp, test_on, train_on, args, results):

    logging.info("N_EXP {}  train on: {}  test on: {}".format(N_exp, train_on, test_on))
    args.type = train_on

    all_predictions = {}

    model = None
    model_file = None
    
    for exp_nr in range(N_exp):
        if results is None:
            results = pd.DataFrame(columns=args.result_keys.split(" "))
        model, model_file, dataloaders, results = make_run(exp_nr, test_on, train_on, args, results, all_predictions)
        
    logging.info("\n\n train on {} / test on {} => model info -- for results: {}".format(train_on, test_on, model.getinfo()))
    return all_predictions, model, results, model.getinfo(), model_file, dataloaders



def run_X_test_model(model_file, train_on, test_on_list, args, results):
    
    all_predictions = {}
    exp_type = misc.get_model_info(model_file)
    exp_nr = 0

    (model, model_path, _train_dirs, test_dirs, _train_percs, output_dir) = initialize(args, test_on_list)                    
    test.test_run(exp_nr, model, model_file, model_path, exp_type, train_on, test_dirs, test_on_list, results, all_predictions, args)

    return results, all_predictions, model.getinfo()

