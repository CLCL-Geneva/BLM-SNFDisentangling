'''
Created on Nov 16, 2022

@author: vivi
'''

import sys
import os
import argparse
import json

import logging

import utils.misc as misc
import run_funcs


def test_X_test(N_exp, args, train_on, d_types, make_plot=False):
    
    results = None
    model_file = None
    model_name = None
    all_predictions = {}
    
    logging.info("\n\nArgs after changing: {}\n".format(args))

    for d_type in train_on:
        if os.path.isdir(args.data_dir + "/" + d_type):
            args.type = d_type
            logging.info("\t\trunning {} experiments training on {} and testing on {}".format(N_exp, d_type, d_types))
            predictions, _model, results, model_name, _model_file, _dataloaders = run_funcs.run_experiment(N_exp, d_types, d_type, args, results)
            all_predictions["train_on " + d_type] = predictions
            
    filename = misc.make_results_filename(args, "X_test_train-" + "_".join(args.train_perc.split(" ")), model_name)
    results.to_csv(filename, sep="\t")
    logging.info("Results written to {}".format(filename))
    
    filename = misc.make_results_filename(args, "predictions___X_test_train-" + "_".join(args.train_perc.split(" ")), model_name) + ".json"
    with open(filename, 'w') as file:
        file.write(json.dumps(all_predictions, indent=4, sort_keys=True)) 
    logging.info("Predictions written to {}".format(filename))


def run_X_exp(args):
    
    os.makedirs(args.results_dir, exist_ok=True)

    logfile = args.results_dir + "/" + misc.get_run_info(args) + ".log"
    print("Logging run info in file {}".format(logfile))
    
    logging.basicConfig(filename=logfile, filemode='w', level=logging.DEBUG)  
    logging.info("_______\nrun_experiments.py {}\n__________\n".format(args))
    logging.info("\n\nRunning args: {}\n".format(args))

    logging.info("Logging run info in file {}".format(logfile))
    logging.info("\n_____________________________\nProcessing data:\n\t transformer = {}\n\t data_dir = {}\n\t results_dir = {}\n________________________________\n".format(args.transformer, args.data_dir, args.results_dir))
    logging.info("Training percentages: {}".format(args.train_perc))

    d_types = args.proc_types.split(" ")    
    test_X_test(args.N_exp, args, d_types, d_types, make_plot=False)




def run_exp(args):
    
    os.makedirs(args.results_dir, exist_ok=True)

    logfile = args.results_dir + "/" + misc.get_run_info(args) + ".log"    
    print("Logging run info in file {}".format(logfile))

    logging.basicConfig(filename=logfile, filemode='w', level=logging.DEBUG)  
    logging.info("_______\nrun_experiments.py {}\n__________\n".format(args))
    logging.info("\n\nRunning args: {}\n".format(args))
    logging.info("Logging run info in file {}".format(logfile))    
    logging.info("\n_____________________________\nProcessing data:\n\t transformer = {}\n\t data_dir = {}\n\t results_dir = {}\n________________________________\n".format(args.transformer, args.data_dir, args.results_dir))
    logging.info("Training percentages: {}".format(args.train_perc))

    d_types = args.proc_types.split(" ")
    test_X_test(args.N_exp, args, [args.type], d_types, make_plot=False)



def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--config_file', type=str, default="",
                        help='a path to a json file containing all the information about the pretrained model (including parameters).\nIf this option is provided, there is no need to specify the parameters listed below (safer to do that, so the correct parameter values are used).')

    parser.add_argument("--cuda", default=True, action="store_true", help="Enable GPU")

    parser.add_argument("--data_dir", default="data/BLMagr4dev/", help="data directory")
    parser.add_argument("--results_dir", default="results/BLMagr4dev", help="results directory")

    parser.add_argument("--transformer", default="electra", choices=["bert", "roberta", "electra", "flaubert"], help="transformer model to use to produce (attn/sentence) representations")

    parser.add_argument("--sent_emb_dim", default=768, help="Dimension of sentence embedding")
    parser.add_argument("--seq_size", "-s", type=int, default=7, help="Size of the matrix for VAE initialisation.")

    parser.add_argument("--sent_sampling", default = "joint", choices = ["simple", "joint"], help="Sampling method for the sentence VAE")    
    parser.add_argument("--latent_sent_dim_cont", default=5, help="Dimension of the latent layer for sentence embedding (used for the 2-level analysis)")
    parser.add_argument("--latent_sent_categ", default=0, help="Number of categories for the latent layer")
    parser.add_argument("--latent_sent_categ_N", default=3, help="Number of possible values for the categories for the latent layer")
    parser.add_argument("--beta_sent", default=1.0, help="Define a beta quantity.")

    parser.add_argument("--sampling", default = "joint", choices = ["simple", "joint"], help="Sampling method for the VAE")
    parser.add_argument("--latent", "-z", default=5, help="Latent variable dimensions.")    
    parser.add_argument("--categorical_dim", default=0, help="Number of categories for the latent layer")
    parser.add_argument("--categ_N", default=3, help="Number of categories for the latent layer")
    parser.add_argument("--beta", "-b", default=1.0, help="Define a beta quantity.")

    parser.add_argument("--N_exp", default=1, help="Number of experiments to run for each configuration")
    parser.add_argument("--Xtest", action="store_true", help="If specified, perform cross-testing of all configurations")

    parser.add_argument("--epochs", "-e", type=int, default = 1, help="Number of epochs.")
    parser.add_argument("--lr", "-l", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", "-bs", type=int, default=100, help="batch size")

    parser.add_argument("--train", "-t", action="store_true", help="Load embeddings from file and compute training.")
    parser.add_argument("--train_perc", default="1.0 1.0 1.0", help="Percentage (or absolute number) of the available training data of each type  to use")
    parser.add_argument("--indices", default="", help="Subset of input sequence to use as training data")

    parser.add_argument("--valid", default="", help="Use additional data from the other types for validation. This is given as a string, and datasets that should be paired for the purpose of obtaining validation data should be separated by dash while the pairing should be separated by space, like this: 'type_I-type_II type_II-type_III' ")
    parser.add_argument("--valid_perc", default=0.2, help="How much of the training data to be used for validation")

    parser.add_argument("--test", action="store_true", help="If true, do only the evaluation step.")

    parser.add_argument("--baseline", "-ba", action="store_true", default=False, help="System baselines")
    parser.add_argument("--baseline_sys", default="ffnn",  choices=["ffnn", "cnn", "cnn_seq"], help="Baseline set-up")
    
    parser.add_argument("--sys", default="vae", choices=["vae", "vae_1d_2level", "two_level_vae", "two_level_vae_subnets"], help="System to use for experiments")
    
    parser.add_argument("--type", "-ty", choices=["type_I", "type_II", "type_III"], default="type_I", help="For running specific experiments, process type I, II or III. (this would be the data to train on)")
    parser.add_argument("--proc_types", default="type_I type_II type_III", help="These are the directories with the data to test on, but I use it also for cross-testing, where it is both for choosing the training and the test data")

    parser.add_argument("--result_keys", default="run train_on test_on TP FP FN TN P R F1 Acc")

    args = parser.parse_args()

    if args.config_file != "":
        misc.load_config(args.config_file, args)

    if args.latent_sent_categ == 0:
        args.sent_sampling = "simple"  
    else:
        args.sent_sampling = "joint"
        
    if args.categorical_dim == 0:
        args.sampling = "simple"  
    else:
        args.sampling = "joint"
    
        
    if args.Xtest:
        run_X_exp(args)
    else:
        run_exp(args)


if __name__ == '__main__':
    main()
