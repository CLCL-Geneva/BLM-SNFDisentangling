
import os
import argparse

from datetime import datetime
import logging

import pandas as pd

import utils.plots as plots
import utils.utils as utils
import run_funcs



def test_X_test(N_exp, args, d_types, make_plot=True):
    
    results = None
    
    logging.info("\n\nArgs after changing: {}\n".format(args))
       
    for d_type in d_types:        
        results, model_name = run_funcs.run_X_experiment(N_exp, d_types, d_type, args, results)

    logging.info("Results:\n")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logging.info(results)
                
    #print_avgs(results)
    #results.to_csv(output_dir + "/results_X_test.tsv", sep="\t")
    filename = make_results_filename(args, "X_test_train-" + "_".join(args.train_perc.split(" ")), model_name)
    results.to_csv(filename, sep="\t")
    logging.info("Resuts written to {}".format(filename))
    
    ## I changed the way the results are returned -- this will need processing the results to work
    X_test_results = get_results(results)   
            
    if make_plot:
        plots.plot_X_test(X_test_results,args.transformer,"data type","F1","Cross-testing analysis",args.results_dir + "/cross_testing_analysis_" + args.transformer + "_.png")  


def get_results(results):
    
    res = {}
    for i, row in results.iterrows():
        if row["train_on"] not in res:
            res[row["train_on"]] = {}
        if row["test_on"] not in res[row["train_on"]]:
            res[row["train_on"]][row["test_on"]] = []
        res[row["train_on"]][row["test_on"]].append(row["F1"])
            
    return res


def print_avgs(results):
    logging.info("\n\n F1 averages: \n\n")
    for x in results.keys():
        for y in results[x].keys():
            logging.info("train on {}, test on {} => {}".format(x, y, sum(results[x][y])/len(results[x][y])))

            
def make_results_filename(args, pref, model_name):
    #results_dir = "/home/vivi/work/Projects/BLMs/results/"
    return args.results_dir + "/results_" + pref + "_" + args.transformer + "_" + model_name + "_"+ datetime.now().strftime("%d-%b-%Y_%H:%M") + ".tsv"       


def plot_latent_layer(args, d_types, model_file):

    #'''
    (model, _model_path, train_dirs, test_dirs, _train_percs, output_dir) = run_funcs.initialize_and_load_model(args, model_file, d_types)
  
    latent_values = {}
     
    for d_type in d_types:
        latent_values[d_type] = {"train": [], "test": []}
        run_funcs.get_latent_values(model, train_dirs[d_type], latent_values[d_type], "train", args.device, args.batch_size)
        run_funcs.get_latent_values(model, test_dirs[d_type], latent_values[d_type], "test", args.device, args.batch_size)
    
    N = 300 ## random sample these many values, otherwise we can't see what is going on
    #plots.plot_latent_values(latent_values, "i", "latent values", "latent values for " + args.transformer, output_dir + "/latent_values_" + args.transformer, N)
    plots.plot_latent_values_separately(latent_values, "i", "latent values", "latent values for " + args.transformer, output_dir + "/latent_values_separately_" + args.transformer, N)
    plots.plot_latent_values_histograms(latent_values, "latent values for " + args.transformer, output_dir + "/latent_values_histograms_" + args.transformer)





def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', '--config_file', type=str,
                        help='a path to a json file containing all the information about the pretrained model (including parameters).\nIf this option is provided, there is no need to specify the parameters listed below (safer to do that, so the correct parameter values are used).')

    parser.add_argument("--cuda", action="store_true", help="Enable GPU")

    parser.add_argument("--data_dir", default="data/BLM-AgrF", help="data directory")
    parser.add_argument("--results_dir", default="results/BLM_AgrF/", help="results directory")

    parser.add_argument("--transformer", default="roberta", choices=["bert", "roberta", "electra", "flaubert"], help="transformer model to use to produce (attn/sentence) representations")
    parser.add_argument("--layer", default=-1, help="Instead of using the sentence embedding on the last layer, use the ([CLS]) embedding at the given level (0:11) to represent sentences")
    
    parser.add_argument("--attention", default=-1, help="Instead of using the sentence embedding, use the attention weights at the given level (0:11) to represent sentences")
    parser.add_argument("--attn_comb", default="sum", choices=["sum", "max"], help="how to combine the attention matrices on the requested level")
    parser.add_argument("--attn_filter", default=True, action="store_true", help="how to combine the attention matrices on the requested level")

    parser.add_argument("--sent_emb_dim", default=768, help="Dimension of sentence embedding")
    parser.add_argument("--seq_size", "-s", type=int, default=7, help="Size of the matrix for VAE initialisation.")
    
    parser.add_argument("--latent_sent_dim", default=16, help="Dimension of the latent layer for sentence embedding (used for the 2-level analysis)")
    parser.add_argument("--beta_sent", default=1.0, help="Define a beta quantity.")

    parser.add_argument("--beta", "-b", default=1.0, help="Define a beta quantity.")
    parser.add_argument("--latent", "-z", default=5, help="Latent variable dimensions.")
    
    parser.add_argument("--categorical_dim", default=1, help="Number of categories for the latent layer")
    parser.add_argument("--categ_N", default=2, help="Number of categories for the latent layer")
    parser.add_argument("--probe_latent", action="store_false", default=False, help="Change the values on the discrete portion of the latent to see what happens")
        
    parser.add_argument("--epochs", "-e", type=int, default = 120, help="Number of epochs.")
    parser.add_argument("--lr", "-l", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", "-bs", type=int, default=100, help="batch size")

    parser.add_argument("--train", "-t", action="store_true", help="Load embeddings from file and compute training.")
    parser.add_argument("--train_perc", default="1.0 1.0 1.0", help="Percentage (or absolute number) of the available training data of each type  to use")

    parser.add_argument("--shuffle", action="store_true", default=False, help="Use shuffled training.")
    parser.add_argument("--indices", default="", help="The indices (from the input sequence) to be used as training data. If all should be used, then the string should be empty")

    parser.add_argument("--valid", action="store_true", default=True, help="Use additional data from the other types for validation")

    parser.add_argument("--test", action="store_true", help="If true, do only the evaluation step.")

    parser.add_argument("--combine", action="store_true", default=True, help="When using training from multiple subsets, whether to combine them before training, or fine tune a model trained from previous parts. (tuning doesn't work well)")
        
    parser.add_argument("--baseline", "-ba", action="store_false", default=False, help="System baselines")
    parser.add_argument("--baseline_sys", default="ffnn",  choices=["ffnn", "cnn", "cnn_seq"], help="Baseline set-up")
    
    parser.add_argument("--sys", default="vae", choices=["vae", "vae_1d", "vae_1d-xseq", "dual_vae", "dual_vae_1dxseq"], help="System to use for experiments")    
    parser.add_argument("--sampling", default = "joint", choices = ["simple", "joint"], help="Sampling method for the VAE")
    
    parser.add_argument("--type", "-ty", choices=["type_I", "type_II", "type_III"], default="type_I", help="For running specific experiments, process type I, II or III. (this would be the data to train on)")
    parser.add_argument("--proc_types", default="type_I type_II type_III", help="These are the data to test on, but I use it also for cross-testing, where it is both for choosing the training and the test data")    

    parser.add_argument("--result_keys", default="run train_on test_on TP FP FN TN P R F1 Acc")

    args = parser.parse_args()
    
    utils.load_config(args)

        
    logfile = "logs/" + utils.get_run_info(args) + ".log"
    logging.basicConfig(filename=logfile, filemode='w', level=logging.DEBUG)  
    logging.info("_______\nrun_experiments.py {}\n__________\n".format(args))
        
    d_types = args.proc_types.split(" ")
    N_exp = 2
    
    
    args.valid = False
    args.probe_latent = False
    
    args.train_perc = "1.0 1.0 1.0"

    for args.transformer in ["electra"]:
        
        os.makedirs(args.results_dir, exist_ok=True)
        args.baseline = True
         
        for args.baseline_sys in ["ffnn"]:            
            test_X_test(N_exp, args, d_types, make_plot=True)
       
        args.baseline = False

        args.probe_latent = False
        args.sampling = "joint"
        for args.categorical_dim in range(1, 2):
            test_X_test(N_exp, args, d_types, make_plot=True)    

        args.sampling = "simple"
        for args.latent in [7, 9]:
            test_X_test(N_exp, args, d_types, make_plot=True)    
            


if __name__ == '__main__':
    main()