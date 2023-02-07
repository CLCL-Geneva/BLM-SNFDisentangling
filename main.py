'''
Created on Nov 16, 2022

@author: vivi
'''

from datetime import datetime
import argparse
import os

import utils.plots as plots
import utils.run_funcs as run_funcs

import train
import test

import pandas as pd
results_keys = ["run", "train_on", "test_on", "TP", "FP", "FN", "TN", "P", "R", "F1", "Acc", "agreement_error", "coordination", "mis_num", "N1_alter", "N2_alter"]
results_dir = "results/"



def test_X_test(N_exp, args, d_types):
    
    results = pd.DataFrame(columns=results_keys)
    
    print("\n\nArgs after changing: {}\n".format(args))
       
    for d_type in d_types:        
        results, model_name = run_funcs.run_X_experiment(N_exp, d_types, d_type, args, results)

    print("Results:\n")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(results)
                
    #print_avgs(results)
    #results.to_csv(output_dir + "/results_X_test.tsv", sep="\t")
    filename = make_results_filename("X_test_train-" + "_".join(args.train_perc.split(" ")), model_name)
    results.to_csv(filename, sep="\t")
    print("Resuts written to {}".format(filename))
    
    ## I changed the way the results are returned -- this will need processing the results to work
    X_test_results = get_results(results)   
            
    plots.plot_X_test(X_test_results,args.transformer,"data type","F1","Cross-testing analysis",results_dir + "/cross_testing_analysis_" + args.transformer + "_.png")  


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
    print("\n\n F1 averages: \n\n")
    for x in results.keys():
        for y in results[x].keys():
            print("train on {}, test on {} => {}".format(x, y, sum(results[x][y])/len(results[x][y])))
            
def make_results_filename(pref, model_name):
    os.makedirs(results_dir, exist_ok=True)
    return results_dir + "/results_" + pref + "_" + model_name + "_"+ datetime.now().strftime("%d-%b-%Y_%H:%M") + ".tsv"       


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data/BLM-AgrF/", help="data directory")

    parser.add_argument("--transformer", default="bert", choices=["bert", "flaubert"], help="transformer model to use to produce (attn/sentence) representations")
    
    parser.add_argument("--sent_emb_dim", default=768, help="Dimension of sentence embedding")
        
    parser.add_argument("--epochs", "-e", type=int, default = 100, help="Number of epochs.")
    parser.add_argument("--lr", "-l", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", "-bs", type=int, default=100, help="batch size")
    parser.add_argument("--seq_size", "-s", type=int, default=7, help="Size of the matrix for VAE initialisation.")
    parser.add_argument("--model", "-v", help="path to the saved model parameters.")
    parser.add_argument("--train", "-t", action="store_true", help="Load embeddings from file and compute training.")
    parser.add_argument("--test", action="store_false", help="If true, do only the evaluation step.")
    parser.add_argument("--cuda", action="store_true", help="Enable GPU")
    
    #parser.add_argument("--train_perc", default="1.0 1.0 1.0", help="Percentage of the available training data of each type  to use")
    parser.add_argument("--train_perc", default="2073 2073 2073", help="Percentage of the available training data of each type  to use")
    
    parser.add_argument("--baseline_sys", choices=["ffnn", "cnn"], default="ffnn", help="Baseline set-up")
    
    parser.add_argument("--type", "-ty", choices=["type_I", "type_II", "type_III"], default="type_I", help="Process type I, II or III.")

    parser.add_argument("--n_exps", default=1, help="Number of experiments to run")

    args = parser.parse_args()
    
    print("Running tests with args: {}".format(args))

    args.data_dir = os.path.abspath(args.data_dir)
    
    args.attention = -1
    
    d_types = ["type_I", "type_II", "type_III"]
    test_X_test(args.n_exps, args, d_types)



if __name__ == '__main__':
    main()