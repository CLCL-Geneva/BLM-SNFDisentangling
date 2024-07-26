

import argparse

from datetime import datetime

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

import pickle

import pandas as pd
import numpy as np
import random
import torch

from datetime import datetime

from sklearn.preprocessing import normalize

import utils.embeddings as embeddings
import utils.data_extractor as data_extractor
import utils.misc as misc

import vaes.sampling as sampling

import train, test, utils_sr

from VAE import VariationalAutoencoder


max_instances = 4000

def load_embeddings(input_file):
    with open(input_file, "rb") as f:
        sentence_embeddings = pickle.load(f)
    return sentence_embeddings        
    #return normalize_embs(sentence_embeddings)
            
def normalize_embs(embeddings):    
    normalized_embs = normalize(np.array(list(embeddings.values())), axis=1)
    return dict(zip(embeddings.keys(), normalized_embs))


def load_data(input_file, args):
    return pd.read_csv(input_file, sep = "\t")


def encode_sentence(sentence, model, tokenizer):
    print("Encoding sentence: {}".format(sentence))
    
    token_ids = torch.tensor([tokenizer.encode(sentence)])        
    last_layer = model(token_ids)[0]
    representation = last_layer[:, 0, :].cpu().detach().numpy()
    
    return np.asarray(representation, dtype='float32').squeeze()


def extract_emb(sequences_file, outfile, args):
    vecs = {}

    (tokenizer, model) = embeddings.load_transformer(args)
    if os.path.isfile(outfile):
        print("Skipping file {} -- already processed ({})".format(sequences_file, outfile))
    else:
        sequences = pd.read_csv(sequences_file, sep="\t", index_col = 0)
        for i, row in sequences.iterrows():
            for c in sequences.columns:
                vecs[row[c]] = encode_sentence(row[c], model, tokenizer)
        # save the embeddings

        print("Save embeddings to {} ...".format(outfile))
        with open(outfile, 'wb') as handle:
            pickle.dump(vecs, handle, protocol=pickle.HIGHEST_PROTOCOL)        
        
        ## clean up    
        del vecs
        torch.cuda.empty_cache()
    
    del tokenizer   
    del model



def initialize(args):

    output_dir, embs_file, sent_file = initialize_setup(args)
    model, model_dir = initialize_model_and_sampler(output_dir, args)
          
    return (model, model_dir, output_dir, embs_file, sent_file)



def initialize_setup(args):
    # enable cuda
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        print("cuda available...")
    else:
        args.device = torch.device("cpu")
        print("cuda not available...")
        args.cuda = False

    # setting the seed for reproducibility
    print("Setting seeds...")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    emb_type = args.transformer + "_sentembs" 

    output_dir = args.data_dir + "/output/" + emb_type +  "/"

    os.makedirs(args.data_dir + "/embeddings/" + emb_type, exist_ok = True)
    embs_file = args.data_dir + "/embeddings/" + emb_type + "/embeddings.pkl"
    sent_file = args.data_dir + "/sequences/sequences.csv"

    if args.extract:
        print("Extracting training embeddings for {} to {} ...".format(args.data_dir, embs_file))
        extract_emb(sent_file, embs_file, args)
                   
    return (output_dir, embs_file, sent_file)




def initialize_model_and_sampler(output_dir, args):

    timestamp = datetime.now().strftime("%d-%b-%Y_%H:%M")     
    
    #sampling method initialization
    print("Initializing sampler...")
    sampler = None
    if args.sampling == "joint":
        sampler = sampling.JointSampling(args.latent, args.categorical_dim, args.categ_N, args.device)
    else:  
        sampler = sampling.simpleSampling(args.latent)

    model = VariationalAutoencoder(args.sent_emb_dim, sampler)

    model_name = model.getinfo()        
    model_dir = output_dir + "/model_" + model_name + "_" + args.sampling + "-sampling_" + str(args.train_perc) + "-train_perc_" + str(args.epochs) + "-epochs_" + str(args.lr) + "-lr_" + timestamp + "/"
    
    return model, model_dir



def make_train_test(embs, sent_seqs, patterns, args):
    max_instances = 4000
    data = make_data(sent_seqs, embs, patterns, args)
    if len(data[0]) > max_instances:
        print("downsampling the data to {} instances".format(max_instances))
        data, _rest = split_data(data, max_instances/len(data[0]))
    else:
        max_instances = len(data[0])

    train_ratio = 0.8
    print("Splitting {} instances into train/test (with ratio {}:{})".format(len(data[0]), int(train_ratio*max_instances), int((1-train_ratio)*max_instances)))
    return split_data(data, train_ratio, args.train_test_split)


def split_data(data, ratio, train_test_split = "random"):
    
    (x, x_pos, x_negs, x_p, x_negs_p) = data
    print("splitting {} data instances into train:test = {}:{}".format(len(x), ratio, 1-ratio))
    inds = select_indices(x_p, ratio, train_test_split)
    zipped_lists = tuple(zip(inds, x, x_pos, x_negs, x_p, x_negs_p))

    return select_data(zipped_lists, 1), select_data(zipped_lists, 0)


def select_indices(pos_patterns, ratio, train_test_split):
    N = len(pos_patterns)
    patts = set(pos_patterns)
    n = len(patts)

    ## try to sample relatively uniformly w.r.t each patterns
    n_p = int(N * ratio/n) + 1

    if train_test_split == "sep":
        patts = random.sample(patts, int(n*ratio))

    inds = [0] * N
    for p in patts:
        p_inds = [i for i, x in enumerate(pos_patterns) if x == p]
        if len(p_inds) < n_p:
            print("number of instances {} lower than the wanted sample {} for pattern *{}*".format(len(p_inds), n_p, p))
        else:
            p_inds = random.sample(p_inds, n_p)
        for i in p_inds:
            inds[i] = 1

    return inds

def select_data(zipped_lists, val):
    data = [(e, e_pos, e_negs, e_p, e_negs_p) for i, e, e_pos, e_negs, e_p, e_negs_p in zipped_lists if i == val]
    x, x_pos, x_negs, x_p, x_negs_p = zip(*data)    
    return (x, x_pos, x_negs, x_p, x_negs_p)


    
def make_data(sent_seqs, embs, patterns, args):
    
    N_negs = args.nr_negs
    data = {}
    
    x = []       ## input
    x_pos = []   ## output (sentence that has the same structure)
    x_negs = []  ## negatives (N_negs sentences that have a different structure)
    x_p = []     ## pattern of input sentence
    x_negs_p = [] ## patterns of negative sentences

    for p in patterns:
        data[p] = list(sent_seqs[p])
    
    for p in data:
        for s_i in data[p]:
            for s_j in data[p]:
                x.append(embs[s_i])
                x_p.append(p)
                x_pos.append(embs[s_j])
                
                negs, negs_p = utils_sr.choose_negatives(embs, p, data, N_negs, args.negatives) 
                x_negs.append(negs)
                x_negs_p.append(negs_p)
    
    print("Processed {} instances".format(len(x)))
    
    return (x, x_pos, x_negs, x_p, x_negs_p)
            
            

def train_model(model, model_dir, epochs, optimizer, train_data, args):
    if args.train_perc < 1.0:
        print("Taking {} of the available training data ({} instances) for actual training (because we have too much)".format(args.train_perc, len(train_data)))
        train_data, _rest = split_data(train_data, args.train_perc)

    train_valid_ratio = 0.8
    print("Splitting selected training data into train and dev ({} ratio)".format(train_valid_ratio))
    train_split, valid_split = split_data(train_data, train_valid_ratio)
        
    trainloader = data_extractor.data_loader(list(train_split), int(args.batch_size), True)
    validloader = data_extractor.data_loader(list(valid_split), int(args.batch_size), True)

    return trainloader, train.training(trainloader, validloader, model, epochs, optimizer, args.sent_emb_dim, float(args.lr), args.beta_sent, args.latent, args.device, args.batch_size, model_dir, args)


def test_model(model, test_data, args):
    testloader = data_extractor.data_loader(list(test_data), int(args.batch_size), False)    
    res = test.testing(testloader, model, args.device)
    return res, testloader


def write_results(model, res, args):

    results = pd.DataFrame(res)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("results:\n{}".format(results))

    os.makedirs(args.results_dir, exist_ok=True)
    data_name = os.path.basename(os.path.normpath(args.data_dir))
    filename = args.results_dir + "/results_" + data_name + "_" + args.transformer + "_" + model.getinfo() + "_" + datetime.now().strftime("%d-%b-%Y_%H:%M") + ".tsv"
    results.to_csv(filename, sep="\t")
    print("Results written to {}".format(filename))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_file', type=str,
                        help='a path to a json file containing all the information about the pretrained model (including parameters).\nIf this option is provided, there is no need to specify the parameters listed below (safer to do that, so the correct parameter values are used).')
 
    parser.add_argument("--data_dir", default="../data/sentences/sent_RO/", help="data directory")
    parser.add_argument("--results_dir", default="../results/sentenceEmbeddingAnalysis/RO/", help="results directory")

    parser.add_argument("--transformer", default="electra", choices=["bert", "roberta", "electra"], help="transformer model to use to produce (attn/sentence) representations")
    parser.add_argument("--sent_emb_dim", default=768, help="The size of the embedding")
    
    parser.add_argument("--latent", default=5, help="Dimension of the latent layer for sentence embedding (used for the 2-level analysis)")
    parser.add_argument("--beta_sent", default=1.0, help="Define a beta quantity.")

    parser.add_argument("--categorical_dim", default=0, help="Number of categories for the latent layer")
    parser.add_argument("--categ_N", default=2, help="Number of categories for the latent layer")

    parser.add_argument("--epochs", "-e", type=int, default = 1, help="Number of epochs.")
    parser.add_argument("--lr", "-l", type=float, default=1e-3, help="Learning rate.")   #5e-4
    parser.add_argument("--batch_size", "-bs", type=int, default=100, help="batch size")
    parser.add_argument("--model", "-v", help="path to the saved model parameters.")
    parser.add_argument("--extract", "-r", action="store_false", help="If extract, it ONLY extracts bert embeddings from sentences.")

    parser.add_argument("--N_exp", default=1, help="number of experiments to run")

    parser.add_argument("--train", "-t", action="store_true", help="Load embeddings from file and compute training.")
    parser.add_argument("--test", action="store_true", help="If true, do only the evaluation step.")
    parser.add_argument("--cuda", action="store_true", help="Enable GPU")

    parser.add_argument("--train_perc", default=1.0, help="Percentage of the available training data of each type  to use")

    parser.add_argument("--negatives", default="minimal", choices = ["all", "random", "minimal"], help="How to choose the negative examples (random from examples that have different patterns, or choose minimally different patterns")
    parser.add_argument("--nr_negs", default = 7, help="The number of negative examples to choose")

    parser.add_argument("--train_test_split", default="random", choices=["sep", "random"], help="How to split the data between train and test -- whether to have patterns in the test that have not be seen in the train, or whether to do random split.")


    args = parser.parse_args()

    args.sampling = "joint"
    if args.categorical_dim == 0:
        args.sampling = "simple"

    print("ARGS: {}".format(args))


    patterns = ["np-s vp-s", 
               "np-s pp1-s vp-s", 
               "np-s pp1-p vp-s", 
               "np-s pp1-s pp2-s vp-s", 
               "np-s pp1-p pp2-s vp-s", 
               "np-s pp1-s pp2-p vp-s", 
               "np-s pp1-p pp2-p vp-s", 
               "np-p vp-p", 
               "np-p pp1-s vp-p", 
               "np-p pp1-p vp-p", 
               "np-p pp1-s pp2-s vp-p", 
               "np-p pp1-p pp2-s vp-p", 
               "np-p pp1-s pp2-p vp-p", 
               "np-p pp1-p pp2-p vp-p"]

    (output_dir, embs_file, sent_file) = initialize_setup(args)
    
    embs = load_embeddings(embs_file)
    sent_seqs = load_data(sent_file, args)

    results = []
    for _n in range(args.N_exp):

        model, model_path = initialize_model_and_sampler(output_dir, args)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

        (train_data, test_data) = make_train_test(embs, sent_seqs, patterns, args)
        _trainloader, best_model_file = train_model(model, model_path, args.epochs, optimizer, train_data, args)

        checkpoint = torch.load(best_model_file, map_location=args.device)
        model.load_state_dict(checkpoint)

        res, _testloader = test_model(model, test_data, args)
        results.append(pd.DataFrame.from_dict(res))

        res = pd.concat(results)
        write_results(model, res, args)


if __name__ == '__main__':
    main()
