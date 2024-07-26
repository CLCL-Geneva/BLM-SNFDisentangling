
import os
import json
import sys
import logging

import random

import pickle

from utils import data_extractor
import utils.misc as misc

import torch
import numpy as np

random.seed(1)
torch.manual_seed(1)


#max_N = 1000  ## used during development to process smaller data subsets
max_N = None

def load_embeddings(input_dir, train_percentage=1.0,  seq_indices = ""):
    
    logging.info("Loading data from {}".format(input_dir))

    data = []
    for f in ["x.pt", "y.pt", "truth.pt", "labels.pt", "x_templates.pt", "y_templates.pt"]:
        pt_file = input_dir + "/" + f
        if os.path.isfile(pt_file):
            data.append(torch.load(pt_file))
        
    logging.info("input tensor shape: {}".format(data[0].shape))
    
    if seq_indices != "":
        seq_indices = torch.tensor(np.fromstring(seq_indices, dtype=int, sep=' ')) - 1  ## to map them onto python indices
        data[0] = torch.index_select(data[0], 1, seq_indices)
        logging.info("input tensor shape after selection: {}".format(data[0].shape))
    
    if train_percentage == 1.0:
        return data
    if train_percentage < 1.0:
        val = int(train_percentage * len(data[0]))
    else:
        val = int(train_percentage)  ## because we can give exact size wanted instead of percentage
    
    print("Sampling {} instances from {}".format(val, len(data[0])))
    
    inds = random.sample(range(len(data[0])), val)
    return [[d[i] for i in inds] for d in data]



def load_labels_dict(args, data_type = None):

    if data_type is None:
        data_type = args.type

    _output_dir, emb_type = misc.make_output_dir_name(args)
    labels_file = args.data_dir + "/" + data_type + "/datasets/train/embeddings/" + emb_type + "/labels_dict.json"
    labels = None
    with open(labels_file, "r") as f:
        labels = json.load(f)
        
    return labels


def load_transformer(args):

    if args.transformer == "bert":
        logging.info("loading BERT tokenizer and model")
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        model = BertModel.from_pretrained("bert-base-multilingual-cased", output_attentions=False)
    elif args.transformer == "roberta":
        logging.info("loading RoBERTa tokenizer and model")
        from transformers import XLMRobertaModel, XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        model = XLMRobertaModel.from_pretrained("xlm-roberta-base", output_attentions=False)
    elif args.transformer == "flaubert":
        logging.info("loading FlauBERT tokenizer and model")
        from transformers import FlaubertModel, FlaubertTokenizer
        tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_uncased")
        model = FlaubertModel.from_pretrained("flaubert/flaubert_base_uncased", output_attentions=False)
    elif args.transformer == "electra":
        from transformers import ElectraTokenizer, ElectraModel
        tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
        model = ElectraModel.from_pretrained("google/electra-base-discriminator")
    else:
        sys.exit("Unknown transformer option: {}".format(args.transformer))

    return (tokenizer, model)


def extract_emb(input_dir, out_dir, args, dim):
    logging.info("Preparing the data for training...")

    os.makedirs(out_dir, exist_ok = True)  

    (tokenizer, model) = load_transformer(args)
    
    for file in ["x.json", "y.json", "truth_bool.json", "labels.json", "x_templates.json", "y_templates.json"]:
        logging.info("Processing {} ... ".format(file))
        full_path = input_dir + "/" + file
        if "x.json" in file:
            logging.info("Data x...")
            file_path = out_dir + "/x"
            is_label = False
            seq = True
        elif "y.json" in file:
            logging.info("Data y...")
            file_path = out_dir + "/y"
            is_label = False
            seq = True
        elif "truth" in file:
            file_path = out_dir + "/truth"
            dim = 1
            is_label = True
            seq = True
        elif "labels" in file:
            file_path = out_dir + "/labels"
            dim = 1
            is_label = True
            seq = True
        elif "x_templates" in file:
            file_path = out_dir + "/x_templates"
            dim = 1
            is_label = True
            seq = True
        elif "y_templates" in file:
            file_path = out_dir + "/y_templates"
            dim = 1
            is_label = True
            seq = True
        else:
            continue # to avoid errors when we are looking at a file that we dont need

        if os.path.isfile(full_path):
            output = file_path + ".pt"
            if os.path.isfile(output):
                logging.info("Skipping file {} -- already processed ({})".format(full_path, output))
            else:
                with open(full_path, "r", encoding="utf-8") as file:
                    raw_data = json.load(file)

                if max_N is not None:
                    raw_data = raw_data[:max_N]

                labels_dict = None
                if ("labels" in full_path) or ("templates" in full_path):
                    labels = misc.get_labels(raw_data)
                    labels_dict = {labels[i]:i for i in range(len(labels))}
                    with open(file_path + "_dict.json", "w") as l_file:
                        json.dump(labels_dict, l_file)
                    logging.info("Labels/templates dictionary written to {}_dict.json".format(file_path))

                data = data_extractor.divide_in_batches(raw_data, args.batch_size)

                for i, batch in enumerate(data):
                    logging.info("BATCH {} ({})".format(i,len(batch)))
                    batch_vecs = data_extractor.wrapper(batch, seq, model, dim=dim, label=is_label, labels_dict = labels_dict, tokenizer=tokenizer, device=args.device)
                    # concatenate back the data in a single file instead of batches
                    if i == 0:
                        vecs = batch_vecs.clone()
                    else:
                        vecs = torch.cat((vecs, batch_vecs))

                # save the embeddings
                logging.info("Save embeddings to {} ...".format(output))
                torch.save(vecs, output)

                torch.cuda.empty_cache()
