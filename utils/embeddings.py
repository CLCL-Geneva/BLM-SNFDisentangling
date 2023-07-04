
import os
import json
import sys
import logging

import random

import pickle

from utils import data_extractor

import torch
import numpy as np

from transformers import utils

random.seed(1)
torch.manual_seed(1)


#max_N = 1000  ## used during development to process smaller data subsets
max_N = None

def load_embeddings(input_dir, train_percentage=1.0,  seq_indices = ""):
    x = torch.load(input_dir + "/x.pt")
    y = torch.load(input_dir + "/y.pt")
    t = torch.load(input_dir + "/truth.pt")
    
    logging.info("input tensor shape: {}".format(x.shape))
    
    if seq_indices != "":
        seq_indices = torch.tensor(np.fromstring(seq_indices, dtype=int, sep=' ')) - 1  ## to map them onto python indices
        x = torch.index_select(x, 1, seq_indices)
        logging.info("input tensor shape after selection: {}".format(x.shape))
    
    if train_percentage == 1.0:
        return x, y, t
    if train_percentage < 1.0:
        val = int(train_percentage * len(x))
    else:
        val = int(train_percentage)  ## because we can give exact size wanted instead of percentage
    
    inds = random.sample(range(len(x)), val)
    return [x[i] for i in inds], [y[i] for i in inds], [t[i] for i in inds]


def load_transformer(args):

    if args.transformer == "bert":
        logging.info("loading BERT tokenizer and model")
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        model = BertModel.from_pretrained("bert-base-multilingual-cased", output_attentions=True)
    elif args.transformer == "roberta":
        logging.info("loading RoBERTa tokenizer and model")
        from transformers import XLMRobertaModel, XLMRobertaTokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        model = XLMRobertaModel.from_pretrained("xlm-roberta-base", output_attentions=True)    
    elif args.transformer == "flaubert":
        logging.info("loading FlauBERT tokenizer and model")
        from transformers import FlaubertModel, FlaubertTokenizer
        tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_uncased")
        model = FlaubertModel.from_pretrained("flaubert/flaubert_base_uncased", output_attentions=True)    
    elif args.transformer == "electra":
        from transformers import ElectraTokenizer, ElectraModel
        tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
        model = ElectraModel.from_pretrained("google/electra-base-discriminator")
    else:
        sys.exit("Unknown transformer option: {}".format(args.transformer))

    return (tokenizer, model)


def getMaxLength(dir_list, args, tokenizer = None):
    
    dim = 0
    
    if tokenizer is None:
        (tokenizer, _) = load_transformer(args)
    
    for input_dir in dir_list:        
        for f in ["x.json", "y.json"]:
            full_path = input_dir + "/" + f
            with open(full_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            
            max_len = 0
            for _i, batch in enumerate(data):
                for sent in batch:
                    tk, _inds = data_extractor.get_tokens(tokenizer.tokenize(sent), args.attn_filter)
                    if len(tk) > max_len:
                        max_len = len(tk)
                    del tk
            if max_len > dim:
                dim = max_len
        
    logging.info("Maximum sentence length in the data: {}".format(dim))
    return dim    



def extract_emb(input_dir, out_dir, args, dim):
    logging.info("Preparing the data for training...")

    os.makedirs(out_dir, exist_ok = True)  

    (tokenizer, model) = load_transformer(args)
    
    for file in ["x.json", "y.json", "truth_bool.json"]:
        logging.info("Processing {} ... ".format(file))
        full_path = input_dir + "/" + file
        if "x" in file:
            logging.info("Data x...")
            file_path = out_dir + "/x"
            is_label = False
            seq = True
        elif "y" in file:
            logging.info("Data y...")
            file_path = out_dir + "/y"
            is_label = False
            seq = True
        elif "truth" in file:
            file_path = out_dir + "/truth"
            dim = 1
            is_label = True
        else:
            continue # to avoid errors when we are looking at a file that we dont need
        
        output = file_path + ".pt"
        if os.path.isfile(output):
            logging.info("Skipping file {} -- already processed ({})".format(full_path, output))
        else:
            with open(full_path, "r", encoding="utf-8") as file:
                raw_data = json.load(file)
                
            if max_N is not None:
                raw_data = raw_data[:max_N]
    
            data = data_extractor.divide_in_batches(raw_data, args.batch_size)
            
            for i, batch in enumerate(data):
                logging.info("BATCH {} ({})".format(i,len(batch)))
                batch_vecs = data_extractor.wrapper(batch, seq, model, dim=dim, label=is_label, layer=args.layer, attn=args.attention, tokenizer=tokenizer, device=args.device, comb=args.attn_comb)
                logging.info("BATCH VECS: ".format(batch_vecs.size()))
    
                # we concatenate back the data in a single file instead of batches
                if i == 0:
                    vecs = batch_vecs.clone()
                else:
                    vecs = torch.cat((vecs, batch_vecs))
                del batch_vecs
    
            del data
    
            # save the embeddings
            logging.info("Save embeddings to {} ...".format(output))
            torch.save(vecs, output)
    
            del vecs
            torch.cuda.empty_cache()
        
    del model
    
    
    
def extract_complete_emb(data_dir, tokenizer, model, representations, outdir):
    
    for file in ["x.json", "y.json", "truth_bool.json"]:
        logging.info("Processing {}/{} ... ".format(data_dir, file))
        full_path = data_dir + "/" + file
        if "x" in file:
            logging.info("Data x...")
        elif "y" in file:
            logging.info("Data y...")
        else:
            continue # to avoid errors when we are looking at a file that we don't need
        
        with open(full_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)
            
            i = 0
            for sent_list in raw_data:
                i += 1
                if i%100 == 0:
                    logging.info("\t {} instances ...".format(i))
                for sentence in sent_list:
                    if not sentence in representations:
                        input = tokenizer(sentence, return_tensors="pt")
                        tokens = tokenizer.tokenize(sentence)
                        outputs = model(**input)
                        
                        sentence_repr = {"sentence": sentence, "tokens": tokens, "input": input, "output": outputs}
                        
                        representations[sentence] = len(representations) + 1
                        
                        outfile = outdir + "/" + str(representations[sentence]) + "_complete_output.pkl"
                        with open(outfile, 'wb') as handle:
                            pickle.dump(sentence_repr, handle, protocol=pickle.HIGHEST_PROTOCOL)        
 

    
if __name__ == '__main__':

    tokens = ['La', 'conférence', 'sur', 'l', '[UNK]', 'histoire', 'dont', 'l', '[UNK]', 'organi', '##zate', '##ur', 'm', '[UNK]', 'a', 'par', '##lé', 'a', 'commencé', 'plus', 'tard', 'que', 'prévu', '.']
    filt_tokens, indices = data_extractor.get_tokens(tokens, True)
    print("filtered tokens: {}".format(filt_tokens))
    print("indices: {}".format(indices))

