import sys
import logging
import json

import random
import re

import numpy as np
import pandas as pd

import seaborn as sns

from sklearn.metrics import precision_recall_curve

import utils.misc as misc


def check_disc(vectors, patterns, args):

    vectors_dict = {}
    
    for i in range(len(vectors)):
        v = make_disc_patt(vectors[i], args)
        if v not in vectors_dict:
            vectors_dict[v] = {}

        p = patterns[i]
        if p not in vectors_dict[v]:
            vectors_dict[v][p] = 0
            
        vectors_dict[v][p] += 1

    patts = sorted(list(set(patterns)))
    vecs = sorted(list(set(vectors_dict.keys())))

    print("{} patterns and {} vectors".format(len(patts), len(vecs)))

    rows = []
    index = []
        
    for v in vecs:
        index.append(v)
        row = {} 
        for p in patts:
            if p in vectors_dict[v]:
                row[p] = vectors_dict[v][p]
            else:
                row[p] = 0
        rows.append(row)
    
    vec_patt_map = pd.DataFrame(rows, index=index, columns=patts)
    
    print("vectors-patterns map:\n{}".format(vec_patt_map))
    
    sns.heatmap(vec_patt_map, cmap="Blues")
    


def make_disc_patt(my_list, args): 
    
    my_list = [str(int(x)) for x in my_list]
    
    n = args.categ_N  
    for i in range(1, args.categorical_dim):
        my_list.insert(i*n + i-1, "-")
        
    return "".join(my_list)


#_____________________________________

def find_threshold(y_pred, y_true):
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2*recall*precision/(recall + precision + sys.float_info.epsilon)   
       
    logging.info('Best threshold: {}'.format(thresholds[np.argmax(f1_scores)]))    
    logging.info('Best F1-Score: {}'.format(np.max(f1_scores)))
    
    return float(thresholds[np.argmax(f1_scores)])

            
##_________________________________________________________

            
def choose_negatives(embs, p, data, N_negs, choice_type):

    neg_patts = choose_patterns(p, list(data.keys()), choice_type)

    if choice_type == "all":
        N_negs = len(neg_patts)

    return sample_negatives(embs, neg_patts, data, N_negs)


def choose_patterns(pos_p, all_p, choice_type):
    all_p.remove(pos_p)
    
    if choice_type in ["random", "all"]:
        return all_p    
    
    if choice_type == "minimal":
        ## how to choose minimally different patterns
        chunks = {"vp": -1, "pp1": 1, "pp2": 2}
        patts = []
        for chunk in chunks.keys():
            if chunk in pos_p:
                patts.append(alter_pattern(chunk, chunks[chunk], pos_p))
            
        patts.extend(add_pp(pos_p))
        
        return patts

    print("Unknown option, making random selection of negative patterns")
    return all_p


def alter_pattern(chunk, ind, patt):    
    p = patt.split(" ")
    if len(p) > ind:
        p[ind] = switch_nr(p[ind])

    ## if the chunk is a VP (which is always last in our data), switch the number of the subject (which is always first in our data)        
    if ind == -1:
        p[0] = switch_nr(p[0])
    
    return " ".join(p)


# add a pp to the pattern if the length of the pattern is 2 (np-vp) or 3 (np, pp, vp) 
def add_pp(patt):
    p1 = patt.split(" ")
    p2 = patt.split(" ")
    
    if len(p1) == 2:
        p1.insert(1, "pp1-s")
        p2.insert(1, "pp1-p")
        return [" ".join(p1), " ".join(p2)]
    
    elif len(p1) == 3:
        p1.insert(2, "pp2-s")
        p2.insert(2, "pp2-p")
        return [" ".join(p1), " ".join(p2)]

    return []


def switch_nr(chunk):
    
    nr = {"p": "s", "s": "p"}
    
    c = chunk.split("-")
    c[-1] = nr[c[-1]]
            
    return "-".join(c)
            
            
            
def sample_negatives(embs, neg_patts, data, N_negs):
    negs = []
    negs_p = []
    
    for neg_p in choose_negative_patts(neg_patts, N_negs):
        negs.append(embs[random.choice(data[neg_p])])
        negs_p.append(neg_p)
        
    return negs, negs_p


def choose_negative_patts(neg_patts, N_negs):
    
    if len(neg_patts) >= N_negs:
        return random.sample(neg_patts, N_negs)
    
    return random.choices(neg_patts, k=N_negs)

#____________________________________________________________


def add_value(hash, key, val):
    if key not in hash:
        hash[key] = []
    hash[key].append(val)

