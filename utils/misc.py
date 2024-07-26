
from datetime import datetime
import logging
import json
import re
import glob

import random
import pandas as pd

import numpy as np

import copy
from scipy import stats

import utils.embeddings as embeddings
import utils.data_extractor as data_extractor


## __________________________________________________________
## config file


def load_config(config_file, args):
        
    if config_file is not None:
        with open(config_file) as json_file:
            config_info = json.load(json_file)

        for arg in config_info:
            setattr(args, arg, config_info[arg])

    
def save_config(model_path, args):
    config_file = f'{model_path}/config.json'
    config_info = dict()
    config_info['model_path'] = model_path
    for arg in vars(args):
        if arg != "device":
            config_info[arg] = getattr(args, arg)

    with open(config_file, "w") as json_file:
        json.dump(config_info, json_file)
        


##_____________________________________________________________________
## loading the dataloader and the labels


def get_data(data_dir, data_percentage, batch_size, args):
    data_sets = embeddings.load_embeddings(data_dir, data_percentage, args.indices)
    return data_extractor.data_loader(data_sets, batch_size, False)


##___________________________________________________________________
## to randomize instances
def get_randomized_inds(N, ones):
    split = [1]*ones + [0]*(N-ones)
    random.shuffle(split)
    return split
        
## ____________________________________________________________________
## reading the answer labels (for computing error statistics)

def get_answer_labels(data_dir):
    
    with open(data_dir + "/labels.json", "r", encoding="utf-8") as file:
        data = json.load(file)    

    labels = data[0]

    ## remove the label for the correct sentence. In the current datasets it is either "True" or "Correct"
    if "True" in labels:
        labels.remove("True")
    elif "Correct" in labels:
        labels.remove("Correct")

    return labels



##___________________________________________________________________

def make_results_filename(args, pref, model_name):
    return args.results_dir + "/results_" + pref + "_" + args.transformer + "_" + model_name + "_"+ datetime.now().strftime("%d-%b-%Y_%H:%M") + ".tsv"

def get_model_file(model_path):
    files = glob.glob(model_path + "/*.pth")
    if len(files) > 0:
        return files[0]
    print("Model not found in directory {}".format(model_path))
    return None




## ____________________________________________________________________    
## a string containing main arguments info

def make_output_dir_name(args):

    emb_type = args.transformer + "_sentembs"
    output_dir =  args.data_dir + "/" + args.type + "/output/" + emb_type +  "/"

    return output_dir, emb_type


def get_run_info(args):
    
    timestamp = datetime.now().strftime("%d-%b-%Y_%H:%M")
    
    model_name = ""
    if args.baseline:
        model_name = args.baseline_sys
    else:
        model_name = args.sys + "_" + args.sampling + "-sampling"
        
    return model_name + "_" + "-".join(args.train_perc.split(" ")) + "-train_perc_" + str(args.epochs) + "-epochs_" + str(args.lr) + "-lr_" + timestamp


def get_model_info(model_file):
    
    model_type = re.compile(r".*\/model_(.*)\/best_model__train_(.*)_expNr.*\.pth")
    m = model_type.search(model_file) 
    
    if m:
        return m.group(1)
    
    return "model_unmatched"

##_____________________________________________________________________

def get_validation_data(d_type, train_dirs, args):

    validation = get_pairings(args.valid)
    valid_data = []

    for t in validation[d_type]:
        valid_t = embeddings.load_embeddings(train_dirs[t], train_percentage=args.valid_perc)
        if valid_data == []:
            valid_data = copy.deepcopy(valid_t)
        else:
            for i in range(len(valid_t)):
                valid_data[i].extend(valid_t[i])
    return valid_data


#    args.valid = "type_I_Fun-type_I_Lex type_II_Fun-type_II_Lex type_III_Fun-type_III_Lex"
def get_pairings(valid_data_string):

    validation = {}
    for group in valid_data_string.split(" "):
        d_types = group.split("-")
        for x in d_types:
            validation[x] = []
            for y in d_types:
                if x != y:
                    validation[x].append(y)

    return validation


####_____________________
## misc


def add_value(my_dict, key):
    
    if key in my_dict:
        my_dict[key] += 1
    else:
        my_dict[key] = 1

##______________________________

## asumption: stride < kernel
def get_padd(dim, kernel, stride):  

    if dim % kernel == 0:
        return 0

    if stride == 1:
        return max(0, kernel-dim)
    
    #print("padd = {} - ({} - {} * (int({}/{})-1))".format(kernel, dim, stride, dim, stride))

    #return kernel - (dim - stride * (int(dim/stride)-1))
    return kernel - (dim - stride * (int(dim/stride)))

def get_out_padd(OutDim, InDim, kernel, stride, padding):
    return OutDim - (InDim-1) * stride + 2 * padding - kernel


#________________________________

def convert_ind(i, channels, max_x, max_y):
    ch = int(i/(max_x * max_y))
    i = i%(max_x * max_y)
    x = int(i/max_y)
    y = i%max_y
    return (ch,x,y)


##__________________________________

def get_ranges(latents):
    print("latents shape: {}".format(latents.shape))
    ranges = []
    for i in range(len(latents[0])):
        ranges.append([min(latents[:,i]), max(latents[:,i])])

    return ranges

#________________________________________


## compute the "difference" between two samples -- convert them to histograms, and subtract those
# (using shared intervals! -- so make a common list first, make histogram and take the intervals,
# then convert the orginal list to histograms using the shared intervals, and then make the difference)
def compute_diff(list1, list2, bin_edges):
    hist1, _ = np.histogram(list1, bins=bin_edges)
    hist2, _ = np.histogram(list2, bins=bin_edges)

    return hist1 - hist2


def get_dist(my_dict, p1, p2, bin_edges):
    if (p1 not in my_dict) or (p2 not in my_dict):
        return 0   ## is this correct? the distance quantifies the difference between two sets of values, so if one of those doesn't exist, what should be the difference?

    return compute_dist(my_dict[p1], my_dict[p2], bin_edges)

## compute distance between to samples as the euclidean distance between their histograms (computed over the same bins)
def compute_dist(list1, list2, bin_edges):
    hist1, _ = np.histogram(list1, bins=bin_edges)
    hist2, _ = np.histogram(list2, bins=bin_edges)

    hist1_norm = np.linalg.norm(hist1)
    hist2_norm = np.linalg.norm(hist2)

    if hist1_norm * hist2_norm == 0:
        return 0

    # return np.linalg.norm(hist1-hist2)
    return abs(1 - np.dot(hist1, hist2) / (hist1_norm * hist2_norm))



def minimal_diff_pairs(my_list, diff):
    diffs = {}
    for i in range(len(my_list)-1):
        p1 = my_list[i].split(" ")
        for j in range(i + 1, len(my_list)):
            p2 = my_list[j].split(" ")
            res = has_min_diff(p1, p2, diff)
            if res:
                (p1_str,p2_str,d) = res  ## has_min_diff will order the patterns, such that they are always consistent  (e.g. the longer one is always first, the singular form is always first,
                if d not in diffs:
                    diffs[d] = []
                diffs[d].append([p1_str, p2_str])

    pairs = []
    pair_diffs = []

    for d in diffs:
        for p in diffs[d]:
            pairs.append(p)
            pair_diffs.append(d)

    print("pairs = {}".format(pairs))
    print("pair diffs = {}".format(pair_diffs))

    return pairs, pair_diffs


# check if a pair of patterns are minimally different, w.r.t. the difference type
# (nr -- grammatical nr, len -- extra chunk, agr -- only subject and verb numbers differ, the rest of the pattern is the same
#
# order the patterns, such that they are always consistent  (e.g. the longer one is always first, the singular form is always first,
def has_min_diff(a, b, diff):

    print("Comparing patterns **{}**   **{}**".format(a, b))
    d1 = list(set(a) - set(b))
    d2 = list(set(b) - set(a))

    if diff == "len":
        # return len(d1) + len(d2) == 1
        print("d1 = {} / d2 = {}".format(d1, d2))
        if len(d1) + len(d2) != 1:
            return False
        if len(d1) == 1:  ## return the longer pattern as first in the pair
            return (" ".join(a), " ".join(b), d1[0])
        return (" ".join(b), " ".join(a), d2[0])

    if diff == "nr":
        if len(d1) != 1 or len(d2) != 1:
            return False
        # return (set(d1[0][:-1]) == set(d2[0][:-1]))  ## the grammatical number is on the last position of the strings
        if (set(d1[0][:-1]) != set(d2[0][:-1])):
            return False
        if d1[0][-1] == "s":
            return (" ".join(a), " ".join(b), d1[0][:-2])
        return (" ".join(b), " ".join(a), d1[0][:-2])

    if diff == "agr":  ## subject and verb have different number in the two patterns, but everything else is the same
        # return set(d1).union(d2) == set(["np-s", "vp-s", "np-p", "vp-p"])
        if not subj_and_verbs(set(d1).union(d2)):
            return False
        set_int = list(set.intersection(set(a),set(b)))
        if len(set_int) == 0:
            set_int = ["none"]
        if d1[0][-1] == "s":
            return (" ".join(a), " ".join(b), "_".join(set_int))
        return (" ".join(b), " ".join(a), "_".join(set_int))

    return False

def subj_and_verbs(patt_set):

    if patt_set == set(["np-s", "vp-s", "np-p", "vp-p"]):
        return True

    if patt_set == set(["subj-s", "vp-s", "subj-p", "vp-p"]):
        return True

    return False


## to remove prefix and roc chunk info from patterns, and see whether this way they cluster more cleanly
# (do they appear in the answer set? I thought no, but maybe yes. In that case it may be debatable whether this kind of analysis is a good idea.
# the issue is that these kinds of chunks are not really relevant for the BLM agreement task, so they may not "feature" on the latent layer)
def simplify_patterns(patterns):
    return [simplify(p) for p in patterns]

def simplify(pattern):
    pattern = pattern.lower()
    pattern = re.sub(r"(prefix|roc[-_].*?)\s+","",pattern)
    pattern = re.sub(r' p2_none', "", pattern)
    return pattern


def get_labels(raw_data):
    labels = set()
    for x in raw_data:
        labels.update(x)

    return sorted(list(labels))


def compute_activations(latents, latent_vectors, x_conv, x_masked, patterns, trace):
    patterns_list = list(set(patterns))

    print("len(patterns) = {}, len(latent_vectors), len(x_conv) = {}, len(masked) = {}, len(patterns_list) = {}".format(
        len(patterns), len(latent_vectors), len(x_conv), len(x_masked), len(patterns_list)))

    cnn_latents_map = {}
    activations = {i: {} for i in latents}
    for ch in trace:
        for latent_index in trace[ch]:
            for i in trace[ch][latent_index]:
                cnn_latents_map[i] = latent_index
                activations[latent_index][i] = {p: [] for p in patterns_list}

            # print("\tactivations[{}] = {}".format(latent_index, activations[latent_index]))

    # print("\ncnn_latents_map = {}".format(cnn_latents_map))
    # print("len(x_conv) = {}, len(x_conv[0]) = {}".format(len(x_conv), len(x_conv[0])))
    for i in range(len(latent_vectors)):
        # print("  processing vector {}".format(i))
        v = latent_vectors[i]
        p = patterns[i]
        for cnn_ind in cnn_latents_map:
            latent_index = cnn_latents_map[cnn_ind]
            # print("\tcnn_ind = {} => latent_index = {}".format(cnn_ind, latent_index))
            # print("\t\tactivations[{}][{}] = {}".format(latent_index, cnn_ind, activations[latent_index][cnn_ind]))
            # print("\t\t\t value to add x_conv[{}][{}] = {}".format(i, cnn_ind, x_conv[i][cnn_ind]))
            activations[latent_index][cnn_ind][p].append(x_conv[i][cnn_ind])

    print("activations for {} indices computed ({})".format(len(activations), list(activations.keys())))

    for latent_index in activations:
        cnn_inds = list(activations[latent_index].keys())
        for cnn_ind in cnn_inds:
            for p in patterns_list:
                if activations[latent_index][cnn_ind][p] == []:
                    del activations[latent_index][cnn_ind][p]

            if len(activations[latent_index][cnn_ind]) == 0:
                del activations[latent_index][cnn_ind]

    return activations, cnn_latents_map


## filter out the activations information, to keep for each pattern only what distinguishes it from any of the others
def reduce_activations(activations):
    print("Reducing activations ...")

    latent_inds = list(activations.keys())

    for latent_ind in latent_inds:
        # print("\t latent index = {}".format(latent_ind))

        inds = list(activations[latent_ind].keys())
        for cnn_ind in inds:

            # print("\t\t cnn_ind = {}".format(cnn_ind))
            patterns = list(activations[latent_ind][cnn_ind].keys())
            histograms = {p: np.histogram(activations[latent_ind][cnn_ind][p], bins=20) for p in
                          patterns}  ## the histogram is a tuple (bin_counts, histogram)

            diff = False
            for i in range(len(patterns)):
                p1 = patterns[i]
                # print("\t\t  vals[{}] = {}".format(p1, activations[latent_ind][cnn_ind][p1]))
                # print("\t\t  hist[{}] = {}".format(p1, histograms[p1]))

                j = i + 1
                while j < len(patterns) and not diff:
                    p2 = patterns[j]

                    # print("\t\t  vals[{}] = {}".format(p2, activations[latent_ind][cnn_ind][p2]))
                    # print("\t\t  hist[{}] = {}".format(p2, histograms[p2]))

                    score = compare_value_sets(p1, p2, activations[latent_ind][cnn_ind], histograms)
                    diff = (score.pvalue <= 0.05)
                    j += 1

                    # print("\t\t\t ks_2samp({}, {}) = {}".format(p1, p2, score))

            if not diff:
                print("deleting activations for latent unit {} / cnn unit {}".format(latent_ind, cnn_ind))
                del activations[latent_ind][cnn_ind]

        if len(list(activations[latent_ind].keys())) == 0:
            del activations[latent_ind]

    print("Activations information after reduction based on Kolmogorov-Smirnov pairwise values histogram comparison:")
    print("\t{} latent units".format(len(activations)))
    for l_i in activations:
        print("\t\t unit {} => {} inputs".format(l_i, len(activations[l_i])))


def compare_value_sets(p1, p2, activations, histograms):
    # return stats.ks_2samp(histograms[p1][1], histograms[p2][1])
    return stats.ks_2samp(activations[p1], activations[p2])

#_____________________________________________________________
## write info tracking (activations and comparisons) to file

def write_activations(activations, cnn_ind_dec, filename):
    
    rows = []    
    #activations[lat_ind][cnn_ind][p1]
    for lat_ind in activations:       
        for cnn_ind in activations[lat_ind]:
            (ch, x, y) = cnn_ind_dec[cnn_ind]
            for p in activations[lat_ind][cnn_ind]:
                rows.append({"latent": lat_ind, "cnn_output_unit": cnn_ind, "channel": ch, "emb_x": x, "emb_y": y, "pattern": p, "activations": " ".join([str(x) for x in activations[lat_ind][cnn_ind][p]])})

    df = pd.DataFrame(rows)
    df.to_csv(filename+"_activations.tsv", sep="\t")
    print("CNN activations written to {}".format(filename+"_activations.tsv"))
                

    
def write_distribution_comparisons(diff, values, filename):

    rows = []
    #values[x][y][lat_ind][ch][" vs. ".join([p1, p2])]
    for x in values:
        for y in values[x]:
            for lat_ind in values[x][y]:
                for ch in values[x][y][lat_ind]:
                    for patt_pair in values[x][y][lat_ind][ch]:
                        #print("pattern pair: {}".format(patt_pair))
                        [p1,p2] = patt_pair.split(" vs. ")
                        rows.append({"emb_x": x, "emb_y": y, "latent": lat_ind, "channel": ch, "pattern_1": p1, "pattern_2": p2, "dist": values[x][y][lat_ind][ch][patt_pair]})
                        
    df = pd.DataFrame(rows)
    df.to_csv(filename + "_" + diff + "_distances.tsv", sep="\t")
    print("Value distribution distances for minimally different patterns w.r.t. {} written to {}".format(diff, filename + "_" + diff + "_distances.tsv"))

#_____________________________________________________________


def get_units_index(weights):
    return {i: weights[i] for i in range(len(weights)) if weights[i] != 0}


def get_section(i, params):
    # (ch, seq, x, y) = convert_linear_index(i, params["cnn_output_shape"])
    (ch, x, y) = convert_linear_index(i, params["cnn_output_shape"])
    # print("section for {}: ({} {} {} {})".format(i, ch, seq, x, y))
    return [ch, i, convert_to_section(x, y, params["kernel_size"], params["stride"], params["cnn_input_shape"])]


def convert_linear_index(i, cnn_out_shape):
    '''
    (_ch, seq, x, y) = cnn_out_shape
    n = i % (seq * x * y)
    k = n % (x * y)
    return (int(i/(seq*x*y)), int(n/(x*y)), int(k/x), k%y)
    '''
    (_ch, x, y) = cnn_out_shape
    n = i % (x * y)
    return (int(i / (x * y)), int(n / x), n % x)


def convert_to_section(x, y, kernel_size, stride, cnn_in_shape):
    '''
    (_, kx, ky) = kernel_size
    (_, sx, sy) = stride
    (_,x_dim, y_dim) = cnn_in_shape
    '''
    (kx, ky) = kernel_size
    (sx, sy) = stride
    (x_dim, y_dim) = cnn_in_shape
    return (x * sx, y * sy, min(x * sx + kx, x_dim), min(y * sy + ky, y_dim))

if __name__ == '__main__':

    filename = "/home/vivi/work/Projects/BLMs/results/sentenceEmbeddingAnalysis/results_agreement_FR_electra_SentenceVAE2D_32x24__k-15x15_s-1x1___simple-sampler__latent-size_5_fine_26-Mar-2024_13:42.tsv"
    df = pd.read_csv(filename, sep="\t")
    headers = df.columns
    
    print("df:\n{}".format(df))
    f1_s = []
    
    for idx, row in df.iterrows():
        if row['set-up'] == "base":
            print("Getting F1s from row: {}".format(row))
            f1_i = []
            for h in headers:
                if "_F1" in h:
                    print("adding F1 score {} for {}".format(row[h], h))
                    f1_i.append(row[h])
            print(" f1_i = {} ({})".format(f1_i, np.mean(f1_i)))
            f1_s.append(np.mean(f1_i))
            
    print("F1-s: {}, mean = {}, std = {}".format(f1_s, np.mean(f1_s), np.std(f1_s)))

    
