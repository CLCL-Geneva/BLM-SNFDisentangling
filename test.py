
import sys
import os

import torch
import torch.nn as nn
import json
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

import utils.losses as losses  
#import plot


def testing(testloader, labels, model, device, data_type):
    model = model.to(device)
    # set to evaluation mode
    model.eval()

    results = {"agreement_error":0, "coordination":0, "mis_num":0, "N1_alter":0, "N2_alter":0}   ## to avoid NaNs
            
    # initialitation of the variables for evaluation
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    with torch.no_grad():
        for idx, (seq_x, seq_y, truth) in enumerate(testloader):
            
            truth = torch.flatten(truth)
            seq_labels = labels[idx]
            
            seq = seq_x.squeeze(2)   #.transpose(-1, -2)
            model_output = model(seq.to(device))
            #output = output.view(1,output.shape[0])
            output = model_output["output"].view(1,-1)
            
            '''
            if base:
                ## for the baseline setting compare the inputs and outputs using cosine
                
                scores = torch.bmm(seq_x, seq_y).squeeze()
                results.write("BMM scores: {}".format(scores))
                results.write("Similarity average: {}".format(torch.mean(scores)))

            else:
            '''
            (max_ind, _pred) = losses.prediction(output, seq_y, device)
                        
            if truth[max_ind] == 1:
                tp += 1
                tn += len(truth)-1
            else:
                fp += 1
                fn += 1
                tn += len(truth)-2
                results[seq_labels[max_ind]] += 1

                ### Score measures
                
    print("\nTP = {}, FP = {}, FN = {}, TN = {}\n".format(tp, fp, fn, tn))
    results["TP"] = tp
    results["FP"] = fp
    results["FN"] = fn
    results["TN"] = tn

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    f1 = (2*precision*recall)/(precision+recall + sys.float_info.epsilon)
    
    results["P"] = precision
    results["R"] = recall
    results["F1"] = f1
    results["Acc"] = accuracy
    
    print("Results: {}".format(results))
        
    return results


