
import sys
import os

import torch

import utils.losses as losses

results_keys = ["run", "train on", "test on", "TP", "FP", "FN", "TN", "P", "R", "F", "Acc", "agreement_error", "coordination", "mis_num", "N1_alter", "N2_alter"]

def testing(testloader, labels, model, device):
    model = model.to(device)
    # set to evaluation mode
    model.eval()

    results = {}
    
    # initialitation of the variables for evaluation
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    with torch.no_grad():
        for idx, (seq_x, seq_y, truth) in enumerate(testloader):
            
            truth = torch.flatten(truth)
            seq_labels = labels[idx]
            
            seq = seq_x.squeeze(2)   #.transpose(1, 2)
            output = model(seq.to(device))

            (max_ind, _pred) = losses.prediction(output, seq_y, device)
                        
            if truth[max_ind] == 1:
                tp += 1
                tn += len(truth)-1
            else:
                fp += 1
                fn += 1
                tn += len(truth)-2
                add_error(results, seq_labels[max_ind])

        ### Score measures
        
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
        results["accuracy"] = accuracy
        
        print("Results: {}".format(results))
        
    return results

            
def add_error(errors, err_type):
    
    if err_type not in errors:
        errors[err_type] = 0
        
    errors[err_type] += 1
