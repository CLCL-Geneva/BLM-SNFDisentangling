
import sys
import logging

import torch

import utils.losses as losses  

    

def testing(testloader, labels, model, device, result_keys, mask = None):
    model = model.to(device)
    # set to evaluation mode
    model.eval()
    
    try:
        model.sampler.is_training = False
    except AttributeError:
        logging.info("model does not have sampler. skipping the is_training setting.")    

    results = {x: 0 for x in result_keys}
                
    # initialitation of the variables for evaluation
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    with torch.no_grad():
        for idx, (seq_x, seq_y, truth) in enumerate(testloader):
            
            truth = torch.flatten(truth)
            seq_labels = labels[idx]
            
            seq = seq_x.squeeze(2)
            model_output = model(seq.to(device), mask)
            output = model_output["output"].view(1,-1)
            
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
                
    logging.info("\nTP = {}, FP = {}, FN = {}, TN = {}\n".format(tp, fp, fn, tn))
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
    
    if mask is not None:
        logging.info("\n Analysis for discrete latent probe mask = {}".format(mask))
        
    logging.info("Results: {}".format(results))
        
    return results

