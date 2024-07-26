
import sys

import torch


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

import losses

def testing(testloader, model, device):

    model = model.to(device)
    model.eval()

    results = {}
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    preds = []
    true = []

    pred_p = []
    true_p = []
    all_ps = []
        
    with torch.no_grad():
        for _idx, batch in enumerate(testloader):
            (x, x_pos, x_negs, x_pos_p, x_negs_p) = batch[:5]
                        
            model_output = model(x.to(device))
            output = model_output["output"]
            negs = torch.stack(x_negs).to(device)
            negs = torch.transpose(negs.squeeze(), 0, 1)

            x_pos = x_pos.reshape(x.shape[0], 1, -1)

            x_pos_p = list(x_pos_p)
            x_negs_p = list(zip(*x_negs_p))

            preds.extend(losses.prediction(output, torch.cat((x_pos.to(device), negs), dim=1), device))
            for i in range(negs.shape[0]):   ## for each batch add the true labels
                true.extend([1] + [0]*negs.shape[1])
                all_ps.append(x_pos_p[i])
                all_ps.extend(list(x_negs_p[i]))
                true_p.append(x_pos_p[i])

    for i in range(len(preds)):
        if preds[i] == true[i]:
            if preds[i] == 0:
                tn += 1
            else:
                tp += 1
                pred_p.append(all_ps[i])
        elif preds[i] == 1:
            fp += 1
            pred_p.append(all_ps[i])
        else:
            fn += 1
                                               
    print("\nTP = {}, FP = {}, FN = {}, TN = {}\n".format(tp, fp, fn, tn))
    results["TP"] = tp
    results["FP"] = fp
    results["FN"] = fn
    results["TN"] = tn

    precision = tp/(tp+fp + sys.float_info.epsilon)
    recall = tp/(tp+fn + sys.float_info.epsilon)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    f1 = (2*precision*recall)/(precision+recall + sys.float_info.epsilon)

    ## overall scores, as for a binary task    
    results["P"] = precision
    results["R"] = recall
    results["F1"] = f1
    results["Acc"] = accuracy

    ## per patterns scores
    cr = classification_report(true_p, pred_p, output_dict=True)
    results["pattern_f1_macro_avg"] = cr["macro avg"]["f1-score"]
    results["pattern_accuracy"] = cr["accuracy"]

    patts = sorted(list(set(all_ps)))
    results["patterns_list"] = patts
    results["CM_patterns"] = confusion_matrix(true_p, pred_p, labels=patts)
    

    f1_s = f1_score(true_p, pred_p, average=None, labels=patts)
    prec_s = precision_score(true_p, pred_p, average=None, labels=patts)
    rec_s = recall_score(true_p, pred_p, average=None, labels=patts)

    for i in range(len(patts)):
        p = patts[i]
        results[p + "_Prec"] = prec_s[i]
        results[p + "_Rec"] = rec_s[i]
        results[p + "_F1"] = f1_s[i]

    print("Results: {}".format(results))

    return [results]
