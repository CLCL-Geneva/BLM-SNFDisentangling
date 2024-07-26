
import sys
import logging

import torch
import numpy as np

import utils.losses as losses  
import utils.misc as misc
import utils.embeddings as embeddings

import run_funcs

def testing_one(testloader, model, device, result_keys, args, mask = None, sent_mask = None, latent_part = "latent_vec", data_type = None):

    model = model.to(device)
    # set to evaluation mode
    model.eval()

    results = {x: 0 for x in result_keys}                
    results, tp, fp, tn, fn,  all_truth, all_predictions, latent_values = get_predictions(model, testloader, latent_part, device, results, mask, sent_mask, args, data_type = data_type)
                
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
        logging.info("\n Analysis for sequence latent probe mask = {}".format(mask))

    if sent_mask is not None:
        logging.info("\n Analysis for sentence latent probe mask = {}".format(sent_mask))

    logging.info("Results after testing: {}".format(results))
    return results, all_truth, all_predictions, latent_values



def testing(testloaders, model, device, result_keys, args, mask = None, sent_mask = None, latent_part = "latent_vec", test_on = ""):

    results = []
    all_truth = {}
    all_predictions = {}
    latent_values = {}

    for lang in testloaders:
        results_lang, all_truth_lang, all_predictions_lang, latent_values_lang = testing_one(testloaders[lang], model, device, result_keys, args, mask=mask, sent_mask = sent_mask, latent_part=latent_part, data_type = test_on + "_" + lang)
        results_lang["test_lang"] = lang
        results.append(results_lang)
        all_truth[lang] = all_truth_lang
        all_predictions[lang] = all_predictions_lang
        latent_values[lang] = latent_values_lang

    return results, all_truth, all_predictions, latent_values


def get_predictions(model, dataloader, latent_part, device, results, mask, sent_mask, args, data_type=None):

    is_two_level = check_model(args, "two_level")
    labels_dict = {v:k for (k,v) in embeddings.load_labels_dict(args, data_type=data_type).items()}

    # initialitation of the variables for evaluation
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    all_predictions = []
    all_truth = []
    latent_values = []

    with torch.no_grad():
        for _idx, batch in enumerate(dataloader):

            (seq_x, seq_y, truth, seq_labels) = batch[:4]

            truth = truth.view(truth.shape[0],-1).cpu().detach().numpy()
            seq_labels = seq_labels.view(seq_labels.shape[0],-1).cpu().detach().numpy()
            
            seq = seq_x.squeeze(2)
            if is_two_level:
                model_output = model(seq.to(device), mask=mask, sent_mask = sent_mask)                
            else:
                model_output = model(seq.to(device), mask=mask)
                
            output = model_output["output"].view(1,-1)

            if "VAE" in model.__class__.__name__ :  
                latent_values.extend(model_output[latent_part].tolist())
            
            max_ind, _scores = losses.prediction(output, seq_y, device)
                        
            for i in range(len(truth)):
                if int(truth[i][max_ind[i]]) == 1:
                    tp += 1
                    tn += len(truth[i])-1
                else:
                    fp += 1
                    fn += 1
                    tn += len(truth[i])-2
                    misc.add_value(results, labels_dict[int(seq_labels[i][max_ind[i]])])

                all_truth.append(int(truth[i][max_ind[i]]))
                all_predictions.append(labels_dict[int(seq_labels[i][max_ind[i]])])

    return results, tp, fp, tn, fn,  all_truth, all_predictions, latent_values



##_____________________________________________________________________________________

def test_run(exp_nr, model, model_file, model_path, exp_type, train_on, test_dirs, results, all_predictions, args):
    logging.info("\n\nTesting final model ...")
    logging.info("\n\nTesting model {}".format(model_file))
    
    try:    
        run_funcs.load_model(model, model_file, args)
    except RuntimeError:
        logging.info("ERROR loading model for testing from {}".format(model_file))

    testloaders = {}
    for test_on in test_dirs:
        logging.info("Testing on {}".format(test_on))
        res, predictions, testloaders[test_on] = test_model(model, model_path, test_on, test_dirs[test_on], exp_type + "__test_" + test_on, args)
        res["train_on"] = train_on
        res["test_on"] = test_on
        res["run"] = exp_nr

        results = results.append(res, ignore_index=True)
        all_predictions["test_on " + test_on] = predictions
                    
    return testloaders, results
    
##____________________________________________________________________________________

def test_model(model, model_path, data_type, test_dir, exp_type, args):

    testloader = misc.get_data(test_dir, 1.0, args.batch_size, args)
        
    logging.info("Testing {} data ...".format(data_type)) 
    logging.info("result keys: {}".format(args.result_keys))   
    res, truth, predictions, _latent_values = testing_one(testloader, model, args.device, args.result_keys, args, data_type = data_type) ## if I give latent_part = "latent_vec_disc" or "mean" I can get the separate parts and look at clustering for different portions of the latents

    all_predictions = {"no mask": {"truth": truth, "label": predictions}}

    return res, all_predictions, testloader


##____________________________________________________________________________________



def mask_one_latent(model, testloader, args, all_predictions, res):

    logging.info("\nMasking sequence latent\n")

    categorical_dim = args.categorical_dim
    categ_N = args.categ_N
    cont_latent = args.latent

    N = categorical_dim * categ_N + cont_latent

    for i in range(categorical_dim):
        mask = np.ones(N, dtype=np.float32)
        mask[i*categ_N:(i+1)*categ_N] = 0
        apply_mask(mask, None, testloader, model, args, all_predictions, res)

    for i in range(categorical_dim * categ_N, N):
        mask = np.ones(N, dtype=np.float32)
        mask[i] = 0
        apply_mask(mask, None, testloader, model, args, all_predictions, res)


def mask_two_level_latent(model, testloader, args, all_predictions, res):

    logging.info("\nMasking sentence latent\n")

    categorical_dim = args.latent_sent_categ
    categ_N = args.latent_sent_categ_N
    cont_latent = args.latent_sent_dim_cont

    N = categorical_dim * categ_N + cont_latent

    for i in range(categorical_dim):
        mask = np.ones(N, dtype=np.float32)
        mask[i*categ_N:(i+1)*categ_N] = 0
        apply_mask(None, mask, testloader, model, args, all_predictions, res)

    for i in range(categorical_dim * categ_N, N):
        mask = np.ones(N, dtype=np.float32)
        mask[i] = 0
        apply_mask(None, mask, testloader, model, args, all_predictions, res)

    mask_one_latent(model, testloader, args, all_predictions, res)


#_______________________________________


def get_mask_str(mask_array):
    if mask_array is None:
        return "NA"
    return "".join([str(int(x)) for x in mask_array])

def negate(mask):
    return np.array([abs(x-1) for x in mask], dtype=np.float32)


def apply_mask(seq_mask, sent_mask, testloader, model, args, all_predictions, res):

    if seq_mask is not None:
        seq_mask = torch.from_numpy(seq_mask).to(args.device)

    if sent_mask is not None:
        sent_mask = torch.from_numpy(sent_mask).to(args.device)

    probe_res, truth, predictions, _latents = testing(testloader, model, args.device, args.result_keys, args, mask = seq_mask, sent_mask = sent_mask)
    
    mask_str = "Seq_" + get_mask_str(seq_mask) + "-sent_" + get_mask_str(sent_mask)
    all_predictions["probe_mask_" + mask_str] = {"truth": truth, "label": predictions}
    
    for k in probe_res:
        res[k + "_probe_mask_" + mask_str] = probe_res[k]


#_____________________________________________________________________

def check_model(args, part_name):
    return part_name in args.sys
