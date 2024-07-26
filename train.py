
import os

import logging

import torch

import numpy as np

np.random.seed(1)
torch.manual_seed(1)

import utils.losses as losses
import utils.misc as misc
import utils.embeddings as embeddings
import utils.data_extractor as data_extractor
import test
import run_funcs

def training(trainloader, validloader, model, exp_type, optimizer, betas, args, model_dir):
    model = model.to(args.device)
    os.makedirs(model_dir, exist_ok=True)

    min_valid_loss = np.inf

    print("Training for {} epochs".format(args.epochs))

    for epoch in range(args.epochs):
        train_loss = 0.0
        model.train()

        logging.info("\n\nEPOCH n° {}".format(epoch+1))
        for _idx, train_batch in enumerate(trainloader):

            (seq, y, truth) = train_batch[:3]
            optimizer.zero_grad()
            model_output = model(seq.to(args.device))
                            
            loss = losses.compute_vae_loss(model, seq, y, truth, model_output, betas, args)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            del model_output, seq, y, loss

        valid_loss = 0.0
        model.eval()

        logging.info("Validation step...")
        with torch.no_grad():
            for _idx, val_batch in enumerate(validloader):
                (val_x, val_y, val_t) = val_batch[:3]                
                model_output = model(val_x.to(args.device))

                loss = losses.compute_vae_loss(model, val_x, val_y, val_t, model_output, betas, args)
                valid_loss += loss

            del model_output, val_x, val_y, loss

        train_loss /= len(trainloader)
        valid_loss /= len(validloader)
        logging.info(f'Epoch: {epoch+1}/{args.epochs}.. Training loss: {train_loss}.. Validation Loss: {valid_loss}') 

        if min_valid_loss > valid_loss:
            logging.info("Validation Loss Decreased from " + str(min_valid_loss) + " to " + str(valid_loss) +  ".\n Saving the model to " + str(model_dir) + "/best_model_" + str(exp_type) + ".pth")
            min_valid_loss = valid_loss
            run_funcs.save_model(model, model_dir + "/best_model_" + exp_type + ".pth")

    misc.save_config(model_dir, args)

    logging.info("\n\nTesting final model on dev data ...")
    test.testing_one(validloader, model, args.device, args.result_keys, args, latent_part = "latent_vec", data_type=args.type)
    
    return model_dir + "/best_model_" + exp_type + ".pth"



##_________________________________________________________________________________________
## a training run

def train_run(exp_nr, model, model_path, train_dirs, train_percs, train_on, args):
    
    # ADAM optimizers
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    exp_type = "_train_" + train_on + "_expNr_" + str(exp_nr+1)
    
    logging.info("Training model {} on {} data".format(model_path, train_on))
    train_data = embeddings.load_embeddings(train_dirs[train_on], train_percs[train_on], args.indices)

    logging.info("checking validation data settings: {}".format(args.valid))
    if args.valid != "":
        valid_data = misc.get_validation_data(train_on, train_dirs, args)
        model_file, trainloader = train_model(model, model_path, exp_type, optimizer, train_data, args, valid_data=valid_data)
    else:
        model_file, trainloader = train_model(model, model_path, exp_type, optimizer, train_data, args)

    return model_file, exp_type, trainloader



##______________________________________________________________________________________
## loading the data and training the model


def train_model(model, model_path, exp_type, optimizer, train_data, args, valid_data=None):
        
    (train_part, valid_part) = make_train_valid(train_data, valid_data, args)

    trainloader = data_extractor.data_loader(train_part, int(args.batch_size), True)
    validloader = data_extractor.data_loader(valid_part, int(args.batch_size), True)

    logging.info("\n\nTraining...")
    logging.info("\ttraining data: {} instances ({} batches) \n\tvalidation data: {} instances  ({} batches)".format(len(train_part[0]), len(trainloader), len(valid_part[0]), len(validloader)))

    return training(trainloader, validloader, model, exp_type, optimizer, [args.beta, args.beta_sent], args, model_path), trainloader
 
##____________________________________________________________________________________________
## separating train and validation data


def make_train_valid(train_data, valid_data, args):

    N = len(train_data[0])
    split = misc.get_randomized_inds(N, int(N * (1-args.valid_perc)))

    if valid_data is None or len(valid_data) == 0:
        valid_data = [[] for i in range(len(train_data))]

    for i in range(len(train_data)):
        valid_data[i].extend([train_data[i][split[x]] for x in range(N) if split[x] == 0])
        train_data[i] = [train_data[i][split[x]] for x in range(N) if split[x] == 1]

    return (train_data, valid_data)

