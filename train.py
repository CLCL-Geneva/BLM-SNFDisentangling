
import os

import logging

import torch
import numpy as np

np.random.seed(1)
torch.manual_seed(1)

import utils.losses as losses
import utils.utils as utils


def training(trainloader, validloader, model, exp_type, epochs, optimizer, betas, args, model_dir):    
    model = model.to(args.device)
    os.makedirs(model_dir, exist_ok=True)

    min_valid_loss = np.inf

    train_loss_list = []
    valid_loss_list = []

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()

        logging.info("\n\nEPOCH n° {}".format(epoch+1))
        for _idx, (seq, y, truth) in enumerate(trainloader):

            optimizer.zero_grad()

            if args.shuffle:   ## shuffle sentences in sequence
                idx = torch.randperm(seq.shape[1])
                seq = seq[:,idx]

            model_output = model(seq.to(args.device))
                            
            loss = losses.compute_vae_loss(seq, y, truth, args.baseline, model_output, betas, args.latent, args.latent_sent_dim, args.sent_emb_dim, args.batch_size, args.device)
                        
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            del model_output, seq, y, loss

        valid_loss = 0.0
        model.eval()

        logging.info("Validation step...")
        with torch.no_grad():
            for _idx, (val_x, val_y, val_t) in enumerate(validloader):
                if args.shuffle:
                    idx = torch.randperm(val_x.shape[1])
                    val_x = val_x[:,idx]
                
                model_output = model(val_x.to(args.device))
                                
                loss = losses.compute_vae_loss(val_x, val_y, val_t, args.baseline, model_output, betas, args.latent, args.latent_sent_dim, args.sent_emb_dim, args.batch_size, args.device)

                valid_loss += loss

            del model_output, val_x, val_y, loss

        train_loss /= len(trainloader)
        valid_loss /= len(validloader)
        logging.info(f'Epoch: {epoch+1}/{epochs}.. Training loss: {train_loss}.. Validation Loss: {valid_loss}') 

        if min_valid_loss > valid_loss:
            logging.info("Validation Loss Decreased from " + str(min_valid_loss) + " to " + str(valid_loss) + ".\n Saving the model to " + str(model_dir) + "/best_model_" + str(exp_type) + ".pth")
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), model_dir + "/best_model_" + exp_type + ".pth")

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

    utils.save_config(model_dir, args)
    return model_dir + "/best_model_" + exp_type + ".pth"

