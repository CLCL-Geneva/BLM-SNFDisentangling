
import os
import logging

import torch
import torch.nn as nn
import numpy as np

np.random.seed(1)
torch.manual_seed(1)

import losses
import utils_sr as utils

import test

def training(trainloader, validloader, model, epochs, optimizer, sent_emb_dim, lr, beta, latent_sent_dim, device, batch_size, model_dir, args):
    model = model.to(device)
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir + "/best_model_VAE.pth"

    min_valid_loss = np.inf

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()

        print("\n\nEPOCH n°", epoch+1)
        for _idx, batch in enumerate(trainloader):

            (x, pos, negs) = batch[:3]

            optimizer.zero_grad()
            model_output = model(x.to(device), pos=pos.to(device))
                            
            loss = losses.compute_vae_loss(x, pos, negs, model_output, beta, latent_sent_dim, sent_emb_dim, batch_size, device, model, args)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        valid_loss = 0.0
        model.eval()

        print("Validation step...")
        with torch.no_grad():
            for _idx, batch in enumerate(validloader):
                
                (val_x, val_pos, val_negs) = batch[:3]
                
                model_output = model(val_x.to(device), pos=val_pos.to(device))
                                
                loss = losses.compute_vae_loss(val_x, val_pos, val_negs, model_output, beta, latent_sent_dim, sent_emb_dim, batch_size, device, model, args)
                valid_loss += loss

        train_loss /= len(trainloader)
        valid_loss /= len(validloader)
        print(f'Epoch: {epoch+1}/{epochs}.. Training loss: {train_loss}.. Validation Loss: {valid_loss}') 

        if min_valid_loss > valid_loss:
            print("Validation Loss Decreased from ", min_valid_loss, " to ", valid_loss, ".\n Saving the model to {}".format(model_path))
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)

    return model_path

