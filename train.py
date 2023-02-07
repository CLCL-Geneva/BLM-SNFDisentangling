
import os

import torch
import torch.nn as nn
import numpy as np

np.random.seed(1)
torch.manual_seed(1)

import utils.losses as losses


#train.training(trainloader, validloader, model, exp_type, epochs, optimizer, args.sent_emb_dim, args.test, float(args.lr), model_path)
def training(trainloader, validloader, model, exp_type, epochs, optimizer, args, model_dir):
    model = model.to(args.device)
    os.makedirs(model_dir, exist_ok=True)

    min_valid_loss = np.inf

    train_loss_list = []
    valid_loss_list = []

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()

        print("\n\nEPOCH n°", epoch+1)
        for _idx, (seq_x, y, truth) in enumerate(trainloader):

            optimizer.zero_grad()
            
            seq = seq_x.squeeze(2)  
            model_output = model(seq.to(args.device))
                            
            loss = losses.compute_vae_loss(seq, y, truth, model_output, args.device)
            loss.backward()
            train_loss += loss.item()
            
            optimizer.step()

            del model_output, seq, loss, seq_x, y

        valid_loss = 0.0
        model.eval()

        print("Validation step...")
        with torch.no_grad():
            for _idx, (val_x, val_y, val_t) in enumerate(validloader):
                seq = val_x.squeeze(2)                
                model_output = model(seq.to(args.device))
                                
                loss = losses.compute_vae_loss(seq, val_y, val_t, model_output, args.device)
                valid_loss += loss

            del model_output, seq, loss, val_x, val_y

        train_loss /= len(trainloader)
        valid_loss /= len(validloader)
        print(f'Epoch: {epoch+1}/{epochs}.. Training loss: {train_loss}.. Validation Loss: {valid_loss}') 

        if min_valid_loss > valid_loss:
            print("Validation Loss Decreased from ", min_valid_loss, " to ", valid_loss, ".\n Saving the model.")
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), model_dir + "/best_model_" + exp_type + ".pth")

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

    return model_dir + "/best_model_" + exp_type + ".pth"

