
import os

import torch
import numpy as np

import utils.losses as losses



def training(trainloader, validloader, model, e, optimizer, train_size, eval_, lr, device, data_type, model_dir):
    model = model.to(device)
    os.makedirs(model_dir, exist_ok=True)

    min_valid_loss = np.inf

    train_loss_list = []
    valid_loss_list = []

    for epoch in range(e):
        train_loss = 0.0
        model.train()

        print("\n\nEPOCH n°", epoch+1)
        for _idx, (seq_x, y, truth) in enumerate(trainloader):

            optimizer.zero_grad()
            
            seq = seq_x.squeeze(2).transpose(1, 2)  # from (1, 7, 1, 768) to (1, 768, 7)
            output = model(seq.to(device))

            loss = losses.max_margin(output.to(device), y.to(device), truth.to(device), device)
            #loss = losses.reconstruction_loss(output.to(device), y.to(device), truth.to(device), device)
            loss.backward()
            train_loss += loss
            
            optimizer.step()

            del output, seq, loss, seq_x, y, truth

        valid_loss = 0.0
        model.eval()

        print("Validation step...")
        with torch.no_grad():
            for _idx, (val_x, val_y, val_t) in enumerate(validloader):
                seq = val_x.squeeze(2).transpose(1, 2) # from (20, 7, 1, 768) to (20, 768, 7)
                output = model(seq.to(device))

                loss = losses.max_margin(output.to(device), val_y.to(device), val_t.to(device), device)

                valid_loss += loss

        del output, seq, loss, val_x, val_y

        train_loss /= len(trainloader)
        valid_loss /= len(validloader)
        print(f'Epoch: {epoch+1}/{e}.. Training loss: {train_loss}.. Validation Loss: {valid_loss}') 

        if min_valid_loss > valid_loss:
            print("Validation Loss Decreased from ", min_valid_loss, " to ", valid_loss, ".\n Saving the model.")
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), model_dir + "/best_model.pth")

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

    return train_loss_list, valid_loss_list
