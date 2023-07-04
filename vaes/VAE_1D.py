
import math

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_size):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim    
        self.nc = embed_dim
        # Size of the sequence
        self.seq_size = seq_size

        self.hidden_dim = int(self.nc * 2)

        self.lin1 = nn.Linear(self.nc*self.seq_size, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin3 = nn.Linear(self.hidden_dim, self.latent_dim*2)
        

    def forward(self, x):
        
        batch_size = x.shape[0]

        x = x.reshape(batch_size, math.prod(x.shape[1:]))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        mean_var = torch.relu(self.lin3(x))  ## sentence representation
        
        mean, logvar = mean_var.view(-1, self.latent_dim, 2).unbind(-1) # separate mean and var in two different vectors
        
        return mean, logvar



class Decoder_answer(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_size):
        super(Decoder_answer, self).__init__()

        self.latent_dim = latent_dim
        self.nc = embed_dim
        # Size of the sequence
        self.seq_size = seq_size

        self.hidden_dim = int(self.nc / 2)
    
        self.lin3 = nn.Linear(self.hidden_dim, self.nc)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin1 = nn.Linear(self.latent_dim, self.hidden_dim)


    def forward(self, x):

        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        
        return x



class VariationalAutoencoder(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_size, sampler):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(embed_dim, latent_dim, seq_size)
        self.decoder_answer = Decoder_answer(embed_dim, latent_dim, seq_size)
        self.sampler = sampler


    def forward(self, x):
        # encoder + reparametrization = latent variable
        mean, logvar = self.encoder(x)
        
        #reparametrization
        latent_vec = self.sampler(mean, logvar) 
        
        # decoder
        answer = self.decoder_answer(latent_vec).squeeze()
                
        return {"output": answer, "latent_vec": latent_vec, "mean" : mean, "logvar" : logvar}
    
    
    def getinfo(self):
        return "VAE_1D"
