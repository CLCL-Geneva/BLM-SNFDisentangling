import math

import torch
import torch.nn as nn

hid_ch = 32
emb_x_dim = 32  #32

kernel_size = (3, 15, 15)  #(3, 10, 10)

class Encoder(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_size):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = int(embed_dim/2)
        self.hidden_ch = hid_ch
        
        # Size of the sequence
        self.seq_size = seq_size
    
        if embed_dim == 768:        ## 768 for sentence embedding, otherwise an attention matrix
            self.x_dim = emb_x_dim 
        else:
            self.x_dim = int(math.sqrt(embed_dim))        

        self.y_dim = int(embed_dim/self.x_dim)
        
        (k1seq, k1x, k1y) = kernel_size
        n_conv_layers = 1
    
        nx = int((self.x_dim - k1x + n_conv_layers))
        ny = int((self.y_dim - k1y + n_conv_layers))
        nseq = int((self.seq_size - k1seq + n_conv_layers))
           
        self.conv1 = nn.Conv3d(1, self.hidden_ch, kernel_size=(k1seq, k1x, k1y)) 
        self.lin1 = nn.Linear(nx * ny * nseq * self.hidden_ch, self.latent_dim)


    def forward(self, x):
        
        batch_size = x.shape[0]
    
        x = x.view(batch_size, 1, self.seq_size, self.x_dim, self.y_dim)
        x = torch.relu(self.conv1(x))
        x = x.view(batch_size, -1)
                
        return self.lin1(x)


class Decoder_mirror(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_size):
        super(Decoder_mirror, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = int(embed_dim/2)
        self.hidden_ch = hid_ch

        self.seq_size = seq_size

        if embed_dim == 768:        ## 768 for sentence embedding, otherwise an attention matrix
            self.x_dim = emb_x_dim 
        else:
            self.x_dim = int(math.sqrt(embed_dim))        

        self.y_dim = int(embed_dim)/self.x_dim
        
        (k4seq, k4x, k4y) = kernel_size
        n_conv_layers = 1
        
        self.nx = int((self.x_dim - k4x + n_conv_layers))
        self.ny = int((self.y_dim - k4y + n_conv_layers))
        self.nseq = int((self.seq_size - k4seq + n_conv_layers))
           
        self.lin1 = nn.Linear(self.latent_dim, self.nx * self.ny * self.nseq * self.hidden_ch)
        self.convT4 = nn.ConvTranspose3d(self.hidden_ch, 1, kernel_size=(k4seq, k4x, k4y)) 


    def forward(self, z):

        batch_size = z.shape[0]
        
        x = torch.relu(self.lin1(z)) 
        x = x.view(batch_size, self.hidden_ch, self.nseq,  self.nx, -1)
        x = self.convT4(x)
        
        return x



class Decoder_answer(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_size):
        super(Decoder_answer, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = int(embed_dim/2)
        self.hidden_ch = hid_ch

        self.seq_size = 1

        if embed_dim == 768:        ## 768 for sentence embedding, otherwise an attention matrix
            self.x_dim = emb_x_dim 
        else:
            self.x_dim = int(math.sqrt(embed_dim))        

        self.y_dim = int(embed_dim)/self.x_dim
    
    
        (_, k4x, k4y) = kernel_size
        k4seq = 1
        n_conv_layers = 1
       
        self.nx = int((self.x_dim - k4x + n_conv_layers))
        self.ny = int((self.y_dim - k4y + n_conv_layers))
        self.nseq = int((self.seq_size - k4seq + n_conv_layers))
           
        self.lin1 = nn.Linear(self.latent_dim, self.nx * self.ny * self.nseq * self.hidden_ch)
        self.convT4 = nn.ConvTranspose3d(self.hidden_ch, 1, kernel_size=(k4seq, k4x, k4y)) 


    def forward(self, z):

        batch_size = z.shape[0]
        
        x = torch.relu(self.lin1(z))
        x = x.view(batch_size, self.hidden_ch, self.nseq, self.nx, -1)
        x = self.convT4(x)
        
        return x



class VariationalAutoencoder(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_size, sampler):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(embed_dim, sampler.latent_size_in, seq_size)
        self.decoder_mirror = Decoder_mirror(embed_dim, sampler.latent_size_sample, seq_size)
        self.decoder_answer = Decoder_answer(embed_dim, sampler.latent_size_sample, seq_size)
        self.sampler = sampler


    def forward(self, x):
        # encoder + reparametrization = latent variable
        latent_vec = self.encoder(x)
        
        #reparametrization
        latent_sampled_vec, mean, logvar = self.sampler(latent_vec) 
        
        # decoder
        recon_x = self.decoder_mirror(latent_sampled_vec).squeeze()
        answer = self.decoder_answer(latent_sampled_vec).squeeze()
                
        return {"recon_input": recon_x, "output": answer, "latent_vec": latent_sampled_vec, "mean" : mean, "logvar" : logvar}
    
    def getinfo(self):
        return "Dual-VAE_" + str(self.encoder.x_dim) + "x" + str(self.encoder.y_dim) + "_" + self.sampler.getinfo() 
