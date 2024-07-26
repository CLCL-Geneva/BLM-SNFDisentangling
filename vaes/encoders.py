'''
Created on Feb 28, 2024

@author: vivi
'''

import sys

import math

import numpy as np

import torch
import torch.nn as nn

from utils.mask_weights import MaskedLinear
from utils.misc import get_padd

np.random.seed(1)
torch.manual_seed(1)
#torch.use_deterministic_algorithms(True)


class Encoder(nn.Module):
    def __init__(self, embed_dim, latent_disc, latent_cont, kernel_size, stride, hidden_channels = 32, x_dim = 32, device = "cuda:0"):
        super(Encoder, self).__init__()

        self.device = device

        self.embed_dim = embed_dim
        self.latent_disc = latent_disc
        self.latent_cont = latent_cont
        self.latent_dim = self.latent_disc + 2 * self.latent_cont

        self.hidden_ch = hidden_channels
            
        self.x_dim = x_dim 
        self.y_dim = int(self.embed_dim/self.x_dim)
 
        self.kernel_size = kernel_size 
        self.stride = stride


    def param_info(self):
        return str(self.x_dim) + "x" + str(self.y_dim) + "_k-" + "x".join(map(str,list(self.kernel_size))) + "_s-" + "x".join(map(str,list(self.stride)))
    
    def getinfo(self):
        return "enc_" + self.param_info()
    

##________________________________________________________________________________

'''
Sequence encoders
'''
    
    
class Encoder_3D(Encoder):
    def __init__(self, embed_dim, latent_disc, latent_cont, seq_size, kernel_size = [3, 15, 15], stride = [1, 10, 10], hidden_channels = 32, x_dim = 32, device = "cuda:0"):
        super(Encoder_3D, self).__init__(embed_dim, latent_disc, latent_cont, kernel_size, stride, hidden_channels, x_dim, device)

        self.seq_size = seq_size
                           
        (kseq, kx, ky) = self.kernel_size
        (sseq, sx, sy) = self.stride
        
        self.padding = (get_padd(self.seq_size, kseq, sseq),get_padd(self.x_dim, kx, sx), get_padd(self.y_dim, ky, sy))

        self.nx = int((self.x_dim + 2*self.padding[1] - kx)/stride[1]+1)
        self.ny = int((self.y_dim + 2*self.padding[2] - ky)/stride[2]+1)
        self.nseq = int((self.seq_size + 2*self.padding[0] - kseq)/stride[0]+1)
        self.N = self.nx * self.ny * self.nseq * self.hidden_ch

        self.conv = nn.Conv3d(1, self.hidden_ch, kernel_size=self.kernel_size, stride=stride, padding=self.padding)
        self.lin = nn.Linear(self.N, self.latent_dim)


    def forward(self, x): 
        
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, self.seq_size, self.x_dim, self.y_dim)
        x = torch.relu(self.conv(x))
        x = x.view(batch_size, -1)
        return self.lin(x)

    
class Encoder_3D_subnets(Encoder_3D):
    
    def __init__(self, embed_dim, latent_disc, latent_cont, seq_size, cont_latent = 5, kernel_size = [3, 15, 15], stride = [1, 10, 10], hidden_channels = 32, x_dim = 32, device = "cuda:0" ):
        super(Encoder_3D_subnets, self).__init__(embed_dim, latent_disc, latent_cont, seq_size, kernel_size, stride, hidden_channels, x_dim, device)
        
        self.nr_masked_units = self.latent_disc + self.latent_cont
        
        self.lin_mask = MaskedLinear(self.N, self.nr_masked_units, self.nr_masked_units, device=self.device)
        self.lin_sd = nn.Linear(self.N, self.latent_cont)


    def forward(self, x):        
        batch_size = x.shape[0]
    
        x = x.view(batch_size, 1, self.seq_size, self.x_dim, self.y_dim)
        x = self.conv(x)
        x = torch.tanh(x.view(batch_size, -1))
        x_masked = self.lin_mask(x)
        x_sd = self.lin_sd(x).to(self.device)

        return torch.cat((x_masked, x_sd), dim=1)

    def getinfo(self):
        return "enc-sparse_" + self.param_info()



##________________________________________________________________________________________________

'''
Sentence encoders
'''
    
class Encoder_2D(Encoder):
    def __init__(self, embed_dim, latent_disc, latent_cont, kernel_size = [15, 15], stride = [10, 10], hidden_channels = 32, x_dim = 32, device = "cuda:0" ):
        super(Encoder_2D, self).__init__(embed_dim, latent_disc, latent_cont, kernel_size, stride, hidden_channels, x_dim, device)
                
        (kx, ky) = self.kernel_size
        (sx, sy) = self.stride

        self.padding = (get_padd(self.x_dim, kx, sx), get_padd(self.y_dim, ky, sy))

        self.nx = int((self.x_dim + 2*self.padding[0] - kx)/stride[0]+1)
        self.ny = int((self.y_dim + 2*self.padding[1] - ky)/stride[1]+1)
        self.N = self.nx * self.ny * self.hidden_ch
        
        self.conv = nn.Conv2d(1, self.hidden_ch, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)        
        self.lin = nn.Linear(self.N, self.latent_dim)

        print("\tpadding = {} / nx = {} / ny = {} / N = {}".format(self.padding, self.nx, self.ny, self.N))

    def forward(self, x):        
        batch_size = x.shape[0]

        x = x.view(batch_size, 1, self.x_dim, self.y_dim)
        x = torch.relu(self.conv(x)) 
        x = x.view(batch_size, -1)
        
        return self.lin(x)

    def encode(self, x):
        batch_size = x.shape[0]

        x = x.view(batch_size, 1, self.x_dim, self.y_dim)
        x_conv = torch.relu(self.conv(x))
        x_lin = self.lin(x_conv.view(batch_size, -1))

        return(x_lin, x_conv, x_lin[:,self.latent_disc + self.latent_cont])


class Encoder_2D_subnets(Encoder_2D):
    def __init__(self, embed_dim, latent_disc, latent_cont, kernel_size = [15, 15], stride = [10, 10], hidden_channels = 32, x_dim = 32, device = "cuda:0" ):
        super(Encoder_2D_subnets, self).__init__(embed_dim, latent_disc, latent_cont, kernel_size, stride, hidden_channels, x_dim, device)

        self.nr_masked_units = self.latent_disc + self.latent_cont
                     
        self.lin_mask = MaskedLinear(self.N, self.nr_masked_units, self.nr_masked_units, device=self.device)
        self.lin_sd = nn.Linear(self.N, self.latent_cont)



    def forward(self, x):        

        batch_size = x.shape[0]
    
        x = x.view(batch_size, 1, self.x_dim, self.y_dim)
        x_conv = self.conv(x)
        x_conv = torch.relu(x_conv.view(batch_size, -1))
        x_masked = self.lin_mask(x_conv)
        x_sd = self.lin_sd(x_conv).to(self.device)

        return torch.cat((x_masked, x_sd), dim=1)
        

    def encode(self, x):        

        batch_size = x.shape[0]
    
        x = x.view(batch_size, 1, self.x_dim, self.y_dim)
        x_conv = self.conv(x)
        x_conv = torch.relu(x_conv.view(batch_size, -1))
        x_masked = self.lin_mask(x_conv)
        x_sd = self.lin_sd(x_conv).to(self.device)

        return (torch.cat((x_masked, x_sd), dim=1), x_conv, x_masked)

##___________________________________________________________

class Encoder_1D(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_size):
        super(Encoder_1D, self).__init__()

        self.latent_dim = latent_dim
    
        # Bert vector dim: 768
        self.nc = embed_dim
        # Size of the sequence
        self.seq_size = seq_size

        self.hidden_dim = int(self.nc * 2)
    
        # Fully connected layers
        self.lin1 = nn.Linear(self.nc*self.seq_size, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin3 = nn.Linear(self.hidden_dim, self.latent_dim*2)
        

    def forward(self, x):
        
        #print("x shape: {}".format(x.shape))
        batch_size = x.shape[0]

        x = x.reshape(batch_size, math.prod(x.shape[1:]))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        mean_var = torch.relu(self.lin3(x))  ## sentence representation
        
        mean, logvar = mean_var.view(-1, self.latent_dim, 2).unbind(-1) # separate mean and var in two different vectors
        
        return mean, logvar



class Encoder_1D_seq(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_size):
        super(Encoder_1D_seq, self).__init__()

        self.latent_dim = latent_dim
        self.nc = embed_dim    
        self.hidden_dim = int(self.nc * 2)

        self.seq_size = seq_size
        
        (k1seq, k1x) = (3, 3)
        (k2seq, k2x) = (3, 3)
        (k3seq, k3x) = (3, 3)
        s = 1
        
        # Fully connected layers
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(k1seq, k1x), stride=(s,1)) 
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(k2seq, k2x)) 
        self.conv3 = nn.Conv2d(8, 16, kernel_size=(k3seq, k3x))
        
        #ny = self.seq_size - k1y - k2y - k3y + 3
        nx = int((self.nc - k1x - k2x - k3x + 3))
        self.fc = nn.Linear(nx * 16, self.latent_dim * 2)
        

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.view(batch_size, 1, self.seq_size, self.nc)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))  ## sentence representation
        x = x.view(batch_size, -1)
    
        return self.fc(x)


class Encoder_1D_sent(nn.Module):
    def __init__(self, embed_dim, latent_dim, latent_dim_disc=0):
        super(Encoder_1D_sent, self).__init__()

        self.latent_dim = latent_dim
        self.latent_dim_disc = latent_dim_disc

        self.embed_dim = embed_dim
        self.hidden = int(self.embed_dim / 2)

        self.lin1 = nn.Linear(self.embed_dim, self.hidden)
        self.lin2 = nn.Linear(self.hidden, self.latent_dim * 2 + self.latent_dim_disc)

    def forward(self, x):
        batch_size = x.shape[0]

        out = torch.relu(self.lin1(x))
        out = out.view(batch_size, -1)
        return self.lin2(out)

    def encode(self, x):
        batch_size = x.shape[0]

        out = torch.relu(self.lin1(x))
        out = out.view(batch_size, -1)
        x_lin = self.lin2(out)
        return(x_lin, out, x_lin[:,self.latent_dim_disc + self.latent_dim])
