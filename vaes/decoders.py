'''
Created on Feb 28, 2024

@author: vivi
'''

import math

import numpy as np

import torch
import torch.nn as nn

from utils.mask_weights import RevMaskedLinear
from utils.misc import get_padd, get_out_padd

np.random.seed(1)
torch.manual_seed(1)
#torch.use_deterministic_algorithms(True)

class Decoder(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_size, kernel_size, stride, hidden_channels = 32, x_dim = 32, y_dim = -1, device = "cuda:0"):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_ch = hidden_channels
        self.embed_dim = embed_dim

        self.seq_size = seq_size

        self.x_dim = x_dim
        if y_dim == -1:
            self.y_dim = int(self.embed_dim/self.x_dim)
        else:
            self.y_dim = y_dim
    
        self.kernel_size = kernel_size
        self.stride = stride    

    def param_info(self):
        return str(self.x_dim) + "x" + str(self.y_dim) + "_k-" + "x".join(map(str,list(self.kernel_size))) + "_s-" + "x".join(map(str,list(self.stride)))

    def getinfo(self):
        return "dec_" + self.param_info()


## ____________________________________________________________
## decoders for sequences

class Decoder_answer_3D(Decoder):            
    def __init__(self, embed_dim, latent_dim, seq_size, kernel_size, stride, hidden_channels = 32, x_dim = 32, device = "cuda:0"):
        super(Decoder_answer_3D, self).__init__(embed_dim, latent_dim, seq_size, kernel_size, stride, hidden_channels, x_dim = x_dim, device = device)
        
        self.kernel_size[0] = 1
        
        (kseq, kx, ky) = self.kernel_size
        (sseq, sx, sy) = self.stride
                
        self.padding = (get_padd(self.seq_size, kseq, sseq),get_padd(self.x_dim, kx, sx), get_padd(self.y_dim, ky, sy))

        self.nseq = int((self.seq_size +2*self.padding[0] - kseq)/stride[0] + 1)
        self.nx = int((self.x_dim + 2*self.padding[1] - kx)/stride[1]+1)
        self.ny = int((self.y_dim + 2*self.padding[2] - ky)/stride[2]+1)
        self.N = self.nx * self.ny * self.nseq * self.hidden_ch
           
        self.output_padding = (get_out_padd(self.seq_size, self.nseq, kseq, stride[0], self.padding[0]), 
                          get_out_padd(self.x_dim, self.nx, kx, stride[1], self.padding[1]), 
                          get_out_padd(self.y_dim, self.ny, ky, stride[2], self.padding[2]))
           
        self.lin = nn.Linear(self.latent_dim, self.N)
        self.convT = nn.ConvTranspose3d(self.hidden_ch, 1, kernel_size=(kseq,kx,ky), stride=stride, padding=self.padding, output_padding=self.output_padding)


    def forward(self, z):

        batch_size = z.shape[0]
        
        z = torch.relu(self.lin(z))
        z = z.view(batch_size, self.hidden_ch, self.nseq, self.nx, -1)
        return self.convT(z)

        

    
class Decoder_answer_3D_subnets(Decoder_answer_3D):
    
    def __init__(self, embed_dim, latent_dim, seq_size = 1, kernel_size = [3, 15, 15], stride = [1, 10, 10], hidden_channels = 32, x_dim = 32, device = "cuda:0"):
        super(Decoder_answer_3D_subnets, self).__init__(embed_dim, latent_dim, seq_size, kernel_size, stride, hidden_channels, x_dim, device)

        self.lin = RevMaskedLinear(self.latent_dim, self.N, self.N, device=device)
        self.convT = nn.ConvTranspose3d(self.hidden_ch, 1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=self.output_padding)


    def forward(self, z):
        batch_size = z.shape[0]

        z = torch.relu(self.lin(z))
        z = z.view(batch_size, self.hidden_ch, self.nseq, self.nx, -1)       
        return self.convT(z)
    
    def getinfo(self):
        return "dec-sparse_" + self.param_info()



#_________________________________

class Decoder_mirror_3D(Decoder):
    def __init__(self, embed_dim, latent_dim, seq_size, kernel_size = [3, 15, 15], stride = [1, 10, 10], hidden_channels = 32, x_dim = 32, device = "cuda:0"):
        super(Decoder_mirror_3D, self).__init__(embed_dim, latent_dim, seq_size, kernel_size, stride, hidden_channels, x_dim, device)

        (kseq, kx, ky) = self.kernel_size
        (sseq, sx, sy) = self.stride
                
        self.padding = (get_padd(self.seq_size, kseq, sseq),get_padd(self.x_dim, kx, sx), get_padd(self.y_dim, ky, sy))

        self.nseq = int((self.seq_size +2*self.padding[0] - kseq)/stride[0] + 1)
        self.nx = int((self.x_dim + 2*self.padding[1] - kx)/stride[1]+1)
        self.ny = int((self.y_dim + 2*self.padding[2] - ky)/stride[2]+1)
        self.N = self.nx * self.ny * self.nseq * self.hidden_ch
           
        self.output_padding = (get_out_padd(self.seq_size, self.nseq, kseq, stride[0], self.padding[0]), 
                          get_out_padd(self.x_dim, self.nx, kx, stride[1], self.padding[1]), 
                          get_out_padd(self.y_dim, self.ny, ky, stride[2], self.padding[2]))
           
        self.lin = nn.Linear(self.latent_dim, self.N)
        self.convT = nn.ConvTranspose3d(self.hidden_ch, 1, kernel_size=(kseq,kx,ky), stride=stride, padding=self.padding, output_padding=self.output_padding)
    

    def forward(self, z):

        batch_size = z.shape[0]
        
        x = torch.relu(self.lin(z)) 
        x = x.view(batch_size, self.hidden_ch, self.nseq,  self.nx, -1)
        return self.convT(x)

    def getinfo(self):
        return "dec-mirror-seq_" + self.param_info()

    
##_______________________________________________________________________________________
## sentence decoder    
    
class Decoder_2D(Decoder):
    def __init__(self, embed_dim, latent_dim, kernel_size = [15, 15], stride = [10, 10], hidden_channels = 32, x_dim = 32, y_dim = -1, device = "cuda:0"):
        super(Decoder_2D, self).__init__(embed_dim, latent_dim, 1, kernel_size, stride, hidden_channels, x_dim, y_dim, device)

        (kx, ky) = self.kernel_size
        (sx, sy) = self.stride
                
        self.padding = (get_padd(self.x_dim, kx, sx), get_padd(self.y_dim, ky, sy))
       
        self.nx = int((self.x_dim + 2*self.padding[0] - kx)/stride[0]+1)
        self.ny = int((self.y_dim + 2*self.padding[1] - ky)/stride[1]+1)
        self.N = self.nx * self.ny * self.hidden_ch 
           
        self.output_padding = ( 
                          get_out_padd(self.x_dim, self.nx, kx, stride[0], self.padding[0]),
                          get_out_padd(self.y_dim, self.ny, ky, stride[1], self.padding[1]))
           
        self.lin = nn.Linear(self.latent_dim, self.N)
        self.convT = nn.ConvTranspose2d(self.hidden_ch, 1, kernel_size=(kx, ky), stride=self.stride, padding=self.padding, output_padding=self.output_padding) 


    def forward(self, z):
        batch_size = z.shape[0]

        z = self.lin(z)
        z = z.view(batch_size, self.hidden_ch, self.nx, -1)
        return torch.relu(self.convT(z))

    def getinfo(self):
        return "sent_dec_" + self.param_info()



class Decoder_2D_subnets(Decoder_2D):
    def __init__(self, embed_dim, latent_dim, kernel_size = [15, 15], stride = [10, 10], hidden_channels = 32, x_dim = 32, device = "cuda:0"):
        super(Decoder_2D_subnets, self).__init__(embed_dim, latent_dim, kernel_size, stride, hidden_channels, x_dim = x_dim, device = device)

        self.lin = RevMaskedLinear(self.latent_dim, self.N, self.N, device=device)


    def forward(self, z):
        batch_size = z.shape[0]

        z = torch.relu(self.lin(z))
        z = z.view(batch_size, self.hidden_ch, self.nx, -1)
        return self.convT(z)


    def getinfo(self):
        return "sent_dec_sparse_" + self.param_info()
    
    
    
##______________________________________________________________________


class Decoder_answer_1D(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_size):
        super(Decoder_answer_1D, self).__init__()

        self.latent_dim = latent_dim
    
        self.nc = embed_dim
        self.seq_size = seq_size

        self.hidden_dim = int(self.nc / 2)
    
        # Fully connected layers
        self.lin3 = nn.Linear(self.hidden_dim, self.nc)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin1 = nn.Linear(self.latent_dim, self.hidden_dim)


    def forward(self, x):

        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        
        return x



class Decoder_answer_1D_seq(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_size):
        super(Decoder_answer_1D_seq, self).__init__()

        self.latent_dim = latent_dim
        self.nc = embed_dim    
        self.hidden_dim = int(self.nc * 2)
        # Size of the sequence
        self.seq_size = seq_size
        (k1seq, k1x) = (1, 3)
        (k2seq, k2x) = (1, 3)
        (k3seq, k3x) = (1, 3)
        s = 1
        
        # Fully connected layers
        self.conv1 = nn.ConvTranspose2d(4, 1, kernel_size=(k1seq, k1x), stride=(s,1)) 
        self.conv2 = nn.ConvTranspose2d(8, 4, kernel_size=(k2seq, k2x)) 
        self.conv3 = nn.ConvTranspose2d(16, 8, kernel_size=(k3seq, k3x))
        
        self.nx = int((self.nc - k1x - k2x - k3x + 3))
        self.fc = nn.Linear(self.latent_dim, self.nx * 16)


    def forward(self, x):
        batch_size = x.shape[0]

        x = torch.relu(self.fc(x))
        x = x.view(batch_size, 16, -1, self.nx)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv2(x))
        return torch.relu(self.conv1(x))  ## sentence representation


class Decoder_1D_sent(nn.Module):
    def __init__(self, embed_dim, latent_dim, latent_dim_disc=0):
        super(Decoder_1D_sent, self).__init__()

        self.latent_dim = latent_dim
        self.latent_dim_disc = latent_dim_disc

        self.embed_dim = embed_dim
        self.hidden = int(self.embed_dim / 2)

        self.lin1 = nn.Linear(self.latent_dim + self.latent_dim_disc, self.hidden)
        self.lin2 = nn.Linear(self.hidden, self.embed_dim)

    def forward(self, z):
        x = self.lin1(z)
        return self.lin2(x)


    