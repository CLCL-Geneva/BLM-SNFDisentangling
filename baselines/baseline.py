'''
'''

import math

import torch
import torch.nn as nn

from utils.misc import get_padd


class BaselineFFNN(nn.Module):

    def __init__(self, embed_dim, seq_size):
        '''
        Constructor
        '''
        super(BaselineFFNN, self).__init__()

        self.nc = embed_dim
        self.seq_size = seq_size

        self.hidden_dim = int(self.nc * 2)
    
        self.lin1 = nn.Linear(self.nc*self.seq_size, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin3 = nn.Linear(self.hidden_dim, self.nc)
        
        print(self)


    def forward(self, x, mask=False):
        batch_size = x.shape[0]

        x = x.reshape(batch_size, x.shape[1]*x.shape[2])
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))  ## sentence representation
        
        return {"output": x}
    
    def getinfo(self):
        return "Baseline-FFNN"
    

        

class BaselineCNN_1DxSeq(nn.Module):

    def __init__(self, embed_dim, seq_size):
        super(BaselineCNN_1DxSeq, self).__init__()

        self.nc = embed_dim
        self.seq_size = seq_size

        (k1seq, k1x) = (3, 3)
        (k2seq, k2x) = (3, 3)
        (k3seq, k3x) = (3, 3)
        s = 1
        
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(k1seq, k1x), stride=(s,1)) 
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(k2seq, k2x)) 
        self.conv3 = nn.Conv2d(8, 16, kernel_size=(k3seq, k3x))
        
        nx = int((self.nc - k1x - k2x - k3x + 3))
        self.fc = nn.Linear(nx * 16, self.nc)
        
        print(self)


    def forward(self, x, mask=False):
        batch_size = x.shape[0]
        
        x = x.view(batch_size, 1, self.seq_size, self.nc)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))  ## sentence representation
         
        x = x.view(batch_size, -1)
        x = torch.relu(self.fc(x))
        
        return {"output": x}
     
    def getinfo(self):
        return "Baseline_CNN_1DxSeq"
   

hid_ch = 32
emb_x_dim = 32  #32

kernel_size = (3, 15, 15)  #(3, 10, 10)
stride = (1, 1, 1)

class BaselineCNN(nn.Module):
    def __init__(self, embed_dim, seq_size):
        super(BaselineCNN, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_ch = hid_ch
        
        # Size of the sequence
        self.seq_size = seq_size
    
        if self.embed_dim == 768:        ## 768 for sentence embedding, otherwise an attention matrix
            self.x_dim = emb_x_dim 
        else:
            self.x_dim = int(math.sqrt(self.embed_dim))        

        self.y_dim = int(self.embed_dim/self.x_dim)
    
        self.kernel_size = kernel_size 
        self.stride = stride
                
        (kseq, kx, ky) = kernel_size
        (sseq, sx, sy) = stride

        self.padding = (get_padd(self.seq_size, kseq, sseq),get_padd(self.x_dim, kx, sx), get_padd(self.y_dim, ky, sy))

        self.nx = int((self.x_dim + 2*self.padding[1] - kx)/stride[1]+1)
        self.ny = int((self.y_dim + 2*self.padding[2] - ky)/stride[2]+1)
        self.nseq = int((self.seq_size + 2*self.padding[0] - kseq)/stride[0]+1)
        self.N = self.nx * self.ny * self.nseq * self.hidden_ch

        self.conv = nn.Conv3d(1, self.hidden_ch, kernel_size=self.kernel_size, stride=stride, padding=self.padding)
        self.lin = nn.Linear(self.nx * self.ny * self.nseq * self.hidden_ch, self.embed_dim)
        
        print(self)


    def forward(self, x, mask=False):
        
        batch_size = x.shape[0]
    
        x = x.view(batch_size, 1, self.seq_size, self.x_dim, self.y_dim)
        x = torch.relu(self.conv(x))        
        x = x.view(batch_size, -1)
        x = torch.relu(self.lin(x))

        return {"output": x}
    
    def getinfo(self):
        return "Baseline_CNN_" + str(self.x_dim) + "x" + str(self.y_dim)

        
