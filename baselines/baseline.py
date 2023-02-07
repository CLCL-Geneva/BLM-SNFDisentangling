'''
'''

import math

import torch
import torch.nn as nn

class BaselineFFNN(nn.Module):

    def __init__(self, embed_dim, seq_size):
        '''
        Constructor
        '''
        super(BaselineFFNN, self).__init__()

        # Bert vector dim: 768
        self.nc = embed_dim
        # Size of the sequence
        self.seq_size = seq_size

        self.hidden_dim = int(self.nc * 2)
    
        # Fully connected layers
        self.lin1 = nn.Linear(self.nc*self.seq_size, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin3 = nn.Linear(self.hidden_dim, self.nc)
        
        print(self)


    def forward(self, x):
        batch_size = x.shape[0]

        x = x.reshape(batch_size, x.shape[1]*x.shape[2])
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))  ## sentence representation
        
        return {"output": x}
    
    def getinfo(self):
        return "Baseline-FFNN"
    

        

class BaselineCNN(nn.Module):

    def __init__(self, embed_dim, seq_size):
        super(BaselineCNN, self).__init__()

        # Bert vector dim: 768
        self.nc = embed_dim
        # Size of the sequence
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
        self.fc = nn.Linear(nx * 16, self.nc)
        
        print(self)


    def forward(self, x):
        batch_size = x.shape[0]
        
        x = x.view(batch_size, 1, self.seq_size, self.nc)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))  ## sentence representation
         
        x = x.view(batch_size, -1)
        x = torch.relu(self.fc(x))
        
        return {"output": x}
     
    def getinfo(self):
        return "Baseline_CNN"
   
