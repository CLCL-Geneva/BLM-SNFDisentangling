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
    
        (k1seq, k1x, k1y) = kernel_size
        n_conv_layers = 1
    
        nx = int((self.x_dim - k1x + n_conv_layers))
        ny = int((self.y_dim - k1y + n_conv_layers))
        nseq = int((self.seq_size - k1seq + n_conv_layers))
           
        # Convolutional layers
        self.conv1 = nn.Conv3d(1, self.hidden_ch, kernel_size=(k1seq, k1x, k1y)) 

        self.lin1 = nn.Linear(nx * ny * nseq * self.hidden_ch, self.embed_dim)
        
        print(self)


    def forward(self, x, mask=False):
        
        batch_size = x.shape[0]
    
        #print("x before reshape: {}".format(x.shape))
        #print("\treshaping to (1,{},{},{})".format(self.seq_size, self.x_dim, self.y_dim))
        
        #x = x.view(batch_size, 1, x.shape[1], x.shape[2], x.shape[3])
        x = x.view(batch_size, 1, self.seq_size, self.x_dim, self.y_dim)
        #x = torch.reshape(x, (batch_size, 1, self.seq_size, self.x_dim, self.y_dim))
        #print("x after reshape: {}".format(x.shape))

        x = torch.relu(self.conv1(x))
        #print("x after conv 1: {}".format(x.shape))
        
        x = x.view(batch_size, -1)
        #print("x after reshape: {}".format(x.shape))
                
        x = torch.relu(self.lin1(x))
        #print("mean_var after lin1: {}".format(mean_var.shape))

        return {"output": x}
    
    def getinfo(self):
        return "Baseline_CNN_" + str(self.x_dim) + "x" + str(self.y_dim)

        
