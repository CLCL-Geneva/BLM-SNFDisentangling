
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_size):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.nc = embed_dim    
        self.hidden_dim = int(self.nc * 2)
        # Size of the sequence
        self.seq_size = seq_size
        (k1seq, k1x) = (3, 3)
        (k2seq, k2x) = (3, 3)
        (k3seq, k3x) = (3, 3)
        s = 1
        
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(k1seq, k1x), stride=(s,1)) 
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(k2seq, k2x)) 
        self.conv3 = nn.Conv2d(8, 16, kernel_size=(k3seq, k3x))

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



class Decoder_answer(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_size):
        super(Decoder_answer, self).__init__()

        self.latent_dim = latent_dim
        self.nc = embed_dim    
        self.hidden_dim = int(self.nc * 2)
        # Size of the sequence
        self.seq_size = seq_size
        (k1seq, k1x) = (1, 3)
        (k2seq, k2x) = (1, 3)
        (k3seq, k3x) = (1, 3)
        s = 1
        
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
        x = torch.relu(self.conv1(x))  ## sentence representation
              
        return x



class Decoder_mirror(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_size):
        super(Decoder_mirror, self).__init__()

        self.latent_dim = latent_dim
        self.nc = embed_dim    
        self.hidden_dim = int(self.nc * 2)
        # Size of the sequence
        self.seq_size = seq_size
        (k1seq, k1x) = (3, 3)
        (k2seq, k2x) = (3, 3)
        (k3seq, k3x) = (3, 3)
        s = 1
        
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
        x = torch.relu(self.conv1(x))  ## sentence representation
              
        return x



class VariationalAutoencoder(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_size, sampler):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(embed_dim, latent_dim, seq_size)
        self.decoder_answer = Decoder_answer(embed_dim, latent_dim, seq_size)
        self.decoder_mirror = Decoder_mirror(embed_dim, latent_dim, seq_size)
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
        return "dual_VAE_1DxSeq"
