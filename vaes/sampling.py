'''
Created on Aug 8, 2022

@author: vivi
'''

import numpy as np

import torch
from torch.nn.modules import Module
from torch.distributions import Gamma
from torch.nn import functional as F

np.random.seed(1)
torch.manual_seed(1)
#torch.use_deterministic_algorithms(True)


class Sampling(Module):
    
    def __init__(self, latent_dim = None, categorical_dim = 0):
        super(Sampling, self).__init__()
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.is_training = True
        
        

class simpleSampling(Sampling):
    
    def __init__(self, latent_dim):
        super(simpleSampling, self).__init__()
        self.latent_dim = latent_dim
        self.latent_size_in = latent_dim * 2
        self.latent_size = latent_dim
        

    def forward(self, latent_vec):
        mean, logvar = latent_vec.view(-1, self.latent_dim, 2).unbind(-1) # separate mean and var in two different vectors
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean, mean, logvar      
    
    def getLatentInfo(self):
        return {"cont": self.latent_dim, 
                "disc": 0}

    def getinfo(self):
        return "simple-sampler" + "__latent-size_" + str(self.latent_dim) 


    
#_______________________________________________________


class JointSampling(Sampling):

    def __init__(self, latent_dim, N_categ, categ_N, device, is_continuous = True, is_discrete = True, eps:float = 1e-12, temperature: float = 0.5, anneal_rate: float = 3e-5):
        super(JointSampling, self).__init__(latent_dim)
        
        self.N_categ = N_categ  ## the number of categories e.g. phrase type, grammatical number, etc.
        self.categ_N = categ_N   ## the (max) number of potential values for a category (e.g. grammatical number would have 2 for English, but this is the max over all categories) 
        self.cont_len = latent_dim 
        
        self.latent_size_in = self.N_categ * self.categ_N + self.latent_dim * 2
        self.latent_size = self.N_categ * self.categ_N + self.latent_dim

        self.device = device
        
        self.eps = eps
        self.temperature = temperature
        self.anneal_rate = anneal_rate

        if self.cont_len == 0:
            self.is_continuous = False
        else:
            self.is_continuous = is_continuous
        self.is_discrete = is_discrete
        
        self.is_training = True
        

    def forward(self, latent):        
        batch_len = latent.shape[0]
        ## separate the latent vector produced by the encoder into the discrete and the continuous portions, and reparameterize
        discrete_len = self.N_categ * self.categ_N
        return self.reparameterize({"disc": latent[:,:discrete_len].view(batch_len, self.N_categ, self.categ_N), "cont": latent[:,discrete_len:]}, batch_len)

    
    def getLatentInfo(self):
        return {"cont": self.cont_len, "disc": self.N_categ * self.categ_N}



    ## original code for JointVAE at https://github.com/Schlumberger/joint-vae/blob/master/jointvae/models.py
    def reparameterize(self, latent_dist, batch_len):
        latent_sample = []

        mean_all = torch.zeros((batch_len, self.N_categ * self.categ_N + self.cont_len), dtype=torch.float)
        logvar_all = torch.ones((batch_len, self.N_categ * self.categ_N + self.cont_len), dtype=torch.float)

        if self.is_discrete:
            latent_sample_disc = []
            for alpha in latent_dist['disc']:
                #alpha_nonzero = torch.log(torch.where(alpha == 0, self.eps, alpha.to(torch.double)))   ## for when alphas are not logits, but the actual distribution
                #disc_sample = self.sample_gumbel_softmax(F.softmax(alpha_nonzero, dim=1))

                disc_sample = self.sample_gumbel_softmax(F.softmax(alpha, dim=1))   ## alpha may be considered as logits of a distribution, or the distribution itself -- this depends on the implementation of the encoder
                latent_sample_disc.append(disc_sample.view(-1).to(self.device))
            latent_sample.append(torch.stack(latent_sample_disc))
            mean = None
            logvar = None         

        if self.is_continuous:
            mean, logvar = latent_dist['cont'].view(-1, self.cont_len, 2).unbind(-1)
            cont_sample = self.sample_normal(mean, logvar).to(self.device)
            latent_sample.append(cont_sample)
            mean_all[:,-self.cont_len:] = mean
            logvar_all[:,-self.cont_len:] = logvar

        # Concatenate continuous and discrete samples into one large sample
        return torch.cat(latent_sample, dim=1), mean, logvar
    

    def sample_normal(self, mean, logvar):
        
        zero_mean = torch.zeros(mean.shape, dtype=torch.float)
        one_std = torch.ones(logvar.shape, dtype=torch.float)
        
        if self.training:
            std = torch.exp(0.5 * logvar).to(self.device)
            eps = torch.normal(zero_mean, one_std).to(self.device)
            return mean.to(self.device) + std * eps
        else:
            # Reconstruction mode
            return mean.to(self.device)
                

    def sample_gumbel_softmax(self, alpha):
        if self.is_training:
            return F.gumbel_softmax(alpha, self.temperature, hard=False, dim=1)

        '''
        # In reconstruction mode, pick most likely sample
        _, max_alpha = torch.max(alpha, dim=1)
        one_hot_samples = torch.zeros(alpha.size())
        # On axis 1 of one_hot_samples, scatter the value 1 at indices
        # max_alpha. Note the view is because scatter_ only accepts 2D
        # tensors.
        one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
        return one_hot_samples
        '''
        return F.gumbel_softmax(alpha, self.temperature, hard=True, dim=1)
    
    

    def getinfo(self):
        return "joint-sampler" + "__latent-cont-size_" + str(self.cont_len)  + "__latent-disc-size_" + str(self.N_categ) + "x" + str(self.categ_N) 
    