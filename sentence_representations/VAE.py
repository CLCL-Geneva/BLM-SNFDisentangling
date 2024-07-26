'''
Created on Feb 28, 2023

@author: vivi
'''

import math

import numpy as np

import torch
import torch.nn as nn

from torch.nn.functional import normalize

from vaes import encoders, decoders, VAE



class VariationalAutoencoder(nn.Module):
    def __init__(self, embed_dim, sampler, kernel_size = [15, 15], stride = [1, 1], hidden_channels = 32, x_dim = 32, device="cuda:0"):
        super(VariationalAutoencoder, self).__init__()

        self.embed_dim = embed_dim
        self.sampler = sampler

        self.latent_disc = self.sampler.getLatentInfo()["disc"]
        self.latent_cont = self.sampler.getLatentInfo()["cont"]
        
        self.latent_dim = self.latent_cont + self.latent_disc
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.hidden_channels = hidden_channels
        
        print("initializing sentence encoder")
        self.encoder = encoders.Encoder_2D(self.embed_dim, self.latent_disc, self.latent_cont, self.kernel_size, self.stride, self.hidden_channels, x_dim, device)

        print("initializing sentence decoder")
        self.decoder = decoders.Decoder_2D(self.embed_dim, self.latent_dim, self.kernel_size, self.stride, self.hidden_channels, x_dim = x_dim, device = device)



    def forward(self, x, pos=None, mask=None, mask_add=None):
        latent_vec = self.encoder(x)        
        sampled_latent_vec, mean, logvar = self.sampler(latent_vec)

        sampled_latent_vec = self.mask_latent(sampled_latent_vec, mask, mask_add)

        x_recon = self.decoder(sampled_latent_vec).squeeze()
                        
        return {"output": x_recon, 
                "latent_vec": latent_vec, "mean" : mean, "logvar" : logvar,
                "sampled_latent_vec": sampled_latent_vec,
                "latent_vec_disc": latent_vec[:,:self.latent_disc], 
                "sampled_latent_vec_disc": sampled_latent_vec[:,:self.latent_disc],
                "sampled_latent_vec_cont": sampled_latent_vec[:,self.latent_disc:]}


    def encode(self, x):
        self.sampler.is_training = False
        latent_vec = self.encoder(x)        
        sampled_latent_vec, mean, logvar = self.sampler(latent_vec) 
                
        return {"sampled_latent_vec": sampled_latent_vec, 
                "latent_vec": latent_vec, "mean" : mean, "logvar" : logvar,
                "sampled_latent_vec": sampled_latent_vec,
                "latent_vec_disc": latent_vec[:,:self.latent_disc],
                "sampled_latent_vec_disc": sampled_latent_vec[:,:self.latent_disc],
                "sampled_latent_vec_cont": sampled_latent_vec[:,self.latent_disc:]}

    def mask_latent(self, vec, mask, mask_add):
        if mask is not None:
            v = vec * mask
            if mask_add is not None:
                return v + mask_add
            return v
        return vec
    
    def getinfo(self):
        return "SentenceVAE2D_" + str(self.encoder.x_dim) + "x" + str(self.encoder.y_dim) + "__k-" + "x".join(map(str,list(self.kernel_size))) + "_s-" + "x".join(map(str,list(self.stride))) + "___" + self.sampler.getinfo()
    
    
      
    

class VariationalAutoencoder_sparse(VariationalAutoencoder):
    def __init__(self, embed_dim, sampler, kernel_size = [15, 15], stride = [1, 1], hidden_channels = 32, x_dim = 32, device="cuda:0"):
        super(VariationalAutoencoder_sparse, self).__init__(embed_dim, sampler, kernel_size = kernel_size, stride = stride, hidden_channels = hidden_channels, x_dim = x_dim, device=device)

        self.encoder = encoders.Encoder_2D_subnets(embed_dim, self.latent_disc, self.latent_cont, kernel_size = kernel_size, stride = stride, hidden_channels = hidden_channels, x_dim = x_dim, device = device )
        self.decoder = decoders.Decoder_2D_subnets(embed_dim, self.latent_dim, kernel_size = kernel_size, stride = stride, hidden_channels = hidden_channels, x_dim = x_dim, device = device)


    def get_encoder_params(self):
        return {"cnn_input_shape": (self.encoder.x_dim, self.encoder.y_dim),
                "cnn_output_shape": (self.encoder.hidden_ch, self.encoder.nx, self.encoder.ny),
                "kernel_size": self.encoder.kernel_size, 
                "stride": self.encoder.stride, 
                "conv_kernels": self.encoder.conv.weight.detach().cpu().numpy(),
                "linear_masked_weights": self.encoder.lin_mask.get_weights_array(),
                "latent_info": self.sampler.getLatentInfo()}


    def encode(self, x):
        self.sampler.is_training = False
        (latent_vec, x_conv, x_masked) = self.encoder.encode(x)        
        sampled_latent_vec, mean, logvar = self.sampler(latent_vec) 
                
        return {"sampled_latent_vec": sampled_latent_vec, 
                "latent_vec": latent_vec, "mean" : mean, "logvar" : logvar,
                "sampled_latent_vec": sampled_latent_vec,
                "latent_vec_disc": latent_vec[:,:self.latent_disc],
                "sampled_latent_vec_disc": sampled_latent_vec[:,:self.latent_disc],
                "sampled_latent_vec_cont": sampled_latent_vec[:,self.latent_disc:],
                "x_conv": x_conv,
                "x_masked": x_masked}


    def get_weights(self):
        return self.encoder.lin_mask.get_weights()

    def get_masks(self):
        return self.encoder.lin_mask.get_masks()

    def getinfo(self):
        return "SentenceVAE_sparse_" + self.encoder.getinfo() + "_" + self.decoder.getinfo()

