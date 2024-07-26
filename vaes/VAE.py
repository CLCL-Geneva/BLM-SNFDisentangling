
import sys 

import numpy as np

import torch
import torch.nn as nn

from vaes import encoders
from vaes import decoders
        
np.random.seed(1)
torch.manual_seed(1)
#torch.use_deterministic_algorithms(True)

class VariationalAutoencoder(nn.Module):
    def __init__(self, embed_dim, seq_size, sampler, kernel_size = [3, 15, 15], stride = [1, 15, 15], hidden_channels = 32, x_dim = 32, device="cuda:0"):
        super(VariationalAutoencoder, self).__init__()

        self.sampler = sampler
        
        self.disc = self.sampler.getLatentInfo()["disc"]
        self.cont = self.sampler.getLatentInfo()["cont"]
        self.latent_size = self.sampler.latent_size
        
        self.encoder = encoders.Encoder_3D(embed_dim, self.disc, self.cont, seq_size, kernel_size, stride, hidden_channels, x_dim, device)
        self.decoder = decoders.Decoder_answer_3D(embed_dim, self.latent_size, 1, kernel_size, stride, hidden_channels, x_dim, device)


    def forward(self, x, mask = None):
    
        latent_vec = self.encoder(x) 
        latent_sampled_vec, mean, logvar = self.sampler(latent_vec)
         
        if mask is not None:
            latent_sampled_vec = self.mask_latent(latent_sampled_vec, mask)
        
        answer = self.decoder(latent_sampled_vec).squeeze()
                
        return {"output": answer, 
                "latent_vec": latent_vec,
                "latent_sampled_vec": latent_sampled_vec,
                "latent_vec_disc": latent_vec[:,:self.disc],  
                "latent_sampled_vec_disc": latent_sampled_vec[:,:self.disc],  
                "mean" : mean, "logvar" : logvar}
    

    def mask_latent(self, vec, mask):
        return vec * mask
        
            
    def getinfo(self):
        return "VAE_" + self.encoder.getinfo() + "_" + self.decoder.getinfo() + "_" + self.sampler.getinfo()
    
    
    
    
class VariationalAutoencoder_enc_sparse(VariationalAutoencoder):
    def __init__(self, embed_dim, seq_size, sampler, kernel_size = [3, 15, 15], stride = [1, 10, 10], hidden_channels = 32, x_dim = 32, device="cuda:0"):
        super(VariationalAutoencoder_enc_sparse, self).__init__(embed_dim, seq_size, sampler, kernel_size, stride, hidden_channels, x_dim, device)
        
        self.encoder = encoders.Encoder_3D_subnets(embed_dim, self.disc, self.cont, seq_size, kernel_size, stride, hidden_channels, x_dim, device)


    def forward(self, x, mask = None):
    
        latent_vec = self.encoder(x)
        latent_sampled_vec, mean, logvar = self.sampler(latent_vec)
         
        if mask is not None:
            latent_sampled_vec = self.mask_latent(latent_sampled_vec, mask)
        
        answer = self.decoder(latent_sampled_vec).squeeze()
                
        return {"output": answer, 
                "latent_vec": latent_vec,
                "latent_vec_disc": latent_vec[:,:self.disc],  
                "latent_sampled_vec": latent_sampled_vec,
                "latent_sampled_vec_disc": latent_sampled_vec[:,:self.disc],    
                "mean" : mean, "logvar" : logvar, "sampling": self.sampler.getLatentInfo()}
    
            
    def get_weights(self):
        return self.encoder.lin_mask.get_weights()

    def get_weights_array(self):
        return np.transpose(np.array(self.encoder.lin_mask.get_weights_array()))

    '''
    def get_masks(self):
        return self.encoder.lin_mask.get_masks()

    def get_masks_as_tensors(self):
        return self.encoder.lin_mask.get_masks_as_tensors()

    def load_masks(self, masks_array):
        self.encoder.lin_mask.load_masks(masks_array)
    '''

    def get_encoder_params(self):
        return {"cnn_input_shape": (self.encoder.seq_size, self.encoder.x_dim, self.encoder.y_dim),
                "cnn_output_shape": (self.encoder.hidden_ch, self.encoder.nseq, self.encoder.nx, self.encoder.ny),
                "kernel_size": self.encoder.kernel_size, 
                "stride": self.encoder.stride, 
                "conv_kernels": self.encoder.conv1.weight.detach().cpu().numpy(),
                "linear_masked_weights": self.get_weights_array(),
                "latent_info": self.sampler.getLatentInfo()}

          
    def getinfo(self):
        return "VAE_enc_sparse_" + self.encoder.getinfo() + "_" + self.decoder.getinfo() +  "_" + self.sampler.getinfo()




class VariationalAutoencoder_sparse(VariationalAutoencoder_enc_sparse):
    def __init__(self, embed_dim, seq_size, sampler, kernel_size = [3, 15, 15], stride = [1, 10, 10], hidden_channels = 32, x_dim = 32, device="cuda:0"):
        super(VariationalAutoencoder_sparse, self).__init__(embed_dim, seq_size, sampler, kernel_size, stride, hidden_channels, x_dim, device)
                
        self.decoder = decoders.Decoder_answer_3D_subnets(embed_dim, self.latent_size, 1, kernel_size, stride, hidden_channels, x_dim, device)


    def get_encoder_params(self):
        return {"cnn_input_shape": (self.encoder.seq_size, self.encoder.x_dim, self.encoder.y_dim),
                "cnn_output_shape": (self.encoder.hidden_ch, self.encoder.nseq, self.encoder.nx, self.encoder.ny),
                "kernel_size": self.encoder.kernel_size, 
                "stride": self.encoder.stride, 
                "conv_kernels": self.encoder.conv1.weight.detach().cpu().numpy(),
                "linear_masked_weights": self.get_weights_array(),
                "latent_info": self.sampler.getLatentInfo()}

          
    def getinfo(self):
        return "VAE_sparse_" + self.encoder.getinfo() + "_" + self.decoder.getinfo() + "_" + self.sampler.getinfo()


#_____________________________________________________________________

class DualVAE(VariationalAutoencoder):
    def __init__(self, embed_dim, seq_size, sampler, kernel_size = [3, 15, 15], stride = [1, 10, 10], hidden_channels = 32, x_dim = 32, device="cuda:0"):
        super(DualVAE, self).__init__(embed_dim, seq_size, sampler, kernel_size, stride, hidden_channels, x_dim, device)
        
        self.decoder_mirror = decoders.Decoder_mirror_3D(embed_dim, self.latent_size, seq_size, kernel_size, stride, hidden_channels, x_dim, device)


    def forward(self, x, mask=None):

        latent_vec = self.encoder(x)
        latent_sampled_vec, mean, logvar = self.sampler(latent_vec) 

        recon_x = self.decoder_mirror(latent_sampled_vec).squeeze()
        answer = self.decoder_answer(latent_sampled_vec).squeeze()
                
        return {"output": answer,
                "recon_input": recon_x, 
                "latent_vec": latent_vec,
                "latent_vec_disc": latent_vec[:,:self.disc],  
                "latent_sampled_vec": latent_sampled_vec,
                "latent_sampled_vec_disc": latent_sampled_vec[:,:self.disc],    
                "mean" : mean, "logvar" : logvar, "sampling": self.sampler.getLatentInfo()}
        
        
    def getinfo(self):
        return "dualVAE_" + self.encoder.getinfo() + "_" + self.decoder.getinfo() + "_" + self.sampler.getinfo()
    
    
#______________________________________________________________________


class VariationalAutoencoder_2level(nn.Module):
    def __init__(self, embed_dim, seq_size, samplers, 
                 kernel_size_seq = [4, 4], stride_seq = [1, 1], hidden_channels_seq = 32, 
                 kernel_size_sent = [15, 15], stride_sent = [1, 1], hidden_channels_sent = 40,
                 x_dim = 32, device="cuda:0"):
        super(VariationalAutoencoder_2level, self).__init__()

        self.seq_size = seq_size
        self.embed_dim = embed_dim

        self.sampler_sent = samplers[1]
        self.sampler = samplers[0]

        self.latent_sent_disc = self.sampler_sent.getLatentInfo()["disc"]
        self.latent_sent_cont = self.sampler_sent.getLatentInfo()["cont"]
        
        ## dimension of the sampled latent sentence, as input to the sequence VAE        
        self.latent_sent = self.latent_sent_cont + self.latent_sent_disc
        
        self.latent_seq_disc = self.sampler.getLatentInfo()["disc"]
        self.latent_seq_cont = self.sampler.getLatentInfo()["cont"]

        self.latent_seq = self.latent_seq_disc + self.latent_seq_cont

        print("2l_VAE: initializing sentence encoder")
        self.sentence_encoder = encoders.Encoder_2D(self.embed_dim, self.latent_sent_disc, self.latent_sent_cont, kernel_size_sent, stride_sent, hidden_channels_sent, x_dim, device)

        print("2l_VAE: initializing sentence decoder")
        self.sentence_decoder = decoders.Decoder_2D(self.embed_dim, self.latent_sent, kernel_size_sent, stride_sent, hidden_channels_sent, x_dim = x_dim, device = device)
        
        print("2l_VAE: initializing sequence encoder")
        self.encoder_seq = encoders.Encoder_2D(self.seq_size * self.latent_sent, self.latent_seq_disc, self.latent_seq_cont, kernel_size_seq, stride_seq, hidden_channels_seq, self.seq_size, device)

        print("2l_VAE: initializing sequence decoder")
        #self.decoder_answer_seq = decoders.Decoder_2D(self._embed_dim, self._latent_dim_seq, kernel_size_seq, stride_seq, hidden_channels_seq, x_dim = self._embed_dim, y_dim = 1, device = device)
        self.decoder_answer_seq = decoders.Decoder_2D(self.embed_dim, self.latent_seq, kernel_size_sent, stride_sent, hidden_channels_seq, x_dim=x_dim, device=device)
        
        

    def forward(self, x, mask=None, sent_mask = None):
        input_shape = x.shape

        x = x.view(input_shape[0] * self.seq_size, -1)

        sent_latent_vec = self.sentence_encoder(x)
        sent_sampled_latent_vec, sent_mean, sent_logvar = self.sampler_sent(sent_latent_vec) 

        if sent_mask is not None:
            sent_sampled_latent_vec = self.mask_latent(sent_sampled_latent_vec, sent_mask)
        
        sent_recon = self.sentence_decoder(sent_sampled_latent_vec)
        sent_recon = sent_recon.view(input_shape)
        seqs = sent_sampled_latent_vec.view(input_shape[0], self.seq_size, self.latent_sent)
        
        seq_latent_vec = self.encoder_seq(seqs)
        seq_sampled_latent_vec, seq_mean, seq_logvar = self.sampler(seq_latent_vec)

        if mask is not None:
            seq_sampled_latent_vec = self.mask_latent(seq_sampled_latent_vec, mask)
         
        answer = self.decoder_answer_seq(seq_sampled_latent_vec).squeeze()

        return {"output": answer, 
                "latent_vec": seq_latent_vec,
                "latent_vec_disc": seq_latent_vec[:,:self.latent_seq_disc],
                "latent_sampled_vec": seq_sampled_latent_vec,
                "latent_sampled_vec_disc": seq_sampled_latent_vec[:,:self.latent_seq_disc],
                "mean" : seq_mean, "logvar" : seq_logvar,
                "sent_disc_dim": self.latent_sent_disc,
                "recon_sent": sent_recon, 
                "sent_latent_vec": sent_latent_vec, 
                "sent_latent_vec_disc": sent_latent_vec[:,:self.latent_sent_disc],
                "sent_sampled_latent_vec": sent_sampled_latent_vec, 
                "sent_mean": sent_mean, "sent_logvar": sent_logvar }
 

    def encode(self, x):
        self.sampler_sent.is_training = False
        input_shape = x.shape
        x = x.view(input_shape[0] * self.seq_size, -1)
        (latent_vec, x_conv, x_masked) = self.sentence_encoder.encode(x)
        sampled_latent_vec, mean, logvar = self.sampler_sent(latent_vec)

        return {"sampled_latent_vec": sampled_latent_vec,
                "latent_vec": latent_vec, "mean": mean, "logvar": logvar,
                "sampled_latent_vec": sampled_latent_vec,
                "latent_vec_disc": latent_vec[:, :self.latent_sent_disc],
                "sampled_latent_vec_disc": sampled_latent_vec[:, :self.latent_sent_disc],
                "sampled_latent_vec_cont": sampled_latent_vec[:, self.latent_sent_disc:],
                "x_conv": x_conv,
                "x_masked": x_masked}


    def encode___(self, x):
        self.sampler_sent.is_training = False
        latent_vec = self.sentence_encoder(x)        
        sampled_latent_vec, mean, logvar = self.sampler_sent(latent_vec) 
                
        return {"sampled_latent_vec": sampled_latent_vec, 
                "sent_latent_vec": latent_vec, "mean" : mean, "logvar" : logvar,
                "sent_latent_vec_disc": latent_vec[:,:self.latent_sent_disc], 
                "sampled_latent_vec_disc": sampled_latent_vec[:,:self.latent_sent_disc],
                "sampled_latent_vec_cont": sampled_latent_vec[:,self.latent_sent_disc:]}


 
    def mask_latent(self, vec, mask):
        return vec * mask
        
        
    def getinfo(self):
        return "2level_VAE_sent-latent_" + self.sentence_encoder.getinfo() + "_seq-latent_" + self.encoder_seq.getinfo()

    

class VariationalAutoencoder_2level_subnets(nn.Module):
    def __init__(self, embed_dim, seq_size, samplers, 
                 kernel_size_seq = [4, 4], stride_seq = [1, 1], hidden_channels_seq = 32, 
                 kernel_size_sent = [15, 15], stride_sent = [15, 15], hidden_channels_sent = 40,
                 x_dim = 32, device="cuda:0"):
        super(VariationalAutoencoder_2level_subnets, self).__init__()

        self.seq_size = seq_size
        self.embed_dim = embed_dim

        self.sampler_sent = samplers[1]
        self.sampler = samplers[0]

        self.latent_sent_disc = self.sampler_sent.getLatentInfo()["disc"]
        self.latent_sent_cont = self.sampler_sent.getLatentInfo()["cont"]
        
        ## dimension of the sampled latent sentence, as input to the sequence VAE        
        self.latent_sent = self.latent_sent_cont + self.latent_sent_disc
        
        self.latent_seq_disc = self.sampler.getLatentInfo()["disc"]
        self.latent_seq_cont = self.sampler.getLatentInfo()["cont"]

        self.latent_seq = self.latent_seq_disc + self.latent_seq_cont

        print("2l_VAE: initializing sentence encoder")
        self.sentence_encoder = encoders.Encoder_2D_subnets(self.embed_dim, self.latent_sent_disc, self.latent_sent_cont, kernel_size_sent, stride_sent, hidden_channels_sent, x_dim = x_dim, device=device)

        print("2l_VAE: initializing sentence decoder")
        self.sentence_decoder = decoders.Decoder_2D_subnets(self.embed_dim, self.latent_sent, kernel_size_sent, stride_sent, hidden_channels_sent, x_dim = x_dim, device = device)
        
        print("2l_VAE: initializing sequence encoder")
        self.encoder_seq = encoders.Encoder_2D(self.seq_size * self.latent_sent, self.latent_seq_disc, self.latent_seq_cont, kernel_size_seq, stride_seq, hidden_channels_seq, x_dim = self.seq_size, device=device)

        print("2l_VAE: initializing sequence decoder")
        self.decoder_answer_seq = decoders.Decoder_2D(self.embed_dim, self.latent_seq, kernel_size_sent, [1,1], hidden_channels_sent, x_dim=x_dim, device=device)

    def forward(self, x, mask=None, sent_mask = None):
        input_shape = x.shape
                        
        x = x.view(input_shape[0] * self.seq_size, -1)

        sent_latent_vec = self.sentence_encoder(x)
        sent_sampled_latent_vec, sent_mean, sent_logvar = self.sampler_sent(sent_latent_vec) 

        if sent_mask is not None:
            sent_sampled_latent_vec = self.mask_latent(sent_sampled_latent_vec, sent_mask)
        
        sent_recon = self.sentence_decoder(sent_sampled_latent_vec)
        sent_recon = sent_recon.view(input_shape)
        
        seqs = sent_sampled_latent_vec.view(input_shape[0], self.seq_size, self.latent_sent)
        
        seq_latent_vec = self.encoder_seq(seqs)
        seq_sampled_latent_vec, seq_mean, seq_logvar = self.sampler(seq_latent_vec)

        if mask is not None:
            seq_sampled_latent_vec = self.mask_latent(seq_sampled_latent_vec, mask)
         
        answer = self.decoder_answer_seq(seq_sampled_latent_vec).squeeze()
                
        return {"output": answer, 
                "latent_vec": seq_latent_vec,
                "latent_vec_disc": seq_latent_vec[:,:self.latent_seq_disc],
                "latent_sampled_vec": seq_sampled_latent_vec,
                "latent_sampled_vec_disc": seq_sampled_latent_vec[:,:self.latent_seq_disc],
                "mean" : seq_mean, "logvar" : seq_logvar,
                "sent_disc_dim": self.latent_sent_disc,
                "recon_sent": sent_recon, 
                "sent_latent_vec": sent_latent_vec, 
                "sent_latent_vec_disc": sent_latent_vec[:,:self.latent_sent_disc],
                "sent_sampled_latent_vec": sent_sampled_latent_vec, 
                "sent_mean": sent_mean, "sent_logvar": sent_logvar }

    def get_encoder_params(self):
        return {"cnn_input_shape": (self.sentence_encoder.x_dim, self.sentence_encoder.y_dim),
                "cnn_output_shape": (self.sentence_encoder.hidden_ch, self.sentence_encoder.nx, self.sentence_encoder.ny),
                "kernel_size": self.sentence_encoder.kernel_size,
                "stride": self.sentence_encoder.stride,
                "conv_kernels": self.sentence_encoder.conv.weight.detach().cpu().numpy(),
                "linear_masked_weights": self.sentence_encoder.lin_mask.get_weights_array(),
                "latent_info": self.sampler_sent.getLatentInfo()}

    def encode(self, x):
        self.sampler_sent.is_training = False
        input_shape = x.shape
        x = x.view(input_shape[0] * self.seq_size, -1)
        (latent_vec, x_conv, x_masked) = self.sentence_encoder.encode(x)
        sampled_latent_vec, mean, logvar = self.sampler_sent(latent_vec)

        return {"sampled_latent_vec": sampled_latent_vec,
                "latent_vec": latent_vec, "mean": mean, "logvar": logvar,
                "sampled_latent_vec": sampled_latent_vec,
                "latent_vec_disc": latent_vec[:, :self.latent_sent_disc],
                "sampled_latent_vec_disc": sampled_latent_vec[:, :self.latent_sent_disc],
                "sampled_latent_vec_cont": sampled_latent_vec[:, self.latent_sent_disc:],
                "x_conv": x_conv,
                "x_masked": x_masked}

    def get_weights(self):
        return self.sentence_encoder.lin_mask.get_weights()

    def get_masks(self):
        return self.sentence_encoder.lin_mask.get_masks()

    def mask_latent(self, vec, mask):
        return vec * mask
        
        
    def getinfo(self):
        return "2level_VAE_sparse_sent-latent_" + self.sentence_encoder.getinfo() + "_seq-latent_" + self.encoder_seq.getinfo()






class VariationalAutoencoder_1D_2level(nn.Module):
    def __init__(self, embed_dim, seq_size, samplers,
                 kernel_size_seq = [4, 4], stride_seq = [1, 1], hidden_channels_seq = 32,
                 kernel_size_sent=[15, 15], stride_sent=[15, 15], 
                 x_dim = 32, device="cuda:0"):
        super(VariationalAutoencoder_1D_2level, self).__init__()

        self.seq_size = seq_size
        self.embed_dim = embed_dim

        self.sampler_sent = samplers[1]
        self.sampler = samplers[0]

        self.latent_sent_disc = self.sampler_sent.getLatentInfo()["disc"]
        self.latent_sent_cont = self.sampler_sent.getLatentInfo()["cont"]

        ## dimension of the sampled latent sentence, as input to the sequence VAE
        self.latent_sent = self.latent_sent_cont + self.latent_sent_disc

        self.latent_seq_disc = self.sampler.getLatentInfo()["disc"]
        self.latent_seq_cont = self.sampler.getLatentInfo()["cont"]

        self.latent_seq = self.latent_seq_disc + self.latent_seq_cont

        print("2l_VAE: initializing sentence encoder")
        #embed_dim, latent_dim, latent_dim_disc = 0
        self.sentence_encoder = encoders.Encoder_1D_sent(self.embed_dim, self.latent_sent_cont, self.latent_sent_disc)

        print("2l_VAE: initializing sentence decoder")
        self.sentence_decoder = decoders.Decoder_1D_sent(self.embed_dim, self.latent_sent_cont, self.latent_sent_disc)

        print("2l_VAE: initializing sequence encoder")
        self.encoder_seq = encoders.Encoder_2D(self.seq_size * self.latent_sent, self.latent_seq_disc,
                                               self.latent_seq_cont, kernel_size_seq, stride_seq, hidden_channels_seq,
                                               self.seq_size, device)

        print("2l_VAE: initializing sequence decoder")
        self.decoder_answer_seq = decoders.Decoder_2D(self.embed_dim, self.latent_seq, kernel_size_sent, stride_sent,
                                                      hidden_channels_seq, x_dim=x_dim, device=device)

    def forward(self, x, mask=None, sent_mask=None):
        input_shape = x.shape

        x = x.view(input_shape[0] * self.seq_size, -1)

        sent_latent_vec = self.sentence_encoder(x)
        sent_sampled_latent_vec, sent_mean, sent_logvar = self.sampler_sent(sent_latent_vec)

        if sent_mask is not None:
            sent_sampled_latent_vec = self.mask_latent(sent_sampled_latent_vec, sent_mask)

        sent_recon = self.sentence_decoder(sent_sampled_latent_vec)
        sent_recon = sent_recon.view(input_shape)
        seqs = sent_sampled_latent_vec.view(input_shape[0], self.seq_size, self.latent_sent)

        seq_latent_vec = self.encoder_seq(seqs)
        seq_sampled_latent_vec, seq_mean, seq_logvar = self.sampler(seq_latent_vec)

        if mask is not None:
            seq_sampled_latent_vec = self.mask_latent(seq_sampled_latent_vec, mask)

        answer = self.decoder_answer_seq(seq_sampled_latent_vec).squeeze()

        return {"output": answer,
                "latent_vec": seq_latent_vec,
                "latent_vec_disc": seq_latent_vec[:, :self.latent_seq_disc],
                "latent_sampled_vec": seq_sampled_latent_vec,
                "latent_sampled_vec_disc": seq_sampled_latent_vec[:, :self.latent_seq_disc],
                "mean": seq_mean, "logvar": seq_logvar,
                "sent_disc_dim": self.latent_sent_disc,
                "recon_sent": sent_recon,
                "sent_latent_vec": sent_latent_vec,
                "sent_latent_vec_disc": sent_latent_vec[:, :self.latent_sent_disc],
                "sent_sampled_latent_vec": sent_sampled_latent_vec,
                "sent_mean": sent_mean, "sent_logvar": sent_logvar}

    def encode(self, x):
        self.sampler_sent.is_training = False
        input_shape = x.shape
        x = x.view(input_shape[0] * self.seq_size, -1)
        (latent_vec, x_conv, x_masked) = self.sentence_encoder.encode(x)
        sampled_latent_vec, mean, logvar = self.sampler_sent(latent_vec)

        return {"sampled_latent_vec": sampled_latent_vec,
                "latent_vec": latent_vec, "mean": mean, "logvar": logvar,
                "sampled_latent_vec": sampled_latent_vec,
                "latent_vec_disc": latent_vec[:, :self.latent_sent_disc],
                "sampled_latent_vec_disc": sampled_latent_vec[:, :self.latent_sent_disc],
                "sampled_latent_vec_cont": sampled_latent_vec[:, self.latent_sent_disc:],
                "x_conv": x_conv,
                "x_masked": x_masked}



    def mask_latent(self, vec, mask, mask_add):
        if mask is not None:
            v = vec * mask
            if mask_add is not None:
                return v + mask_add
            return v
        return vec


    def get_encoder_params(self):
        return {"lin_input_shape": (self.encoder.embed_dim, 1),
                "lin_output_shape": (self.encoder.self.latent_dim * 2 + self.encoder.latent_dim_disc, 1),
                "latent_info": self.sampler.getLatentInfo()}

    def getinfo(self):
        return "SentenceVAE1D_2level___" + self.sampler.getinfo()
