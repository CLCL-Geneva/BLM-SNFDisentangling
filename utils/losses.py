
import logging

import numpy as np

from sentence_representations.losses import discrete_latent_loss_logit, latent_loss___max_margin
from sentence_representations.losses import max_margin as max_margin_sent

import torch
import torch.nn.functional as F

from torchmetrics.functional import pairwise_cosine_similarity

torch.manual_seed(1)


max_marg = torch.nn.MarginRankingLoss(margin=1)
mm_target = torch.Tensor(1)
mse = torch.nn.MSELoss() 

cos_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')

EPS = 1e-12

def compute_vae_loss(model, input_seq, y, truth, model_output, betas, args):

    beta = betas[0]
    recon_alpha = 100.0 ##0.01 ## 0.007
    disc_beta = 1.0  #1000

    output = model_output["output"]
    
    n = 1
    recon_loss = max_margin(output, y, truth, args.device)

    logging.info("max margin recon loss = {}".format(recon_loss))

    if "recon_input" in model_output:
        seq_recon_loss = recon_alpha * reconstruction_loss_seq(model_output["recon_input"], input_seq, args.device)
        logging.info("sequence reconstruction loss = {}".format(seq_recon_loss))

        recon_loss += seq_recon_loss
        n += 1

    if "recon_sent" in model_output:
        sent_recon_loss = reconstruction_loss_sent(model_output["recon_sent"], input_seq, args.device)
        logging.info("sentence reconstruction loss = {}".format(sent_recon_loss))
        
        recon_loss += sent_recon_loss
        n += 1
    
    kl_loss = 0
    n = 0
    
    if ("mean" in model_output) and (model_output["mean"] is not None):
        mean = model_output["mean"]
        logvar = model_output["logvar"]
            
        kl_loss += kl_divergence(mean.to(args.device), logvar.to(args.device)) * float(beta) * args.latent/args.sent_emb_dim
        n += 1

        logging.info("kl loss: {}".format(kl_loss))


    #'''
    if "sent_mean" in model_output:
        mean = model_output["sent_mean"]
        logvar = model_output["sent_logvar"]
    
        sent_kl_loss = kl_divergence(mean.to(args.device), logvar.to(args.device)) * float(betas[1]) * args.latent_sent_dim_cont/args.sent_emb_dim
        kl_loss += sent_kl_loss
        n += 1

        logging.info("sent kl loss = {}".format(sent_kl_loss))
    #'''
        

    if n > 0:
        #kl_loss /= n
        logging.info("kl_loss = {}".format(kl_loss))

    disc_loss = 0

    if "sent_latent_vec" in model_output and hasattr(model.sampler_sent, "N_categ"):

        if hasattr(model, "sampler_sent"):
            disc_loss = disc_beta * discrete_latent_loss_logit(model.sampler_sent, model_output, args.device, vec_part = "sent_latent_vec_disc")
            #disc_loss = disc_beta * discrete_latent_loss__max_margin(model, model_output, input_seq, device)
        elif hasattr(model, "sampler"):
            disc_loss = disc_beta * discrete_latent_loss_logit(model.sampler, model_output, args.device, vec_part = "latent_vec_disc" )
        logging.info("sent disc loss: {}".format(disc_loss))
        #n += 1

    elif "latent_vec_disc" in model_output and hasattr(model.sampler, "N_categ"):
        disc_loss = disc_beta * discrete_latent_loss_logit(model.sampler, model_output, args.device, vec_part = "latent_vec_disc")
        logging.info("seq disc loss: {}".format(disc_loss))
        #n += 1

    return recon_loss + kl_loss + disc_loss


def kl_divergence(mean, logvar):
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1), dim=0)
    return kld_loss


def reconstruction_loss_seq(recon_seq, seq, device):
    batch_len = seq.shape[0]
    seq_len = seq.shape[1]

    ## if the sentence representation and y are 3D (like when using attention matrices) transform them into vectors
    input_vec = seq.reshape(batch_len, seq_len, -1).to(device)
    output_vec = recon_seq.reshape(batch_len, seq_len, -1).to(device)

    #input_vec = torch.sigmoid(input_vec).to(device)
    #output_vec = torch.sigmoid(output_vec).to(device)

    #return torch.sigmoid(torch.mean(input_vec * output_vec)**2)
    return F.mse_loss(output_vec, input_vec, reduction="mean").div(batch_len)


def reconstruction_loss_sent(recon_sent, input_seq, device):
    
    dim = input_seq.shape[-1]
    recon_sent = recon_sent.reshape(-1, dim)
    sents = input_seq.reshape(-1, dim)
    
    N = sents.shape[0]
    negs = [torch.stack([sents[(i+1) % N], sents[(i+2) %N], sents[(i+3) %N]]) for i in range(N)]
    
    return max_margin_sent(recon_sent, sents, negs, device)
    

def max_margin(sentence_repr, y, truth, device, pos_only=False):

    loss = 0
    batch_len = y.shape[0]
    zeros = torch.zeros(batch_len, dtype=torch.float64).to(device)
    zeros = zeros.reshape(batch_len, 1)    
    
    ## if the sentence representation and y are 3D (like when using attention matrices) transform them into vectors
    y = y.reshape(batch_len, y.shape[1], -1).to(device)    
    sentence_repr = sentence_repr.reshape(batch_len, -1, 1).to(device)
    
    scores = torch.bmm(y, sentence_repr).squeeze()

    true_ind = torch.argmax(truth,dim=1)

    n = scores.size()[1]

    scores_pos = torch.squeeze(torch.stack([scores[i,true_ind[i]] for i in range(len(true_ind))]))
    scores_neg = torch.stack([scores[i,j] for i in range(len(true_ind)) for j in range(n) if j != true_ind[i]])
    scores_neg = scores_neg.reshape(batch_len,-1)

    if pos_only:
        return 1-sum(scores_pos.tolist())/batch_len

    mm_target = torch.ones(batch_len)
    for i in range(n-1):
        loss += max_marg(scores_pos,scores_neg[:,i],mm_target.to(device))
        
    return loss/(n-1.)


def prediction(sentence_repr, y, device):
        
    batch_len = y.shape[0]        

    ## for cosine loss
    sentence_repr = sentence_repr.reshape(batch_len, 1, -1).expand(-1, y.shape[1], -1)  #un-2D the output and expand for cosine computation
    y = y.reshape(batch_len, y.shape[1], -1)

    scores = F.cosine_similarity(sentence_repr, y.to(device), dim=-1)
    maxes = torch.max(scores,dim=1)
    
    return maxes.indices, scores


