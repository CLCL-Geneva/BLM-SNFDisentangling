
import logging

import torch
import torch.nn.functional as F

from torchmetrics.functional import pairwise_cosine_similarity

torch.manual_seed(1)


max_marg = torch.nn.MarginRankingLoss(margin=1)
mm_target = torch.Tensor(1)
mse = torch.nn.MSELoss() 

cos_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')


def compute_vae_loss(input_seq, y, truth, base, model_output, betas, latent_dim, latent_sent_dim, sent_emb_dim, batch_size, device):
   
    beta = betas[0]
    recon_alpha = 0.01    

    ## the dual model, for example, reconstructs the input and predicts the answer (two "parallel" decoders) 
    output = model_output["output"]
    
    n = 1
    recon_loss = max_margin(output, y, truth, device)
    #recon_loss = max_margin(output, y, truth, device, pos_only=True)    
    #recon_loss = max_margin_cos(output, y, truth, device)
     
    #logging.info("max margin recon loss = {}".format(recon_loss))


    if "recon_input" in model_output:
        recon_loss += recon_alpha * reconstruction_loss_seq(model_output["recon_input"], input_seq, device)
        n += 1
        
    
    if "recon_sent" in model_output:
        recon_loss += recon_alpha * reconstruction_loss_seq(model_output["recon_sent"], input_seq, device)        
        n += 1
    
    #logging.info("recon loss: {}".format(recon_loss))

    recon_loss /= n

    kl_loss = 0
    n = 0
    
    if ("mean" in model_output) and (model_output["mean"] is not None):
        #logging.info("has mean")
        mean = model_output["mean"]
        logvar = model_output["logvar"]
    
        # KL Divergence
        kl_loss = kl_divergence(mean.to(device), logvar.to(device)) * float(beta) * latent_dim/sent_emb_dim  ## should we multiply by the sequence length as well?
        # Beta-VAE Loss as in Higgins et al. (2017)
        n += 1

    if "sent_mean" in model_output:
        #logging.info("has mean")
        mean = model_output["sent_mean"]
        logvar = model_output["sent_logvar"]
    
        # KL Divergence
        kl_loss = kl_divergence(mean.to(device), logvar.to(device)) * float(betas[1]) * latent_sent_dim/sent_emb_dim
        n += 1

    if n > 0:
        kl_loss /= n

    loss = recon_loss + kl_loss
    
    return loss




def beta_vae_loss_H(recon_loss, kl_loss, beta, latent_dim, sent_emb_dim):
    # Normalisation -> to avoid KLD degradation
    kld_weight = latent_dim/sent_emb_dim
    
    loss = recon_loss + beta * kld_weight * kl_loss
     
    logging.info("loss={}\t\trec={}\tkl={}\tkl_w={}\tbeta={}".format(loss, recon_loss, kl_loss, kld_weight, beta))
    return loss



def kl_divergence(mean, logvar):
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1), dim=0)
    return kld_loss


'''
def reconstruction_loss(sentence_repr, y, truth, device):
    #true_y = y[torch.argmax(truth,dim=1)]
    #true_y = torch.take_along_dim(y, torch.argmax(truth, dim=1), dim=1)

    batch_len = y.shape[0]
    
    ## if the sentence representation and y are 3D (like when using attention matrices) transform them into vectors
    y = y.reshape(batch_len, y.shape[1], -1)    
    sentence_repr = sentence_repr.reshape(batch_len, -1)

    inds = torch.argmax(truth, dim=1)
    true_y = torch.stack([y[i][inds[i]] for i in range(len(torch.flatten(inds)))]).squeeze()

    output_vec = sentence_repr
    input_vec = true_y

    input_vec = torch.sigmoid(input_vec)
    output_vec = torch.sigmoid(output_vec)

    # Bernoulli distribution
    #loss = F.binary_cross_entropy_with_logits(output_vec, input_vec, reduction="sum").div(batch_len)
    #loss = F.binary_cross_entropy(output_vec, input_vec, reduction="sum").div(batch_len)
    #loss = F.mse_loss(output_vec, input_vec, reduction="sum").div(batch_len)
    
    target = torch.ones(batch_len, dtype=torch.float64)
    loss = F.cosine_embedding_loss(output_vec, input_vec, target).div(batch_len)
    
    return loss
'''


def reconstruction_loss(sentence_repr, y, truth, device):

    batch_len = y.shape[0]
    
    ## if the sentence representation and y are 3D (like when using attention matrices) transform them into vectors
    y = y.reshape(batch_len, y.shape[1], -1)    
    sentence_repr = sentence_repr.reshape(batch_len, -1)

    inds = torch.argmax(truth, dim=1)
    true_y = torch.stack([y[i][inds[i]] for i in range(len(torch.flatten(inds)))]).squeeze()

    return cosine_loss(true_y, sentence_repr, batch_len, device)



def cosine_loss(input_vec, output_vec, batch_len, device):    
    #input_vec = torch.sigmoid(input_vec)
    #output_vec = torch.sigmoid(output_vec)
    
    target = torch.ones(batch_len, dtype=torch.float64)
    loss = F.cosine_embedding_loss(output_vec.to(device), input_vec.to(device), target.to(device)).div(batch_len)
    
    return loss


def reconstruction_loss_seq(recon_x, x, device):

    batch_len = x.shape[0]
    seq_len = x.shape[1]
    
    ## if the sentence representation and y are 3D (like when using attention matrices) transform them into vectors
    x = x.reshape(batch_len, seq_len, -1)    
    recon_x = recon_x.reshape(batch_len, seq_len, -1)
    #x = x.reshape(batch_len, -1)    
    #recon_x = recon_x.reshape(batch_len, -1)

    input_vec = torch.sigmoid(x).to(device)
    output_vec = torch.sigmoid(recon_x).to(device)
    #input_vec = x.to(device)
    #output_vec = recon_x.to(device)

    # Bernoulli distribution
    #loss = F.binary_cross_entropy_with_logits(output_vec, input_vec, reduction="sum").div(batch_len)
    #loss = F.binary_cross_entropy(output_vec, input_vec, reduction="sum").div(batch_len)
    loss = F.mse_loss(output_vec, input_vec, reduction="sum").div(batch_len)    
    #loss = F.cosine_embedding_loss(output_vec, input_vec, target).div(batch_len)
    
    #logging.info("seq recon loss = {}".format(loss))
    
    return loss



def max_margin(sentence_repr, y, truth, device, pos_only=False):

    loss = 0
    batch_len = y.shape[0]
    
    ## if the sentence representation and y are 3D (like when using attention matrices) transform them into vectors
    y = y.reshape(batch_len, y.shape[1], -1).to(device)    
    sentence_repr = sentence_repr.reshape(batch_len, -1, 1).to(device)
    
    scores = torch.bmm(y, sentence_repr).squeeze()
    true_ind = torch.argmax(truth,dim=1)
    n = scores.size()[1]

    mm_target = torch.ones(batch_len)

    scores_pos = torch.squeeze(torch.stack([scores[i,true_ind[i]] for i in range(len(true_ind))]))
    scores_neg = torch.stack([scores[i,j] for i in range(len(true_ind)) for j in range(n) if j != true_ind[i]])
    scores_neg = scores_neg.reshape(batch_len,-1)

    if pos_only:
        return 1-sum(scores_pos.tolist())/batch_len

    for i in range(n-1):
        loss += max_marg(scores_pos,scores_neg[:,i],mm_target.to(device))
        
    return loss/(n-1.)



def max_margin_cos(sentence_repr, y, truth, device, pos_only=False):

    loss = 0
    batch_len = y.shape[0]
    n = y.shape[1]
    
    ## if the sentence representation and y are 3D (like when using attention matrices) transform them into vectors
    y = y.reshape(batch_len, y.shape[1], -1)    
    sentence_repr = sentence_repr.reshape(batch_len, -1)
    
    true_inds = torch.argmax(truth,dim=1)
    true_y = torch.stack([y[i][true_inds[i]] for i in range(len(torch.flatten(true_inds)))]).squeeze()

    scores_pos = cosine_loss(true_y, sentence_repr, batch_len, device)
    
    if pos_only:
        return scores_pos

    for j in range(n-1):
        false_y = torch.stack([y[i][false_ind(j,true_inds[i])] for i in range(len(torch.flatten(true_inds)))]).squeeze()
        scores_neg = cosine_loss(false_y, sentence_repr, batch_len, device) 
        #loss += max_marg(scores_pos,scores_neg,mm_target.to(device))
        loss += max(0, 1-scores_pos+scores_neg)
        
    return loss/(n-1.)


def false_ind(j, true_ind):
    if j == true_ind:
        return j+1
    return j




def prediction(sentence_repr, y, device):
        
    batch_len = y.shape[0]        
        
    ## if the sentence representation and y are 3D (like when using attention matrices) transform them into vectors
    y = y.reshape(batch_len, y.shape[1], -1)    
    sentence_repr = sentence_repr.reshape(batch_len, -1)
    
    pred = pairwise_cosine_similarity(sentence_repr, y.squeeze().to(device)).squeeze()
    
    max_prob = torch.max(pred)
    max_ind = torch.argmax(pred).item()
    return (max_ind, (pred >= max_prob).float())


