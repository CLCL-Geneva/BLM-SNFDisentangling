
from operator import mul
from functools import reduce

import numpy as np

import torch
import torch.nn.functional as F

from torch.nn.functional import normalize
from torchmetrics.functional import pairwise_cosine_similarity

torch.manual_seed(1)

max_marg = torch.nn.MarginRankingLoss(margin=1)
mm_target = torch.Tensor(1)
mse = torch.nn.MSELoss(reduction='none')

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

EPS = 1e-12

#compute_vae_loss(x, pos, negs, model_output, beta, latent_sent_dim, sent_emb_dim, batch_size, device)
def compute_vae_loss(input_x, pos, negs, model_output, beta, latent_sent_dim, sent_emb_dim, batch_size, device, model, args):

    output = model_output["output"]  
    
    loss = max_margin(output, pos, negs, device)

    mean = model_output["mean"]
    logvar = model_output["logvar"]

    ## no kl loss because the sentences only have structural similarity -- does this make sense? I am not sure, but using the kl factor causes the system not to learn anything ...
    #'''
    if mean is not None and logvar is not None:
        kl_loss = kl_divergence(mean.to(device), logvar.to(device)) * float(beta) * latent_sent_dim/sent_emb_dim  ## should we multiply by the sequence length as well?
        print("\tkl_loss = {}".format(kl_loss))
        loss += kl_loss
    #'''
    

    if args.categorical_dim > 0:
        '''
        ## compute the loss on the latent layer to force distinction between sentences with different patterns
        lat_loss = latent_loss___max_margin(model, model_output, pos, negs, device)
        print("\tdisc latent loss (max_marg)= {}".format(lat_loss))

        loss += lat_loss
        '''

        #'''
        lat_loss = discrete_latent_loss_logit(model.sampler, model_output, device)
        print("\tdisc latent loss = {}".format(lat_loss))
        
        loss += lat_loss
        #'''
        
    '''
    if "output_pos" in model_output:
        #loss += max_margin(model_output["output_pos"], input_x, negs, device)
        #loss += kl_divergence(model_output["mean_pos"].to(device), model_output["logvar_pos"].to(device)) * float(beta) * latent_sent_dim/sent_emb_dim
        loss += loss_latents(model_output, device, input_x.shape[0])
    '''
    
    return loss


def kl_divergence(mean, logvar):
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1), dim=0)
    return kld_loss


def cross_ent(sentence_repr, pos, device):

    batch_len = pos.shape[0]    
    sentence_repr = sentence_repr.view(batch_len, -1)
    pos = pos.view(batch_len, -1).to(device)

    recon_loss = F.binary_cross_entropy(torch.sigmoid(sentence_repr),
                                            torch.sigmoid(pos))
    return recon_loss * sentence_repr.shape[1]


def max_margin(sentence_repr, pos, negs, device):

    #print("reconstructed sentence representation (one vec): {}".format(sentence_repr[0][:20]))

    #print("sentence_repr shape: {}".format(sentence_repr.shape))
    #print("pos tensor shape: {}".format(pos.shape))
    #print("negatives length {} and shape of elem {}".format(len(negs), negs[0].shape))

    batch_len = pos.shape[0]
    zeros = torch.zeros(batch_len, dtype=torch.float64).to(device)
    zeros = zeros.reshape(batch_len, 1)
        
    sentence_repr = sentence_repr.reshape(batch_len, -1, 1)
    pos = pos.reshape(batch_len, 1, -1)

    #negs = torch.stack(negs).transpose(0,1)
    negs = torch.stack(negs)   
    if negs.shape[0] != batch_len:
        negs = negs.transpose(0,1)
       
    '''
    #print("sentence negs shape: {}".format(negs.shape))
    scores_pos = torch.bmm(pos.to(device), sentence_repr).reshape(batch_len, -1)
    scores_negs = torch.mean(torch.bmm(negs.to(device), sentence_repr), 1)
    '''   

    #'''        
    all_sents = torch.cat((pos, negs), 1)
    scores = torch.bmm(all_sents.to(device), sentence_repr)
    #print("scores shape after bmm: {}".format(scores.shape))
    
    scores = scores.reshape(batch_len, -1)
    #print("scores shape after reshape: {}".format(scores.shape))

    #scores = torch.softmax(scores, 1) ## softwax is not good here
    scores = torch.nn.functional.normalize(scores, dim=1)    
    
    scores_pos = scores[:,0]
    scores_negs = torch.mean(scores[:,1:], 1)
  
    #print("\tmax margin losses:\n\tpos: {}\n\tnegs: {}".format(torch.mean(scores_pos), torch.mean(scores_negs)))
    #print("\tloss = {}".format(torch.mean(torch.maximum(zeros, 1 - scores_pos + scores_negs))))
    #'''
      
    return torch.mean(torch.maximum(zeros, 1 - scores_pos + scores_negs))


def max_margin_cos(sentence_repr, pos, negs, device):

    batch_len = pos.shape[0]
    zeros = torch.zeros(batch_len, dtype=torch.float64).to(device)
    zeros = zeros.reshape(batch_len, 1)
    
    sentence_repr = sentence_repr.reshape(batch_len, -1)
    pos = pos.reshape(batch_len, -1).to(device)
    negs = torch.stack(negs).to(device)

    #print("sentence repr: {}".format(sentence_repr))
    #print("pos: {}".format(pos))

    scores_pos = cos(sentence_repr, pos)
    scores_negs = torch.mean(torch.stack([cos(sentence_repr, neg_i) for neg_i in negs]), 0)

    #print("\nreconstruction losses:\n\tpos: {}\n\tnegs: {}".format(torch.mean(scores_pos), torch.mean(scores_negs)))
    
    return torch.mean(torch.maximum(zeros, 1 - scores_pos + abs(scores_negs)))
    #return torch.maximum(zeros, 1 - scores_pos + scores_negs)


def max_margin_mse(sentence_repr, pos, negs, device):

    batch_len = pos.shape[0]
    zeros = torch.zeros(batch_len, dtype=torch.float64).to(device)
    zeros = zeros.reshape(batch_len, 1)
    
    sentence_repr = sentence_repr.reshape(batch_len, -1)
    pos = pos.reshape(batch_len, -1).to(device)
    negs = torch.stack(negs).to(device) ## will have dimensions N_negs x batch_size x dim_of_vector

    scores_pos = torch.mean(mse(sentence_repr, pos), -1)
    scores_negs = torch.mean(torch.mean(mse(sentence_repr, negs), -1), 0)

    return torch.mean(torch.maximum(zeros, scores_pos + 1 - scores_negs))
    #return torch.mean(scores_pos + 1 - scores_negs)


def max_margin_nce(sentence_repr, pos, negs, device):

    batch_len = pos.shape[0]
    zeros = torch.zeros(batch_len, dtype=torch.float64).to(device)
    zeros = zeros.reshape(batch_len, 1)
        
    sentence_repr = sentence_repr.reshape(batch_len, -1, 1)
    pos = pos.reshape(batch_len, 1, -1)
    negs = torch.stack(negs).transpose(0,1)   
    
    scores_pos = torch.sigmoid(torch.bmm(pos.to(device), sentence_repr)).squeeze()
    scores_negs = torch.sum(torch.sigmoid(torch.bmm(negs.to(device), sentence_repr)).squeeze(), 1)
    
    #return torch.max(torch.log(scores_pos/(scores_pos + scores_negs)))
    #return torch.sum(torch.exp(scores_pos/(scores_pos + scores_negs)))
    return torch.mean(scores_pos / (scores_pos + scores_negs))




#_______________________________________________________________

def latent_loss___max_margin(model, model_output, pos, negs, device, vec_part = "latent_vec_disc"):
    #vec_part = "sampled_latent_vec_disc"    ## use only the discrete part, because we only want to enforce that the discrete part of the the positive vectors is the same, while the continuous part (encoding semantics or that kind of stuff) need not be close across examples
    input_latent = model_output[vec_part].to(device)
    #print("input latent shape: {}".format(input_latent.shape))
    
    pos_latent = model.encode(pos.to(device))[vec_part]
    negs_latents = [model.encode(negs_i.to(device))[vec_part] for negs_i in negs]

    #print("computing max margin for tensors of shape {} / {} tensors of shape {}".format(pos_latent.shape, len(negs_latents), negs_latents[0].shape))    
    #return max_margin(input_latent, pos_latent, negs_latents, device)
    return max_margin_mse(input_latent, pos_latent, negs_latents, device)
    #return max_margin_nce(input_latent, pos_latent, negs_latents, device)
    #return max_margin_cos(input_latent, pos_latent, negs_latents, device)


def loss_latents(model_output, device, batch_size):
    loss = mse(model_output["sampled_latent_vec_disc"].to(device), model_output["sampled_latent_vec_pos_disc"].to(device))
    #print("latents (discrete part) loss: {}".format(loss))
    #cos_target = torch.ones(batch_size)
    #loss = cos_loss(model_output["sampled_latent_vec_disc"].to(device), model_output["sampled_latent_vec_pos_disc"].to(device), cos_target.to(device))
    
    #print("Sampled discrete latents: {}, {}".format(model_output["sampled_latent_vec_disc"][0], model_output["sampled_latent_vec_pos_disc"][0]))
    
    return loss



#___________________
## discrete latent loss from https://github.com/Schlumberger/joint-vae/blob/master/jointvae/training.py


def discrete_latent_loss_logit___(sampler, model_output, device, vec_part = "latent_vec_disc"):
    """
    Calculates the KL divergence between a set of categorical distributions
    and a set of uniform categorical distributions.

    Parameters
    ----------
    alphas : list
        List of the alpha parameters of a categorical (or gumbel-softmax)
        distribution. For example, if the categorical atent distribution of
        the model has dimensions [2, 5, 10] then alphas will contain 3
        torch.Tensor instances with the parameters for each of
        the distributions. Each of these will have shape (N, D).
    """


    alpha = model_output[vec_part].to(device)
    #print("alpha (a.k.a. log of distrib for discrete latent part) {} = {}".format(alpha.shape, alpha))

    batch_size = alpha.shape[0]

    N_categ = sampler.N_categ  ## nr of categ
    categ_N = sampler.categ_N  ## nr of values per categ

    alpha = alpha.reshape(batch_size, N_categ, categ_N)

    '''    
    # Calculate negative entropy of each row
    #neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
    neg_entropy = torch.sum(torch.exp(alpha) * alpha, dim=1)
    #print("alpha negative entropy = {}".format(neg_entropy))
    
    # Take mean of negative entropy across batch
    mean_neg_entropy = torch.mean(neg_entropy, dim=0)
    # KL loss of alpha with uniform categorical variable
    kl_loss = log_dim + mean_neg_entropy
    #print("disc kl loss = {}".format(kl_loss))
    return kl_loss
    '''
    
    print("\t\talpha = {}".format(alpha))
    neg_entropy = torch.sum(torch.exp(alpha) * alpha, dim=2)
    #neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
    print("\t\tneg_entropy = {}".format(neg_entropy))
    
    mean_neg_entropy = torch.mean(neg_entropy)
    print("\t\tmean_neg_entropy = {}".format(mean_neg_entropy))    
    
    kl_loss = torch.exp(mean_neg_entropy)
    #kl_loss = torch.log(torch.tensor(N_categ* categ_N)).to(device) + mean_neg_entropy
    
    return kl_loss




def discrete_latent_loss_logit(sampler, model_output, device, vec_part = "latent_vec_disc"):
    """
    Calculates the KL divergence between a set of categorical distributions
    and a set of uniform categorical distributions.
    """
    
    alpha = model_output[vec_part].to(device)
    #print("alpha (a.k.a. log of distrib for discrete latent part) {} = {}".format(alpha.shape, alpha))

    batch_size = alpha.shape[0]

    N_categ = sampler.N_categ  ## nr of categ
    categ_N = sampler.categ_N  ## nr of values per categ
    
    alpha = alpha.reshape(batch_size, N_categ, categ_N)
    alpha = torch.softmax(alpha, dim=2)

    '''    
    neg_entropy = torch.sum(torch.exp(alpha) * alpha, dim=1)

    # Take mean of negative entropy across batch
    mean_neg_entropy = torch.mean(neg_entropy, dim=0)
    # KL loss of alpha with uniform categorical variable
    kl_loss = log_dim + mean_neg_entropy
    #print("disc kl loss = {}".format(kl_loss))
    return kl_loss
    '''
    
    #neg_entropy = torch.sum(torch.exp(alpha) * alpha, dim=2)
    neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)    
    mean_neg_entropy = torch.mean(neg_entropy)
    
    kl_loss = torch.exp(mean_neg_entropy)
    #kl_loss = torch.log(torch.tensor(N_categ* categ_N)).to(device) + mean_neg_entropy
    
    return kl_loss



#___________________________________


def prediction(x, y, device):
        
    #print("x_hat: {}".format(x.shape))
    #print("y: {}".format(y.shape))
    
    ## for max margin loss
    '''        
    ## because bmm requires 3d tensors
    if len(x.shape) == 2:
        x = x.unsqueeze(1)
    x = x.transpose(-2,-1)
    
    if len(y.shape) == 2:
        y = y.unsqueeze(1)
    else:
        y = y.transpose(0,1)
    
    scores = torch.bmm(y.to(device), x)#.squeeze()
    '''
    
    batch_len = x.shape[0]
    
    ## for cosine loss
    
    y = y.reshape(batch_len, y.shape[1], -1)
    #y = y.transpose(0,1)
    #print("tensors after reshaping:\n\t x shape: {} \t y shape: {}".format(x.shape, y.shape))
    
    x = x.reshape(batch_len, 1, -1).expand(-1, y.shape[1], -1)  #un-2D the output        
    scores = F.cosine_similarity(x, y.to(device), dim=-1)   ## do not take the absolute value of this!!!!
    
    #x = x.reshape(batch_len, 1, -1)
    #y = y.transpose(1,2)
    #scores = torch.bmm(x.to(device), y.to(device))

    '''      
    n_negs = len(y) - 1

    max_i = torch.argmax(scores).item()
    p = [0] * (n_negs+1)
    p[max_i] = 1
    '''

    p = torch.zeros_like(scores)
    maxes = torch.max(scores,dim=1)
    for i in range(batch_len):
        p[i][maxes.indices[i]] = 1
    
    #print("preds: {}".format(p))
    
    return p.reshape(-1)



###_______________________________________________________________________________________________________________
### https://github.com/Schlumberger/joint-vae/blob/master/jointvae/training.py

    #def _loss_function(self, data, recon_data, latent_dist):
def compute_vae_loss__joint_vae(input_x, pos, negs, model_output, beta, latent_sent_dim, sent_emb_dim, batch_size, device, model):
    
    """
    Calculates loss for a batch of data.
    Parameters
    ----------
    data : torch.Tensor
        Input data (e.g. batch of images). Should have shape (N, C, H, W)
    recon_data : torch.Tensor
        Reconstructed data. Should have shape (N, C, H, W)
    latent_dist : dict
        Dict with keys 'cont' or 'disc' or both containing the parameters
        of the latent distributions as values.
    """
    
    '''
        cont_capacity : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_z).
            Parameters to control the capacity of the continuous latent
            channels. Cannot be None if model.is_continuous is True.
        disc_capacity : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_c).
            Parameters to control the capacity of the discrete latent channels.
            Cannot be None if model.is_discrete is True.    
    '''
    
    cont_capacity=[0.0, len(model_output["mean"]), 25000, 30]
    disc_capacity=[0.0, len(model_output["sampled_latent_vec_disc"]), 25000, 30]
    vec_size = sent_emb_dim
    num_steps = 10000
    
    losses = {'recon_loss': [], 'kl_loss': [], 'loss': [], 'kl_loss_disc': [], 'kl_loss_cont': []}
    
    #print("model output: {}".format(model_output))
    
    output = model_output["output"]  
    batch_size = pos.shape[0]
    
    '''
    # Reconstruction loss is pixel wise cross-entropy
    recon_loss = F.binary_cross_entropy(torch.sigmoid(output.view(batch_size, -1)).to(device),
                                        torch.sigmoid(pos.view(batch_size, -1)).to(device))
    # F.binary_cross_entropy takes mean over pixels, so unnormalise this
    recon_loss *= reduce(mul, pos.shape[1:], 1)
    '''
    
    #recon_loss = max_margin_cos(output, pos, negs, device)
    recon_loss = max_margin(output, pos, negs, device)

    # Calculate KL divergences
    kl_cont_loss = 0  # Used to compute capacity loss (but not a loss in itself)
    kl_disc_loss = 0  # Used to compute capacity loss (but not a loss in itself)
    cont_capacity_loss = 0
    disc_capacity_loss = 0

    if (not hasattr(model.sampler, "is_continuous")) or model.sampler.is_continuous:  ## the simple sampler doesn't have is_continuous attribute, but its output should be processed
        # Calculate KL divergence
        mean = model_output["mean"]
        logvar = model_output["logvar"]
        
        kl_cont_loss = _kl_normal_loss(mean, logvar, losses)
        # Linearly increase capacity of continuous channels
        cont_min, cont_max, cont_num_iters, cont_gamma = cont_capacity
        # Increase continuous capacity without exceeding cont_max
        cont_cap_current = (cont_max - cont_min) * num_steps / float(cont_num_iters) + cont_min
        cont_cap_current = min(cont_cap_current, cont_max)
        # Calculate continuous capacity loss
        cont_capacity_loss = cont_gamma * torch.abs(cont_cap_current - kl_cont_loss)

    if hasattr(model.sampler, "is_discrete") and model.sampler.is_discrete:
        # Calculate KL divergence
        kl_disc_loss = _kl_multiple_discrete_loss(model, model_output["sampled_latent_vec_disc"], losses, device) # * float(beta) * latent_sent_dim/sent_emb_dim
        # Linearly increase capacity of discrete channels
        disc_min, disc_max, disc_num_iters, disc_gamma = \
            disc_capacity
        # Increase discrete capacity without exceeding disc_max or theoretical
        # maximum (i.e. sum of log of dimension of each discrete variable)
        disc_cap_current = (disc_max - disc_min) * num_steps / float(disc_num_iters) + disc_min
        disc_cap_current = min(disc_cap_current, disc_max)
        # Require float conversion here to not end up with numpy float
        #disc_theoretical_max = sum([float(np.log(disc_dim)) for disc_dim in self.model.latent_spec['disc']])
        #disc_cap_current = min(disc_cap_current, disc_theoretical_max)
        # Calculate discrete capacity loss
        disc_capacity_loss = disc_gamma * torch.abs(disc_cap_current - kl_disc_loss)

    # Calculate total kl value to record it
    #kl_loss = kl_cont_loss + kl_disc_loss
    kl_loss = kl_disc_loss

    # Calculate total loss
    #total_loss = recon_loss + cont_capacity_loss + disc_capacity_loss
    total_loss = recon_loss + beta * kl_loss

#    # Record losses
#    if model.training:
    losses['recon_loss'].append(recon_loss.item())
    losses['kl_loss'].append(kl_loss.item())
    losses['loss'].append(total_loss.item())
    if hasattr(model.sampler, "is_discrete") and model.sampler.is_discrete:
        losses['kl_loss_disc'].append(kl_disc_loss)
    if (not hasattr(model.sampler, "is_continuous")) or model.sampler.is_continuous:
        losses['kl_loss_cont'].append(kl_cont_loss)

    #print("losses: {}".format(losses))
    #print("total loss: {}".format(total_loss))

    # To avoid large losses normalise by number of pixels
    #return total_loss / vec_size
    return total_loss


def _kl_normal_loss(mean, logvar, losses):
    """
    Calculates the KL divergence between a normal distribution with
    diagonal covariance and a unit normal distribution.
    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (N, D) where D is dimension
        of distribution.
    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (N, D)
    """
    # Calculate KL divergence
    kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
    # Mean KL divergence across batch for each latent variable
    kl_means = torch.mean(kl_values, dim=0)
    # KL loss is sum of mean KL of each latent variable
    kl_loss = torch.sum(kl_means)

    #losses['kl_loss_cont'].append(kl_loss.item())

    return kl_loss

def _kl_multiple_discrete_loss(model, alphas, losses, device):
    """
    Calculates the KL divergence between a set of categorical distributions
    and a set of uniform categorical distributions.
    Parameters
    ----------
    alphas : list
        List of the alpha parameters of a categorical (or gumbel-softmax)
        distribution. For example, if the categorical atent distribution of
        the model has dimensions [2, 5, 10] then alphas will contain 3
        torch.Tensor instances with the parameters for each of
        the distributions. Each of these will have shape (N, D).
    """

    ## reshape the alphas to have the last dimensions (N, D) -- they are flattened in the latent layer
    ## the first dimension is the batch size
    alphas = alphas.reshape(alphas.shape[0], model.sampler.N_categ, model.sampler.categ_N)
    
    # Calculate kl losses for each discrete latent
    kl_losses = [_kl_discrete_loss(alpha, device) for alpha in alphas]
    # Total loss is sum of kl loss for each discrete latent
    kl_loss = torch.sum(torch.cat(kl_losses))
    
    losses['kl_loss_disc'].append(kl_loss.item())

    return kl_loss

def _kl_discrete_loss(alpha, device):
    """
    Calculates the KL divergence between a categorical distribution and a
    uniform categorical distribution.
    Parameters
    ----------
    alpha : torch.Tensor
        Parameters of the categorical or gumbel-softmax distribution.
        Shape (N, D)
    """
    
    EPS = 1e-12

    #print("sampled discrete vector for computing discrete loss: {}".format(alpha))
    
    #disc_dim = int(alpha.size()[-1])
    disc_dim = alpha.shape[-1]
    #print("disc dim = {} / log = {}".format(disc_dim, np.log(disc_dim)) )
    #log_dim = torch.Tensor([np.log(disc_dim)])  #.to(device)

    # Calculate negative entropy of each row
    #neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
    alpha_nonzero = torch.where(alpha == 0, EPS, alpha.to(torch.double)) ## to avoid having values above 1
    
    neg_entropy = -1 * torch.sum(alpha * torch.log(alpha_nonzero), dim=1)  
    #print("neg entropy: {}".format(neg_entropy))      

    '''
    # Take mean of negative entropy across batch
    mean_neg_entropy = torch.mean(neg_entropy, dim=0)

    ## why is this a scalar??? it should be a tensor
    print("mean neg entropy: {}".format(mean_neg_entropy))

    # KL loss of alpha with uniform categorical variable
    
    #return np.log(disc_dim) + mean_neg_entropy
    return mean_neg_entropy
    '''
    
    return neg_entropy
