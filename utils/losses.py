
import torch
import torch.nn.functional as F

torch.manual_seed(1)


max_marg = torch.nn.MarginRankingLoss(margin=1)
mm_target = torch.Tensor(1)
mse = torch.nn.MSELoss() 

cos_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')


#y, truth, mlp, base, model_output, beta, latent_dim, sent_emb_dim, batch_size, device
def compute_vae_loss(input_seq, y, truth, model_output, device):
   
    recon_alpha = 0.01    

    ## the dual model, for example, reconstructs the input and predicts the answer (two "parallel" decoders) 
    output = model_output["output"]
    
    n = 1
    recon_loss = max_margin(output, y, truth, device)

    if "recon_input" in model_output:
        recon_loss += recon_alpha * reconstruction_loss_seq(model_output["recon_input"], input_seq, device)
        n += 1
        
    
    #if 'xxx' in model_output:    ## just to skip the next part
    if "recon_sent" in model_output:
        recon_loss += recon_alpha * reconstruction_loss_seq(model_output["recon_sent"], input_seq, device)        
        n += 1
    
    recon_loss /= n
    
    return recon_loss




def reconstruction_loss(sentence_repr, y, truth, device):

    batch_len = y.shape[0]
    
    y = y.reshape(batch_len, y.shape[1], -1)    
    sentence_repr = sentence_repr.reshape(batch_len, -1)

    inds = torch.argmax(truth, dim=1)
    true_y = torch.stack([y[i][inds[i]] for i in range(len(torch.flatten(inds)))]).squeeze()

    return cosine_loss(true_y, sentence_repr, batch_len, device)



def cosine_loss(input_vec, output_vec, batch_len, device):        
    target = torch.ones(batch_len, dtype=torch.float64)
    loss = F.cosine_embedding_loss(output_vec.to(device), input_vec.to(device), target.to(device)).div(batch_len)
    
    return loss


def reconstruction_loss_seq(recon_x, x, device):
    batch_len = x.shape[0]
    seq_len = x.shape[1]
    
    ## if the sentence representation and y are 3D (like when using attention matrices) transform them into vectors
    x = x.reshape(batch_len, seq_len, -1)    
    recon_x = recon_x.reshape(batch_len, seq_len, -1)

    input_vec = torch.sigmoid(x).to(device)
    output_vec = torch.sigmoid(recon_x).to(device)

    loss = F.mse_loss(output_vec, input_vec, reduction="sum").div(batch_len)    
    
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
    sentence_repr = sentence_repr.reshape(batch_len, -1, 1)
    
#    pred = torch.bmm(y.to(device), sentence_repr.unsqueeze(2)).squeeze() 
    pred = torch.bmm(y.to(device), sentence_repr).squeeze()
    
    max_prob = torch.max(pred)
    max_ind = torch.argmax(pred).item()
    return (max_ind, (pred >= max_prob).float())


