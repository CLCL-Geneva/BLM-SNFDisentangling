
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader


class SeqDataset(Dataset):
    def __init__(self,feature,target,truth):
        self.feature = feature
        self.target = target
        self.truth = truth
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]
        truth = self.truth[idx]
        
        return item,label,truth
    
    def __size__(self):
        return self.feature.size()
    
    

def divide_in_batches(data, batch_size):
    sublists = [data[x:x+batch_size] for x in range(0, len(data), batch_size)]
    return sublists


def data_loader(x, y, z, batch, shuf):
    data = SeqDataset(x, y, z)
    return DataLoader(data, batch_size=batch, shuffle=shuf)


def wrapper(data, seq, model, dim = 768, label = False, attn = -1, tokenizer = None, device = "cpu", comb = "sum"):
    
    with torch.no_grad():
        
        if seq or (attn > -1):
            vecs = torch.randn((len(data), len(data[0]), dim))
            if attn > -1 and not label:
                vecs = torch.randn((len(data), len(data[0]), dim, dim))

            for i, sequence in enumerate(data):
                if not label:
                    vecs[i] = torch.from_numpy(encode_sequence(sequence, model, attn, tokenizer, dim, device, comb))
                else:
                    vs = torch.from_numpy(np.asarray([torch.tensor(-1) if x=="False" else torch.tensor(1) for x in sequence]))
                    vecs[i] = vs.reshape(len(data[0]), 1)

        else:
            if not label:
                vecs = torch.from_numpy(model.encode(data))
            else:
                vecs = torch.from_numpy(np.asarray([torch.tensor(-1) if x=="False" else torch.tensor(1) for x in data])) 

        torch.cuda.empty_cache()
    
    return vecs


def encode_sequence(sequence, model, attn, tokenizer, dim, device, comb):
    #print("Encoding sequence: {}".format(sequence))
    
    representation = []
    for sentence in sequence:
#        print("processing sentence: {}".format(sentence))
        
        if attn == -1:
#            representation.append(model.encode(sentence))
            token_ids = torch.tensor([tokenizer.encode(sentence)])
            
            last_layer = model(token_ids)[0]
#            print("last layer shape: {}".format(last_layer.shape))
            # torch.Size([1, 8, 768])  -> (batch size x number of tokens x embedding dimension)
            
            # The BERT [CLS] token correspond to the first hidden state of the last layer
            #print("sentence embedding: {}".format(last_layer[:, 0, :].shape))
            representation.append(last_layer[:, 0, :].cpu().detach().numpy())
    
        else:
            #print("Processing sentence: {}".format(sentence))
            input = tokenizer(sentence, return_tensors="pt")
            tokens = tokenizer.tokenize(sentence)

            outputs = model(**input) 
    
            representation.append(get_attention(tokens, outputs, attn, dim, comb))
    ## normalize
    
    return np.asarray(representation, dtype='float32').squeeze()


def get_attention(tokens, outputs, attn_lev, dim, comb="sum"):
    
    filtered_tokens, indices = get_tokens(tokens)

    attn = np.zeros((dim,dim))
    for attn_mat in outputs['attentions'][attn_lev][0]:
        attn_mat = recompute_attn_mat(attn_mat, indices)
        
        d = len(attn_mat)
        attn_mat = np.pad(attn_mat, (0, dim-d))
        
        if comb == "sum":   
            np.add(attn, attn_mat, attn)
        else:  ##assume the other option is "max" 
            np.fmax(attn, attn_mat, out=attn)

    if comb == "sum":
        attn = attn/len(outputs['attentions'][attn_lev][0])
        
    return attn


def make_complete_seq(x, y, truth):
    data = torch.zeros((x.size(0), x.size(1)+1, x.size(2), x.size(3)))
    for i in range(x.size(0)):
        true_ind = torch.argmax(truth[i],dim=1)
        data[i] = torch.cat((x[i], y[true_ind].unsqueeze(0).unsqueeze(1)), 0)
    
    return data



## if the setting requires the filtering of the attention matrices (remove UNK and reform token-split words), then count the length accordingly
def get_tokens(tokens, filter_ts=True):
    
    if not filter_ts:
        return tokens, [[i] for i in range(len(tokens))]
    
    #print("Filtering tokens from {}".format(tokens))    
        
    filtered_tokens = []
    indices = []
    for i in range(len(tokens)):
        if tokens[i] != '[UNK]':
            if i<len(tokens) and '##' in tokens[i]:
                filtered_tokens[-1] += tokens[i]
                indices[-1].append(i)
            else:
                filtered_tokens.append(tokens[i])
                indices.append([i])

    return filtered_tokens, indices


## filter out rows and columns for UNK tokens and combine rows and columns for words that were split by the tokenizer
## indices is a list of indices, each such list represents a "complete" token
## example:
'''
sentence: 'La conf??rence  sur l???histoire  dont l???organizateur m???a parl?? a commenc?? plus tard que pr??vu .'
tokenized tokens: ['La', 'conf??rence', 'sur', 'l', '[UNK]', 'histoire', 'dont', 'l', '[UNK]', 'organi', '##zate', '##ur', 'm', '[UNK]', 'a', 'par', '##l??', 'a', 'commenc??', 'plus', 'tard', 'que', 'pr??vu', '.']
filtered tokens: ['La', 'conf??rence', 'sur', 'l', 'histoire', 'dont', 'l', 'organi##zate##ur', 'm', 'a', 'par##l??', 'a', 'commenc??', 'plus', 'tard', 'que', 'pr??vu', '.']
indices: [[0], [1], [2], [3], [5], [6], [7], [9, 10, 11], [12], [14], [15, 16], [17], [18], [19], [20], [21], [22], [23]]
'''

def recompute_attn_mat(attn_mat, indices):
    
    n = len(indices)
    attn = np.zeros((n,n))
    for i in range(len(indices)):
        attn[i] = make_vec(i, attn_mat, indices)
        attn[:,i] = make_vec(i, np.transpose(attn_mat), indices)
        
    return attn


def make_vec(i, mat, inds):
    return [sum(mat[i][inds[j][0]:inds[j][-1]+1]) for j in range(len(inds))]

    