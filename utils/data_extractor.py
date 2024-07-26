
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self,ListOfLists):
        self.tuples = list(zip(*ListOfLists))
    
    def __len__(self):
        return len(self.tuples)
    
    def __getitem__(self,idx):
        return self.tuples[idx]
    
    def __size__(self):
        return self.tuples.size()


def divide_in_batches(data, batch_size):
    sublists = [data[x:x+batch_size] for x in range(0, len(data), batch_size)]
    return sublists

def data_loader(ListOfLists, batch, shuf):    
    data = SeqDataset(ListOfLists)
    return DataLoader(data, batch_size=batch, shuffle=shuf)


def wrapper(data, seq, model, dim = 768, label = False, labels_dict = None, tokenizer = None, device = "cpu"):
    
    with torch.no_grad():
        
        if seq:
            if label:
                vecs = torch.randn((len(data), len(data[0])))
            else: 
                vecs = torch.randn((len(data), len(data[0]), dim))

            for i, sequence in enumerate(data):
                if not label:
                    vecs[i] = torch.from_numpy(encode_sequence(sequence, model, tokenizer, dim, device))
                elif labels_dict is None:
                    vecs[i] = torch.from_numpy(np.asarray([torch.tensor(-1) if x=="False" else torch.tensor(1) for x in sequence]))
                else:
                    vecs[i] = torch.from_numpy(np.asarray([torch.tensor(labels_dict[x]) for x in sequence]))

        else:
            if not label:
                vecs = torch.from_numpy(model.encode(data))
            else:
                if labels_dict is None:
                    vecs = torch.from_numpy(np.asarray([torch.tensor(-1) if x=="False" else torch.tensor(1) for x in data]))
                else:
                    vecs = torch.from_numpy(np.asarray([torch.tensor(labels_dict[x]) for x in data])) 

        torch.cuda.empty_cache()
    
    return vecs


def encode_sequence(sequence, model, tokenizer, dim, device):

    representation = []
    for sentence in sequence:
        token_ids = torch.tensor([tokenizer.encode(sentence)])
        last_layer = model(token_ids)[0]
        #The BERT/RoBERTa/Electra [CLS] token corresponds to the first hidden state of the last layer
        #print("sentence embedding: {}".format(last_layer[:, 0, :].shape))
        representation.append(last_layer[:, 0, :].cpu().detach().numpy())

    return np.asarray(representation, dtype='float32').squeeze()
