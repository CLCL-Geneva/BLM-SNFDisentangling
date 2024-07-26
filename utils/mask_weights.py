'''
Created on Aug 10, 2023

@author: vivi
'''

import logging

import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid(x):
    return float(1./(1.+np.exp(-x)))


'''___________________________________________________
    1D mask classes 
'''


class Mask(nn.Module):
## based on https://github.com/lolemacs/continuous-sparsification/blob/master/models/layers.py#L21

    def __init__(self, n, device="cuda:0", mask_initial_value:float = 0., eps:float = 1e-12, temperature: float = 0.000001):
        super(Mask, self).__init__()
        
        self.mask_initial_value = mask_initial_value
        self.device = device
        self.nr_masked_units = n

        self.eps = eps

        self.start_temp = temperature
        self.temperature = temperature
     
        self.weight = nn.Parameter(torch.Tensor(self.nr_masked_units))
        nn.init.constant_(self.weight, math.sqrt(self.nr_masked_units))

        self.init_mask()
        self.mask = torch.sigmoid(self.temperature * self.mask_weight)
        
    def init_mask(self):
        self.mask_weight = nn.Parameter(torch.Tensor(self.nr_masked_units))
        nn.init.uniform_(self.mask_weight)

    ## Gumbel-Softmax style, to enforce disjoint subnetwork -- one mask corresponds to one weight in the weight matrix that is used to compute values for the next layer! 
    ## the mask is actually a distribution probability over subnetworks (how likely it is that this edge (weight) belongs to each of the subnetworks
    ## use GS to sample fro mthis probability distribution to enfore that one edge (weight) belongs to only one subnetwork
    def compute_mask(self):
        return torch.softmax(self.mask_weight/self.temperature, 0)
        #return F.gumbel_softmax(self.mask_weight, self.temperature, hard=False)
            
    def prune(self):
        self.mask_weight.data = torch.clamp(self.temperature * self.mask_weight.data, max=self.mask_initial_value)   

    def forward(self, x):      

        self.mask = self.compute_mask()
        masked_weight = self.weight.to(self.device) * self.mask.to(self.device)

        return x.to(self.device) * masked_weight.to(self.device)
    
    def get_masks(self):
        return self.mask.to(self.device).detach().cpu().numpy() 

    def get_masks_as_tensors(self):
        return self.mask
    
    def load_mask(self, mask):
        self.mask_weight = nn.Parameter(mask)
    
    def get_weights(self):
        return (self.weight.to(self.device) * self.mask.to(self.device)).detach().cpu().numpy()
        
        
    def checkpoint(self):
        self.init_weight.data = self.weight.clone()       
        
    def rewind_weights(self):
        self.weight.data = self.init_weight.clone()       
        
    ## if the model improved, update the temperature: decrease it to push the system towards more "discrete" subnetworks
    ## I found that this doesn't improve learning, so made this trivial for code compatibility
    def update_temperature(self, temp=None):
        '''
        self.temperature *= self.temp_decrease_rate
        self.temp_decrease_rate += 0.01  ## increase this to slow down the temperature decrease
        ## maybe this factor (0.01) should be 1/nr_epochs, to ensure that it remains low, but not too low
        '''
        self.temperature = self.start_temp
        
        
class MaskedLinear(nn.Module):
    #### based on: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, input_len: int, output_len: int, nr_masked_outputs: int, bias: bool = True,
                 device="cuda:0", dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(MaskedLinear, self).__init__()

        self.in_features = input_len
        self.out_features = output_len
        self.nr_masked_units = nr_masked_outputs ## the number of subnets is the same as the number of masked features: one subnet models one feature

        self.device = device

        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        ## we mask each
        self.masked_weights = nn.Parameter(torch.ones((self.in_features, self.out_features)), requires_grad=False)
        self.masks = nn.ModuleList([Mask(self.nr_masked_units, device=self.device) for _ in range(self.in_features)])

        self.reset_parameters()



    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(self.out_features * self.in_features))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        for m in self.masks:
            m.init_mask()
            
            
    def update_temperature(self, temp=None):
        for m in self.masks:
            m.update_temperature(temp)


    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        
        ## apply the masks to the weights of the linear layer (the weights corresponding to sd values are not masked)
        self.masked_weights.data = torch.ones((self.in_features, self.out_features))

        for i in range(self.in_features):
            self.masked_weights.data[...,i,:self.nr_masked_units] = self.masks[i](self.weight[...,i][:self.nr_masked_units])

        return F.linear(inp.to(self.device), torch.transpose(self.masked_weights, -2, -1).to(self.device), self.bias.to(self.device))
 
    
    def get_weights(self):
        logging.info("temperature: {}".format(self.masks[0].temperature))
        weights_list = [m.get_weights() for m in self.masks]
        return np.concatenate(weights_list)

    def get_weights_array(self):
        return np.vstack([m.get_weights() for m in self.masks])

    def get_masks(self):
        logging.info("temperature: {}".format(self.masks[0].temperature))
        masks_list = [m.get_masks() for m in self.masks]
        return np.vstack(masks_list)

    def get_masks_as_tensors(self):
        return [m.get_masks_as_tensors() for m in self.masks]

    def load_masks(self, masks_array):
        for i in range(self.in_features):
            self.masks[i].load_mask(masks_array[i])
            
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
    

##________________________________________________________________________________________________________________________________________


class RevMaskedLinear(nn.Module):
    #### based on: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, input_len: int, output_len: int, nr_masked_outputs: int, bias: bool = True,
                 device="cuda:0", dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(RevMaskedLinear, self).__init__()

        self.in_features = input_len
        self.out_features = output_len
        self.nr_masked_units = nr_masked_outputs ## the number of subnets is the same as the number of masked features: one subnet models one feature

        self.device = device

        self.weight = nn.Parameter(torch.empty((self.in_features, self.out_features), **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(self.in_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        ## we mask each 
        self.masks = nn.ModuleList([RevMask(self.in_features, device=self.device) for _ in range(self.nr_masked_units)])
        self.masked_weights = nn.Parameter(torch.ones((self.out_features, self.in_features)), requires_grad=False)

        self.reset_parameters()



    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(self.out_features * self.in_features))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        for m in self.masks:
            m.init_mask()
            
            
    def update_temperature(self, temp=None):
        for m in self.masks:
            m.update_temperature(temp)


    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        
        ## apply the masks to the weights of the linear layer (the weights corresponding to sd values are not masked)
        self.masked_weights.data = torch.ones((self.out_features, self.in_features))

        for i in range(self.nr_masked_units):
            self.masked_weights.data[...,i,:self.in_features] = self.masks[i](self.weight[...,i][:self.in_features])

        return F.linear(inp.to(self.device), self.masked_weights.to(self.device))    #, self.bias.to(self.device))
    
    def get_weights(self):
        logging.info("temperature: {}".format(self.masks[0].temperature))
        weights_list = [m.get_weights() for m in self.masks]
        return np.concatenate(weights_list)

    def get_weights_array(self):
        return [m.get_weights() for m in self.masks]

    def get_masks(self):
        logging.info("temperature: {}".format(self.masks[0].temperature))
        masks_list = [m.get_masks() for m in self.masks]
        return np.concatenate(masks_list)

    def get_masks_as_tensors(self):
        return [m.get_masks_as_tensors() for m in self.masks]

    def load_masks(self, masks_array):
        for i in range(self.in_features):
            self.masks[i].load_mask(masks_array[i])
            
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
    

##____________________________________________________________________________________________


class RevMask(nn.Module):
## based on https://github.com/lolemacs/continuous-sparsification/blob/master/models/layers.py#L21

    def __init__(self, n, device="cuda:0", mask_initial_value:float = 0., eps:float = 1e-12, temperature: float = 0.000001):
        super(RevMask, self).__init__()
        
        self.mask_initial_value = mask_initial_value
        self.device = device
        self.nr_masked_units = n

        self.eps = eps

        self.temperature = temperature
     
        self.weight = nn.Parameter(torch.Tensor(self.nr_masked_units))
        nn.init.constant_(self.weight, math.sqrt(self.nr_masked_units))

        self.init_mask()
        self.mask = torch.sigmoid(self.temperature * self.mask_weight)
        
        
    def init_mask(self):
        self.mask_weight = nn.Parameter(torch.Tensor(self.nr_masked_units))
        nn.init.uniform_(self.mask_weight)

    ## Gumbel-Softmax style, to enforce disjoint subnetwork -- one mask corresponds to one weight in the weight matrix that is used to compute values for the next layer! 
    ## the mask is actually a distribution probability over subnetworks (how likely it is that this edge (weight) belongs to each of the subnetworks
    ## use GS to sample fro mthis probability distribution to enfore that one edge (weight) belongs to only one subnetwork
    def compute_mask(self):
        return torch.softmax(self.mask_weight/self.temperature, 0)
        #return F.gumbel_softmax(self.masked_weight, self.temperature, hard=False)
            
 
    def forward(self, x):      
        self.mask = self.compute_mask()
        masked_weight = self.weight.to(self.device) * self.mask.to(self.device)
        
        return x.to(self.device) * masked_weight.to(self.device)
    
    def get_masks(self):
        return self.mask.to(self.device).detach().cpu().numpy() 

    def get_masks_as_tensors(self):
        return self.mask
    
    def load_mask(self, mask):
        self.mask.data = nn.Parameter(mask)
    
    def get_weights(self):
        return (self.weight.to(self.device) * self.mask.to(self.device)).detach().cpu().numpy()
                        