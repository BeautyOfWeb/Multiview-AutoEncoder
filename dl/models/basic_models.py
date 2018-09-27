import functools
import collections
import numpy as np

import torch
import torch.nn as nn


def get_attr(obj, attribute_path):
  """Recursively get the 'deep' attributes of any object

  Args:
    obj: any python object with accessible attributes
    attribute_path: a list of attribute names (str); or a 'mixed' list of names and one list/tuple/range

  Example:
    Can only handle the following two cases:
      To get model.decoder.weight, call get_attr(model, ['decoder', 'weight'])
      To get [model.decoders[0].weight, ..., model.decoders[k-1].weight], 
        call get_attr(model, ['decoders', range(k), 'weight'])
        range(k) can be replaced with [0,...,k-1] or ['0',..., 'k-1']

  """
  attr = obj
  for i, p in enumerate(attribute_path):
    if isinstance(p, (list, tuple, range)):
      attr_list = []
      for s in p: 
        attr_list.append(get_attr(attr, [str(s)] + attribute_path[i+1:]))
      return attr_list
    if isinstance(p, str):
      attr = getattr(attr, p)
  return attr


def get_list(x, n, forced=False):
    r"""If x is not a list of length n, duplicate it to be a list of length n
    For a special case when the input is an iterable and len(x) = n, 
     but we still need to copy it to a list of length n, set forced=True
    """
    if forced:
        return [x] * n
    if not isinstance(x, collections.Iterable):
        # Note: np.array, list, str are always iterable
        x = [x] * n
    if len(x) != n:
        x = [x] * n
    return x

def update_model_state(model, state_dict):
    """Update model.state_dict() with state_dict
    
    Args:
        model: torch.nn.Module, model to be updated
        state_dict: In most cases, state_dict = model_pretrained.state_dict()
    """
    state_dict_common = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    state_dict = model.state_dict()
    state_dict.update(state_dict_common) # update will combine state_dict_common to state_dict
    model.load_state_dict(state_dict)


class DenseLinear(nn.Module):
    r"""Multiple linear layers densely connected
    
    Args:
        in_dim: int, number of features
        hidden_dim: iterable of int
        nonlinearity: default nn.ReLU()
                        can be changed to other nonlinear activations
        last_nonlinearity: if True, apply nonlinearity to the last output; default False
        dense: if dense, concatenate all previous intermediate features to current input
        forward_input: should the original input be concatenated to current input used when dense is True
                        if return_all is True and return_layers is None and forward_input is True, 
                            then concatenate input with all hidden outputs as final output
        return_all: if True return all layers
        return_layers: selected layers to output; used only when return_all is True
        bias: if True, use bias in nn.Linear()
        residual: if True, then dense and forward_input is False; add skip connection for residual_layers
        residual_layers: a list or string; add skip connections among these hidden_layers
            if 'all', then set residual_layers = list(range(num_layers))
            if 'all-but-last', then set residual_layers = list(range(num_layers-1))
        return_list: used only when return_all is True; if True, return a list y instead of torch.cat(y, dim=-1)
        
    Shape:
        Input: (N, *, in_dim)
        Output: (N, *, out_dim) 
            out_dim depends on hidden_dim, dense, forward_input (and in_dim), return_all, and return_layers
    
    Attributes:
        A series of weights (and biases) from nn.Linear
    
    Examples:
    
        >>> m = DenseLinear(3, [3,4], return_all=True)
        >>> x = torch.randn(4,3)
        >>> m(x)
    """
    def __init__(self, in_dim, hidden_dim, nonlinearity=nn.ReLU(), last_nonlinearity=False, bias=True,
        dense=True, residual=False, residual_layers='all-but-last', forward_input=False, return_all=False, 
        return_layers=None, return_list=False):
        super(DenseLinear, self).__init__()
        num_layers = len(hidden_dim)
        nonlinearity = get_list(nonlinearity, num_layers)
        bias = get_list(bias, num_layers)
        self.forward_input = forward_input
        self.return_all = return_all
        self.return_layers = return_layers
        self.dense = dense
        self.last_nonlinearity = last_nonlinearity
        self.residual = residual
        self.return_list = return_list
        if self.residual:
            assert not dense and not forward_input
            if residual_layers == 'all':
                residual_layers = list(range(num_layers))
            if residual_layers == 'all-but-last':
                residual_layers = list(range(num_layers-1))
            self.residual_layers = residual_layers
            for i in range(1, len(self.residual_layers)):
                assert hidden_dim[self.residual_layers[i-1]] == hidden_dim[self.residual_layers[i]]
        self.layers = nn.Sequential()
        for i, h in enumerate(hidden_dim):
            self.layers.add_module(f'linear{i}', nn.Linear(in_dim, h, bias[i]))
            if i < num_layers-1 or last_nonlinearity:
                self.layers.add_module(f'activation{i}', nonlinearity[i])
            if dense:
                if i==0 and not forward_input:
                    in_dim = 0
                in_dim += h
            else:
                in_dim = h
            
    def forward(self, x):
        if self.forward_input:
            y = [x]
        else:
            y = []
        out = x
        if self.residual:
            last_layer = self.residual_layers[0]
        for i, (n, m) in enumerate(self.layers._modules.items()):
            out = m(out)
            if n.startswith('activation'):
                y.append(out)
                if self.dense and i < len(self.layers)-1: # if the last layer is nonlinearity, don't cat
                    out = torch.cat(y, dim=-1)
            if self.residual and n.startswith('linear') and int(n[6:]) in self.residual_layers[1:]:
                out += y[last_layer]
                last_layer = int(n[6:])
        if self.return_all:
            if not self.last_nonlinearity: # add last output even if there is no nonlinearity
                y.append(out)
            if self.return_layers is not None:
                return_layers = [i%len(y) for i in self.return_layers]
                y = [h for i, h in enumerate(y) if i in return_layers]
            if self.return_list:
              return y
            else:
              return torch.cat(y, dim=-1)
        else:
            return out
        
    
class FineTuneModel(nn.Module):
    r"""Finetune the last layer(s) (usually newly added) with a pretained model to learn a representation
    
    Args:
        pretained_model: nn.Module, pretrained module
        new_layer: nn.Module, newly added layer
        freeze_pretrained: if True, set requires_grad=False for pretrained_model parameters
        
    Shape:
        - Input: (N, *)
        - Output: 
        
    Attributes:
        All model parameters of pretrained_model and new_layer
    
    Examples:
    
        >>> m = nn.Linear(2,3)
        >>> model = FineTuneModel(m, nn.Linear(3,2))
        >>> x = Variable(torch.ones(1,2))
        >>> print(m(x))
        >>> print(model(x))
        >>> print(FeatureExtractor(model, [0,1])(x))
    """
    def __init__(self, pretrained_model, new_layer, freeze_pretrained=True):
        super(FineTuneModel, self).__init__()
        self.pretrained_model = pretrained_model
        self.new_layer = new_layer
        if freeze_pretrained:
            for p in self.pretrained_model.parameters():
                p.requires_grad = False
                
    def forward(self, x):
        return self.new_layer(self.pretrained_model(x))
    
    
class FeatureExtractor(nn.Module):
    r"""Extract features from different layers of the model
    
    Args:
        model: nn.Module, the model
        selected_layers: an iterable of int or 'string' (as module name), selected layers
        
    Shape:
        - Input: (N,*)
        - Output: a list of Variables, depending on model and selected_layers
        
    Attributes: 
        None learnable
       
    Examples:
    
        >>> m = nn.Sequential(nn.Linear(2,2), nn.Linear(2,3))
        >>> m = FeatureExtractor(m, [0,1])
        >>> x = Variable(torch.randn(1, 2))
        >>> m(x)
    """
    def __init__(self, model, selected_layers=None, return_list=False):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.selected_layers = selected_layers
        if self.selected_layers is None:
            self.selected_layers = range(len(model._modules))
        self.return_list = return_list
    
    def set_selected_layers(self, selected_layers):
        self.selected_layers = selected_layers
        
    def forward(self, x):
        out = []
        for i, (name, m) in enumerate(self.model._modules.items()):
            x = m(x)
            if i in self.selected_layers or name in self.selected_layers:
                out.append(x)
        if self.return_list:
            return out
        else:
            return torch.cat(out, dim=-1)
    

class WeightedFeature(nn.Module):
    r"""Transform features into weighted features
    
    Args:
        num_features: int
        reduce: if True, return weighted mean
        
    Shape: 
        - Input: (N, *, num_features) where * means any number of dimensions
        - Output: (N, *, num_features) if reduce is False (default) else (N, *)
        
    Attributes:
        weight: (num_features)
        
    Examples::
    
        >>> m = WeightedFeature(10)
        >>> x = torch.autograd.Variable(torch.randn(5,10))
        >>> out = m(x)
        >>> print(out)
    """
    def __init__(self, num_features, reduce=False, magnitude=None):
        super(WeightedFeature, self).__init__()
        self.reduce = reduce
        self.weight = nn.Parameter(torch.empty(num_features))
        # initialize with uniform weight
        self.weight.data.fill_(1)
        self.magnitude = 1 if magnitude is None else magnitude
    
    def forward(self, x):
        self.normalized_weight = torch.nn.functional.softmax(self.weight, dim=0)
        # assert x.shape[-1] == self.normalized_weight.shape[0]
        out = x * self.normalized_weight * self.magnitude
        if self.reduce:
            return out.sum(-1)
        else:
            return out

        
class WeightedView(nn.Module):
    r"""Calculate weighted view
    
    Args:
        num_groups: int, number of groups (views)
        reduce_dimension: bool, default False. If True, reduce dimension dim
        dim: default -1. Only used when reduce_dimension is True
        
    Shape: 
        - Input: if dim is None, (N, num_features*num_groups)
        - Output: (N, num_features)
        
    Attributes:
        weight: (num_groups)
        
    Examples:
    
        >>> model = WeightedView(3)
        >>> x = Variable(torch.randn(1, 6))
        >>> print(model(x))
        >>> model = WeightedView(3, True, 1)
        >>> model(x.view(1,3,2))
    """
    def __init__(self, num_groups, reduce_dimension=False, dim=-1):
        super(WeightedView, self).__init__()
        self.num_groups = num_groups
        self.reduce_dimension = reduce_dimension
        self.dim = dim
        self.weight = nn.Parameter(torch.Tensor(num_groups))
        self.weight.data.uniform_(-1./num_groups, 1./num_groups)
    
    def forward(self, x):
        self.normalized_weight = nn.functional.softmax(self.weight, dim=0)
        if self.reduce_dimension:
            assert x.size(self.dim) == self.num_groups
            dim = self.dim if self.dim>=0 else self.dim+x.dim()
            if dim == x.dim()-1:
                out = (x * self.weight).sum(-1)
            else:
                # this is tricky for the case when x.dim()>3
                out = torch.transpose((x.transpose(dim,-1)*self.normalized_weight).sum(-1), dim, -1)
        else:
            assert x.dim() == 2
            num_features = x.size(-1) // self.num_groups
            out = (x.view(-1, self.num_groups, num_features).transpose(1, -1)*self.normalized_weight).sum(-1)
        return out    