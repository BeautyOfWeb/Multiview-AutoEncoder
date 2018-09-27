import sys
import os
import collections

import numpy as np

lib_path = 'I:/code'
if not os.path.exists(lib_path):
  lib_path = '/media/6T/.tianle/.lib'
if os.path.exists(lib_path) and lib_path not in sys.path:
  sys.path.append(lib_path)

import torch
import torch.nn as nn

from dl.models.basic_models import DenseLinear, get_list, get_attr
from dl.utils.train import cosine_similarity

class AutoEncoder(nn.Module):
  r"""Factorization autoencoder
  
  Args:
  
  Shape:
  
  Attributes:
  
  Examples::
  
  
  """
  def __init__(self, in_dim, hidden_dims, num_classes, dense=True, residual=False, residual_layers='all',
    decoder_norm=False, decoder_norm_dim=0, uniform_decoder_norm=False, nonlinearity=nn.ReLU(), 
    last_nonlinearity=True, bias=True):
    super(AutoEncoder, self).__init__()
    self.encoder = DenseLinear(in_dim, hidden_dims, nonlinearity=nonlinearity, last_nonlinearity=last_nonlinearity, 
      dense=dense, residual=residual, residual_layers=residual_layers, forward_input=False, return_all=False, 
      return_layers=None, bias=bias)
    self.decoder_norm = decoder_norm
    self.uniform_decoder_norm = uniform_decoder_norm
    if self.decoder_norm:
      self.decoder = nn.utils.weight_norm(nn.Linear(hidden_dims[-1], in_dim), 'weight', dim=decoder_norm_dim)
      if self.uniform_decoder_norm:
        self.decoder.weight_g.data = self.decoder.weight_g.new_ones(1) # This changed the tensor shape, but it's ok
        self.decoder.weight_g.requires_grad_(False)
    else:
      self.decoder = nn.Linear(hidden_dims[-1], in_dim)
    self.classifier = nn.Linear(hidden_dims[-1], num_classes)
    
  def forward(self, x):
    out = self.encoder(x)
    return self.classifier(out), self.decoder(out)


class MultiviewAE(nn.Module):
  r"""Multiview autoencoder. 

  Args:
    in_dims: a list (or iterable) of integers
    hidden_dims: a list of ints if every view has the same hidden_dims; otherwise a list of lists of ints
    out_dim: for classification, out_dim = num_cls
    fuse_type: default 'sum', add up the outputs of all encoders; require all ouputs has the same dimensions
      if 'cat', concatenate the outputs of all encoders
    dense, residual, residual_layers, nonlinearity, last_nonlinearity, bias are passed to DenseLinear
    decoder_norm: if True, add forward prehook torch.nn.utils.weight_norm  to decoder (a nn.Linear module)
    decoder_norm_dim: default 0; pass to torch.nn.utils.weight_norm
    uniform_decoder_norm: if True, ensure that decoder weight norm is 1 for dim=decoder_norm_dim

  Shape:
    Input: can be a list of tensors or a single tensor which will be splitted into a list
    Output: two heads: score matrix of shape (N, out_dim), concatenated decoder output: (N, sum(in_dims))

  Attributes:
    A list of DenseLinear modules as encoders and decoders
    An nn.Linear as output layer (e.g., class score matrix)

  Examples:
    >>> x = torch.randn(10, 5)
    >>> model = MultiviewAE([2,3], [5, 5], 7)
    >>> y = model(x)
    >>> y[0].shape, y[1].shape

  """
  def __init__(self, in_dims, hidden_dims, out_dim, fuse_type='sum', dense=False, residual=True, 
    residual_layers='all', decoder_norm=False, decoder_norm_dim=0, uniform_decoder_norm=False, 
    nonlinearity=nn.ReLU(), last_nonlinearity=True, bias=True):
    super(MultiviewAE, self).__init__()
    self.num_views = len(in_dims)
    self.in_dims = in_dims
    self.out_dim = out_dim
    self.fuse_type = fuse_type
    if not isinstance(hidden_dims[0], collections.Iterable):
      # hidden_dims is a list of ints, which means all views have the same hidden dims
      hidden_dims = [hidden_dims] * self.num_views
    self.hidden_dims = hidden_dims
    assert len(self.hidden_dims) == self.num_views and isinstance(self.hidden_dims[0], collections.Iterable)
    self.encoders = nn.ModuleList()
    self.decoders = nn.ModuleList()
    for in_dim, hidden_dim in zip(in_dims, hidden_dims):
      self.encoders.append(DenseLinear(in_dim, hidden_dim, nonlinearity=nonlinearity, 
        last_nonlinearity=last_nonlinearity, dense=dense, forward_input=False, return_all=False, 
        return_layers=None, bias=bias, residual=residual, residual_layers=residual_layers))
      decoder = nn.Linear(hidden_dim[-1], in_dim)
      if decoder_norm:
        torch.nn.utils.weight_norm(decoder, 'weight', dim=decoder_norm_dim)
        if uniform_decoder_norm:
          decoder.weight_g.data = decoder.weight_g.new_ones(decoder.weight_g.size())
          decoder.weight_g.requires_grad_(False)
      self.decoders.append(decoder)
    self.fuse_dims = [hidden_dim[-1] for hidden_dim in self.hidden_dims]
    if self.fuse_type == 'sum':
      fuse_dim = self.fuse_dims[0]
      for d in self.fuse_dims:
        assert d == fuse_dim
    elif self.fuse_type == 'cat':
      fuse_dim = sum(self.fuse_dims)
    else:
      raise ValueError(f"fuse_type should be 'sum' or 'cat', but is {fuse_type}")
    self.output = nn.Linear(fuse_dim, out_dim)

  def forward(self, xs):
    if isinstance(xs, torch.Tensor):
      xs = xs.split(self.in_dims, dim=1)
    # assert len(xs) == self.num_views
    encoder_out = []
    decoder_out = []
    for i, x in enumerate(xs):
      out = self.encoders[i](x)
      encoder_out.append(out)
      decoder_out.append(self.decoders[i](out))
    if self.fuse_type == 'sum':
      out = torch.stack(encoder_out, dim=-1).mean(dim=-1)
    else:
      out = torch.cat(encoder_out, dim=-1)
    out = self.output(out)
    return out, torch.cat(decoder_out, dim=-1), torch.cat(encoder_out, dim=-1)


def get_interaction_loss(interaction_mat, w, loss_type='graph_laplacian', normalize=True):
  """Calculate loss on the inconsistency between feature representations w (N*D) 
  and feature interaction network interaction_mat (N*N)
  A trivial solution is all features (row vectors of w) have cosine similarity = 1 or distance = 0
  
  Args:
    interaction_mat: non-negative symmetric torch.Tensor with shape (N, N)
    w: feature representation tensor with shape (N, D)
    normalize: if True, call w = w / w.norm(p=2, dim=1, keepdim=True) /np.sqrt(w.size(0)) 
      for loss_type = 'graph_laplacian' or 'dot_product',
        this makes sure w.norm() = 1 and the row vectors of w have the same norm: len(torch.unique(w.norm(dim=1)))==1
      call loss = loss / w.size(0) for loss_type = 'cosine_similarity'; 
      By doing this we ensure the number of features is factored out; 
      this is useful for combining losses from multi-views.

  See Loss_feature_interaction for more documentation

  """
  if loss_type == 'cosine_similarity':
    # -(|cos(w,w)| * interaction_mat).sum()
    cos = cosine_similarity(w).abs() # get the absolute value of cosine simiarity
    loss = -(cos * interaction_mat).sum()
    if normalize:
      loss = loss / w.size(0)
  elif loss_type == 'graph_laplacian':
    # trace(w' * L * w)
    if normalize:
      w = w / w.norm(p=2, dim=1, keepdim=True) / np.sqrt(w.size(0))
      interaction_mat = interaction_mat / interaction_mat.norm() # this will ensure interaction_mat is normalized
    diag = torch.diag(interaction_mat.sum(dim=1))
    L_interaction_mat = diag - interaction_mat
    loss = torch.diagonal(torch.mm(torch.mm(w.t(), L_interaction_mat), w)).sum()
  elif loss_type == 'dot_product':
    # pairwise distance mat * interaction mat
    if normalize:
      w = w / w.norm(p=2, dim=1, keepdim=True) / np.sqrt(w.size(0))
    d = torch.sum(w*w, dim=1) # if normalize is True, then d is a vector of the same element 1/w.size(0)
    dist = d.unsqueeze(1) + d - 2*torch.mm(w, w.t())
    loss = (dist * interaction_mat).sum()
    # loss = (dist / dist.norm() * interaction_mat).sum() # This is an alternative to 'normalize' loss
  else:
    raise ValueError(f"loss_type can only be 'cosine_similarity', "
                     f"graph_laplacian' or 'dot_product', but is {loss_type}")
  return loss


class Loss_feature_interaction(nn.Module):
  r"""A customized loss function for a graph Laplacian constraint on the feature interaction network
    For factorization autoencoder model, the decoder weights can be seen as feature representations;
    This loss measures the inconsistency between learned feature representations and their interaction network.
    A trivial solution is all features have cosine similarity = 1 or distance = 0

  Args:
    interaction_mat: torch.Tensor of shape (N, N), a non-negative (symmetric) matrix; 
      or a list of matrices; each is an interaction mat; 
      To control the magnitude of the loss, it is preferred to have argument interaction_mat.norm() = 1
    loss_type: if loss_type == 'cosine_similarity', calculate -(cos(m, m).abs() * interaction_mat).sum()
               if loss_type == 'graph_laplacian' (faster), calculate trace(m' * L * m)
               if loss_type == 'dot_product', calculate dist(m) * interaction_mat 
                 where dist(m) is the pairwise distance matrix of features; the name 'dot_product' is misleading
              If all features have norm 1, all three types are equivalent in a sense
              cosine_similarity is preferred because the magnitude of features are implicitly ignored, 
               while the other two will be affected by the magnitude of features.
    weight_path: default ['decoder', 'weight'], with the goal to get w = model.decoder.weight
    normalize: pass it to get_interaction_loss; 
      if True, call w = w / w.norm(p=2, dim=1, keepdim=True) / np.sqrt(w.size(0))
        for loss_type 'graph_laplacian' or 'dot_product',
          this makes sure each row vector of w has the same norm, and w.norm() = 1
        call loss = loss / w.size(0) for loss_type = 'cosine_similarity'; 
      By doing this we ensure the number of features is factored out; 
      this is useful for combining losses from multi-views.
  
  Inputs:
    model: the above defined AutoEnoder model or other model
    or given weight matrix w
    if interaction_mat has shape (N,N), then w has shape (N, D)

  Returns:
    loss: torch.Tensor that can call loss.backward()
  """

  def __init__(self, interaction_mat, loss_type='graph_laplacian', weight_path=['decoder', 'weight'], 
    normalize=True):
    super(Loss_feature_interaction, self).__init__()
    self.loss_type = loss_type
    self.weight_path = weight_path
    self.normalize = normalize
    # If interaction_mat is a list, self.sections will be the used for splitting the weight matrix
    self.sections = None # when interaction_mat is a single matrix, self.sections is None
    if isinstance(interaction_mat, (list, tuple)):
      if normalize: # ensure interaction_mat is normalized
        interaction_mat = [m/m.norm() for m in interaction_mat]
      self.sections = [m.shape[0] for m in interaction_mat]
    else:
      if normalize: # ensure interaction_mat is normalized
        interaction_mat = interaction_mat / interaction_mat.norm()
    if self.loss_type == 'graph_laplacian':
      # precalculate self.L_interaction_mat save some compute for each forward pass
      if self.sections is None:
        diag = torch.diag(interaction_mat.sum(dim=1))
        self.L_interaction_mat = diag - interaction_mat # Graph Laplacian; should I normalize it?
      else:
        self.L_interaction_mat = []
        for mat in interaction_mat:
          diag = torch.diag(mat.sum(dim=1))
          self.L_interaction_mat.append(diag - mat)
    else: # we don't need to store interaction_mat for loss_type=='graph_laplacian'
      self.interaction_mat = interaction_mat
  
  def forward(self, model=None, w=None):
    if w is None:
      w = get_attr(model, self.weight_path)
    if self.sections is None:
      # There is only one interaction matrix; self.interaction_mat is a torch.Tensor
      if self.loss_type == 'graph_laplacian':
        # Used precalculated L_interaction_mat to save some time
        if self.normalize:
          # interaction_mat had already been normalized during initialization
          w = w / w.norm(p=2, dim=1, keepdim=True) / np.sqrt(w.size(0))
        return torch.diagonal(torch.mm(torch.mm(w.t(), self.L_interaction_mat), w)).sum()
      else:
        return get_interaction_loss(self.interaction_mat, w, loss_type=self.loss_type, normalize=self.normalize)
    else:
      # self.interaction_mat is a list of torch.Tensors
      if isinstance(w, torch.Tensor):
        w = w.split(self.sections, dim=0)
      if self.loss_type == 'graph_laplacian': # handle 'graph_laplacian' differently to save time during training
        loss = 0
        for w_, L in zip(w, self.L_interaction_mat):
          if self.normalize: # make sure w_.norm() = 1 and each row vector of w_ has the same norm
            w_ = w_ / w_.norm(p=2, dim=1, keepdim=True) / np.sqrt(w_.size(0))
          loss += torch.diagonal(torch.mm(torch.mm(w_.t(), L), w_)).sum()  
        return loss
      # for the case 'cosine_similarity' and 'dot_product'
      return sum([get_interaction_loss(mat, w_, loss_type=self.loss_type, normalize=self.normalize) 
                  for mat, w_ in zip(self.interaction_mat, w)])


class Loss_view_similarity(nn.Module):
  r"""The input is a multi-view representation of the same set of patients, 
      i.e., a set of matrices with shape (num_samples, feature_dim). feature_dim can be different for each view
    This loss will penalize the inconsistency among different views.
    This is somewhat limited, because different views should have both shared and complementary information
      This loss only encourages the shared information across views, 
      which may or may not be good for certain applications.
    A trivial solution for this is multi-view representation are all the same; then loss -> -1
    The two loss_types 'circle' and 'hub' can be quite different and unstable.
      'circle' tries to make all feature representations across views have high cosine similarity,
      while 'hub' only tries to make feature representations within each view have high cosine similarity;
      by multiplying 'mean-feature' target with 'hub' loss_type, it might 'magically' capture both within-view and 
        cross-view similarity; set as default choice; but my limited experimental results do not validate this;
        instead, 'circle' and 'hub' are dominant, while explicit_target and cal_target do not make a big difference 
    Cosine similarity are used here; To do: other similarity metrics

  Args:
    sections: a list of integers (or an int); this is used to split the input matrix into chunks;
      each chunk corresponds to one view representation.
      If input xs is not a torch.Tensor, this will not be used; assume xs to be a list of torch.Tensors
      sections being an int implies all feature dim are the same, set sections = feature_dim, NOT num_sections!
    loss_type: supose there are three views x1, x2, x3; let s_ij = cos(x_i,x_j), s_i = cos(x_i,x_i)
      if loss_type=='cicle', similarity = s12*s23*target if fusion_type=='multiply'; s12+s23 if fusion_type=='sum'                   
        This is fastest but requires x1, x2, x3 have the same shape
      if loss_type=='hub', similarity=s1*s2*s3*target if fusion_type=='multiply'; 
        similarity=|s1|+|s2|+|s3|+|target| if fusion_type=='sum'
        Implicitly, target=1 (fusion_type=='multiply) or 0 (fusion_type=='sum') if explicit_target is False
        if graph_laplacian is False:
          loss = - similarity.abs().mean()
        else:
          s = similarity.abs(); L_s = torch.diag(sum(s, axis=1)) - s #graph laplacian
          loss = sum_i(x_i * L_s * x_i^T)
    explicit_target: if False, target=1 (fusion_type=='multiply) or 0 (fusion_type=='sum') implicitly
      if True, use given target or calculate it from xs
      # to do handle the case when we only use the explicitly given target
    cal_target: if 'mean-similarity', target = (cos(x1,x1) + cos(x2,x2) + cos(x3,x3))/3
                if 'mean-feature', x = (x1+x2+x3)/3; target = cos(x,x); this requires x1,x2,x3 have the same shape
    target: default None; only used when explicit_target is True
      This saves computation if target is provided in advance or passed as input
    fusion_type: if 'multiply', similarity=product(similarities); if 'sum', similarity=sum(|similarities|);
      work with loss_type
    graph_laplacian:  if graph_laplacian is False:
          loss = - similarity.abs().mean()
        else:
          s = similarity.abs(); L_s = torch.diag(sum(s, axis=1)) - s #graph laplacian
          loss = sum_i(x_i * L_s * x_i^T)

  Inputs:
    xs: a set of torch.Tensor matrices of (num_samples, feature_dim), 
      or a single matrix with self.sections being specified
    target: the target cosine similarity matrix; default None; 
      if not given, first check if self.targets is given; 
        if self.targets is None, then calulate it according to cal_target;
      only used when self.explicit_target is True

  Output:
    loss = -similarity.abs().mean() if graph_laplacian is False # Is this the right way to do it?
      = sum_i(x_i * L_s * x_i^T) if graph_laplacian is True # call get_interaction_loss()
    
  """
  def __init__(self, sections=None, loss_type='hub', explicit_target=False, 
    cal_target='mean-feature', target=None, fusion_type='multiply', graph_laplacian=False):
    super(Loss_view_similarity, self).__init__()
    self.sections = sections
    if self.sections is not None:
      if not isinstance(self.sections, int):
        assert len(self.sections) >= 2  
    self.loss_type = loss_type
    assert self.loss_type in ['circle', 'hub']
    self.explicit_target = explicit_target
    self.cal_target = cal_target
    self.target = target
    self.fusion_type = fusion_type
    self.graph_laplacian = graph_laplacian
    # I got nan losses easily for whenever graph_laplacian is True, especially the following case; did not know why
    # probably I need normalize similarity during every forward?
    assert not (fusion_type=='multiply' and graph_laplacian) and not (loss_type=='circle' and graph_laplacian)

  def forward(self, xs, target=None):
    if isinstance(xs, torch.Tensor):
      # make sure xs is a list of tensors corresponding to multiple views
      # this requires self.sections to valid
      xs = xs.split(self.sections, dim=1) 
    # assert len(xs) >= 2 # comment this to save time for many forward passes
    similarity = 1
    if self.loss_type == 'circle':
      # assert xs[i-1].shape == xs[i].shape
      # this saves computation
      similarity_mats = [cosine_similarity(xs[i-1], xs[i]) for i in range(1, len(xs))]
      similarity_mats = [(m+m.t())/2 for m in similarity_mats] # make it symmetric
    elif self.loss_type == 'hub':
      similarity_mats = [cosine_similarity(x) for x in xs]
    if self.fusion_type=='multiply':
      for m in similarity_mats:
        similarity = similarity * m # element multiplication ensures the larget value to be 1
    elif self.fusion_type=='sum':
      similarity = sum(similarity_mats) / len(similarity_mats) # calculate mean to ensure the largest value to be 1

    if self.explicit_target:
      if target is None:
        if self.target is None:
          if self.cal_target == 'mean-similarity':
            target = torch.stack(similarity_mats, dim=0).mean(0)
          elif self.cal_target == 'mean-feature':
            x = torch.stack(xs, -1).mean(-1) # the list of view matrices must have the same dimension
            target = cosine_similarity(x)
          else:
            raise ValueError(f'cal_target should be mean-similarity or mean-feature, but is {self.cal_target}')
        else:
          target = self.target
      if self.fusion_type=='multiply':
        similarity = similarity * target
      elif self.fusion_type=='sum':
        similarity = (len(similarity_mats)*similarity + target) / (len(similarity_mats) + 1) # Moving average
    similarity = similarity.abs() # ensure similarity to be non-negative
    if self.graph_laplacian:
      # Easily get nan loss when it is True; do not know why
      return sum([get_interaction_loss(similarity, w, loss_type='graph_laplacian', normalize=True) for w in xs]) / len(xs)
    else:
      return -similarity.mean() # to ensure the loss is within range [-1, 0]

    
