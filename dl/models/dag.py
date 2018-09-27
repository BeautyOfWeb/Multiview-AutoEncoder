import numpy as np

import torch
import torch.nn as nn


use_gpu = True
if use_gpu and torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')



class EmbedBigraph(nn.Module):
  r"""Map two sets of nodes from a bipartite graph
  
  Args:
    bigraph: a list, eg: [[0,1,3], [1,2]], input node ids are from 0 to length-1
    in_channels: int
    out_channels: int
    
  Shape:
    - Input: N * in_channels * M, where M = the number of input nodes
    - Output: N * out_channels * L, where L = len(bigraph), the number of output nodes
    
  Attributes:
    a series (length = len(bigraph)) of weights (and biases) of nn.Conv1d
    
  Examples::
  
    >>> bigraph = [[0,1,3], [1,2,3]]
    >>> x = torch.randn(1, 3, 4, device=device)
    >>> model = EmbedBigraph(bigraph, 3, 10)
    >>> model(x).shape
  """
  def __init__(self, bigraph, in_channels, out_channels, bias=True):
    super(EmbedBigraph, self).__init__()
    self.bigraph = bigraph
    self.maps = nn.ModuleList(
      [nn.Conv1d(in_channels, out_channels, kernel_size=len(v), bias=bias)
      for v in self.bigraph]
    )
    
  def forward(self, x):
    out = [self.maps[i](x[:,:,v]) for i, v in enumerate(self.bigraph)]
    out = torch.cat(out, dim=-1)
    return out


# In[4]:


class DAGLayer(nn.Module):
  r"""Build a computatinal graph from DAG
  
  Args:
    dag: dictionary, e.g.: {2:[0,1], 3:[0,1,2]}, node ids are topological order for computation.
    in_channels: int, embedding dimension
    
  Shape:
    - Input: N * in_channels * x, where x = the number of leaf nodes in dag
    - Output: N * in_channels * M, where M = the number of all nodes in dag
    
  Attributes:
    a series (length = len(dag)) of weights (and biases) for a ModuleList of nn.Conv1d
    
  Examples::
  
  >>> dag = {2:[0,1], 3:[0,1,2], 4:[1,2,3], 5:[0,2,3]}
  >>> model = DAGLayer(dag, 10)
  >>> x = torch.randn(1, 10, 2, device=device)
  >>> model(x).shape
  """
  def __init__(self, dag, in_channels, bias=True):
    super(DAGLayer, self).__init__()
    self.dag = dag
    self.embed = nn.ModuleList(
      [nn.Conv1d(in_channels, in_channels, kernel_size=len(v), bias=bias) 
       for k, v in sorted(self.dag.items())]
    )
    
  def forward(self, x):
    out = x
    for i, (k, v) in enumerate(sorted(self.dag.items())):
      y = self.embed[i](out[:,:,v])
      out = torch.cat([out, y], dim=-1)
    return out


# In[5]:


class StackedDAGLayers(nn.Module):
  r"""Stack multiple DAG layers
  
  Args:
    dag: dictionary, e.g.: {2:[0,1], 3:[0,1,2]}, node ids are topological order for computation.
    in_channels_list: a list of int
    residual: if True, use residual connections between two consecutive layers
    bias: default True for nn.Conv1d
    
  Shape:
    - Input: N * in_channels_list[0] * x, where x = the number of leaf nodes in dag
    - Output: N * in_channels_list[-1] * M, where M = the number of all nodes in dag
    
  Attributes:
    a series (length = len(in_channels_list)) of series (length = len(dag)) 
    of weights (and biases) for a ModuleList of nn.Conv1d
    
  Examples::
  
  >>> dag = {2:[0,1], 3:[0,1,2], 4:[1,2,3], 5:[0,2,3]}
  >>> model = StackedDAGLayers(dag, [10, 3, 2, 5], residual=False)
  >>> x = torch.randn(1, 10, 2, device=device)
  >>> model(x).shape
  """
  def __init__(self, dag, in_channels_list, residual=True, bias=True):
    super(StackedDAGLayers, self).__init__()
    self.dag = dag
    self.in_channels_list = in_channels_list
    self.num_layers = len(self.in_channels_list)
    self.residual = residual
    self.layers = nn.ModuleList(
      [DAGLayer(dag, in_channels, bias) for in_channels in in_channels_list]
    )
    self.bottlenecks = nn.ModuleList(
      [nn.Conv1d(in_channels_list[i-1], in_channels_list[i], kernel_size=1, bias=bias)
       for i in range(1, self.num_layers)]
    )
  
  def forward(self, x):
    out = x
    dim = x.size(-1)
    for i in range(self.num_layers):
      out = self.layers[i](out)
      if i > 0 and self.residual:
        out = out + h
      if i < self.num_layers-1:
        h = self.bottlenecks[i](out)
        out = h[:, :, :dim]
    return out


# In[6]:


class Conv1d2Score(nn.Module):
  r"""Calculate a N*out_dim tensor from N*in_dim*seq_len using nn.Conv1d
  Essentially it is a linear layer
  
  Args:
    in_dim: int
    out_dim: int, usually number of classes
    seq_len: int
    
  Shape:
    - Input: N*in_dim*seq_len
    - Output: N*out_dim
    
  Attributes:
    weight (Tensor): the learnable weights of the module of shape 
      out_channels (out_dim) * in_channels (in_dim) * kernel_size (seq_len)
    bias (Tensor): shape: out_channels (out_dim)
    
  Examples::
  
  
    >>> x = torch.randn(2, 3, 4, device=device)
    >>> model = Conv1d2Score(3, 5, 4)
    >>> model(x).shape
  """
  def __init__(self, in_dim, out_dim, seq_len, bias=True):
    super(Conv1d2Score, self).__init__()
    self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=seq_len, bias=bias)
  
  def forward(self, x):
    out = self.conv(x).squeeze(-1)
    return out


# In[7]:


# x = torch.randn(1, 3, 4, device=device)
# model = nn.Sequential(
#   EmbedBigraph(bigraph, 3, 10),
#   StackedDAGLayers(dag, [10, 3, 2,5,1], residual=False)
# )
# model(x).shape


def get_upper_closure(graph, vertices):
  # graph: np.array of shape (N, 2) with two columns being parent, child
  # assume vertices are only from child
  # output a subgraph that can be reached from vertices
  subgraphs = [np.array([p for p in graph if p[1] in vertices])]
  vertices_seen = set(vertices)
  vertices_unseen = set(subgraphs[-1][:,0]).difference(vertices_seen)
  while len(vertices_unseen) > 0:
    # print('seen:{}, unseen:{}'.format(len(vertices_seen), len(vertices_unseen)))
    subgraph = np.array([p for p in graph if p[1] in vertices_unseen])
    if subgraph.shape[0] > 0:
      subgraphs.append(subgraph)
      vertices_seen = vertices_seen.union(vertices_unseen)
      vertices_unseen = set(subgraph[:,0]).difference(vertices_seen)
    else:
      break
  return np.concatenate(subgraphs, axis=0)


def get_topological_order(adj_list):
  """Assume adj_list is a np.array of shape (N, 2), with two columns being: parent > child
  This function will only generate one possible topological order; there can be many
  Parents have higher id
  If it is DAG, then output name_to_id (dictionary mapping node name to int from 0)
  and chain_graph (a list of list) from decedents to ancestors

  Examples::

  >>> adj_list = np.array([[1, 2], [3, 2], [1, 3], [2, 4], [5,3], [1, 5], [2, 6], [5,2]])
  >>> get_topological_order(adj_list)
  """
  # nodes with no children
  nodes = sorted(set(adj_list[:,1]).difference(adj_list[:,0]))
  chain_graph = [nodes]
  name_to_id = {n: i for i, n in enumerate(nodes)}
  subgraph = np.array([s for s in adj_list if s[1] not in nodes])
  while subgraph.shape[0] > 0:
    nodes = sorted(set(subgraph[:,1]).difference(subgraph[:,0]))
    chain_graph.append(nodes)
    if len(nodes) == 0:
      print('There are cycles!')
      return subgraph, name_to_id
    cur_size = len(name_to_id)
    for i, n in enumerate(nodes):
      name_to_id[n] = i + cur_size
    subgraph = np.array([s for s in subgraph if s[1] not in nodes])
    
  cur_size = len(name_to_id)
  nodes = sorted(set(adj_list[:,0]).difference(name_to_id))
  chain_graph.append(nodes)
  for i, n in enumerate(nodes):
    name_to_id[n] = i + cur_size
  return name_to_id, chain_graph