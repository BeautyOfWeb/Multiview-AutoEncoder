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
from autoencoder.autoencoder import *


# Had run in jupyter notebook but not here
# test Loss_view_similarity
x = torch.randn(11, 10)
model = MultiviewAE(in_dims=[2,3,5], hidden_dims=[7], out_dim=11)
loss_fn_g = Loss_view_similarity(sections=7, loss_type='hub', explicit_target=True, 
    cal_target='mean-feature', target=None, fusion_type='multiply', graph_laplacian=False)
loss_fn_d = Loss_view_similarity(sections=7, loss_type='hub', explicit_target=False, 
    cal_target='mean-feature', target=None, fusion_type='sum', graph_laplacian=True)
loss_fn_c = Loss_view_similarity(sections=7, loss_type='hub', explicit_target=True, 
    cal_target='mean-similarity', target=None, fusion_type='sum', graph_laplacian=False)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_history = []
for i in range(100):
  xs = model(x)[-1]
  loss_g = loss_fn_g(xs)
  optimizer.zero_grad()
  loss_g.backward()
  optimizer.step()
  with torch.no_grad():
    loss_d = loss_fn_d(xs)
    loss_c = loss_fn_c(xs)
  loss_history.append([loss_g.item(), loss_d.item(), loss_c.item()])
plt.plot(loss_history)
plt.show()

xs = xs.split(7, dim=1)
circle_similarity_mats = [cosine_similarity(xs[i-1], xs[i]) for i in range(1, len(xs))]
self_similarity_mats = [cosine_similarity(xs[i]) for i in range(len(xs))]
print([(x.mean().item(), x.std().item()) for x in xs])
print([(m.mean().item(), m.std().item())  for m in circle_similarity_mats])
print([(m.mean().item(), m.std().item())  for m in self_similarity_mats])