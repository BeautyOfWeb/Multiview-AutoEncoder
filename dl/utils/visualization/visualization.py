import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA

import torch

def plot_cdf(y, title=''):
  if isinstance(y, torch.Tensor):
    y_sorted = y.contiguous().view(-1).sort(dim=0)[0].detach().cpu().numpy()
  else:
    y_sorted = np.array(y).reshape(-1)
    y_sorted.sort()
  plt.plot(y_sorted, np.linspace(0,1,len(y_sorted)), 'ro', markersize=0.5)
  if title!='':
    plt.title(title)
  plt.show()

def pca(x, n_components=2, verbose=False):
    r"""PCA for 2-D visualization
    """
    if len(x)>10000:
        pca = IncrementalPCA(n_components=n_components)
    else:
        pca = PCA(n_components=n_components)
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy().copy()
    pca.fit(x)
    if verbose:
        print(pca.explained_variance_, pca.noise_variance_)
        plt.title('explained_variance')
        plt.plot(pca.explained_variance_.tolist() + [pca.noise_variance_], 'ro')
        plt.show()
    return pca.fit_transform(x)

def plot_scatter(y_=None, model_=None, x_=None, title='', labels=None, colors=None, size=15, 
                 marker_size=20, folder='.', save_fig=False):
    r"""2D scatter plot
    """
    if y_ is None:
        assert model_ is not None and x_ is not None
        y_ = model_(x_.contiguous())
    if colors is not None:
        assert len(colors) == len(y_)
    else:
        if labels is not None:
            assert len(y_) == len(labels)
            # here is a bug: BASE_COLORS and CSS4_COLORS have overlap
            color = sorted(matplotlib.colors.BASE_COLORS) + sorted(matplotlib.colors.CSS4_COLORS)
            colors = [color[i] for i in labels]
    if isinstance(y_, torch.Tensor):
        y_ = y_.data.cpu().numpy()
    if y_.shape[1] > 2:
        y_ = pca(y_)
    plt.figure(figsize=(size, size))
    plt.scatter(y_[:,0],y_[:,1], c=colors, s=marker_size)
    if save_fig:
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder+'/'+title+'.png', bbox_inches='tight', dpi=200)
    else:
        plt.title(title)
        plt.show()
    plt.close()


def plot_history(history, title='', indices=None, colors='rgbkmc', markers='ov+*,<',
                 labels=['']*6, linestyles=['']*6, markersize=4):
  """Plot curves such as loss history and accuracy history during training

  Args:
    history: N * m numpy array; N is the number of steps, and m is the number of histories.
    indices: a list of selected indices to plot; usually less than four curves are plotted in a single figure.
  """
  if indices is None:
    indices = range(history.shape[1])
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1,1,1)
  for i in indices:
    ax.plot(range(len(history)), history[:,i], color=colors[i], linestyle=linestyles[i], 
            marker=markers[i], markersize=markersize, label=labels[i])
  ax.legend()
  plt.title(title)
  plt.show()


def plot_history_multi_splits(histories, title='Loss', idx=0, labels=['Train', 'Validation', 'Test'],
  colors='rgbk', markers='ov+*', linestyles=['', '', '', ''], markersize=4):
  """Given a list of histories, e.g., [loss_train_his, loss_val_his, loss_test_his], plot them in one figure

  Args:
    Most arguments are passed to plot_history
    histories: a list of list (or np.array), e.g., [loss_train_his, loss_val_his, loss_test_his]; 
      len(loss_train_his) = num_points; len(loss_train_his[0]) = num_losses (for multiple losses)
    title: plot title
    idx: if there are multiple losses, idx specifies which one should be ploted
    labels: the names for different histories; default: ['Train', 'Validation', 'Test']

  """
  if len(histories) != len(labels): # labels must match with histories
    labels = [f'History {i}' for i in range(len(histories))] 
  arrays = []
  labels_included = []
  for i, history in enumerate(histories):
    if len(history)>0: # only include non-empty array
      arrays.append(history)
      labels_included.append(labels[i])
      if len(arrays) > 1: # make sure all arrays have the same shape
        assert np.array(history).shape == prev_shape
      prev_shape = np.array(history).shape
  history = np.array(arrays)
  if history.ndim == 3: # For multiple losses/accuracies, use idx to select one
    history = history[:,:,idx]
  history = history.T  # To use plot_history, make sure history is of shape N * m; N=num_points, m = num of curves
  plot_history(history, title=title, labels=labels_included, colors=colors, markers=markers,
               linestyles=linestyles, markersize=markersize)


def plot_acc_history(acc_his, title='', color='r', marker='v', linestyle='', markersize=2):
  plt.figure(figsize=(10,10))
  plt.title(title)
  plt.plot(range(len(acc_his)), acc_his, color=color, marker=marker, linestyle=linestyle,
          markersize=markersize)
  plt.show()

