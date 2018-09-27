import sys
import os
import copy
lib_path = 'I:/code'
if not os.path.exists(lib_path):
  lib_path = '/media/6T/.tianle/.lib'
if os.path.exists(lib_path) and lib_path not in sys.path:
  sys.path.append(lib_path)

import numpy as np
import sklearn

import torch
import torch.nn as nn

from dl.utils.visualization.visualization import plot_scatter


def cosine_similarity(x, y=None, eps=1e-8):
  """Calculate cosine similarity between two matrices; 

  Args:
    x: N*p tensor
    y: M*p tensor or None; if None, set y = x
    This function do not broadcast
  
  Returns:
    N*M tensor

  """
  w1 = torch.norm(x, p=2, dim=1, keepdim=True)
  if y is None:
    w2 = w1.squeeze(dim=1)
    y = x
  else:
    w2 = torch.norm(y, p=2, dim=1)
  w12 = torch.mm(x, y.t())
  return w12 / (w1*w2).clamp(min=eps)


def adjust_learning_rate(optimizer, lr, epoch, reduce_every=2):
  """Reduce learning rate by 10% every reduce_every iterations
  """
  lr = lr * (0.1 ** (epoch//reduce_every))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def predict(model, x, batch_size=None, train=True, num_heads=1):
  """Calculate model(x)

  Args:
    batch_size: default None; predict in one batch for small model and data.
    train: default True; if False, call torch.set_grad_enabled(False) first. 
    num_heads: default 1; if num_heads > 1, then calculate multi-head output
  """
  if batch_size is None:
    batch_size = x.size(0)
  y_pred = []
  if num_heads > 1:
    y_pred = [[] for i in range(num_heads)] # store decoder output
  prev = torch.is_grad_enabled()
  if train:
    model.train()
    torch.set_grad_enabled(True)
  else:
    model.eval()
    torch.set_grad_enabled(False)
  for i in range(0, len(x), batch_size):
    y_ = model(x[i:i+batch_size])
    if num_heads > 1:
      for i in range(num_heads):
        y_pred[i].append(y_[i])
    else:
      y_pred.append(y_)
  torch.set_grad_enabled(prev)
  
  if num_heads > 1:
    return [torch.cat(y, 0) for y in y_pred]
  else:
    return torch.cat(y_pred, 0)


def plot_data(model, x, y, title='', num_heads=1, batch_size=None):
  """Scatter plot for input layer and output with colors corresponding to labels
  """
  if isinstance(y, torch.Tensor):
    y = y.cpu().detach().numpy()
  plot_scatter(x, labels=y, title=f'Input {title}')
  y_pred = predict(model, x, batch_size, train=False, num_heads=num_heads)
  if num_heads > 1:
    for i in range(num_heads):
      plot_scatter(y_pred[i], labels=y, title=f'Head {i}')
  else:
    plot_scatter(y_pred, labels=y, title='Output')


def plot_data_multi_splits(model, xs, ys, num_heads=1, titles=['Training', 'Validation', 'Test'], batch_size=None):
  """Call plot_data on multiple data splits, typically x_train, x_val, x_test

  Args:
    Most arguments are passed to plot_data
    xs: a list of model input
    ys: a list of target labels
    titles: a list of titles for each data split

  """
  if len(xs) != len(titles): # Make sure titles are of the same length as xs
    titles = [f'Data split {i}' for i in range(len(xs))]
  for i, (x, y) in enumerate(zip(xs, ys)):
    if len(x) > 0 and len(x)==len(y):
      plot_data(model, x, y, title=titles[i], num_heads=num_heads, batch_size=batch_size)
    else:
      print(f'x for {titles[i]} is empty or len(x) != len(y)')


def get_label_prob(labels, verbose=True):
  """Get label distribution
  """
  if isinstance(labels, torch.Tensor):
    unique_labels = torch.unique(labels).sort()[0]
    label_prob = torch.stack([labels==i for i in unique_labels], dim=0).sum(dim=1)
    label_prob = label_prob.float()/len(labels)
  else:
    labels = np.array(labels) # if labels is a list then change it to np.array
    unique_labels = sorted(np.unique(labels))
    label_prob = np.stack([labels==i for i in unique_labels], axis=0).sum(axis=1)
    label_prob = label_prob / len(labels)
  if verbose:
    msg = '\n'.join(map(lambda x: f'{x[0]}: {x[1].item():.2f}', 
                        zip(unique_labels, label_prob)))
    print(f'label distribution:\n{msg}')
  return label_prob


def eval_classification(y_true, y_pred=None, model=None, x=None, batch_size=None, multi_heads=False, 
  cls_head=0, average='weighted', predict_func=None, pred_kwargs=None, verbose=True):
  """Evaluate classification results

  Args:
    y_true: true labels; numpy array or torch.Tensor
    y_pred: if None, then y_pred = model(x)
    model: torch.nn.Module type
    x: input tensor
    batch_size: used for predict(model, x, batch_size)
    multi_heads: If true, the model output a list; Assume the classification head is the first one
    cls_head: only used when multi_heads is True; specify which head is used for classification; default 0
    average: used for sklearn.metrics to calculate precision, recall, f1, auc and ap; default: 'weighted'
    predict_func: if not None, use predict_func(model, x, **pred_kwargs) instead of predict()
    pred_kwargs: dictionary arguments for predict_func

  """
  if isinstance(y_true, torch.Tensor): 
    y_true = y_true.cpu().detach().numpy().reshape(-1)
  num_cls = len(np.unique(y_true))
  auc = -1 # dummy variable for multi-class classification
  average_precision = -1 # dummy variable for multi-class classification
  y_score = None # only used to calculate auc and average_precision for binary classification; will be set later
  if y_pred is None: # Calculate y_pred = model(x) in batches
    if predict_func is None:
      # use predict() defined in this file
      num_heads = 2 if multi_heads else 1 # num_heads >= 2 is to make predict() to process the model as multi-output
      y_ = predict(model, x, batch_size, train=False, num_heads=num_heads)
      y_pred = y_[cls_head] if multi_heads else y_
    else:
      # use customized predict_func with variable keyworded arguments
      y_pred = predict_func(model, x, **pred_kwargs)
  if isinstance(y_pred, torch.Tensor):
    # either input argument is a torch.Tensor or calculate it from model(x) in the last chunk
    y_pred = y_pred.cpu().detach().numpy()
  if isinstance(y_pred, np.ndarray) and y_pred.ndim == 2 and y_pred.shape[1] > 1:
    # y_pred is the class score matrix: n_samples * n_classes
    if y_pred.shape[1] == 2: # for binary classification
      y_score = y_pred[:,1] - y_pred[:,0] # y_score is only useful for calculating auc and average precison 
    y_pred = y_pred.argmax(axis=-1) # only consider top 1 prediction
  if num_cls==2 and y_pred.dtype == np.dtype('float'): # last chunk had not been executed
    # For binary classification, argument y_pred can be the scores for belonging to class 1.
    y_score =  y_pred # Used for calculate auc and average_precision
    y_pred = (y_score > 0).astype('int')
  acc = sklearn.metrics.accuracy_score(y_true, y_pred)
  precision = sklearn.metrics.precision_score(y_true, y_pred, average=average)
  recall = sklearn.metrics.recall_score(y_true, y_pred, average=average)
  f1_score = sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average=average)
  adjusted_mutual_info = sklearn.metrics.adjusted_mutual_info_score(labels_true=y_true, labels_pred=y_pred)
  confusion_mat = sklearn.metrics.confusion_matrix(y_true, y_pred)
  msg = f'acc={acc:.3f}, precision={precision:.3f}, recall={recall:.3f}, fl={f1_score:.3f}, adj_MI={adjusted_mutual_info:.3f}'
  if num_cls == 2:
    # When y_pred is given as an int np.array or tensor, model(x) is not called; 
    # set y_score = y_pred to calculate auc and average precision approximately; 
    # it may not be 100% accurate because I assign y_pred (binary labels) to y_score (which should be probabilities)
    if y_score is None: 
      y_score = y_pred
    auc = sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_score, average=average)
    average_precision = sklearn.metrics.average_precision_score(y_true=y_true, y_score=y_score, average=average)
    msg = msg + f', auc={auc:.3f}, ap={average_precision:.3f}'
  msg = msg + f', confusion_mat=\n{confusion_mat}'
  if verbose:
    print(msg)
    print('report', sklearn.metrics.classification_report(y_true=y_true, y_pred=y_pred))

  return np.array([acc, precision, recall, f1_score, adjusted_mutual_info, auc, average_precision]), confusion_mat


def eval_classification_multi_splits(model, xs, ys, batch_size=None, multi_heads=False, cls_head=0, 
  average='weighted', return_result=True, split_names=['Train', 'Validataion', 'Test'],
  predict_func=None, pred_kwargs=None, verbose=True):
  """Call eval_classification on multiple data splits, e.g., x_train, x_val, x_test with given model

  Args:
    Most arguments are passed to eval_classification
    xs: a list of model input, e.g., [x_train, x_val, x_test]
    ys: a list of targets, e.g., [y_train, y_val, y_test]
    return_results: if True return results on non-empty data splits
    split_names: for print purpose; default: ['train', 'val', 'test']

  """
  res = []
  if len(xs) != len(split_names):
    split_names = [f'Data split {i}' for i in range(len(xs))]
  for i, (x, y) in enumerate(zip(xs, ys)):
    if len(x) > 0:
      print(split_names[i])
      metric = eval_classification(y_true=y, model=model, x=x, batch_size=batch_size, 
                          multi_heads=multi_heads, cls_head=cls_head, average=average,
                          predict_func=predict_func, pred_kwargs=pred_kwargs, verbose=verbose)
      res.append(metric)
  if return_result:
    return res


def run_one_epoch_single_loss(model, x, y_true, loss_fn=nn.CrossEntropyLoss(), train=True, optimizer=None, 
  batch_size=None, return_loss=True, epoch=0, print_every=10, verbose=True):
  """Run one epoch, i.e., model(x), but split into batches
  
  Args:
    model: torch.nn.Module
    x: torch.Tensor
    y_true: target torch.Tensor
    loss_fn: loss function
    train: if False, call model.eval() and torch.set_grad_enabled(False) to save time
    optimizer: needed when train is True
    batch_size: if None, batch_size = len(x)
    return_loss: if True, return epoch loss
    epoch: for print 
    print_every: print epoch_loss if print_every % epoch == 0
    verbose: if True, print batch_loss
  """

  is_grad_enabled = torch.is_grad_enabled()
  if train:
    model.train()
    torch.set_grad_enabled(True)
  else:
    model.eval()
    torch.set_grad_enabled(False)
  loss_history = []
  is_classification = isinstance(y_true.cpu(), torch.LongTensor)
  if is_classification:
    acc_history = []
  if batch_size is None:
    batch_size = len(x)
  for i in range(0, len(x), batch_size):
    y_pred = model(x[i:i+batch_size])
    loss = loss_fn(y_pred, y_true[i:i+batch_size])
    loss_history.append(loss.item())
    if is_classification:
      labels_pred = y_pred.topk(1, -1)[1].squeeze() # only calculate top 1 accuracy
      acc = (labels_pred == y_true[i:i+batch_size]).float().mean().item()
      acc_history.append(acc)
    if verbose:
      msg = 'Epoch{} {}/{}: loss={:.2e}'.format(
        epoch, i//batch_size, (len(x)+batch_size-1)//batch_size, loss.item())
      if is_classification:
        msg = msg + f', acc={acc:.2f}'
      print(msg)
    if train:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  torch.set_grad_enabled(is_grad_enabled)

  loss_epoch = np.mean(loss_history)
  if is_classification:
    acc_epoch = np.mean(acc_history)
  if epoch % print_every == 0:  
    msg = 'Epoch{} {}: loss={:.2e}'.format(epoch, 'Train' if train else 'Test', np.mean(loss_history))
    if is_classification:
      msg = msg + f', acc={np.mean(acc_history):.2f}'
    print(msg)
  if return_loss:
    if is_classification:
      return loss_epoch, acc_epoch, loss_history, acc_history
    else:
      return loss_epoch, loss_history
  

def train_single_loss(model, x_train, y_train, x_val=[], y_val=[], x_test=[], y_test=[], 
    loss_fn=nn.CrossEntropyLoss(), lr=1e-2, weight_decay=1e-4, amsgrad=True, batch_size=None, num_epochs=1, 
    reduce_every=200, eval_every=1, print_every=1, verbose=False, 
    loss_train_his=[], loss_val_his=[], loss_test_his=[], 
    acc_train_his=[], acc_val_his=[], acc_test_his=[], return_best_val=True):
  """Run a number of epochs to backpropagate

  Args:
    Most arguments are passed to run_one_epoch_single_loss
    lr, weight_decay, amsgrad are passed to torch.optim.Adam
    reduce_every: call adjust_learning_rate if cur_epoch % reduce_every == 0
    eval_every: call run_one_epoch_single_loss on validation and test sets if cur_epoch % eval_every == 0
    print_every: print epoch loss if cur_epoch % print_every == 0
    verbose: if True, print batch loss
    return_best_val: if True, return the best model on validation set for classification task 
  """

  def eval_one_epoch(x, targets, loss_his, acc_his, epoch, train=False):
    """Function within function; reuse parameters within proper scope
    """
    results = run_one_epoch_single_loss(model, x, targets, loss_fn=loss_fn, train=train, optimizer=optimizer, 
      batch_size=batch_size, return_loss=True, epoch=epoch, print_every=print_every, verbose=verbose)
    if is_classification:
      loss_epoch, acc_epoch, loss_history, acc_history = results
    else:
      loss_epoch, loss_history = results
    loss_his.append(loss_epoch)
    if is_classification:
      acc_his.append(acc_epoch)

  is_classification = isinstance(y_train.cpu(), torch.LongTensor)
  best_val_acc = -1 # best_val_acc >=0 after the first epoch for classification task
  for i in range(num_epochs):   
    if i == 0:
      optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
    # Should I create a new torch.optim.Adam instance every time I adjust learning rate? 
    adjust_learning_rate(optimizer, lr, i, reduce_every=reduce_every)

    eval_one_epoch(x_train, y_train, loss_train_his, acc_train_his, i, train=True)
    if i % eval_every == 0:
      if len(x_val)>0 and len(y_val)>0:
        eval_one_epoch(x_val, y_val, loss_val_his, acc_val_his, i, train=False) # Set train to be False is crucial!
        if is_classification:
          if acc_val_his[-1] > best_val_acc:
            best_val_acc = acc_val_his[-1]
            best_model = copy.deepcopy(model)
            best_epoch = i
            print('epoch {}, best_val_acc={:.2f}, train_acc={:.2f}'.format(
              best_epoch, best_val_acc, acc_train_his[-1]))
      if len(x_test)>0 and len(y_test)>0:
        eval_one_epoch(x_test, y_test, loss_test_his, acc_test_his, i, train=False) # Set train to be False

  if is_classification:
    if return_best_val and len(x_val)>0 and len(y_val)>0:
      return best_model, best_val_acc, best_epoch
    else:
      return model, acc_train_his[-1], i


def run_one_epoch_multiloss(model, x, targets, heads=[0,1], loss_fns=[nn.CrossEntropyLoss(), nn.MSELoss()], 
  loss_weights=[1,0], other_loss_fns=[], other_loss_weights=[], return_loss=True, batch_size=None, 
  train=True, optimizer=None, epoch=0, print_every=10, verbose=True):
  """Calculate a multi-head model with multiple losses including losses from the outputs and targets (head losses) 
  and regularizers on model parameters (non-head losses).
  
  Args:
    model: A model with multihead; for example, an AutoEncoder classifier, returns classification scores 
      (or regression target) and decoder output (reconstruction of input)
    x: input
    targets: a list of targets associated with multi-head output specified by argument heads; 
      e.g., for an autoencoder with two heads, targets = [y_labels, x]
      targets are not needed to pair with all heads output one-to-one; 
      use arguments heads to specify which heads are paired with targets;
      The elements of targets can be None, too; 
      the length of targets must be compatible with that of loss_weights, loss_fns, and heads
    heads: the index for the heads paired with targets for calculating losses; 
      if None, set heads = list(range(len(targets)))
    loss_fns: a list of loss functions for the corresponding head
    loss_weights: the (non-negative) weights for the above head-losses;
      heads, loss_fns, and loss_weights are closely related to each other; need to handle it carefully
    other_loss_fns: a list of loss functions as regularizers on model parameters
    other_loss_weights: the corresponding weights for other_loss_fns
    return_loss: default True, return all losses
    batch_size: default None; split data into batches
    train: default True; if False, call model.eval() and torch.set_grad_enabled(False) to save time
    optimizer: when train is True, optimizer must be given; default None, do not use for evaluation
    epoch: for print only
    print_every: print epoch losses if epoch % print_every == 0
    verbose: if True, print losses for each batch
  """

  is_grad_enabled = torch.is_grad_enabled()
  if train:
    model.train()
    torch.set_grad_enabled(True)
  else:
    model.eval()
    torch.set_grad_enabled(False)
  if batch_size is None:
    batch_size = len(x)
  
  if len(targets) < len(loss_weights):
    # Some losses do not require targets (using 'implicit' targets in the objective)
    # Add None so that targets for later use
    targets = targets + [None]*(len(loss_weights) - len(targets))
  is_classification = [] # record the indices of targets that is for classification
  has_unequal_size = [] # record the indices of targets that has a different size with input
  is_none = [] # record the indices of the targets that is None
  for j, y_true in enumerate(targets):
    if y_true is not None:
      if len(y_true) == len(x):
        if isinstance(y_true.cpu(), torch.LongTensor):
          # if targets[j] is LongTensor, treat it as classification task
          is_classification.append(j)
      else:
        has_unequal_size.append(j)
    else:
      is_none.append(j)
  loss_history = []
  if len(is_classification) > 0:
    acc_history = []

  if heads is None: # If head is not given, then assume the targets is paired with model output in order
    heads = list(range(len(targets)))
  for i in range(0, len(x), batch_size):
    y_pred = model(x[i:i+batch_size])
    loss_batch = []
    for j, w in enumerate(loss_weights):
      if w>0: # only execute when w>0
        if j in is_none:
          loss_j = loss_fns[j](y_pred[heads[j]]) * w
        elif j in has_unequal_size:
          loss_j = loss_fns[j](y_pred[heads[j]], targets[j]) * w # targets[j] is the same for all batches
        else:
          loss_j = loss_fns[j](y_pred[heads[j]], targets[j][i:i+batch_size]) * w
        loss_batch.append(loss_j)
    for j, w in enumerate(other_loss_weights):
      if w>0:
        # The implicit 'target' is encoded in the loss function itself
        # todo: in addition to argument model, make loss_fns handle other 'dynamic' arguments as well
        loss_j = other_loss_fns[j](model) * w 
        loss_batch.append(loss_j)
    loss = sum(loss_batch)
    loss_batch = [v.item() for v in loss_batch]
    loss_history.append(loss_batch)
    # Calculate accuracy
    if len(is_classification) > 0:
      acc_batch = []
      for k, j in enumerate(is_classification):
        labels_pred = y_pred[heads[j]].topk(1, -1)[1].squeeze()
        acc = (labels_pred == targets[j][i:i+batch_size]).float().mean().item()
        acc_batch.append(acc)
      acc_history.append(acc_batch)
    if verbose:
      msg = 'Epoch{} {}/{}: loss:{}'.format(epoch, i//batch_size, (len(x)+batch_size-1)//batch_size, 
        ', '.join(map(lambda x: f'{x:.2e}', loss_batch)))
      if len(is_classification) > 0:
        msg = msg + ', acc={}'.format(', '.join(map(lambda x: f'{x:.2f}', acc_batch)))
      print(msg)
    if train:
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  torch.set_grad_enabled(is_grad_enabled)

  loss_epoch = np.mean(loss_history, axis=0)
  if len(is_classification) > 0:
    acc_epoch = np.mean(acc_history, axis=0)
  if epoch % print_every == 0:
    msg = 'Epoch{} {}: loss:{}'.format(epoch, 'Train' if train else 'Test', 
      ', '.join(map(lambda x: f'{x:.2e}', loss_epoch)))
    if len(is_classification) > 0:
      msg = msg + ', acc={}'.format(', '.join(map(lambda x: f'{x:.2f}', acc_epoch)))
    print(msg)
    
  if return_loss:
    if len(is_classification) > 0:
      return loss_epoch, acc_epoch, loss_history, acc_history
    else:
      return loss_epoch, loss_history


def train_multiloss(model, x_train, y_train, x_val=[], y_val=[], x_test=[], y_test=[], heads=[0, 1], 
  loss_fns=[nn.CrossEntropyLoss(), nn.MSELoss()], loss_weights=[1,0], other_loss_fns=[], other_loss_weights=[], 
  lr=1e-2, weight_decay=1e-4, batch_size=None, num_epochs=1, reduce_every=100, eval_every=1, print_every=1,
  loss_train_his=[], loss_val_his=[], loss_test_his=[], acc_train_his=[], acc_val_his=[], acc_test_his=[], 
  return_best_val=True, amsgrad=True, verbose=False):
  """Train a number of epochs
  Most of the parameters are passed to run_one_epoch_multiloss

  Args:
    lr, weight_decay, amsgrad are passed to torch.optim.Adam
    reduce_every: call adjust_learning_rate if i % reduce_every == 0; i is the current epoch
    eval_every: run_one_multiloss on validation and test set if i % eval_every == 0
    return_best_val: for classification task, if validation set is available, return the best model on validation set
    print_every: print epoch losses if i % print_every == 0
    verbose: if True, print batch losses
  """
  def eval_one_epoch(x, targets, loss_his, acc_his, epoch, train=False):
    """This is a function within a function; reuse some parameters in the scope of the "outer" function
    """
    results = run_one_epoch_multiloss(model, x, targets=targets, heads=heads, loss_fns=loss_fns, 
                loss_weights=loss_weights, other_loss_fns=other_loss_fns, other_loss_weights=other_loss_weights, 
                return_loss=True, batch_size=batch_size, train=train, optimizer=optimizer, epoch=epoch, 
                print_every=print_every, verbose=verbose)
    if is_classification:
      loss_epoch, acc_epoch, loss_history, acc_history = results
    else:
      loss_epoch, loss_history = results
    # loss_train_his += loss_history
    # acc_train_his += acc_history
    loss_his.append(loss_epoch)
    if is_classification:
      acc_his.append(acc_epoch)

  cls_targets = []
  for i, y_true in enumerate(y_train):
    if isinstance(y_true.cpu(), torch.LongTensor):
      cls_targets.append(i)
  is_classification = len(cls_targets) > 0
  best_val_acc = -1 # After the first iteration, best_val_acc >= 0 

  for i in range(num_epochs):   
    if i == 0: # I did not clear the caches after adjusting the learning rate later; this works, but is it better?
      optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, 
        weight_decay=weight_decay, amsgrad=amsgrad)
    adjust_learning_rate(optimizer, lr, i, reduce_every=reduce_every)

    eval_one_epoch(x_train, y_train, loss_train_his, acc_train_his, i, train=True)
    if i % eval_every == 0:
      if len(x_val)>0 and len(y_val)>0:
        # Must set train=False, otherwise leak data
        eval_one_epoch(x_val, y_val, loss_val_his, acc_val_his, i, train=False) 
        if is_classification:
          cur_val_acc = np.mean(acc_val_his[-1])
          if cur_val_acc > best_val_acc: # Use the mean accuracy for all classification tasks (in most case just one)
            best_val_acc = cur_val_acc
            best_model = copy.deepcopy(model)
            best_epoch = i
            print('epoch {}, best_val_acc={:.2f}, train_acc={:.2f}'.format(
              best_epoch, best_val_acc, np.mean(acc_train_his[-1])))
      if len(x_test)>0 and len(y_test)>0:
        eval_one_epoch(x_test, y_test, loss_test_his, acc_test_his, i, train=False)

  if is_classification:
    if return_best_val and len(x_val)>0 and len(y_val)>0:
      return best_model, best_val_acc, best_epoch
    else:
      return model, np.mean(acc_train_his[-1]), i