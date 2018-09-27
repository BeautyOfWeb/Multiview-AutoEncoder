import os
import functools
import itertools
import collections
import numpy as np
import pandas
from PIL import Image
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from .outlier import normalization
from .train import get_label_prob

def discrete_to_id(targets, start=0, sort=True, complex_object=False):
  """Change discrete variable targets to numeric values

  Args:
    targets: 1-d torch.Tensor or np.array, or a list
    start: the starting index for the first elements
    sort: sort the unique value, so that the 'smaller' values have smaller indices
    complex_object: input is not numeric, but complex objects, e.g., tuple

  Returns:
    target_ids: torch.Tensor or np.array with integer elements starting from start(=0 default)
    cls_id_dict: a dictionary mapping variables to their numeric ids

  """
  if complex_object:
    unique_targets = sorted(collections.Counter(targets))
  else:
    if isinstance(targets, torch.Tensor):
      targets = targets.cpu().detach().numpy()
    else:
      targets = np.array(targets) # if targets is already an np.array, then it does nothing
    unique_targets = np.unique(targets)
    if sort:
      unique_targets = np.sort(unique_targets)
  cls_id_dict = {v: i+start for i, v in enumerate(unique_targets)}
  target_ids = np.array([cls_id_dict[v] for v in targets])
  if isinstance(targets, torch.Tensor):
    target_ids = targets.new_tensor(target_ids)
  return target_ids, cls_id_dict
  

def get_f1_score(m, average='weighted', verbose=False):
  """Given a confusion matrix for binary classification, 
    calculate accuracy, precision, recall, F1 measure
    
  Args:
    m: confusion mat for binary classification
    average: if 'weighted': calculate metrics for each label, then get weighted average (weights are supports)
      if 'average': calculate average metrics for each label
      see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    verbose: if True, print result
  """
  def cal_f1(precision, recall):
    if precision + recall == 0:
      print('Both precision and recall are zero')
      return 0
    return 2*precision*recall / (precision+recall)
  m = np.array(m)
  t0 = m[0,0] + m[0,1]
  t1 = m[1,0] + m[1,1]
  p0 = m[0,0] + m[1,0]
  p1 = m[0,1] + m[1,1]
  prec0 = m[0,0] / p0
  prec1 = m[1,1] / p1
  recall0 = m[0,0] / t0
  recall1 = m[1,1] / t1
  f1_0 = cal_f1(prec0, recall0)
  f1_1 = cal_f1(prec1, recall1)
  if average == 'macro':
    w0 = 0.5
    w1 = 0.5
  elif average == 'weighted':
    w0 = t0 / (t0+t1)
    w1 = t1 / (t0+t1)
  prec = prec0*w0 + prec1*w1
  recall = recall0*w0 + recall1*w1
  f1 = f1_0*w0 + f1_1*w1
  acc = (m[0,0] + m[1,1]) / (t0+t1)
  if verbose:
    print(f'prec0={prec0}, recall0={recall0}, f1_0={f1_0}\n'
         f'prec1={prec1}, recall1={recall1}, f1_1={f1_1}')
  return acc, prec, recall, f1


def dist(params1, params2=None, dist_fn=torch.norm): #pylint disable=no-member
    """Calculate the norm of params1 or the distance between params1 and params2; 
        Common usage calculate the distance between two model state_dicts.
    Args:
        params1: dictionary; with each item a torch.Tensor
        params2: if not None, should have the same structure (data types and dimensions) as params1
    """
    if params2 is None:
        return dist_fn(torch.Tensor([dist_fn(params1[k]) for k in params1]))
    d = torch.Tensor([dist_fn(params1[k] - params2[k]) for k in params1])
    return dist_fn(d)
    
class AverageMeter(object):
    def __init__(self):
        self._reset()
    
    def _reset(self):
        self.val = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def pil_loader(path, format = 'RGB'):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert(format)


class ImageFolder(data.Dataset):
    def __init__(self, root, imgs, transform = None, target_transform = None, 
                 loader = pil_loader, is_test = False):
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.is_test = is_test
    
    def __getitem__(self, idx):
        if self.is_test:
            img = self.imgs[idx]
        else:
            img, target = self.imgs[idx]
        img = self.loader(os.path.join(self.root, img))
        if self.transform is not None:
            img = self.transform(img)
        if not self.is_test and self.target_transform is not None:
            target = self.target_transform(target)
        if self.is_test:
            return img
        else:
            return img, target
    
    def __len__(self):
        return len(self.imgs)

        
def check_acc(output, target, topk=(1,)):
    if isinstance(output, tuple):
        output = output[0]
    maxk = max(topk)
    _, pred = output.topk(maxk, 1)
    res = []
    for k in topk:
        acc = (pred.eq(target.contiguous().view(-1,1).expand(pred.size()))[:, :k]
               .float().contiguous().view(-1).sum(0))
        acc.mul_(100 / target.size(0))
        res.append(acc)
    return res


### Mainly developed for TCGA data analysis
def select_samples(mat, aliquot_ids, feature_ids, patient_clinical=None, clinical_variable='PFI', 
                   sample_type='01', drop_duplicates=True, remove_na=True):
  """Select samples with given sample_type ('01');
     if drop_duplicates is True (by default), remove technical duplicates; 
     and if remove_na is True (default), remove features that have NA;
     If patient_clinical is not None, further filter out samples with clinical_variable being NA
  """
  mat = pandas.DataFrame(mat, columns=feature_ids) # Use pandas to drop NA
  # Select samples with sample_type(='01')
  idx = np.array([[i,s[:12]] for i, s in enumerate(aliquot_ids) if s[13:15]==sample_type])
  # Remove technical duplicate
  if drop_duplicates:
    idx = pandas.DataFrame(idx).drop_duplicates(subset=[1]).values
    mat = mat.iloc[idx[:,0].astype(int)]
  aliquot_ids = aliquot_ids[idx[:,0].astype(int)]
  if remove_na:
  # Remove features that have NA values
    mat = mat.dropna(axis=1)
    feature_ids = mat.columns.values
  mat = mat.values
  if patient_clinical is not None:
    idx = [s[:12] in patient_clinical and not np.isnan(patient_clinical[s[:12]][clinical_variable]) 
           for s in aliquot_ids]
    mat = mat[idx]
    aliquot_ids = aliquot_ids[idx]
  return mat, aliquot_ids, feature_ids


def get_feature_feature_mat(feature_ids, gene_ids, feature_gene_adj, gene_gene_adj, 
                            max_score=1000):
  """Calculate feature-feature interaction matrix based on their mapping to genes 
    and gene-gene interactions:
    feature_feature = feature_gene * gene_gene * feature_gene^T (transpose)
  
  Args:
    feature_ids: np.array([feature_names]), dict {id: feature_name}, or {feature_name: id}
    gene_ids: np.array([gene_names]), dict {id: gene_name}, or {gene_name: id}
    feature_gene_adj: np.array([[feature_name, gene_name, score]]) 
      with rows corresponding to features and columns genes; 
      or (Deprecated) a list (gene) of lists of feature_ids. 
        Note this is different from np.array input; len(feature_gene_adj) = len(gene_ids)
    gene_gene_adj: an np.array. Each row is (gene_name1, gene_name2, score)
    max_score: default 1000. Normalize confidence scores in gene_gene_adj to be in [0, 1]
    
  Returns:
    feature_feature_mat: np.array of shape (len(feature_ids), len(feature_ids))
    
  """
  def check_input_ids(ids):
    if isinstance(ids, np.ndarray) or isinstance(ids, list):
      ids = {v: i for i, v in enumerate(ids)} # Map feature names to indices starting from 0
    elif isinstance(ids, dict):
      if sorted(ids) == list(range(len(ids))):
        # make sure it follows format {feature_name: id}
        ids = {v: k for k, v in ids.items()}
    else:
      raise ValueError(f'The input ids should be a list/np.ndarray/dictionary, '
                       'but is {type(feature_ids)}')
    return ids
  feature_ids = check_input_ids(feature_ids)
  gene_ids = check_input_ids(gene_ids)
  
  idx = []
  if isinstance(feature_gene_adj, list): # Assume feature_gene_adj is a list; this is deprecated
    for i, v in enumerate(feature_gene_adj):
      for j in v:
        idx.append([j, i, 1])
  elif isinstance(feature_gene_adj, np.ndarray) and feature_gene_adj.shape[1] == 3:
    for v in feature_gene_adj: 
      if v[0] in feature_ids and v[1] in gene_ids:
        idx.append([feature_ids[v[0]], gene_ids[v[1]], float(v[2])])
  else:
    raise ValueError('feature_gene_adj should be an np.ndarray of shape (N, 3) '
                     'or a list of lists (deprecated).')
  idx = np.array(idx).T
  feature_gene_mat = torch.sparse.FloatTensor(torch.tensor(idx[:2]).long(), 
                                              torch.tensor(idx[2]).float(), 
                                              (len(feature_ids), len(gene_ids)))
  # Extract a subnetwork from gene_gene_adj
  # Assume there is no self-loop in gene_gene_adj 
  # and it contains two records for each undirected edge
  idx = []
  for v in gene_gene_adj: 
    if v[0] in gene_ids and v[1] in gene_ids:
      idx.append([gene_ids[v[0]], gene_ids[v[1]], v[2]/max_score])
  # Add self-loops
  for i in range(len(gene_ids)):
    idx.append([i, i, 1.])
  idx = np.array(idx).T
  gene_gene_mat = torch.sparse.FloatTensor(torch.tensor(idx[:2]).long(),
                                          torch.tensor(idx[2]).float(),
                                          (len(gene_ids), len(gene_ids)))
  feature_feature_mat = feature_gene_mat.mm(gene_gene_mat.mm(feature_gene_mat.to_dense().t()))
  return feature_feature_mat.numpy()


def get_overlap_samples(sample_lists, common_list=None, start=0, end=12, return_common_list=False):
  """Given a list of aliquot_id lists, find the common sample ids
  
  Args:
    sample_lists: a iterable of sample (aliquot) id lists
    common_list: if None (default), find the interaction of sample_lists; 
      if provided, it should not be a set, because iterating over a set can be different from different runs
    start: default 0; assume sample ids are strings; 
      when finding overlapping samples, only consider a specific range [start, end)
    end: default 12, for TCGA BCR barcode
    return_common_list: if True, return a set containing common list for backward compatiablity,
      returns a sorted common list is a better option
  
  Returns:
    np.array of shape (len(sample_lists), len(common_list))
  """ 
  sample_lists = [[s_id[start:end] for s_id in sample_list] for sample_list in sample_lists]
  if common_list is None:
    common_list = functools.reduce(lambda x,y: set(x).intersection(y), sample_lists)
    if return_common_list:
      return common_list
    common_list = sorted(common_list) # iterate over set can vary from different runs
  for s in sample_lists: # make sure every list in sample_lists contains all elements in common_list
    assert len(set(common_list).difference(s)) == 0 
  idx_lists = np.array([[sample_list.index(s_id) for s_id in common_list] 
                        for sample_list in sample_lists])
  return idx_lists


# Select samples that have target variable(s) is in clinical file
def filter_clinical_dict(target_variable, target_variable_type, target_variable_range, 
                         clinical_dict):
  """Select patients with given target variable, its type and range in clinical data
  To save computation time, I assume all target variable(s) names are in clinical_dict without verification;
  
  Args:
    target_variable: str or a list of strings
    target_variable_type: 'discrete' or 'continuous' or a list of 'discrete' or 'continuous'
    target_variable_range: a list of values for 'continous' type, it is [lower_bound, upper_bound]
      or a list of list; target_variable, target_variable_type, target_variable_range must match
    clinical_dict: a dictionary of dictinaries; 
      first-level keys: patient ids, second-level keys: variable names
  
  Returns:
    clinical_dict: newly constructed clinical_dict with all patients having target_variables
    
  Examples:
    target_variable = ['PFI', 'OS.time'] 
    target_variable_type = ['discrete', 'continuous']
    target_variable_range = [[0, 1], [0, float('Inf')]]
    clinical_dict = filter_clinical_dict(target_variable, target_variable_type, target_variable_range, 
                            patient_clinical)
    assert sorted([k for k, v in patient_clinical.items() if v['PFI'] in [0,1] and not np.isnan(v['OS.time'])]) == 
      sorted(clinical_dict.keys())

  """
  if isinstance(target_variable, str):
    if target_variable_type == 'discrete':
      clinical_dict = {p:v for p, v in clinical_dict.items() 
                       if v[target_variable] in target_variable_range}
    elif target_variable_type == 'continuous':
      clinical_dict = {p:v for p, v in clinical_dict.items() 
                       if v[target_variable] >= target_variable_range[0] 
                       and v[target_variable] <= target_variable_range[1]}
  
  elif isinstance(target_variable, (list, tuple)):
    # Brilliant recursion
    for tar_var, tar_var_type, tar_var_range in zip(target_variable, target_variable_type, target_variable_range):
      clinical_dict = filter_clinical_dict(tar_var, tar_var_type, tar_var_range, clinical_dict)
      
  return clinical_dict


def get_target_variable(target_variable, clinical_dict, sel_patient_ids):
  """Extract target_variable from clinical_dict for sel_patient_ids
  If target_variable is a single str, it is only one line of code
  If target_variable is a list, recursively call itself and return a list of target variables
  
  Assume all sel_patient_ids have target_variable in clinical_dict
  
  """
  if isinstance(target_variable, str):
    return [clinical_dict[s][target_variable] for s in sel_patient_ids]
  elif isinstance(target_variable, (list, str)):
    return [[clinical_dict[s][tar_var] for s in sel_patient_ids] for tar_var in target_variable]


def normalize_continuous_variable(y_targets, target_variable_type, transform=True, forced=False, 
                        threshold=10, rm_outlier=True, whis=1.5, only_positive=True, max_val=1):
  """Normalize continuous variable(s)
    If a variable is 'continuous', then call normalization() in outlier.py
  
  Args:
    y_targets: a np.array or a list of np.array
    target_variable_type: can be a string: 'continous' or 'discrete' (do nothing but return the input)
      or a list of strings
    transform, forced, threshold, rm_outlier, whis, only_positive, max_val are all passed to normalization

  """
  if isinstance(target_variable_type, str):
    if target_variable_type=='continuous':
      y_targets = normalization(y_targets, transform=transform, forced=forced, threshold=threshold, 
                                rm_outlier=rm_outlier, whis=whis, only_positive=only_positive, 
                                max_val=max_val, diagonal=False, symmetric=False)
    return y_targets
  elif isinstance(target_variable_type, list):
    return [normalize_continuous_variable(y, var_type, transform=transform, forced=forced, 
            threshold=threshold, rm_outlier=rm_outlier, whis=whis, only_positive=only_positive, 
            max_val=max_val) for y, var_type in zip(y_targets, target_variable_type)]
  else:
    raise ValueError(f'target_variable_type should be a str or list of strs, but is {target_variable_type}')


def get_label_distribution(ys, check_num_cls=True):
  """Get label distributions for a list of labels
  
  Args:
    ys: an iterable (e.g., list) of labels (1-d numpy.array or torch.Tensor);
      the most common usage is get_label_distribution([y_train, y_val, y_test])
    check_num_cls: only if it is True, ensure that each list of labels will have the same number of classes 
      and also print out the message
    
  Returns:
    label_prob: a list of label distributions (multinomial);
    
  """
  num_cls = 0
  label_probs = []
  for i, y in enumerate(ys):
    if len(y)>0:
      label_prob = get_label_prob(y, verbose=False)
      label_probs.append(label_prob)
      if check_num_cls:
        if num_cls > 0:
          assert num_cls == len(label_probs[-1]), f'{i}: {num_cls} != {len(label_probs[-1])}'
        else:
          num_cls = len(label_probs[-1])
    else:
      label_probs.append([])
  if check_num_cls:
    if isinstance(label_probs, torch.Tensor):
      print('label distribution:\n', torch.stack(label_probs, dim=1))
    else:
      print('label distribution:\n', np.stack(label_probs, axis=1))
  return label_probs


def get_shuffled_data(sel_patient_ids, clinical_dict, cv_type, instance_portions, group_sizes,
                     group_variable_name, seed=None, verbose=True):
  """Shuffle sel_patient_ids and split them into multiple splits, 
    in most cases, train, val and test sets; 
  
  Args:
    sel_patient_ids: a list of object (patient) ids
    clinical_dict: a dictionary of dictionaries; 
      first-level keys: object ids; second-level keys: attribute names;
    cv_type: either 'group-shuffle' or 'instance-shuffle'; in most cases:
      if 'group-shuffle', split groups into train, val and test set according to group_sizes or
      implicitly instance_portions;
      if 'instance-shuffle': split based on instance_portions
    instance_portions: a list of floats; the proportions of samples in each split; 
      when cv_type=='group-shuffle' and group_sizes is given, then instance_portions is not used
    group_sizes: the number of groups in each split; only used when cv_type=='group-shuffle'
    group_variable_name: the attribute name for group information
    
  Returns:
    sel_patient_ids: shuffled object ids
    idx_splits: a list of indices, e.g., [train_idx, val_idx, test_idx]
      sel_patient_ids[train_idx] will get patient ids for training
      
  """
  np.random.seed(seed)
  sel_patient_ids = np.random.permutation(sel_patient_ids)
  num_samples = len(sel_patient_ids)
  idx_splits = []
  if cv_type == 'group-shuffle':
    # for my TCGA project, I used disease types as groups; thus the variable name is named 'disease_types'
    disease_types = sorted({clinical_dict[s][group_variable_name] for s in sel_patient_ids})
    num_disease_types = len(disease_types)
    np.random.shuffle(disease_types)
    type_splits = []
    cnt = 0
    for i in range(len(group_sizes)-1):
      if group_sizes[i] < 0: 
        # use instance_portion as group portions
        assert sum(instance_portions) == 1
        group_sizes[i] = round(instance_portions[i] * num_disease_types)
      type_splits.append(disease_types[cnt:cnt+group_sizes[i]])
      cnt = cnt+group_sizes[i]
      # do not use i to enumerate sel_patient_ids because i is used
      idx_splits.append([j for j, s in enumerate(sel_patient_ids) 
                         if clinical_dict[s][group_variable_name] in type_splits[i]])
    # process the last split
    if group_sizes[-1] >=0: # for most of time, set group_sizes[-1] = num_test_types = -1
      # almost never set group_sizes[-1] = 0, which will be useless
      assert group_sizes[-1] == num_disease_types - sum(group_sizes[:-1])
    if cnt == len(disease_types):
      print('The last group is empty, thus not included')
    else:
      type_splits.append(disease_types[cnt:]) 
      idx_splits.append([i for i, s in enumerate(sel_patient_ids) 
                          if clinical_dict[s][group_variable_name] in type_splits[-1]])
  elif cv_type == 'instance-shuffle':
    # because sel_patient_ids has already been shuffled, we do not need to shuffle indices
    cnt = 0
    assert sum(instance_portions) == 1
    for i in range(len(instance_portions)-1):
      n = round(instance_portions[i]*num_samples)
      idx_splits.append(list(range(cnt, cnt+n)))
      cnt = cnt + n
    # process the last split
    if cnt == num_samples:
      # this can rarely happen
      print('The last split is empty, thus not included')
    else:
      idx_splits.append(list(range(cnt, num_samples)))
  
  def get_type_cnt_msg(p_ids):
    """For a list p_ids, prepare group statistics for printing
    """
    cnt_dict = dict(collections.Counter([clinical_dict[p_id][group_variable_name] 
                                       for p_id in p_ids]))
    return f'{len(cnt_dict)} groups: {cnt_dict}'

  if verbose:
    msg = f'{cv_type}: \n'
    msg += '\n'.join([f'split {i}: {len(v)} samples ({len(v)/num_samples:.2f}), '
                      f'{get_type_cnt_msg(sel_patient_ids[v])}'
                      for i, v in enumerate(idx_splits)])
    print(msg)
  return sel_patient_ids, idx_splits


def target_to_numpy(y_targets, target_variable_type, target_variable_range):
  """y_targets is a list or a list of lists; transform it to numpy array
  For a discrete variable, generate numerical class labels from 0;
  for a continous variable, simply call np.array(y_targets);
  use recusion to handle a list of target variables
  
  Args:
    y_targets: a list of objects (strings/numbers, must be comparable) or lists
    target_variable_type: a string or a list of string ('discrete' or 'continous')
    target_variable_range: only used for sanity check for discrete variables
    
  Returns:
    y_true: a numpy array or a list of numpy arrays of type either float or int
    
  """
  
  if isinstance(target_variable_type, str):
    y_true = np.array(y_targets)
    if target_variable_type == 'discrete':
      unique_cls = np.unique(y_true)
      num_cls = len(unique_cls)
      if sorted(unique_cls) != sorted(target_variable_range):
        print(f'unique_cls: {unique_cls} !=\ntarget_variable_range {target_variable_range}')
      cls_idx_dict = {p.item(): i for i, p in enumerate(sorted(unique_cls))}
      y_true = np.array([cls_idx_dict[i.item()] for i in y_true])
      print(f'Changed class labels for the model: {cls_idx_dict}')
  elif isinstance(target_variable_type, (list, tuple)):
    y_true = [target_to_numpy(y, tar_var_type, tar_var_range) 
              for y, tar_var_type, tar_var_range in 
              zip(y_targets, target_variable_type, target_variable_range)]
  else:
    raise ValueError(f'target_variable_type must be str, list or tuple, '
                     f'but is {type(target_variable_type)}')
  return y_true


def get_mi_acc(xs, y_true, var_names, var_name_length=35):
  """Get mutual information (MI), adjusted MI, the maximal acc from Bayes classifier 
  for a list of discrete predictors xs and target y_true
  For all combinations of xs calculate MI, Adj_MI, and Bayes_ACC

  Args:
    xs: a list of tensors or numpy arrays
    y_true: a tensor or numpy array

  Returns:
    a list of dictionaries with key being the variable name
  """
  if isinstance(xs[0], torch.Tensor):
    xs = [x.cpu().detach().numpy() for x in xs]
  if isinstance(y_true, torch.Tensor):
    y_true = y_true.cpu().detach().numpy()
  result = []
  print('{:^{var_name_length}}\t{:^5}\t{:^6}\t{:^9}'.format('Variable', 'MI', 'Adj_MI', 'Bayes_ACC', 
    var_name_length=var_name_length))
  for i, l in enumerate(itertools.chain.from_iterable(itertools.combinations(range(len(xs)), r) 
                                     for r in range(1, 1+len(xs)))):
    if len(l) == 1:
      new_x = xs[l[0]]
      msg = f'{var_names[i]:^{var_name_length}}\t'
    else: # len(l) > 1
      new_x = [tuple([v.item() for v in s]) for s in zip(*[xs[j] for j in l])]
      new_x = discrete_to_id(new_x, complex_object=True)[0]
      msg = f'{"-".join(map(str, l)):^{var_name_length}}\t'
    mi = sklearn.metrics.mutual_info_score(y_true, new_x)
    adj_mi = sklearn.metrics.adjusted_mutual_info_score(y_true, new_x)
    bayes_acc = (sklearn.metrics.confusion_matrix(y_true, new_x).max(axis=0).sum() / len(y_true))
    result.append({msg: [mi, adj_mi, bayes_acc]})
    msg += f'{mi:^5.3f}\t{adj_mi:^6.3f}\t{bayes_acc:^9.3f}'
    print(msg)
  return result
  # p1 = sklearn.metrics.confusion_matrix(y_true.numpy(), new_x)[:2].reshape(-1)
  # p2 = (np.bincount(y_true.numpy())[:,None] * np.bincount(new_x)).reshape(-1)
  # p = torch.distributions.categorical.Categorical(torch.tensor(p1, dtype=torch.float))
  # q = torch.distributions.categorical.Categorical(torch.tensor(p2, dtype=torch.float))
  # torch.distributions.kl.kl_divergence(p,q)