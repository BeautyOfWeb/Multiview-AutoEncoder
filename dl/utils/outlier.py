"""Functions for remove outliers"""
import numpy as np

def remove_boxplot_outlier(array, whis=1.5, only_positive=True):
  """remove outliers drawn from matplotlib.pyplot.boxplot
  """
  if only_positive:
    q1 = np.percentile(array[array>0], 25)
    q3 = np.percentile(array[array>0], 75)
  else:
    q1 = np.percentile(array, 25)
    q3 = np.percentile(array, 75)
  iqr = q3 - q1
  a_min = q1 - whis*iqr
  a_max = q3 + whis*iqr
  return np.clip(array, a_min, a_max)


def log2_transformation(mat, forced=False, threshold=50):
  """log2 transform
  
  Args:
    mat: np.array
    forced: if forced is True, then do log2 transformation immediately; 
      otherwise use threshold to decide if log2 transformation is necessary
    threshold: float, default 50; 
      if range(mat) / interquartile range > threshold, then do transform
  """
  mat = np.array(mat) # in case arg mat is a list
  if forced:
    return np.log2(mat - mat.min() + 1)
  q1 = np.percentile(mat, 25)
  q3 = np.percentile(mat, 75)
  iqr = q3 - q1
  r = mat.max() - mat.min()
  if (iqr==0 and r>0) or r/iqr > threshold:
    mat = np.log2(mat - mat.min() + 1)
  return mat


def normalization(mat, transform=True, forced=False, threshold=50, rm_outlier=True, whis=1.5, 
                  only_positive=True, max_val=1, diagonal=1, symmetric=True):
  """Normalize interaction/similarity matrix
  
  Args:
    transform: if True, call log2_transform(mat, forced, threshold)
    rm_outlier: if True, call remove_boxplot_outlier(mat, whis, only_positive)
    max_val: if max_val=1, execute mat=mat/mat.max()
    diagonal: if diagonal=1, make diagonal element to be 1
    symmetric: if True, execute mat = (mat+mat.T)/2
  """
  if transform:
    mat = log2_transformation(mat, forced, threshold)
  if rm_outlier:
    mat = remove_boxplot_outlier(mat, whis, only_positive) 
  if max_val == 1:
    mat = mat / mat.max()
  if diagonal == 1:
    mat[range(len(mat)), range(len(mat))] = 1
  if symmetric:
    mat = (mat + mat.T) / 2
  return mat