# -*- coding: utf-8 -*-
"""
Methods

Module for basic analysis methods of the C elegans data set

Experimental data:
Shay Stern, C. Bargman Lab, The Rockefeller University 2016
"""

__author__  = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__docformat__ = 'rest'


import numpy as np

import scipy.stats as stats


##############################################################################
### Binning and Averaging
##############################################################################


def binned_average(x, bin_size = 10, function = np.nanmean):
  """Binned average of a signal"""
  n = len(x);
  r = n % bin_size;
  if r != 0:
    xp = np.pad(x, (0, bin_size - r), mode = 'constant', constant_values = np.nan);
  else:
    xp = x;
  #print xp.shape
  s = len(xp) / bin_size;
  xp.shape = (s,bin_size);
  return function(xp, axis = 1);










############################################################################
### Aligning Data
############################################################################    


  










def align_stages(d, zeros):
  """Align data to specific stages"""
  nworms,ntimes = d.shape;
  lleft = max(zeros); 
  lright = max(ntimes - zeros);
  l = lleft + lright;
  a = np.zeros((nworms,l), dtype = d.dtype);
  for i in range(nworms):
    a[i,lleft-zeros[i]:lleft-zeros[i]+ntimes] = d[i,:];
  return a;



############################################################################
### Feature Analysis and Embedddings
############################################################################    


def jensen_shannon_divergence(p,q):
  """Jensen-Shannon distance between distributions p and q"""
  m = (p+q)/2.0;
  return stats.entropy(p,m) + stats.entropy(q,m);

def bhattacharyya_distance(p,q):
  return np.sqrt(1 - np.sqrt(np.sum(p * q)))

def pca(Y):
  """PCA with temporal (T) and spatial eigenvectors (C)"""
  u,s,C = np.linalg.svd(np.dot(Y,Y.T));
  Winv = np.diag(1.0/np.sqrt(s));
  L = np.dot(Winv, u.T);
  T = np.dot(L, Y);
  pvar = 100*(s)/s.sum();
  return C,T,pvar
 
 
def scales_to_array(data, worms_first = False, order = None):
  """Convert time scale resolved data into image"""
  nworms, nscales, ntimes = data.shape;
  res = np.zeros((nworms*nscales, ntimes));
  if order is None:
    order = range(nworms);
  if worms_first:
    for i in range(nscales):
      res[(i*nworms):((i+1)*nworms),:] = data[order,i,:];
  else:
    for i in range(nworms):
      res[(i*nscales):((i+1)*nscales),:] = data[order[i],:,:];
  return res;
  
  
def distributions_to_array(data, worms_first = False):
  """Convert distributions for worms andtimes into image"""
  nworms, ntimes, ndist = data.shape;
  res = np.zeros((nworms*ndist, ntimes));
  if worms_first:
    for i in range(ndist):
      res[(i*nworms):((i+1)*nworms),:] = data[:,:,i]; 
  else:
    for i in range(nworms):
      res[(i*ndist):((i+1)*ndist),:] = data[i,:,:].T;
  return res;


def isi_onoff(data):
  """Calculate ISIs for a 0/1 classifications of worm time traces"""
  if data.ndim > 1:
    nworms,ntimes = data.shape;
  else:
    data = np.array([data]);
    nworms = 1;
  
  sw = np.diff(data, axis = 1);
  sw[:,0] = -1; # make a down transition initially
  
  dur_up = [];
  times_up = [];
  dur_dw = [];
  times_dw = [];
  for i in range(nworms):
    t_up = np.where(sw[i,:] == 1)[0];  # ups
    t_dw = np.where(sw[i,:] == -1)[0]; # downs
    if len(t_dw) > len(t_up):          #last intervall assumed to be cutoff and thus neglected
      dur_up.append(t_dw[1:] - t_up);
      dur_dw.append(t_up - t_dw[:-1]);
      times_up.append(t_up);
      times_dw.append(t_dw[:-1]);
    else:
      dur_up.append(t_dw[1:] - t_up[:-1]);
      dur_dw.append(t_up - t_dw);
      times_up.append(t_up[:-1]);
      times_dw.append(t_dw);
  
  return (times_up, times_dw, dur_up, dur_dw);