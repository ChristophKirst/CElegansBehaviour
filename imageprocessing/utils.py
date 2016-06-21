# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 00:52:20 2016

@author: ckirst
"""

## smooth for watershed

def smooth(data, nbins = (256, 256), sigma = (10,10)):
  
  import numpy as np
  import scipy.ndimage as ndi
  
  img = np.zeros(nbins);                ## blank image
  
  d = data.copy();
  d = d[~np.any(np.isnan(d),axis = 1), :];
  for i in range(2):
    d[:,i] = d[:,i] - d[:,i].min();
    d[:,i] = (d[:,i] / d[:,i].max()) * (nbins[i]-1);
  
  d = np.asarray(d, dtype = int);
  
  for i in range(2):
    d[d[:,i] > (nbins[i]-1), i] = nbins[i]-1;
  
  for i in xrange(d.shape[0]):          ## draw pixels
    img[d[i,0], d[i,1]] += 1
  
  if sigma is not None:
    img = ndi.gaussian_filter(img, sigma)  ## gaussian convolution
  
  return img;