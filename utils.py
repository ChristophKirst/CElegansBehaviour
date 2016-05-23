# -*- coding: utf-8 -*-
"""
Created on Sun May 22 21:42:04 2016

@author: ckirst
"""

import sys 
# Print iterations progress
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int) 
        barLength   - Optional  : character length of bar (Int) 
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    #sys.stdout.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    #sys.stdout.flush()
    print '%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)
    if iteration == total:
        print("\n")


## smooth and watershed

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