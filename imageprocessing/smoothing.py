"""
Module Utils

Image processing utils
"""

__author__  = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__docformat__ = 'rest'


import numpy as np
import scipy.ndimage as ndi

import utils.range as rg;


def smooth_data(data, bounds = all, nbins = 256, sigma = 10):
  """Smooth histogram of a set of nd points
  
  Arguments:
    data (nx2 array): data points
    binning (list or all): range of the binning as [(xmin,xmax), (ymin, ymax)]
    nbins (tuple): bins in each dimension
    sigma (tuple or None): optional Gaussian filter width
  
  Returns:
    array: smmothed histogram of the data set
    
  Notes:
    This routine is useful for smoothing data e.g. for use with watershed
  """
  
  # remove nans  
  d = data.copy();
  d = d[~np.any(np.isnan(d),axis = 1), :];
  
  # restrict to range
  dim = d.shape[1];
  
  nbins = rg.fixed_length_tuple(nbins, dim);
    
  full_range = [(data[:,i].min(), data[:,i].max()) for i in range(dim)];
  bds = rg.full_range(full_range, bounds);  
  
  if bounds is not all:
    #remove points out of bounds
    for i in range(dim):
      ids = np.logical_and(bds[i][0] <= d[:,i], d[:,i] <= bds[i][1])
      d = d[ids, :];
  #print bds, bounds
  
  for i in range(dim):
    d[:,i] = d[:,i] - bds[i][0];
    d[:,i] = (d[:,i] / bds[i][1]) * (nbins[i]-1);
  
  d = np.asarray(d, dtype = int);
  
  for i in range(dim):
    d[d[:,i] > (nbins[i]-1), i] = nbins[i]-1;
  
  img = np.zeros(nbins); 
  for i in xrange(d.shape[0]):        
    img[d[i,0], d[i,1]] += 1
  
  if sigma is not None:
    sigma = rg.fixed_length_tuple(sigma, dim);
    img = ndi.gaussian_filter(img, sigma, mode = 'constant')
  
  return img;
  
  
def test():
  import numpy as np;
  import matplotlib.pyplot as plt;
  import imageprocessing.smoothing as sth;
  reload(sth);
  
  data = np.random.rand(10,2);
  img = sth.smooth_data(data, bounds = [(0,1),(0,1)], sigma = None, nbins = 100);
  imgs = sth.smooth_data(data, bounds = [(0,1),(0,1)], sigma = 10, nbins = 100);
  plt.figure(1); plt.clf();
  plt.subplot(1,3,1);
  plt.plot(data[:,1], data[:,0], '.');
  plt.xlim(0,1); plt.ylim(0,1);
  plt.gca().invert_yaxis()
  plt.subplot(1,3,2);
  plt.imshow(img);
  plt.subplot(1,3,3);
  plt.imshow(imgs);

  x,y = img.nonzero()
  print (x/100.0);
  print np.sort(data[:,0])