# -*- coding: utf-8 -*-
"""
Behavioural Development

Long Term Behaviour Analysis of C-elegans

Aligning the roaming dwelling data set with the image data set
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import numpy as np;
import matplotlib.pyplot as plt

### Align routines

def average(x, bin_size = 10):
  """Binned Average"""
  n = len(x);
  r = n % bin_size;
  if r != 0:
    xp = np.pad(x, (0, bin_size - r), mode = 'constant', constant_values = np.nan);
  else:
    xp = x;
  #print xp.shape
  s = len(xp) / bin_size;
  xp.shape = (s,bin_size);
  return np.nanmean(xp, axis = 1);


def distance(x,y, search = all, verbose = False): # assumes nx > ny
  """Normalized distance between two signals for different offsets"""
  nx = len(x);
  ny = len(y);
  nk = nx-ny+1;
  
  if nk <= 0:
    return [];
  
  if search is all:
    rng = range(nk);
  else:
    rng = range(search[0], search[1]);
  
  rng = np.array(rng);
  rng = rng[np.logical_and(rng >=0, rng <= nk)];
  #print rng
  
  d = np.ones(nk) * np.nan; 
  for k in rng:
    if verbose and k % 100 == 0:
      print '%d / %d' % (k,len(rng))
    dd = np.abs(x[k:k+ny] - y);
    d[k] = np.nansum(dd) / (ny - np.sum( np.isnan(dd)));
  
  return d;


def align_start(x, y, bin_size = 500, verbose = False, window = 2000, search = 100000):
  """Find initial alignment via downsampled data"""
  
  # resample for coarse alignment
  xa = average(x, bin_size);
  ya = average(y, bin_size); 
  
  #align resampled   
  dd = distance(xa, ya) 
  min_id = np.argmin(dd);

  #search full data
  id_search = bin_size * np.array(min_id + np.array([-2, 2]), dtype = int);
  dd2 = distance(x[:window+search], y[:window], search = id_search)
  start_id = np.nanargmin(dd2);
  
  if verbose:
    plt.figure(2); plt.clf();
    plt.subplot(1,3,1);
    plt.plot(dd)
    plt.scatter([min_id], [dd[min_id]], s = 50, c = 'r');
    plt.title('cooarse');
    plt.subplot(1,3,2);
    plt.plot(dd2)
    plt.scatter([start_id], [dd2[start_id]], s = 50, c = 'r');
    plt.title('fine');
    plt.subplot(1,3,3);
    plt.plot(x[start_id:start_id+window]);
    plt.plot(y[:window])
  
  return start_id;


def align_indices(x,y, verbose = False, skip = 30, window = 400, search = 100, search_conv = 500, start = all, max_iter = 10000, conv_threshold = 0.0, conv_precision = None, conv_consistent = 3):
  "Generates index array a so that x[a] = y, len(x) >= len(y)"""
  
  #precision for matching
  #eps = 10 * np.finfo(float).eps;
  eps = 0.0001;
  if conv_precision is None:
    conv_precision = eps/10;

  nx = len(x);
  ny = len(y);

  a = np.zeros(ny, dtype = int);
  
  #find start
  if start is all:
    ix = align_start(x,y, verbose = verbose);
    print 'start match %d' % ix;
  else:
    ix = start;
  iy = 0;
  
  ii = 0;
  while (ix < nx and iy < ny and ii < max_iter):
    ii+=1;
    # find next significant difference
    print '%d / %d and %d / %d' % (ix, nx, iy, ny);
  
    nmax = min(ny-iy, nx-ix); 
    idiv = np.where(np.logical_or(x[ix:ix+nmax] == 0.0,  np.abs(x[ix:ix+nmax] - y[iy:iy+nmax]) > eps))[0];
    
    if len(idiv) > 0:
      idiv = idiv[0];
      print 'found divergence at %d!' % idiv
      
      ixnew = ix + idiv;
      iynew = iy + idiv;      
      
      # up to difference alignment is good
      a[iy:iynew] = range(ix,ixnew);
      
      # check for convergence
      ix2 = min(ixnew + search_conv, nx);
      iy2 = min(iynew + search_conv, ny);
      if (ix2 - ixnew) != (iy2 - iynew):
        ix2 = ixnew + min(ix2 - ixnew, iy2 - iynew);
        iy2 = iynew + min(ix2 - ixnew, iy2 - iynew);

      iconv = np.where(np.logical_and(x[ixnew:ix2] > conv_threshold, np.abs(x[ixnew:ix2] - y[iynew:iy2]) <= conv_precision))[0];
      
      # at least this number of identical values (avoid error due to random overlap)
      if len(iconv) > conv_consistent and np.all(iconv[:conv_consistent]+1 == iconv[1:conv_consistent+1]):
        iconv = iconv[0];
        print 'found convergence at %d!' % iconv;
        
        # keep the differing elements
        ixnew2 = ixnew + iconv;        
        iynew2 = iynew + iconv;
        a[iynew:iynew2] = range(ixnew,ixnew2);
        
        ix = ixnew2;
        iy = iynew2;
        
      else: # no convergence -> find shift

        ix2 = min(ixnew + window + search, nx);
        iy2 = min(iynew + window, ny);
        
        if ix2 - ixnew <= iy2 - iynew:
          shift = 0;
        else:
          d = distance(x[ixnew:ix2], y[iynew:iy2], verbose = False);
          shift = np.argmin(d);
          print 'shift data %d' % shift;
        
        #print 'shift  is %d' % shift
        if shift == 0:
          #skip some points to make progress
          delta = np.min([skip, nx-ixnew, ny-iynew]);
          print 'delta = %d'% delta;
          a[iynew:iynew+delta] = range(ixnew,ixnew+delta);
          
          ix = ixnew + delta;
          iy = iynew + delta;
        
        else:
          ix = ixnew + shift; 
          iy = iynew;
        
    
    else:
      print 'match to end %d!' % nmax;
      a[iy:iy+nmax] = range(ix,ix+nmax);
      break;
  
  #print 'alignment done!'
  return a;


def average_invalidation_wrong(x, window = 31, invalid = True):
  """Averaging and data removal in the 'wrong way' done by Shay"""
  
  # set invalid points to zero
  xx = x.copy();
  xx[np.isnan(x)] = 0.0;

  # average speed
  xm = np.convolve(xx, np.ones(window)/(1.0 * window), mode = 'same')

  if not invalid:
    return xm;
  
  # invalid indices (min and max speed)
  xlo = xm > 0.0;  # 0 pixel /sec
  xhi = xm <= 30; # 30 pixel / sec

  xmc = xm[np.logical_and(xlo, xhi)];
  
  #index mapping from xm[a] = xmc;
  a = np.where(np.logical_and(xlo, xhi))[0];
  
  return xmc, a;  
  
