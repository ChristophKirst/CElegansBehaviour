"""
Features

Module specifying the features extracted from the experimental worm data
"""

__author__  = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__docformat__ = 'rest'

import numpy as np;


### Averages

def binned_average(x, bin_size = 10):
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
  
def moving_average(x, bin_size = 10):
  """Moving Average"""
  kern = np.ones(int(bin_size))/float(bin_size);
  return np.convolve(x, kern, 'same')

### Motions

def speed(xy, delta = 3, dt = 1.0):
  """Speed of the worm"""
  return np.linalg.norm(xy[delta:]-xy[:-delta], axis = 1) / dt;

def rotation(xy, delta = 3):
  """Rotation angle between subsequent linear paths"""
  dxy = xy[delta:,:] - xy[:-delta,:];
  nrm = np.linalg.norm(dxy, axis = 1)  
  
  dot = np.sum(dxy[1:,:] * dxy[:-1,:], axis = 1);
  nrm1 = nrm[1:];
  nrm0 = nrm[:-1];
  
  inv = np.logical_or(nrm1 == 0, nrm0 == 0);
  nrm1[inv] = np.nan;
  nrm0[inv] = np.nan;
  
  return np.arccos(dot / (nrm1 * nrm0));  
  

def distance(xy, delta = 1, steps = 10, mode = 'valid'):
  v = speed(xy, delta);
  return np.convolve(v, np.ones((steps,)), mode = mode);
  

def twist(xy, delta = 1, steps = 10, mode = 'valid'):
  r = rotation(xy, delta);
  return np.convolve(r, np.ones((steps,)), mode = mode);


def points_in_disk(xy, radius = 100, steps = 10000, steps_forward = all, steps_backward = all):
  """Average number of trajectory points in a disk centered at the point in time window given by steps"""
  n = len(xy);
  pid = np.ones(n) * np.nan;
  
  if steps_forward is all:
    steps_forward = steps;
  if steps_backward is all:
    steps_backward = steps;  
  
  for i in range(n):
    s = max(0, i - steps_backward);
    e = min(n, i + steps_forward);
    d = np.linalg.norm(xy[i] - xy[s:e], axis = 1);
    inv = np.isnan(d);
    ninv = np.sum(inv);
    if ninv < n:
      pid[i] = np.sum( d < radius) / (n - ninv);
  return pid;


def max_distance(xy, steps = 10000, steps_forward = all, steps_backward = all):
  """Maximal distance between the points"""
  n = len(xy);
  md = np.ones(n) * np.nan;
  
  if steps_forward is all:
    steps_forward = steps;
  if steps_backward is all:
    steps_backward = steps;  
  
  for i in range(n):
    s = max(0, i - steps_backward);
    e = min(n, i + steps_forward);
    d = np.linalg.norm(xy[i] - xy[s:e], axis = 1);
    md = np.max(d);
 
  return md;






#def roam_dwelling()