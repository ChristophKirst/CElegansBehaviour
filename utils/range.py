"""
Module Ranges

Utility routines to manipulate ranges
"""

__author__  = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__docformat__ = 'rest'


import numpy as np


def is_list_like(data):
  """Checks if the data is a tuple, list or numpy array
  
  Arguments:
    data (object): object to check
    
  Returns:
    bool: True if object is tuple, list or array
  """
  return isinstance(data, tuple) or isinstance(data, list) or isinstance(data, np.ndarray);


def full_range_1d(full_range, bounds = all, crop = False):
  """Converts bounds to numeric range (min,max) given the full range
     
  Arguments:
    size (tuple): full range
    bounds (tuple, list or all): bound specification, ``all`` is full range
    crop (bool): if True crop bounds to maximal bounds given i full range
      
  Returns:
    tuple: absolute range as pair of numbers
  """     

  if bounds is all:
    return full_range;
  
  if isinstance(bounds, int) or isinstance(bounds, float):
    bounds = (bounds, bounds + 1);    
  
  # bounds not shoulw be a tuple, list or np.array
  if is_list_like(bounds):
    bounds = list(bounds);
  else:
    raise RuntimeError('bounds %s is not a tuple, list or array!' % str(bounds));
  
 
  for i in range(2):
    if bounds[i] is all:
      bounds[i] = full_range[i];
    if crop:
      if bounds[i] < full_range[0]:
        bounds[i] = full_range[0];
      if bounds[i] > full_range[1]:
        bounds[i] = full_range[1];
    
  return tuple(bounds);

    
def full_range(full_range, bounds = all, crop = False):
  """Converts bounds to numeric range [(min,max),...] given the full range
     
  Arguments:
    size (tuple or list): full range in each dimension
    boudns (tuple or list or all): bound specifications in each dimension, ``all`` is full range
    crop (bool): if True crop bounds to maximal bounds given i full range
      
  Returns:
    list: absolute range as pair of integers
  """  
  
  if bounds is all:
    return full_range;
  else:
    return [full_range_1d(fr, bd, crop) for fr,bd in zip(full_range, bounds)];


def fixed_length_list(data, length):
  """Converts data to a list of a fixed length
     
  Arguments:
    data (tuple or list or number): the value to extend to a list
    length (int): desired length of the list
      
  Returns:
    list: list of specified length 
  """
  
  if not is_list_like(data):
    return [data for i in range(length)];
  else:
    if len(data) != length:
      raise RuntimeError('wrong data length %d, expected %d' % (len(data), length));
    return list(data);
    
    
def fixed_length_tuple(data, length):
  """Converts data to a tuple of a fixed length
     
  Arguments:
    data (tuple or list or number): the value to extend to a tuple
    length (int): desired length of the tuple
      
  Returns:
    tuple: tuple of specified length 
  """
  
  return tuple(fixed_length_list(data, length));

  
def test():
  import utils.range as rg
  reload(rg)
  
  print rg.full_range_1d((-10, 50), (30,all))
  print rg.full_range([(-10,30), (0,100)], [all, (all,10)])
  
  print rg.fixed_length_tuple([128,2,3], length = 3)