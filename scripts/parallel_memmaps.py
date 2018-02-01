# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 19:03:21 2018

@author: ckirst
"""


import numpy as np;


def get_offset(filename):
  # Read the header of the file first.
  fp = open(filename, 'rb')
  try:
    version = np.lib.format.read_magic(fp)
    np.lib.format._check_version(version)

    shape, fortran_order, dtype =  np.lib.format._read_array_header(fp, version)
    if dtype.hasobject:
        msg = "Array can't be memory-mapped: Python objects in dtype."
        raise ValueError(msg)
    offset = fp.tell()
  finally:
    fp.close()
  return (shape, dtype, offset, fortran_order);
  
  
def open_memmap(filename, arange = None, mode = 'r+', shape = None, dtype = None, fortran = False):
  if arange is None:
    np.lib.format.open_memmap(filename, mode = mode, shape = shape, dtype = dtype, fortran_order = fortran);
  else:
   shape, dtype, offset, fortran_order = get_offset(filename);
   assert fortran_order is False;
   shape = np.array(shape);
   shape[0] = arange[1] - arange[0];
   shape_offset = np.prod(shape[1:]);
   shape = tuple(shape);
   return np.memmap(filename,  mode = mode,  shape = shape, dtype = dtype, order = 'C', offset = offset + arange[0] * dtype.itemsize * shape_offset);
   
   
if __name__ == '__main__':
  import numpy as np  
  import os
  import scripts.parallel_memmaps as pm
  reload(pm);
  
  fn = 'test.npy';
  x = np.random.rand(50,7);
  np.save(fn, x);
  
  a = pm.open_memmap(fn, arange = [10,20]);
  a[:] = 5.0;
  
  y = np.load(fn);
  
  x[10:20] = 5.0;
  np.all(y == x)
  
  
  #%% test parallel
  import multiprocessing as mp

  def setter(i):
    print [i, i+10]
    a = pm.open_memmap(fn, arange = [i,i+10]);
    a[:] = 5.0;
    a.flush();
    
  pool = mp.Pool(processes=mp.cpu_count());
  pool.map(setter, range(0, len(x)-10+1, 10));
  
  y = np.load(fn);
  np.all(y == 5.0)
  
  
  
  
  