# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 01:08:15 2016

@author: ckirst
"""

# test parallel processing with shared mem

import numpy as np

from joblib import Parallel, delayed, cpu_count

from parallel_tools import createSharedNumpyArray

def su(a,i):
  return a[i].sum();

def execute():
  dims = (100000,4);
  x = createSharedNumpyArray(dims);
  x[:] = np.random.rand(dims[0], dims[1]);
  
  res = Parallel(n_jobs = cpu_count())(delayed(su)(x,i) for i in range(dims[0]));
  



  