# -*- coding: utf-8 -*-
"""
Created on Sun May 22 21:57:11 2016

@author: ckirst
"""


import tempfile
import shutil
import os
import numpy as np



import multiprocessing
import ctypes


def createSharedNumpyArray(dimensions, ctype = ctypes.c_double):
  # create array in shared memory segment
  shared_array_base = multiprocessing.RawArray(ctype, np.prod(dimensions))

  # convert to numpy array vie ctypeslib
  shared_array = np.ctypeslib.as_array(shared_array_base)

  return shared_array.reshape(dimensions);





from joblib import Parallel, delayed
from joblib import load, dump, cpu_count



def parallelIterateOnMemMap(function, data, result, iterations, moveTo = None, cleanup = True, n_jobs = cpu_count()):

    #try:
    #temporary files
    folder = tempfile.mkdtemp()
    data_name = os.path.join(folder, 'data')   
    result_name = os.path.join(folder, 'result')

    #result memmap
    if isinstance(result, np.memmap):
      result_mmap = result;
    if isinstance(result, tuple): # if shape create temp result memmap
      result_mmap = np.memmap(result_name, dtype=data.dtype, shape = result, mode='w+');
    elif isinstance(result, np.ndarray):
      result_mmap = np.memmap(result_name, dtype=data.dtype, shape = result.shape, mode='w+');
    else:
      raise RuntimeError('result should be array, memmap or tuple');

    #input data memmap
    dump(data, data_name)
    data_mmap = load(data_name, mmap_mode='r');

    # Fork the worker processes to perform computation concurrently
    #Parallel(n_jobs=n_jobs)(delayed(function)(data_mmap, result_mmap, i) for i in iterations)
    Parallel(n_jobs=n_jobs)(delayed(function)(data_mmap, result_mmap, i) for i in iterations)
          
     
    #except:
    #    print("Exception inparallel processing!")
    #    try:
    #        shutil.rmtree(folder)
    #    except:
    #        print("Failed to delete: " + folder)
    
    
    if moveTo is None:
        result = np.array(result_mmap);
    else:
        result_mmap.flush();
        shutil.move(result_name, moveTo);
        result = np.memmap(moveTo, dtype=data.dtype, shape = result);
    
    if cleanup:    
      try:
          shutil.rmtree(folder)
      except:
          print("Failed to delete: " + folder)
    
    return result
    
    
    
    ##make methods pickable
#def _pickle_method(method):
#  func_name = method.im_func.__name__
#  obj = method.im_self
#  cls = method.im_class
#  return _unpickle_method, (func_name, obj, cls)
#
#def _unpickle_method(func_name, obj, cls):
#  for cls in cls.mro():
#    try:
#      func = cls.__dict__[func_name]
#    except KeyError:
#      pass
#    else:
#      break
#  return func.__get__(obj, cls)
#
#import copy_reg
#import types
#copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
    