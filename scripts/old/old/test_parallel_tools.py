# -*- coding: utf-8 -*-
"""
Created on Sun May 22 22:18:05 2016

@author: ckirst
"""

import os
import numpy as np;


def f(data, result, i):
  print("[Worker %d] processing iteration %d" % (os.getpid(), i))
  result[i] =  np.sum(data[i,:]);
  

data = np.random.rand(10000,1000);

from parallel_tools import parallelIterateOnMemMap

result = parallelIterateOnMemMap(f, data, (data.shape[0],), range(data.shape[0]))
result.shape

result2 = np.sum(data, axis = 1);

print np.abs(result - result2).max()
