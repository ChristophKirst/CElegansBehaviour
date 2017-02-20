# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 17:52:42 2016

@author: ckirst
"""


# invert a functoin numerically

import numpy as np
import matplotlib.pyplot as plt;
from scipy.special import ellipeinc
from scipy import optimize

def fun(x):
  return ellipeinc(x,0.5);

def inverse(fun, x):
  def err(xx):
    return fun(xx)-x;
  return optimize.newton(err, x)


x = np.linspace(0,2*np.pi,20);
y = fun(x);

plt.figure(1); plt.clf();
plt.subplot(2,1,1)
plt.plot(x, y)
#plt.subplot(2,1,2)

yi = [inverse(fun, xx) for xx in x];
plt.plot(x,yi)



