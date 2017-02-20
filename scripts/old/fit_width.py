# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 22:30:25 2016

@author: ckirst
"""

import numpy as np

import scipy.optimize as opt

w = np.array([ 0.        ,  5.26876963,  6.98778905,  8.22831422,  8.06015729,
        8.60879012,  9.65816254,  9.67394911,  8.86953907,  9.2187652 ,
        9.29279844,  9.15004847,  9.5918781 ,  9.27386605,  9.18127882,
        8.54446129,  8.12247767,  7.50160495,  6.6412495 ,  4.76439766,  0.        ]);
        
xdata = np.linspace(0,1, len(w));


def f(x,a,b):
  return b * np.power(x,a) * np.power(1-x, a) * np.power(0.5, -2*a);
  
def g(x,a,b):
  return a * np.exp(b * ()


popt, pcov = opt.curve_fit(f, xdata, w, p0 = (0.1, 10));

plt.figure(1); plt.clf();
plt.plot(xdata, w)
plt.plot(xdata, f(xdata, *popt), 'r')
plt.title(str(popt))