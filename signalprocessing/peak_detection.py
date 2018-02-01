# -*- coding: utf-8 -*-
"""
Peak detection from Scott and Andrews
"""


import sys
import numpy as np


def find_peaks(v, delta, x = None):
  "Peak detection algortihm by Scott and Andrews"""
  if x is None:
      x = np.arange(len(v))
  
  v = np.asarray(v)
  
  if len(v) != len(x):
      sys.exit('Input vectors v and x must have same length')
  
  if not np.isscalar(delta):
      sys.exit('Input argument delta must be a scalar')
  
  if delta <= 0:
      sys.exit('Input argument delta must be positive')
  
  mn, mx = np.inf, -np.inf
  #mnpos = NaN;
  mxpos = np.nan; #, NaN
  
  lookformax = True
  
  maxid = [];
  maxvalue = [];
  for i in range(len(v)):
      this = v[i]
      if this > mx:
          mx = this
          mxpos = x[i]
      if this < mn:
          mn = this
          #mnpos = x[i]
      
      if lookformax:
          if this < mx-delta:
              maxid.append(mxpos);
              maxvalue.append(mx);
              mn = this
              #mnpos = x[i]
              lookformax = False
      else:
          if this > mn+delta:
              #mintab.append((mnpos, mn))
              mx = this
              mxpos = x[i]
              lookformax = True


  return np.asarray(maxid, dtype = int), np.asarray(maxvalue, dtype = v.dtype);

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import signalprocessing.peak_detection as pd;
    reload(pd);
    series = [0,0,0,2,0,0,0,-2,0,0,0,2,0,0,0,-2,0]
    i,v = pd.find_peaks(series,.3)
    
    plt.figure(1); plt.clf();
    plt.plot(series)
    plt.scatter(i, v, color='blue', s= 50)
    plt.draw()