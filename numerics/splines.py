# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 17:39:38 2016

@author: ckirst
"""




class spline_interpolator:
  def __init__(self, x, )



  
  def spline
  def initalize_splines(self):
    self.s = np.linspace(0, 1, self.nsamples);
    self.
    
    
    
import numpy as np

x = np.linspace(0, 1, 10);
y = np.sin(x);

knots = np.linspace(0,1,5);

from scipy.interpolate import splprep, splrep, splev


tck = splrep(x,y, t = knots);