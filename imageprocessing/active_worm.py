# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:36:05 2016

@author: ckirst
"""
import numpy as np
import numbers

import cv2
import scipy.ndimage as nd

def isnumber(x):
  return isinstance(x, numbers.Number);

def bwdist(a):
    return nd.distance_transform_edt(a == 0)

# Converts a mask to a SDF
def mask2phi(init_a):
    phi = bwdist(init_a) - bwdist(1 - init_a) + np.asarray(init_a, dtype = float)/init_a.max() - 0.5
    return phi

class WormShape:
  def __init__(self, nseg = 20,  l = 2, theta = 0, width  = 9, x0 = 75, y0 = 75):
    if isnumber(l):
      l = np.ones(nseg) * l;
    self.l = l;
    if isnumber(theta):
      theta = np.ones(nseg) * theta;
    self.theta = theta;
    if isnumber(width):
      width = np.ones(nseg) * width;
    self.width = width;
    self.x0 = x0; self.y0 = y0;
    self.nseg = nseg; # number of segments
  
  def midline(self):
    xm = np.zeros(self.nseg+1);
    ym = np.zeros(self.nseg+1);
    xm[0] = self.x0;
    ym[0] = self.y0;
    
    n0 = np.array([0,1], dtype = float);
    cos = np.cos(self.theta);
    sin = np.sin(self.theta);
    
    for i in range(self.nseg):
      n0 = np.array([cos[i] * n0[0] - sin[i] * n0[1], sin[i] * n0[0] + cos[i] * n0[1]]);
      xm[i+1] = xm[i] + n0[0] * self.l[i];
      ym[i+1] = ym[i] + n0[1] * self.l[i];
    
    return np.vstack([xm,ym]).T
    
  def sides(self):
    xm = np.zeros(self.nseg+1);
    ym = np.zeros(self.nseg+1);
    
    xl = np.zeros(self.nseg+2);
    yl = np.zeros(self.nseg+2); 
    
    xr = np.zeros(self.nseg+2);
    yr = np.zeros(self.nseg+2);
    
    xm[0] = self.x0;
    ym[0] = self.y0;
    
    xl[0] = self.x0;
    yl[0] = self.y0;
    
    xr[0] = self.x0;
    yr[0] = self.y0;
        
    
    n0 = np.array([0,1], dtype = float);
    t0 = np.array([1,0], dtype = float);
    cos = np.cos(self.theta);
    sin = np.sin(self.theta);
    
    for i in range(self.nseg):
      n0 = np.array([cos[i] * n0[0] - sin[i] * n0[1], sin[i] * n0[0] + cos[i] * n0[1]]);
      t0 = np.array([cos[i] * t0[0] - sin[i] * t0[1], sin[i] * t0[0] + cos[i] * t0[1]]);
      xm[i+1] = xm[i] + n0[0] * self.l[i];
      ym[i+1] = ym[i] + n0[1] * self.l[i];
      
      xl[i+1] = xm[i] + n0[0] * self.l[i] / 2.0 - t0[0] * self.width[i] / 2.0;
      yl[i+1] = ym[i] + n0[1] * self.l[i] / 2.0 - t0[1] * self.width[i] / 2.0;
      
      xr[i+1] = xm[i] + n0[0] * self.l[i] / 2.0 + t0[0] * self.width[i] / 2.0;
      yr[i+1] = ym[i] + n0[1] * self.l[i] / 2.0 + t0[1] * self.width[i] / 2.0;
      
    xl[-1] = xm[-1];
    yl[-1] = ym[-1];
    
    xr[-1] = xm[-1];
    yr[-1] = ym[-1];
    
    xyl = np.vstack([xl,yl]).T;
    xyr = np.vstack([xr,yr]).T;  
    xym = np.vstack([xm,ym]).T;    
    
    return xyl,xyr,xym
  
  def polygon(self):
    sides = self.sides();
    return np.vstack([sides[0], sides[1][::-1,:]])

  def mask(self, size = [151, 151]):
    mask = np.zeros(tuple(size));
    #poly = np.asarray(self.polygon().T, dtype = np.int32);
    xyl, xyr, xym = self.sides();
    
    for i in range(self.nseg+1):
      poly = np.array([xyl[i,:], xyr[i,:], xyr[i+1,:], xyl[i+1,:]], dtype = np.int32)
      cv2.fillPoly(mask, [poly], 1);
    return mask

  def phi(self, size = [151, 151]):
    return mask2phi(self.mask(size = size));
    
    
if __name__ == "__main__":
  
  import matplotlib.pyplot as plt;
  ws = WormShape();
  mask = ws.mask();
  
  plt.figure(1); plt.clf();
  plt.imshow(mask);
  