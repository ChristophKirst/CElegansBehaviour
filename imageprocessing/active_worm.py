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
  

  def error(self, delta_phi = 1.2):
    
    # regularized cruve information
    idx = np.flatnonzero(np.logical_and(phi <= delta_phi, phi >= -delta_phi))

    if len(idx) > 0:
            # Intermediate output
            if display:
                if np.mod(its, 50) == 0:
                    print('iteration: {0}'.format(its))
                    show_curve_and_phi(fig, I, phi, color)
            else:
                if np.mod(its, 10) == 0:
                    print('iteration: {0}'.format(its))

            # Find interior and exterior mean
            upts = np.flatnonzero(phi <= 0)  # interior points
            vpts = np.flatnonzero(phi > 0)  # exterior points
            u = np.sum(I.flat[upts]) / (len(upts) + eps)  # interior mean
            v = np.sum(I.flat[vpts]) / (len(vpts) + eps)  # exterior mean

            # Force from image information
            F = (I.flat[idx] - u)**2 - (I.flat[idx] - v)**2
            # Force from curvature penalty
            curvature = get_curvature(phi, idx)

            # Gradient descent to minimize energy
            dphidt = F / np.max(np.abs(F)) + alpha * curvature

            # Maintain the CFL condition
            dt = 0.45 / (np.max(np.abs(dphidt)) + eps)

            # Evolve the curve
            phi.flat[idx] += dt * dphidt

            # Keep SDF smooth
            phi = sussman(phi, 0.5)

            new_mask = phi <= 0
            c = convergence(prev_mask, new_mask, thresh, c)

            if c <= 5:
                its = its + 1
                prev_mask = new_mask
            else:
                stop = True

        else:
            break

    # Final output
    if display:
        show_curve_and_phi(fig, I, phi, color)
        plt.savefig('levelset_end.png', bbox_inches='tight')

    # Make mask from SDF
    seg = phi <= 0  # Get mask from levelset

    return seg, phi, its
    
if __name__ == "__main__":
  
  import matplotlib.pyplot as plt;
  ws = WormShape();
  mask = ws.mask();
  
  plt.figure(1); plt.clf();
  plt.imshow(mask);
  