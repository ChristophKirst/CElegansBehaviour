# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:36:05 2016

@author: ckirst
"""
import numpy as np
import numbers

import cv2
import scipy.ndimage as nd

eps = np.finfo(float).eps

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
    self.l = l.copy();
    if isnumber(theta):
      theta = np.ones(nseg) * theta;
    self.theta = theta.copy();
    if isnumber(width):
      width = np.ones(nseg) * width;
    self.width = width.copy();
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
  

  def error(self, image, phi = None, delta_phi = 1.2, epsilon = 0.1, curvature = None):
    
    if phi is None:
      phi = self.phi(size = image.shape);
    
    # regularized cruve information
    idx = np.flatnonzero(np.logical_and(phi <= delta_phi, phi >= -delta_phi))

    if len(idx) == 0:
      return 0;
    
    # interior and exterior means
    #upts = np.flatnonzero(phi <= 0)  # interior points
    #vpts = np.flatnonzero(phi > 0)  # exterior points
    #u = np.sum(image.flat[upts]) / (len(upts) + eps)  # interior mean
    #v = np.sum(image.flat[vpts]) / (len(vpts) + eps)  # exterior mean

    theta_in = 0.5 * (np.tanh(phi.flat[idx] / epsilon) + 1);
    theta_out = 1 - theta_in;
    
    phi_in = phi.flat * theta_in;
    phi_out = phi.flat * theta_out;
    
    mean_in = np.sum(phi_in) / (np.sum(theta_in) + eps); 
    mean_out = np.sum(phi_out) / (np.sum(theta_out) + eps); 

    # error from image information
    error =  np.sum((theta_in * (image.flat[idx] - mean_in ))**2);
    error += np.sum((theta_out *(image.flat[idx] - mean_out))**2);
    
    # error from curvature constraints
    if curvature is not None:
      c = get_curvature(phi, idx);
      error += np.sum(np.abs(c));
    
    return error;
    

  def optimize(self, image):
    import scipy.optimize as opt;
    
    ws = WormShape(nseg = self.nseg, l = self.l, theta = self.theta,
                   width = self.width, x0 = self.x0, y0 = self.y0);
    
    def fun(x):
      ws.l     = x[:self.nseg];
      ws.theta = x[self.nseg:2*self.nseg];
      ws.width = x[2*self.nseg:3*self.nseg];
      ws.x0 = x[-2];
      ws.y0 = x[-1];
      return ws.error(image = image);
    
    x0 = np.hstack([self.l, self.theta, self.width, [self.x0, self.y0]]);
    res = opt.minimize(fun, x0);
    
    return res;





# Compute curvature
def get_curvature(phi, idx):
    dimy, dimx = phi.shape
    yx = np.array([np.unravel_index(i, phi.shape) for i in idx])  # subscripts
    y = yx[:, 0]
    x = yx[:, 1]

    # Get subscripts of neighbors
    ym1 = y - 1
    xm1 = x - 1
    yp1 = y + 1
    xp1 = x + 1

    # Bounds checking
    ym1[ym1 < 0] = 0
    xm1[xm1 < 0] = 0
    yp1[yp1 >= dimy] = dimy - 1
    xp1[xp1 >= dimx] = dimx - 1

    # Get indexes for 8 neighbors
    idup = np.ravel_multi_index((yp1, x), phi.shape)
    iddn = np.ravel_multi_index((ym1, x), phi.shape)
    idlt = np.ravel_multi_index((y, xm1), phi.shape)
    idrt = np.ravel_multi_index((y, xp1), phi.shape)
    idul = np.ravel_multi_index((yp1, xm1), phi.shape)
    idur = np.ravel_multi_index((yp1, xp1), phi.shape)
    iddl = np.ravel_multi_index((ym1, xm1), phi.shape)
    iddr = np.ravel_multi_index((ym1, xp1), phi.shape)

    # Get central derivatives of SDF at x,y
    phi_x = -phi.flat[idlt] + phi.flat[idrt]
    phi_y = -phi.flat[iddn] + phi.flat[idup]
    phi_xx = phi.flat[idlt] - 2 * phi.flat[idx] + phi.flat[idrt]
    phi_yy = phi.flat[iddn] - 2 * phi.flat[idx] + phi.flat[idup]
    phi_xy = 0.25 * (- phi.flat[iddl] - phi.flat[idur] +
                     phi.flat[iddr] + phi.flat[idul])
    phi_x2 = phi_x**2
    phi_y2 = phi_y**2

    # Compute curvature (Kappa)
    curvature = ((phi_x2 * phi_yy + phi_y2 * phi_xx - 2 * phi_x * phi_y * phi_xy) /
                 (phi_x2 + phi_y2 + eps) ** 1.5) * (phi_x2 + phi_y2) ** 0.5

    return curvature
    
    
    
    
    
  
if __name__ == "__main__":
  
  import matplotlib.pyplot as plt;
  ws = WormShape();
  mask = ws.mask();
  
  plt.figure(1); plt.clf();
  plt.imshow(mask);
  