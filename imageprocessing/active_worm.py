# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:36:05 2016

@author: ckirst
"""
import numpy as np
import matplotlib.pyplot as plt
import numbers
import copy

import cv2
import scipy.ndimage as nd

from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev

eps = np.finfo(float).eps

from scripts.analyse_wormshape import analyse_shape


def isnumber(x):
  return isinstance(x, numbers.Number);

def bwdist(a):
    return nd.distance_transform_edt(a == 0)

# Converts a mask to implicit contour
def mask2phi(init_a):
    phi = bwdist(init_a) - bwdist(1 - init_a) + np.asarray(init_a, dtype = float)/init_a.max() - 0.5
    return phi

class WormShape:
  def __init__(self, nseg = 20,  l = 2, theta = 0, width = 9, position = [75, 75], orientation = 0):
    if nseg % 2 != 0:
      raise RuntimeError('WormShape: number of segments expected to be even!')
    
    if isnumber(l):
      l = np.ones(nseg) * l;
    self.l = l.copy();
    if isnumber(theta):
      theta = np.ones(nseg-1) * theta;
    self.theta = theta.copy();
    if isnumber(width):
      width = np.ones(nseg) * width;
    self.width = width.copy();
    self.position = np.array(position);
    self.orientation = orientation;
    self.nseg = nseg; # number of segments
  
  ############################################################################
  ### Constructors
  
  def from_lines(self, center_line, left_line, right_line):
    self.nseg = center_line.shape[0] - 1;
    if self.nseg % 2 != 0:
      raise RuntimeError('WormShape: number of line points expected to be odd!')
    
    n2 = self.nseg / 2;
    
    self.position = center_line[n2,:];
    
    self.l = np.zeros(self.nseg);
    self.theta = np.zeros(self.nseg-1);
    self.width = np.zeros(self.nseg);
    
    t0 = np.array([0,1], dtype = float);
    t1 = center_line[n2,:]-center_line[n2-1,:];
    self.orientation = np.arctan2(t0[0], t0[1]) - np.arctan2(t1[0], t1[1])
    self.l[n2-1] = np.linalg.norm(t1);   
    c = (center_line[n2,:] + center_line[n2-1,:]) / 2;
    distsl = cdist([c], left_line);
    distsr = cdist([c], right_line);
    self.width[n2-1] = distsr.min() + distsl.min();
    
    t0 = t1;
    for i in range(n2):
      t1 = center_line[i+1+n2,:]-center_line[i+n2,:];
      self.l[n2+i] = np.linalg.norm(t1);
      
      self.theta[n2+i-1] = np.arctan2(t0[0], t0[1]) - np.arctan2(t1[0], t1[1])
      
      c = (center_line[i+1+n2,:] + center_line[i+n2,:]) / 2;
      distsl = cdist([c], left_line);
      distsr = cdist([c], right_line);
      self.width[n2+i] = distsr.min() + distsl.min();
      
      t0 = t1;  
    
    t1 = center_line[n2,:]-center_line[n2-1,:];
    for i in range(n2-1):
      t0 = center_line[n2-1-i,:]-center_line[n2-2-i,:];
      self.l[n2-2-i] = np.linalg.norm(t0);
      self.theta[n2-2-i] = np.arctan2(t0[0], t0[1]) - np.arctan2(t1[0], t1[1])
      
      c = (center_line[n2-1-i,:] + center_line[n2-2-i,:]) / 2;
      distsl = cdist([c], left_line);
      distsr = cdist([c], right_line);
      self.width[n2-2-i] = distsr.min() + distsl.min();
      
      t1 = t0;    

     
  def from_center_line(self, center_line, width):
    self.nseg = center_line.shape[0] - 1;
    if self.nseg % 2 != 0:
      raise RuntimeError('WormShape: number of line points expected to be odd!')
    n2 = self.nseg / 2;
    
    self.position = center_line[n2,:];
        
    self.l = np.zeros(self.nseg);
    self.theta = np.zeros(self.nseg-1);
    self.width = width;
    
    t0 = np.array([0,1], dtype = float);
    t1 = center_line[n2,:]-center_line[n2-1,:];
    self.orientation = np.arctan2(t0[0], t0[1]) - np.arctan2(t1[0], t1[1])
    self.l[n2-1] = np.linalg.norm(t1);   
    
    t0 = t1;
    for i in range(n2):
      t1 = center_line[i+1+n2,:]-center_line[i+n2,:];
      self.l[n2+i] = np.linalg.norm(t1);
      self.theta[n2+i-1] = np.arctan2(t0[0], t0[1]) - np.arctan2(t1[0], t1[1])
      t0 = t1;  
    
    t1 = center_line[n2,:]-center_line[n2-1,:];
    for i in range(n2-1):
      t0 = center_line[n2-1-i,:]-center_line[n2-2-i,:];
      self.l[n2-2-i] = np.linalg.norm(t0);
      self.theta[n2-2-i] = np.arctan2(t0[0], t0[1]) - np.arctan2(t1[0], t1[1])
      t1 = t0;
  
  
  def from_image(self, image, sigma = 1, threshold_level = 0.95,
                 npts_contour = 100, npts_sides = 21, smooth = 1.0, verbose = False):
    shape = analyse_shape(image, sigma = sigma, threshold_level = threshold_level, 
                 npts_contour = npts_contour, npts_sides = npts_sides, 
                 smooth = smooth, verbose = verbose, save = None);
    self.from_lines(shape[-3], shape[-2], shape[-1]);
  
  
  def center_line(self):
    xym = np.zeros((self.nseg + 1, 2));
    n2 = self.nseg/2;
    xym[n2,:] = self.position;
    
    cos = np.cos(self.orientation);
    sin = np.sin(self.orientation);
    n0 = np.array([- sin, cos]);
    xym[n2-1,:] = xym[n2,:] - n0 * self.l[n2-1];
        
    cos = np.cos(self.theta);
    sin = np.sin(self.theta);
    n1 = n0.copy();
    for i in range(n2):
      n1 = np.array([cos[n2-1+i] * n1[0] - sin[n2-1+i] * n1[1], sin[n2-1+i] * n1[0] + cos[n2-1+i] * n1[1]]);
      xym[i+n2+1,:] = xym[i+n2,:] + n1 * self.l[i+n2];
    
    for i in range(n2-1):
      n0 = np.array([cos[n2-2-i] * n0[0] + sin[n2-2-i] * n0[1], - sin[n2-2-i] * n0[0] + cos[n2-2-i] * n0[1]]);
      xym[n2-2-i,:] = xym[n2-1-i,:] - n0 * self.l[n2-2-i];
    
    return xym;
    
  
  def sides(self):
    xym = np.zeros((self.nseg+1, 2));
    xyl = np.zeros((self.nseg+2, 2));
    xyr = np.zeros((self.nseg+2, 2));
    
    n2 = self.nseg / 2;
    
    xym[n2,:] = self.position;
    
    cos = np.cos(self.orientation);
    sin = np.sin(self.orientation);
    n0 = np.array([-sin, cos]);
    t0 = np.array([cos, sin]);
    n1 = n0.copy();
    t1 = t0.copy();
    xym[n2-1,:] = xym[n2,:] - n0 * self.l[n2-1];
    
    c = (xym[n2,:] + xym[n2-1,:])/2.0;
    xyl[n2,:] = c - t0 * self.width[n2-1] / 2.0;
    xyr[n2,:] = c + t0 * self.width[n2-1] / 2.0;
    
    cos = np.cos(self.theta);
    sin = np.sin(self.theta);
    

    for i in range(n2):
      k = n2+i-1;
      n0 = np.array([cos[k] * n0[0] - sin[k] * n0[1], sin[k] * n0[0] + cos[k] * n0[1]]);
      t0 = np.array([cos[k] * t0[0] - sin[k] * t0[1], sin[k] * t0[0] + cos[k] * t0[1]]);
      xym[n2+i+1,:] = xym[n2+i,:] + n0 * self.l[n2+i];

      c = (xym[n2+i+1,:] + xym[n2+i,:])/2.0;
      xyl[n2+i+1,:] = c - t0 * self.width[n2+i] / 2.0;
      xyr[n2+i+1,:] = c + t0 * self.width[n2+i] / 2.0;
    
    xyl[-1,:] = xym[-1,:];
    xyr[-1,:] = xym[-1,:];
    
    
    for i in range(n2-1):
      k = n2-2-i;
      n1 = np.array([cos[k] * n1[0] + sin[k] * n1[1], - sin[k] * n1[0] + cos[k] * n1[1]]);
      t1 = np.array([cos[k] * t1[0] + sin[k] * t1[1], - sin[k] * t1[0] + cos[k] * t1[1]]);
      xym[n2-2-i,:] = xym[n2-1-i,:] - n1 * self.l[n2-2-i];
      
      c = (xym[n2-2-i,:] + xym[n2-1-i,:])/2.0;
      xyl[n2-1-i,:] = c - t1 * self.width[n2-2-i] / 2.0;
      xyr[n2-1-i,:] = c + t1 * self.width[n2-2-i] / 2.0;
    
    xyl[0,:] = xym[0,:];
    xyr[0,:] = xym[0,:];
    
    return xyl,xyr,xym
  
  
  def center(self):
    return self.position;
  
  
  def translate(self, xy):
    self.position += np.array(xy);
    
  
  def rotate(self, ang):
    self.orientation += ang;
  
  
  def rotate_point(self, ang, point):
    cline = self.center_line();
    cline = np.dot(cline-point,[[np.cos(ang),np.sin(ang)],[-np.sin(ang),np.cos(ang)]])+point;
    self.from_center_line(cline, self.width);
  
  
  def forward(self, dist, smooth = 1.0):
    cline = self.center_line();
    # make interpolation and move forward 
    cinterp, u = splprep(cline.T, u = None, s = smooth, per = 0)
    us = u + dist;
    x, y = splev(us, cinterp, der = 0); 
    cline = np.array([x,y]).T;
    self.from_center_line(cline, self.width);
  
     
  def stretch(self, factor):
    self.l *= factor;
  
    
  def scale(self, factor):
    self.l *= factor;
    self.width *= factor;
    
  
  def curve(self, mode_amplitudes):
    #changes curvature by the mode amplitudes;
    #print self.theta.shape
    t = np.fft.rfft(self.theta);
    #print t.shape;
    mode_amplitudes = np.array(mode_amplitudes);
    t[:mode_amplitudes.shape[0]] += mode_amplitudes;
    #print t.shape;
    self.theta = np.fft.irfft(t, n = self.nseg-1);
    #print self.theta.shape
  
  
  def widen(self, factor):
    self.width *= factor;
  
  
  def bend(self, bend, front = True, exponent = 40):
    #head tail bend profile
    if front:
      self.theta += bend * np.exp(-exponent * np.linspace(0,1,self.nseg-1));
    else:
      self.theta += bend * np.exp(-exponent * np.linspace(1,0,self.nseg-1));
  
  
  
  
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
  

  def error(self, image, phi = None, delta_phi = 9, epsilon = 2, curvature = None):
    
    if phi is None:
      phi = self.phi(size = image.shape);
    
    # regularized cruve information
    idx = np.flatnonzero(np.logical_and(phi <= delta_phi, phi >= -delta_phi))
    
    # disp
    #import matplotlib.pyplot as plt
    #plt.figure(10); plt.clf();
    #ii = np.zeros(image.shape);
    #ii.flat[idx] = phi.flat[idx];
    #plt.imshow(ii);
    #plt.show();

    if len(idx) == 0:
      return 0;
    
    # interior and exterior means
    #upts = np.flatnonzero(phi <= 0)  # interior points
    #vpts = np.flatnonzero(phi > 0)  # exterior points
    #u = np.sum(image.flat[upts]) / (len(upts) + eps)  # interior mean
    #v = np.sum(image.flat[vpts]) / (len(vpts) + eps)  # exterior mean

    theta_out = 0.5 * (np.tanh(phi / epsilon) + 1);
    theta_in = 1 - theta_out;
    
    img_in = image * theta_in;
    img_out = image * theta_out;
    
    mean_in  = np.sum(img_in ) / (np.sum(theta_in ) + eps); 
    mean_out = np.sum(img_out) / (np.sum(theta_out) + eps); 
    #print mean_in, mean_out
    
    #plt.figure(11); plt.clf();
    #ii = np.zeros(image.shape);
    #plt.subplot(1,2,1);
    #plt.imshow(theta_in);
    #plt.subplot(1,2,2);
    #plt.imshow(theta_out);

    # error from image information
    error =  np.sum((theta_in * (image - mean_in ))**2);
    error += np.sum((theta_out *(image - mean_out))**2);
    
    # error from curvature constraints
    if curvature is not None:
      c = get_curvature(phi, idx);
      error += np.sum(np.abs(c));
    
    return error;
  
  
  def optimize(self, image, maxiter = 30, debug = True, swarmsize = 100, nmodes = 3):
    #import scipy.optimize as opt;
    import pyswarm as pso;
    
    #ws = WormShape(nseg = self.nseg, l = self.l.copy(), theta = self.theta.copy(),
    #               width = self.width.copy(), position = self.position.copy(), orientation = self.orientation);
    ws = copy.deepcopy(self);
    
    def fun(x):
      #print '------'
      ws2 = copy.deepcopy(ws);
      #print ws2.l.shape, ws2.theta.shape, ws2.width.shape

      ws2.bend(x[0], front = True, exponent = 10);
      ws2.bend(x[1], front = False, exponent = 10);
      ws2.forward(x[2]/10);
      #print ws2.l.shape, ws2.theta.shape, ws2.width.shape
      
      ws2.curve(x[3:-3]);
      #print ws2.l.shape, ws2.theta.shape, ws2.width.shape
      ws2.translate(x[-3:-1] * 1);
      #print ws2.l.shape, ws2.theta.shape, ws2.width.shape
      ws2.rotate(x[-1]);      
      return ws2.error(image = image);
    
    #x0 = np.hstack([self.l, self.theta, self.width, self.position]);
    #x0 = np.zeros(3 + nmodes + 0*3);
    lb = np.hstack([[-1, -1, -0.5], [-3 for i in range(nmodes)], [-4, -np.pi/6]]);
    ub = np.hstack([[1, 1, 0.5], [3 for i in range(nmodes)], [4, np.pi/6]]);
    par, fopt = pso.pso(fun, lb, ub, debug = debug, maxiter = maxiter, swarmsize = swarmsize)
    
    self.bend(par[0], exponent = 10, front = True);
    self.bend(par[1], exponent = 10, front = False);
    self.forward(par[2] / 10.0);
    self.curve(par[3:-3]);
    self.translate(par[-3:-1]);
    self.rotate(par[-1]);
    
    return par;   
  
  
  def optimize_old(self, image, method = 'BFGS', options = {'gtol': 1e-12, 'disp': True, 'eps' : 0.1}, nmodes = 3):
    import scipy.optimize as opt;
    
    #ws = WormShape(nseg = self.nseg, l = self.l.copy(), theta = self.theta.copy(),
    #               width = self.width.copy(), position = self.position.copy(), orientation = self.orientation);
    ws = copy.deepcopy(self);
    
    def fun(x):
      #print '------'
      ws2 = copy.deepcopy(ws);
      #print ws2.l.shape, ws2.theta.shape, ws2.width.shape

      ws2.bend(x[0], front = True, exponent = 10);
      ws2.bend(x[1], front = False, exponent = 10);
      ws2.forward(x[2]/10);
      #print ws2.l.shape, ws2.theta.shape, ws2.width.shape
      
      ws2.curve(x[3:-3]);
      #print ws2.l.shape, ws2.theta.shape, ws2.width.shape
      #ws2.translate(x[-3:-1] * 1);
      #print ws2.l.shape, ws2.theta.shape, ws2.width.shape
      #ws2.rotate(x[-1]);      
      return ws2.error(image = image);
    
    #x0 = np.hstack([self.l, self.theta, self.width, self.position]);
    x0 = np.zeros(3 + nmodes + 0*3);
    
    #x0 = np.zeros(1);
    #res = opt.minimize(fun, x0, jac = False, hess = False, tol = 10);
    bnds = [(None, None) for i in x0];
    bnds[0] = (-0.2, 0.2);
    bnds = tuple(bnds);
    bnds = None;
    def p(x):
      print x;
    res = opt.minimize(fun, x0, jac = False, method = method, options = options, bounds = bnds, callback = p)
    
    par = res['x'];
    self.bend(par[0], exponent = 10, front = True);
    self.bend(par[1], exponent = 10, front = False);
    self.forward(par[2] / 10.0);
    self.curve(par[3:-3]);
    #self.translate(par[-3:-1]);
    #self.rotate(par[-1]);
    
    return res; 
  

  def optimize_parameter(self, image, method = 'BFGS', options = {'gtol': 1e-6, 'disp': True}):
    import scipy.optimize as opt;
    
    ws = WormShape(nseg = self.nseg, l = self.l, theta = self.theta,
                   width = self.width, position = self.position);
    
    def fun(x):
      ws.l     = x[:self.nseg];
      ws.theta = x[self.nseg:2*self.nseg];
      #ws.width = x[2*self.nseg:3*self.nseg];
      ws.position = x[-2:];
      return ws.error(image = image);
    
    #x0 = np.hstack([self.l, self.theta, self.width, self.position]);
    x0 = np.hstack([self.l, self.theta, self.position]);
    #res = opt.minimize(fun, x0, jac = False, hess = False, tol = 10);
    res = opt.minimize(fun, x0, jac = False, method = method, options = options)
    
    par = res['x'];
    self.l = par[:self.nseg];
    self.theta = par[self.nseg:2*self.nseg];
    #self.width = par[2*self.nseg:3*self.nseg];
    self.position = par[-2:];    
    
    return res;


  def plot(self, image = None, ccolor = 'black', lcolor = 'green', rcolor = 'red', ax = None):
    xyl, xyr, xym = self.sides();
    if ax is None:
      ax = plt.gca();
    
    if image is not None:
      ax.imshow(image);
    ax.scatter(xyl[:,0], xyl[:,1], c = lcolor);
    ax.scatter(xyr[:,0], xyr[:,1], c = rcolor);
    ax.scatter(xym[:,0], xym[:,1], c = ccolor);




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
  