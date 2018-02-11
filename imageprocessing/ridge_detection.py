# -*- coding: utf-8 -*-
"""
Ridge Detection
"""

import numpy as np

def mask_size(sigma, derivative):
 max_mask = [3.09023230616781, 3.46087178201605, 3.82922419517181];	
 return int(np.ceil(max_mask[derivative] * sigma));

import scipy.stats as sts
import scipy.ndimage as ndi

def phi(x, sigma, derivative):
  if derivative == 0:
    return sts.norm.cdf(x/sigma);
  elif derivative == 1:
    t = x /sigma;
    return 1/np.sqrt(2 * np.pi) / sigma * np.exp(-0.5 * t * t);
  else:
    t = x /sigma;
    return -x * 1/np.sqrt(2 * np.pi) / sigma**3 * np.exp(-0.5 * t * t);
  
def gauss_filter_1d(sigma, derivative):
  n = mask_size(sigma,derivative);
  g = np.zeros(2*n+1);
  i = np.arange(-n+1,n);
  g[1:-1] = phi(-i + 0.5, sigma, derivative) - phi(-i - 0.5, sigma, derivative);
  if derivative == 0:
    g[0] = 1.0 - phi(n-0.5, sigma, derivative);
  else:
    g[0] = -phi(n-0.5, sigma, derivative);
  g[-1] = phi(-n+0.5, sigma, derivative);
  return g;

def gaussian_derivative(image, sigma, dx, dy):
  gdx = gauss_filter_1d(sigma, dx);
  gdy = gauss_filter_1d(sigma, dy);
  img = ndi.filters.convolve1d(image, gdx, axis = 0);
  img = ndi.filters.convolve1d(img,   gdy, axis = 1);
  return img;


def eigenvalues_vectors(dxx, dxy, dyy):
  i0 = dxy == 0;
  i1 = np.logical_not(i0);
  theta = np.zeros(dxx.shape);
  t = np.zeros(dxx.shape);
  s = np.zeros(dxx.shape);
  c = np.ones(dxx.shape);
  e1 = np.zeros(dxx.shape);
  e2 = np.zeros(dxx.shape);
  
  e1[i0] = dxx[i0];
  e2[i0] = dyy[i0];
  
  theta[i1] = 0.5 * (dyy[i1] - dxx[i1]) / dxy[i1];   
  t[i1] = 1.0 / (np.abs(theta[i1]) + np.sqrt(theta[i1] * theta[i1] + 1.0));
  i2 = theta < 0;
  t[i2] = - t[i2];
  c[i1] = 1.0 / np.sqrt(t[i1] * t[i1] + 1.0);
  s[i1] = t[i1] * c[i1];
  e1[i1] = dxx[i1] - t[i1] * dxy[i1];
  e2[i1] = dyy[i1] + t[i1] * dxy[i1];
  
  n1 = c;
  n2 = -s;  
  
  evals = np.zeros(dxx.shape + (2,));
  evecs = np.zeros(dxx.shape + (2,2,)); 
  
  i0 = np.abs(e1) > np.abs(e2);
  evals[i0,0] = e1[i0];
  evals[i0,1] = e2[i0];
  evecs[i0,0,0] = n1[i0];
  evecs[i0,0,1] = n2[i0];
  evecs[i0,1,0] = -n2[i0];
  evecs[i0,1,1] = n1[i0];
  
  i0 = np.abs(e1) < np.abs(e2);
  evals[i0,0] = e2[i0];
  evals[i0,1] = e1[i0];
  evecs[i0,0,0] = -n2[i0];
  evecs[i0,0,1] = n1[i0];
  evecs[i0,1,0] = n1[i0];
  evecs[i0,1,1] = n2[i0];
  
  i0 = np.abs(e1) == np.abs(e2);
  if i0.sum() > 0:
    i1 = np.logival_and(i0, e1 < e2);
    evals[i1] = e1[i1];
    evals[i1,1] = e2[i1];
    evecs[i1,0,0] = n1[i1];
    evecs[i1,0,1] = n2[i1];
    evecs[i0,1,0] = -n2[i0];
    evecs[i0,1,1] = n1[i0];
    
    i1 = np.logical_and(i0, e1 >= e2);
    evals[i1,0] = e2[i1];
    evals[i1,1] = e1[i1];
    evecs[i1,0,0] = -n2[i1];
    evecs[i1,0,1] = n1[i1];
    evecs[i1,1,0] = n1[i1];
    evecs[i1,1,1] = n2[i1];
    
  return evals, evecs;
  
 
def ridge_points(image, sigma, mode = 'light'):
  dx  = gaussian_derivative(image, sigma, 1, 0);
  dy  = gaussian_derivative(image, sigma, 0, 1);
  dxx = gaussian_derivative(image, sigma, 2, 0);
  dxy = gaussian_derivative(image, sigma, 1, 1);
  dyy = gaussian_derivative(image, sigma, 0, 2);
  evals, evecs = eigenvalues_vectors(dxx, dxy, dyy);
  
  e = evals[:,:,0];
  if mode == 'light':
    i0 = -e > 0;
  else:
    i0 = e > 0;
  
  n1 = evecs[i0][:,0,0];
  n2 = evecs[i0][:,0,1];
  a = dxx[i0] * n1 * n1 + 2.0 * dxy[i0] * n1 * n2 + dyy[i0] * n2 * n2;
  b = dx[i0] * n1 + dy[i0] * n2;
  
  i1 = a != 0;
    
  t = - b[i1] / a[i1];
  
  p1 = n1[i1] * t;
  p2 = n2[i1] * t;
  
  i2 = np.logical_and(np.abs(p1) <= 0.6, np.abs(p2) <= 0.6);
  
  normals = np.zeros(image.shape + (2,));
  points  = np.zeros(image.shape + (2,));
  
  x,y = np.where(i0);
  x = x[i1][i2];
  y = y[i1][i2];
  ridges = np.zeros(image.shape, dtype = bool);
  ridges[x,y] = True;
  
  normals[ridges,0] = n1[i1][i2];
  normals[ridges,1] = n2[i1][i2];
  points[ridges,0] = p1[i2];
  points[ridges,1] = p2[i2];
 
  return ridges,normals,points,evals,evecs

  
def detect_ridges(image, sigma, lower, higher = None, mode = 'light', return_info = False):
  ridges,normals,points,evals,evecs = ridge_points(image, sigma, mode);
  if mode == 'light':
    e = -evals[:,:,0];
  else:
    e = evals[:,:,0];
  
  i0 = e < lower;
  if higher is not None:
    i0 = np.logical_or(i0, e > higher);
  ridges[i0] = False;
  
  if not return_info:
    return ridges;
  else:
    return ridges, normals, points, evals, evecs


import matplotlib.pyplot as plt  
 
def plot(ridges, normals, points, image = None):
  if image is not None:
    plt.imshow(image, interpolation = 'none');
  
  x,y = np.where(ridges);
  pt = (np.vstack([x,y]).T - points[x,y]).T;
  pt = pt[[1,0]];
  plt.scatter(*pt, s = 40, c = 'w');
  for xi,yi in zip(x,y):
    p0 = -points[xi,yi] + [xi,yi];
    p1 = p0 + 5 * normals[xi,yi];
    plt.plot([p0[1],p1[1]], [p0[0], p1[0]], c = 'w');