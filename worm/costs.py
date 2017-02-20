# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 17:30:48 2016

@author: ckirst
"""
import numpy as np

from interpolation.resampling import resample
from interpolation.curve import Curve;

import worm.geometry as wgeo

### Cost functions

# each non nan distance left/rightt is counted + renormalized

import scipy.spatial.distance as spd

def cost_from_skeleton(model, skeleton, soft_max = 2.0):
  c = model.center;
  dist = spd.cdist(c, skeleton);
  edist = np.exp(-soft_max * dist);
  return np.sum(edist / np.sum(edist, axis =1)[:,None] * dist, axis = 1)


def cost_from_distance(res, weight_shape = 0.1, weight_head_tail = 0.9):
  dl, xy_l, dr, xy_r, dh, hm, dt, tm = res;
  wlr = weight_shape;
  wht = weight_head_tail;
  cost = 0;
  
  for d in [dl, dr]:
    idx_nan = np.isnan(d);
    idx_good = np.logical_not(idx_nan);
    if np.any(idx_good):
      #dd = d[idx_good];
      #cost += wlr * np.sum(dd*dd) / np.sum(idx_good);
      cost += wlr * np.sum(d[idx_good]) / np.sum(idx_good);
    #else:
    #  cost += 0.0; # no penalty for no overlaps at this point

  nht = 2;  
  if dh is None:
    nht -=1;
    dh = 0.0;
  if dt is None:
    nht -=1;
    dt = 0.0;
  if nht > 0:
    cost += wht * (dh+dt) / nht;
  return cost;
    


def cost_from_countour(model, contour, head_tail_xy = None, verbose = False, weight_head_tail = 0.9, weight_shape = 0.1):
  if head_tail_xy is None:
    head_tail_xy = wgeo.head_tail_from_contour_discrete(contour, ncontour = all, delta = 0.3, smooth = 1.0, with_index = False, verbose = False, save = None);
  
  left,right,normals = model.shape(with_normals=True);   
  res = wgeo.distance_shape_to_contour_discrete(left,right,normals,contour,
                                           search_radius=[15,20], min_alignment=0, match_head_tail=head_tail_xy,
                                           verbose = verbose);                                            
  return cost_from_distance(res, weight_head_tail = weight_head_tail, weight_shape = weight_shape)
  




def cost_from_image(model, image, nsamples = 100, nparameter = 50):  
  cntrs = wgeo.contours_from_image(image, sigma = 1, absolute_threshold = None, threshold_factor = 0.9, 
                             verbose = False, save = None);
  contour = Curve(resample(cntrs[0], nsamples), nparameter = nparameter);     
  return cost_from_countour(model, contour);
                                              


#def cost_func(model, parameter, contour, head_tail_xy = None, verbose= False, weight_distances = 0.1, weight_head_tail = 0.9, weight_shape = 0.1):
#  """Cost function"""
#  model.set_parameter(parameter);
#  cost =  cost_from_countour(model, contour, head_tail_xy = head_tail_xy, verbose = verbose, weight_head_tail = weight_head_tail, weight_shape = weight_shape);
#  
#  # ensure optimal distances
#  d = np.diff(model.center, axis = 0);
#  d = np.sqrt(np.sum(d*d, axis = 1));
#  l0 = model.length / len(d);
#  d -= l0;
#  cost += weight_distances * np.sum(d*d) / len(d);
#  return cost;

def cost_func(model, parameter, skeleton, weight_distances = 1.0):
  """Cost function"""
  model.set_parameter(parameter);
  cost =  cost_from_skeleton(model, skeleton);
  cost = np.sum(cost);
  
  # ensure optimal distances
  d = np.diff(model.center, axis = 0);
  d = np.sqrt(np.sum(d*d, axis = 1));
  l0 = model.length / len(d);
  d -= l0;
  cost += weight_distances * np.sum(d*d) / len(d);
  return cost;




#def cost_func_grad(model, parameter, contour, head_tail_xy = None, epsilon = 0.1, verbose = False,  weight_distances = 0.1, weight_head_tail = 0.9, weight_shape = 0.1):
#  """Numerical approximation of the gradient of the cost function"""
#  nparameter = parameter.shape[0];
#  c0 = cost_func(model, parameter, contour, head_tail_xy = head_tail_xy, verbose = verbose, weight_head_tail = weight_head_tail, weight_shape = weight_shape, weight_distances = weight_distances);
#  if verbose:
#    print 'costs at 0: %f' % c0;
#  epsilon = np.array(epsilon);
#  grad = np.zeros(nparameter);
#  if epsilon.ndim == 0:
#    epsilon = np.ones(nparameter) * epsilon;
#  for i in range(nparameter):
#    p = parameter.copy(); p[i] += epsilon[i];
#    cs = cost_func(model, p, contour, head_tail_xy = head_tail_xy, weight_head_tail = weight_head_tail, weight_shape = weight_shape, weight_distances = weight_distances);
#    grad[i] = (cs - c0) / epsilon[i];
#  return grad;
#                                                                
   
def cost_func_grad(model, parameter, skeleton, epsilon, weight_distances = 1.0, verbose = True):
  """Numerical approximation of the gradient of the cost function"""
  nparameter = parameter.shape[0];
  c0 = cost_func(model, parameter, skeleton, weight_distances = weight_distances);
  if verbose:
    print 'costs at 0: %f' % c0;
  epsilon = np.array(epsilon);
  grad = np.zeros(nparameter);
  if epsilon.ndim == 0:
    epsilon = np.ones(nparameter) * epsilon;
  for i in range(nparameter):
    p = parameter.copy(); p[i] += epsilon[i];
    cs = cost_func(model, p, skeleton, weight_distances = weight_distances);
    grad[i] = (cs - c0) / epsilon[i];
  return grad;
                          
   
