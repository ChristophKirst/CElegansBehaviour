# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 18:07:27 2016

@author: ckirst
"""

## match worm to contour -> gradient descent 

import numpy as np
import matplotlib.pyplot as plt
import worm.model as wm;
import worm.geometry as wgeo
reload(wgeo); reload(wm);

import analysis.experiment as exp
import scipy.ndimage.filters as filters
from interpolation.curve import Curve
from interpolation.resampling import resample as resample_curve

# load image

t0 = 500000;

img = exp.load_img(wid = 80, t= t0);  
imgs = filters.gaussian_filter(np.asarray(img, float), 1.0);

w = wm.WormModel(npoints = 20);  
w.from_image(img, verbose = True);

w.move_forward(0.1);
w.rotate(0.1);


plt.figure(1); plt.clf();
plt.subplot(1,2,1)
w.plot(image = imgs)
plt.subplot(1,2,2);
cntrs = wgeo.contour_from_image(imgs, sigma = 1, absolute_threshold = None, threshold_factor = 0.9, 
                          verbose = True, save = None);
cntr = resample_curve(cntrs[0], 100);
contour = Curve(cntr, nparameter = 50);                        
                      
                      
plt.figure(2); plt.clf();
head_tail_xy = wgeo.head_tail_from_contour(cntrs, ncontour = all, delta = 0.3, smooth = 1.0, with_index = False,
                            verbose = True, save = None, image = imgs);


left,right,normals = w.shape(with_normals=True);  
plt.figure(3); plt.clf()
reload(wgeo)
res = wgeo.distance_shape_to_contour_discrete(left,right,normals,contour,
                                              search_radius=[5,20], min_alignment=0, match_head_tail=head_tail_xy,
                                              verbose = True);
 


reload(wgeo)

def cost_func(model, parameter, contours):
  """Cost function"""
  model.set_parameter(parameter);
  return wgeo.cost_from_contours(model, contours);


def cost_func_grad(model, parameter, contour, epsilon = 0.1):
  """Numerical approximation of the gradient of the cost function"""
  nparameter = parameter.shape[0];
  c0 = cost_func(model, parameter, contour);
  epsilon = np.array(epsilon);
  grad = np.zeros(nparameter);
  if epsilon.ndim == 0:
    epsilon = np.ones(nparameter) * epsilon;
  for i in range(nparameter):
    p = parameter.copy(); p[i] += epsilon[i];
    cs = cost_func(model, p, contour);
    grad[i] = (cs - c0) / epsilon[i];
  return grad;
                                                                
   
    
wgeo.cost_from_distance(res)
wgeo.cost_from_contours(w, cntrs)
                                         
### Test GD on this

t0 = 500000;

img = exp.load_img(wid = 80, t= t0);  
imgs = filters.gaussian_filter(np.asarray(img, float), 1.0);

w = wm.WormModel(npoints = 20);  
w.from_image(img, verbose = False);

w.move_forward(0.1);
w.rotate(0.1);


cntrs = wgeo.contour_from_image(imgs, sigma = 1, absolute_threshold = None, threshold_factor = 0.9, 
                                verbose = False, save = None);

plt.figure(1); plt.clf();
w.plot(image = imgs)
plt.plot(cntrs[0][:,0], cntrs[0][:,1])


par = w.get_parameter()
npar = par.shape[0]

eps = 0.1 * np.ones(npar);
ik = w.theta.shape[0];
eps[:ik] = 1;
eps[:] = 0.1;

grad = cost_func_grad(w, par, cntrs, epsilon = eps)
print grad

# try gradient descent
nsteps = 100;
sw = 0.1;
par_1 = par;
for i in range(nsteps):
  par_0 = par_1;
  gg = grad;
  gg[ik:] *= 0.1;
  par_1 = par_0 - sw * (gg / np.abs(gg).sum());
  print 'cost = %f' % cost_func(w, par_1, cntrs);
  grad = cost_func_grad(w, par_1, cntrs, epsilon = eps)
  
  plt.figure(10); plt.clf();
  w.set_parameter(par_1);
  w.plot(image = imgs);
  plt.plot(cntrs[0][:,0], cntrs[0][:,1])
  
  plt.draw();
  plt.pause(0.1);




