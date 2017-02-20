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


import worm.costs as wc
reload(wc);
      
### Gradient descent

t0 = 500000;

img = exp.load_img(wid = 80, t= t0);  
#imgs = filters.gaussian_filter(np.asarray(img, float), 1.0);


w = wm.WormModel(npoints = 10);  
w.from_image(img, verbose = False);

w.move_forward(0.1);
w.rotate(0.1);


contours = wgeo.contours_from_image(img, sigma = 1, absolute_threshold = None, threshold_factor = 0.9, 
                                verbose = False, save = None);
contour = Curve(resample_curve(contours[0], 100), nparameter = 50);                                   

head_tail_xy = wgeo.head_tail_from_contour_discrete(contour, ncontour = all, delta = 0.3, smooth = 1.0, with_index = False,
                            verbose = True, save = None, image = imgs);

plt.figure(1); plt.clf();
w.plot(image = img)
contour.plot(with_points=False);

par = w.get_parameter()
npar = par.shape[0]

eps = 0.1 * np.ones(npar);
ik = w.center.shape[0];
eps[:ik] = 0.5;


grad = wc.cost_func_grad(w, par, contour, head_tail_xy=head_tail_xy, epsilon = eps, verbose = True)
print grad

# try gradient descent
nsteps = 100;
sw = 0.5;
par_1 = par;
for i in range(nsteps):
  par_0 = par_1;
  gg = grad;
  gg_n = np.abs(gg).sum();
  #if gg_n > 10:
  gg = sw * gg/ gg_n;
  #else:
  #  gg = sw * gg;
    
  par_1 = par_0 - gg;
  #print 'cost = %f' % wc.cost_func(w, par_1, contour);

  
  fig = plt.figure(10); plt.clf();
  w.set_parameter(par_1);
  w.plot(image = imgs);
  contour.plot(with_points=False);
  
  grad = wc.cost_func_grad(w, par_1, contour, head_tail_xy=head_tail_xy, epsilon = eps, verbose = True,
                           weight_head_tail = 10, weight_shape =5.0, weight_distances=5.0);
  
  plt.draw();
  plt.pause(0.2);



### Near self intersection to self intersection ?
reload(wgeo)


t0 = 500000 + 25620 - 4;
i = 0;

i = i+1;


t0 = 400000 + 25620 - 4;
i = i+1;
img = exp.load_img(wid = 80, t= t0+i, smooth = 1);  

plt.figure(3); plt.clf();
wgeo.center_from_image_skeleton(img, verbose= True, npoints=20, absolute_threshold = None, threshold_factor = 0.85)


w = wm.WormModel(npoints = 15);  
plt.figure(1); plt.clf();
w.from_image(img, verbose = True, nneighbours=1);

plt.figure(2); plt.clf();
w.plot(image = img);

w.move_forward(0.1);
w.rotate(0.1);


contours = wgeo.contours_from_image(imgs, sigma = 1, absolute_threshold = None, threshold_factor = 0.9, 
                                verbose = False, save = None);
contour = Curve(resample_curve(contours[0], 100), nparameter = 50);                                   

head_tail_xy = wgeo.head_tail_from_contour_discrete(contour, ncontour = all, delta = 0.3, smooth = 1.0, with_index = False,
                            verbose = True, save = None, image = imgs);

plt.figure(1); plt.clf();
w.plot(image = imgs)
contour.plot(with_points=False);

par = w.get_parameter()
npar = par.shape[0]

eps = 0.1 * np.ones(npar);
ik = w.center.shape[0];
eps[:ik] = 0.5;




### 














plt.figure(20); plt.clf();
left,right,normals = w.shape(with_normals=True);

reload(wgeo)
reload(wc)

dl, xyl, dr, xyr, dh, hm, dt, tm = wgeo.distance_shape_to_contour_discrete(left,right,normals,contour,
                                           search_radius=[15,20], min_alignment=0, match_head_tail=head_tail_xy,
                                           verbose = True);   

wc.cost_func(w, par_1, contour, head_tail_xy=head_tail_xy, verbose = True,
             weight_head_tail = 10, weight_shape =5.0, weight_distances=5.0);

