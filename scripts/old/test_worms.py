# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:21:43 2016

@author: ckirst
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters  

from interpolation.curve import Curve
from interpolation.resampling import resample as resample_curve  
import analysis.experiment as exp

import worm.model as wmod;
import worm.geometry as wgeo;

reload(wgeo); reload(wmod);

def plot(curve, *args, **kwargs):
  plt.plot(curve[:,0], curve[:,1], *args, **kwargs);
  plt.scatter(curve[:,0], curve[:,1], c = 'k', s = 40);

### Initialize Worm from Image
npoints = 21;
nobs = npoints*2-2; #d differences from the sides / head tail only once  
worm = wmod.WormModel(npoints = npoints);
nparameter = worm.nparameter; #full number of parameter
nobs = worm.ndistances;

t0 = 500000;
threshold_factor = 0.9;
absolute_threshold = None;

img = exp.load_img(wid = 80, t = t0);  
imgs = filters.gaussian_filter(np.asarray(img, float), 1.0); 
worm.from_image(img, absolute_threshold = absolute_threshold, threshold_factor = threshold_factor, verbose = False);

success, center, left, right, width = wgeo.shape_from_image(img, 
                             absolute_threshold = absolute_threshold,
                             threshold_factor = threshold_factor, npoints = npoints, 
                             verbose = False, save = False);

plt.figure(1); plt.clf(); 
#worm.plot(image = imgs)
worm.plot(image = imgs)
plot(center, 'r');


### Compare Worm to Contour

cntrs = wgeo.contour_from_image(img, sigma = 1, absolute_threshold = absolute_threshold, threshold_factor = threshold_factor, 
                                verbose = False, save = None);
cntr = resample_curve(cntrs[0], 100); #assume outer contour is first, only match to this
contour = Curve(cntr, nparameter = 50);
contour.plot(with_points=False);


plt.figure(2); plt.clf();
plt.subplot(1,2,1)
dist = worm.distance_to_contour(contour, verbose = True);
worm.plot(color = 'k')
plt.subplot(1,2,2);
plt.plot(dist[:20])
plt.plot(dist[20:])


worm0 = worm.copy();


worm = worm0.copy();
worm.move_forward(0.4);

plt.figure(2); plt.clf(); 
worm0.plot(color = 'r');
worm.plot(color = 'b')
plt.axis('equal')





### Special Images

t = 529202;

t = t + 1;
img = exp.load_img(wid = 80, t= t);
#img = exp.load_img(wid = 80, t= 529204);
#img = exp.load_img(wid = 80, t= 529218);

plt.figure(3); plt.clf();
sh = worm.from_image(img, verbose = True);
plt.subplot(2,3,1)
worm.plot(color = 'k')

