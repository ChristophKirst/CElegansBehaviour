# -*- coding: utf-8 -*-
"""
worm shape detection module

Provides shape quantification routines for the analysis of C. elegans movies
"""

#import time
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
import scipy.ndimage.filters as filters

from scipy.interpolate import splprep, splev


import experiment as exp
from analysis import Analysis


class WormShapeAnalysis(Analysis):
  
  def __init__(self, name = "WormShape", function = None, parameter = None, with_default_parameter = True):
    self.name = name;
    self.function = function;
    self.parameter = parameter;
    self.with_default_parameter = with_default_parameter;
  
  def tag(self, **kwargs):
    """Tag for this analysis"""
    if isinstance(name, basestring):
      tag = name;
    else:
      tag = '';
      
    tagfunc = tag_from_function(function = self.function,  with_default_parameter = self.with_default_parameter, parameter = self.parameter, **kwargs);
    
    return tag_join(tag, tagfunc);

     
  def run(self, data = None):
    """Run the analysis"""




img = exp.load_img(t=408000);

#plt.figure(10); plt.clf();
#plt.imshow(img, cmap = 'gray', clim = (0,256),  interpolation="nearest");
#plt.figure(11); plt.clf();
#plt.hist(img.flatten(), bins = 256);


## smooth image -> contour from plt

imgs = filters.gaussian_filter(np.asarray(img, float), 1);

plt.figure(12); plt.clf();
plt.subplot(1,3,1)
plt.imshow(img, interpolation ='nearest')
plt.subplot(1,3,2)
plt.imshow(imgs)
plt.subplot(1,3,3)
plt.imshow(img, cmap = 'gray')
cs = plt.contour(imgs, levels = [0.95 * threshold_otsu(imgs)])
plt.show()

v = cs.collections[0].get_paths()[0].vertices

# calculate spline and curvature

plt.figure(18); plt.clf();
plt.subplot(1,3,1)
plt.scatter(v[:,0], -v[:,1])


# gaussian smoothing of points ??

# interpolation
cinterp, u = splprep(v.T, u = None, s = 1.0, per = 1) 
us = np.linspace(u.min(), u.max(), 100)
x, y = splev(us[:-1], cinterp, der = 0)

plt.subplot(1,3,2)
plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
plt.plot(x, y, 'b--')
plt.scatter(x, y, s = 1)
plt.show()

# curvature along the points
dx, dy = splev(us[:-1], cinterp, der = 1)
d2x, d2y = splev(us[:-1], cinterp, der = 2)
k = (dx * d2y - dy * d2x)/np.power(dx**2 + dy**2, 1.5);


# find tail and head via peak detection

from signalprocessing.peak_detection import find_peaks

peaks = find_peaks(np.abs(k), delta = 0.15);

imax = np.sort(np.asarray(peaks[np.argsort(peaks[:,1])[-2:],0], dtype = int))

print 'max k at %s' % str(imax)

plt.scatter(x[imax], y[imax], s=150, color='r')


plt.subplot(1,3,3)
plt.plot(np.abs(k))



# calcualte paths along both sides

nsteps = 50;

u1 = np.linspace(us[imax[0]], us[imax[1]], nsteps)
x1, y1 =  splev(u1, cinterp, der = 0);

u2 = np.linspace(us[imax[0]], us[imax[1]]-1, nsteps);
u2 = np.mod(u2,1);
x2, y2 = splev(u2, cinterp, der = 0);

plt.subplot(1,3,2);
plt.plot(x1,y1, 'g', linewidth= 4)
plt.plot(x2,y2, 'y', linewidth= 4)

# midline (simple)

xm = (x1 + x2) / 2;
ym = (y1 + y2) / 2;
plt.plot(xm,ym,'b')

# plot some segments
for i in range(len(xm)):
    plt.plot([x1[i], x2[i]], [y1[i], y2[i]], 'm')

# worm center
xym = np.vstack([xm,ym]);
xymintp, u = splprep(xym, u = None, s = 1.0, per = 0);

xc, yc = splev([0.5], xymintp, der = 0)
plt.scatter(xc, yc, color = 'k')

