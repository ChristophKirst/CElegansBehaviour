# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:12:24 2016

@author: ckirst
"""


#import time
import numpy as np
import matplotlib.pyplot as plt

import experiment as exp

import scipy.ndimage.morphology as morph
from skimage.filters import threshold_otsu
import scipy.ndimage.filters as filters

from scipy.interpolate import splprep, splev

from imageprocessing.active_contour import chanvese

import cv2

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





















### Other ways -> active contours etc
#
#mask = np.zeros(img.shape)
#mask[20:100, 20:100] = 1
#
##res = chanvese(img, mask, max_its=1000, display=True, alpha=.5, thresh= 1)
#
#
#res2 = chanvese(img, res[0], max_its=1000, display=True, alpha=.1, thresh= 0.4)
#
#plt.figure(19); plt.clf(); 
#plt.subplot(1,3,1)
#plt.imshow(res2[0])
#plt.subplot(1,3,2)
#plt.imshow(img < threshold_otsu(img))
#
#### basic image processing
#
### thresholding
#
#plt.subplot(1,3,3)
#plt.imshow(imgo )
#
## get the contour
#plt.figure(20); plt.clf();
#cs = plt.contour(np.asarray(imgo, dtype=float), levels = [0.5]);
#
#
#
#
#bw = imgo > threshold_otsu(imgo);
#
#plt.figure(11); plt.clf();
#plt.subplot(1,3,1);
#plt.imshow(imgo, cmap = 'gray', clim = (0,256),  interpolation="nearest");
#plt.subplot(1,3,2);
#plt.hist(img.flatten(), bins = 256);
#plt.subplot(1,3,3)
#plt.imshow(bw)
#
#
#
#### open cv 
#imgo = morph.binary_opening(img < threshold_otsu(img));
#imgo = 10 * np.asanyarray(imgo, dtype = 'uint8')
#
#im2, contours, hierarchy = cv2.findContours(imgo,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
#
#plt.figure(12); plt.clf();
#plt.subplot(1,2,1)
#plt.imshow(img)
#mask = np.zeros(img.shape)
#mask[contours[0][:,0][:,1], contours[0][:,0][:,0]] = 100;
#plt.subplot(1,2,2)
#plt.imshow(mask)





### active conotours new
#import numpy as np
#import matplotlib.pyplot as plt
#from skimage.color import rgb2gray
#from skimage import data
#from skimage.filters import gaussian
#from skimage.segmentation import active_contour
#
#
#img = data.astronaut()
#img = rgb2gray(img)
#
#s = np.linspace(0, 2*np.pi, 400)
#x = 220 + 100*np.cos(s)
#y = 100 + 100*np.sin(s)
#init = np.array([x, y]).T
#
#
#snake = active_contour(gaussian(img, 3),
#                       init, alpha=0.015, beta=10, gamma=0.001)
#
#fig = plt.figure(figsize=(7, 7))
#ax = fig.add_subplot(111)
#plt.gray()
#ax.imshow(img)
#ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
#ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
#ax.set_xticks([]), ax.set_yticks([])
#ax.axis([0, img.shape[1], img.shape[0], 0])
#
#
#
#
#
#
#img = data.text()
#
#x = np.linspace(5, 424, 100)
#y = np.linspace(136, 50, 100)
#init = np.array([x, y]).T
#
#snake = active_contour(gaussian(img, 1), init, bc='fixed',
#                       alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)
#
#fig = plt.figure(figsize=(9, 5))
#ax = fig.add_subplot(111)
#plt.gray()
#ax.imshow(img)
#ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
#ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
#ax.set_xticks([]), ax.set_yticks([])
#ax.axis([0, img.shape[1], img.shape[0], 0])
#
#plt.show()
