# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 00:23:54 2016

@author: ckirst
"""

# image processing pipeline adn worm shape

import matplotlib.pyplot as plt
import analysis.experiment as exp
import analysis.plot as fplt

import worm.model as wmod
import worm.geometry as wgeo

t0 = 401700;
img = exp.load_img(t= t0, sigma = None)


fig = plt.figure(1); plt.clf();
plt.imshow(img, cmap = 'gray', interpolation = 'none')
plt.axis('off')

fig.savefig('pipeline_input.png', facecolor = 'white')

#smooth

fig = plt.figure(2); plt.clf();
img = exp.load_img(t= t0, sigma=1.0)
plt.imshow(img, cmap = 'gray', interpolation = 'none')
plt.axis('off')

fig.savefig('pipeline_sigma.png', facecolor = 'white')

#contour


cts = wgeo.contours_from_image(img, verbose= False)

img = exp.load_img(t= t0, sigma=1.0)

fig = plt.figure(3); plt.clf();
plt.imshow(img, cmap = 'gray', interpolation = 'none')
for c in cts:
  plt.plot(c[:,0], c[:,1])
plt.xlim(0, 151); plt.ylim(0,151)
plt.axis('off')

fig.savefig('pipeline_contour.png', facecolor = 'white')


#curvature
import numpy as np
from skimage.filters import threshold_otsu
from scipy.interpolate import splev, splprep

threshold_factor = 0.95
level = threshold_factor * threshold_otsu(img);
pts = wgeo.detect_contour(img, level)[0];

ncontour = 100
smooth = 1.0
cinterp, u = splprep(pts.T, u = None, s = smooth, per = 1) 
us = np.linspace(u.min(), u.max(), ncontour)
x, y = splev(us[:-1], cinterp, der = 0)

dx, dy = splev(us[:-1], cinterp, der = 1)
d2x, d2y = splev(us[:-1], cinterp, der = 2)
k = (dx * d2y - dy * d2x)/np.power(dx**2 + dy**2, 1.5);

from signalprocessing.peak_detection import find_peaks
nextra = 20;
delta = 0.3
kk = -k;
kk = np.hstack([kk[-nextra:], kk, kk[:nextra]]);
peaks = find_peaks(kk, delta = delta);

if peaks.shape[0] > 0:
  peaks[:,0] -= nextra;
  peaks = peaks[peaks[:,0] < k.shape[0],:];
  peaks = peaks[peaks[:,0] >= 0,:];

imax = np.sort(np.asarray(peaks[np.argsort(peaks[:,1])[-2:],0], dtype = int))

fig = plt.figure(4); plt.clf();
plt.plot(k, 'k')
plt.scatter(imax, k[imax], c = 'gray', s= 400);
plt.scatter(peaks[:,0], -peaks[:,1], c = 'red', s= 100);
fig.savefig('pipeline_curvature_peak.png', facecolor = 'white')

### side lines

npoints = 22;
u1 = np.linspace(us[imax[0]], us[imax[1]], npoints)
x1, y1 =  splev(u1, cinterp, der = 0);
left = np.vstack([x1,y1]).T;

u2 = np.linspace(us[imax[0]], us[imax[1]]-1, npoints);
u2 = np.mod(u2,1);
x2, y2 = splev(u2, cinterp, der = 0);
right = np.vstack([x2,y2]).T;

fig = plt.figure(5); plt.clf();
plt.imshow(img, cmap = 'gray');
plt.plot(left[:,0], left[:,1], c = 'r', linewidth = 1.5);
plt.plot(right[:,0], right[:,1], c = 'g', linewidth = 1.5);
plt.xlim(0,151); plt.ylim(0,151)
plt.axis('off')
fig.savefig('pipeline_left_right.png', facecolor = 'white')


s,center,l,r,width = wgeo.shape_from_image(img)

fig = plt.figure(6); plt.clf();
plt.imshow(img, cmap = 'gray');
plt.plot(left[:,0], left[:,1], c = 'r', linewidth = 1.5);
plt.plot(right[:,0], right[:,1], c = 'g', linewidth = 1.5);
plt.plot(center[:,0], center[:,1], c = 'b', linewidth = 2);
plt.xlim(0,151); plt.ylim(0,151)
plt.axis('off')
fig.savefig('pipeline_center.png', facecolor = 'white')


# theta angles , width profile etc


theta = wgeo.theta_from_center(center)
theta = theta[0]

fig = plt.figure(7); plt.clf();
plt.plot(theta, c= 'r', linewidth = 1)
fig.savefig('pipeline_theta.png', facecolor = 'white')


fig = plt.figure(8); plt.clf();
plt.plot(width, c= 'b', linewidth = 1)
fig.savefig('pipeline_width.png', facecolor = 'white')



# fail for self intersections

t0 =  500000 + 25620 +3;
img = exp.load_img(t=t0)
fig = plt.figure(8); plt.clf();
plt.imshow(img, cmap = 'gray');
plt.xlim(0,151); plt.ylim(0,151)
plt.axis('off')
fig.savefig('pipeline_fail.png', facecolor = 'white')


t0 =  500000 + 25620 -5;
img = exp.load_img(t=t0, sigma = 1.0)
fig = plt.figure(8); plt.clf();
plt.imshow(img, cmap = 'gray');
plt.xlim(0,151); plt.ylim(0,151)
plt.axis('off')


w = wmod.WormModel(npoints = 22)
w.from_image(img, verbose = True)

fig = plt.figure(9); plt.clf();
w.plot(image = img, cmap = 'gray', ccolor='b');

plt.xlim(0,151); plt.ylim(0,151)
plt.axis('off')
fig.savefig('pipeline_fail_0.png', facecolor = 'white')


plt.clf();
t1 = t0 + 5
img = exp.load_img(t=t1, sigma = 1.0)
w1 = w.copy();
w1.bend(9)

w1.move_forward(0.06)
#w1.plot(image = img, cmap = 'gray', ccolor='b');

w1.bend(-2,head=False)
w1.center += [0,2]
w1.plot(image = img, cmap = 'gray', ccolor='b');
# make 'movie' 

plt.xlim(0,151); plt.ylim(0,151)
plt.axis('off')
fig.savefig('pipeline_fail_1.png', facecolor = 'white')




plt.clf();
t1 = t0 + 10
img = exp.load_img(t=t1, sigma = 1.0)
w1 = w.copy();
w1.bend(1)

w1.move_forward(0.12)
#w1.plot(image = img, cmap = 'gray', ccolor='b');

w1.bend(-1,head=False)
w1.center += [0,2]
w1.plot(image = img, cmap = 'gray', ccolor='b');
# make 'movie' 

plt.xlim(0,151); plt.ylim(0,151)
plt.axis('off')
fig.savefig('pipeline_fail_2.png', facecolor = 'white')


plt.clf();
t1 = t0 + 12
img = exp.load_img(t=t1, sigma = 1.0)
w2 = w.copy();
w2.from_image(img)
w2.move_forward(0.25)
w2.bend(-12, exponent=3)
#w1.plot(image = img, cmap = 'gray', ccolor='b');

#w1.bend(-1,head=False)
#w1.center += [0,2]
w2.plot(image = img, cmap = 'gray', ccolor='b');
# make 'movie' 

plt.xlim(0,151); plt.ylim(0,151)
plt.axis('off')
fig.savefig('pipeline_fail_3.png', facecolor = 'white')


plt.clf();
t1 = t0 + 22
img = exp.load_img(t=t1, sigma = 1.0)
w2 = w.copy();
w2.from_image(img)
#w2.move_forward(0.25)
#w2.bend(-12, exponent=3)
#w1.plot(image = img, cmap = 'gray', ccolor='b');

#w1.bend(-1,head=False)
#w1.center += [0,2]
w2.plot(image = img, cmap = 'gray', ccolor='b');
# make 'movie' 

plt.xlim(0,151); plt.ylim(0,151)
plt.axis('off')
fig.savefig('pipeline_fail_4.png', facecolor = 'white')







