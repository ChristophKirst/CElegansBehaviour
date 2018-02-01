# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 17:25:31 2017

@author: ckirst
"""


import numpy as np
import matplotlib.pyplot as plt;

import copy
import time

import worm.model as wm;
import worm.geometry as wg;




#%% load data

t0 = 513478;
t0 = 524273 - 3;

import analysis.experiment as exp
img = exp.load_img(wid = 80, t= t0);

plt.imshow(img)



#%% preprocess - smmoth

import scipy.ndimage.filters as filters

imgs =  filters.gaussian_filter(np.asarray(img, float), sigma = 1.0);

plt.figure(2); plt.clf();
plt.imshow(imgs)


#%% gradient

import cv2
imgl = cv2.Laplacian(imgs,cv2.CV_64F)

import skimage.filters.rank as rkf
from skimage.morphology import disk

imgn = imgs.copy();
imgn[imgn < 50] = 50;
imgn[imgn > 100] = 100;
imgn = np.array((imgn -50) / 50 * 255, dtype = 'uint16');

imgg = rkf.gradient(imgn, disk(1))


plt.figure(3); plt.clf();
for i,j in enumerate([imgn, imgg]):
  plt.subplot(1,2,i+1);
  plt.imshow(j)



#%% process images

import imageprocessing.contours as cts;

cont = cts.detect_contour(imgs, level = 77 );


from scipy.interpolate import splev, splprep, UnivariateSpline

contn = [];
for i,c in enumerate(cont):
  ncontour = 100; smooth = 1.0;
  cinterp, u = splprep(c.T, u = None, s = 1.0, per = 1) 
  us = np.linspace(u.min(), u.max(), ncontour)
  x, y = splev(us, cinterp, der = 0);
  contn.append(np.vstack([x,y]).T);


plt.figure(3); plt.clf();

for i,j in enumerate([img, imgg]):
  plt.subplot(1,2,1+i)
  plt.imshow(j);
  for c in contn:
    plt.plot(c[:,0], c[:,1], 'r')


cout = contn[0];
cin = contn[1];


#%% optimze via active contours

#import skimage.segmentation as seg

#contopt = seg.active_contour(img, cout, alpha = 0.0, beta = 0.01, w_line = -0.01, w_edge = 10, gamma = 0.1, bc = 'periodic',  max_px_move=1.0, max_iterations=2500)
#
#plt.figure(7); plt.clf();
#plt.imshow(img);
#plt.plot(contopt[:,0], contopt[:,1], 'r');
#plt.plot(cout[:,0], cout[:,1], 'b');




#%% head tail detection

reload(wg)

plt.figure(5); plt.clf()
ht, idx = wg.head_tail_from_contour_discrete(cout, with_index = True, verbose = True, image = imgs);



#%% reparametrize contour from head

i = idx[0];

cout0 = np.vstack([cout[i:], cout[1:(i+1)]]);

k = wg.curvature_from_contour(cout0);

plt.figure(7); plt.clf();
plt.subplot(1,2,1);
plt.imshow(imgs);
plt.plot(cout0[:,0], cout0[:,1], 'r');
plt.subplot(1,2,2)
plt.plot(k)


#%%

#generate shape from segmentable image

reload(wg)
import analysis.experiment as exp;
img = exp.load_img(wid = 80, t = 524273 - 1);

plt.figure(1); plt.clf();

m = wm.WormModel(npoints = 31)
m.from_image(img, verbose = True);


l,r = m.shape();

plt.subplot(2,3,5)
plt.plot(l[:10,0], l[:10,1], 'w')


#%%

img2 = exp.load_img(wid = 80, t = 524273 );

plt.figure(2); plt.clf();
cont = wg.contours_from_image(img2, verbose = True)

cout = cont[0];
if len(cont) > 1:
  cin  = cont[1];
else:
  cin = [];
  

plt.figure(5); plt.clf()
ht, idx = wg.head_tail_from_contour_discrete(cout, with_index = True, verbose = True, image = img2);

i = idx[0];

cout0 = np.vstack([cout[i:], cout[1:(i+1)]]);

#%%

# distance between left and left contour and right and right contour

# resample contour

ncont = cout0.shape[0];
ncont2 = int(ncont/2);

cout0l = wg.resample_curve(cout0[:ncont2], npoints = 31);
cout0r = wg.resample_curve(cout0[-1:ncont2:-1], npoints = 31);

nrml = np.linalg.norm(cout0r - l, axis = 1)
nrmr = np.linalg.norm(cout0r - r, axis = 1)

plt.figure(7); plt.clf();

plt.subplot(1,2,1);
plt.plot(nrmr, 'r');
plt.plot(nrml, 'b')

plt.subplot(1,2,2);
plt.imshow(img2);
plt.plot(cout0r[:,0], cout0r[:,1], 'r')
plt.plot(cout0l[:,0], cout0l[:,1], 'b')

plt.plot(l[:,0], l[:,1], 'g');
plt.plot(r[:,0], r[:,1], 'y');


#%%


def warp_shape_to_contour(contour, shape):
  """given shape try to wrap it onto a detected contour, assumes head at shape[0] and tail at shape[n/2]"""
  
  # match the head and tails
  
  
  
  
  




#%% optimize worm shape profile toimage given an intial estimate





  