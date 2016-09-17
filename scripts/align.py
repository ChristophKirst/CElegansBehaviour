# -*- coding: utf-8 -*-
"""
Worm raw image data alignment and compression module
"""

import numpy as np
import matplotlib.pyplot as plt


import analysis.experiment as exp
import scipy.ndimage.filters as filters

t0 = 250000
img1 = exp.load_img(wid = 80, t= t0);
img1 = filters.gaussian_filter(np.asarray(img1, float), 1.0);

img2 = exp.load_img(wid = 80, t= t0+1);
img2 = filters.gaussian_filter(np.asarray(img2, float), 1.0);


plt.figure(1); plt.clf();
plt.subplot(3,2,1);
plt.imshow(img1);
plt.subplot(3,2,2);
plt.imshow(img2);
plt.subplot(3,2,3);
plt.hist(img1.flatten(),bins = 256)
plt.subplot(3,2,4);
plt.hist(img2.flatten(), bins = 256)

def preprocess(img, threshold = 90):
  imgt = img.copy();
  imgt[imgt > threshold] = threshold;
  return threshold - imgt;
  
img1t = preprocess(img1);
img2t = preprocess(img2);

th = 90;
plt.subplot(3,2,5);
plt.imshow(img1t)
plt.subplot(3,2,6);
plt.imshow(img2t)

### center of mass
def center_of_mass(img):
  size= img.shape;
  x = np.linspace(-(size[0]-1) * 0.5, (size[0]-1) * 0.5, size[0]);
  y = np.linspace(-(size[1]-1) * 0.5, (size[1]-1) * 0.5, size[1]);
  xx,yy = np.meshgrid(x,y);
  xx = xx * img; 
  yy = yy * img;
  n = size[0] * size[1];
  return np.array([1.0 * xx.sum() / n, 1.0 * yy.sum() / n]);



plt.figure(7); plt.clf();
plt.subplot(1,3,1)
plt.imshow(img1t);
plt.subplot(1,3,2);
plt.imshow(img2t);
plt.subplot(1,3,3)
plt.imshow(img1t - img2t)

## what is the original position:

pos = exp.load(wid = 80)
xy10 = pos[t0];
xy20 = pos[t0+1]

xy1 = center_of_mass(img1t)
xy2 = center_of_mass(img2t)

xy10 - xy1
xy20 - xy2



def align_images(img1, img2):
  
  
def 
