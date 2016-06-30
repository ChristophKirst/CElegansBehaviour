# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 17:00:11 2016

@author: ckirst
"""

import numpy as np

import experiment as exp
import scipy.ndimage.filters as filters
from skimage.filters import threshold_otsu

# load image
img = exp.load_img(wid = 80, t= 529204);
img = exp.load_img(wid = 80, t= 529218);

imgs = filters.gaussian_filter(np.asarray(img, float), 1.0);

threshold_level = 0.95;

level = threshold_level * threshold_otsu(imgs)


import numpy as np
from numpy import ma

import matplotlib as mpl
import matplotlib._contour as _contour

_mask = None;
_corner_mask = None;

_corner_mask = mpl.rcParams['contour.corner_mask']

nchunk = 0;

z = ma.asarray(imgs, dtype=np.float64); 

Ny, Nx = z.shape;
x, y= np.meshgrid(np.arange(Nx), np.arange(Ny));

contour_generator = _contour.QuadContourGenerator(x, y, z.filled(), _mask, _corner_mask, nchunk)




vertices = contour_generator.create_contour(level - 20)


import matplotlib.pyplot as plt

plt.figure(100); plt.clf();
plt.imshow(imgs);

for j in range(len(vertices)):
  plt.scatter(vertices[j][:,0], vertices[j][:,1])


