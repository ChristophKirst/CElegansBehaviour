# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 01:18:39 2016

@author: ckirst
"""

#%%

import worm.geometry as wgeo

reload(wgeo)


import interpolation.resampling as ir

threshold = 75;
cntrs = wgeo.contours_from_image(imgt, sigma = 1, absolute_threshold = threshold, verbose = False, save = None);
nc = len(cntrs);

contour_size = 100;
contour_inner = 20;
if nc == 1:
  cntrs = (ir.resample(cntrs[0], contour_size),);
else:
  cntrs = (ir.resample(cntrs[0], contour_size-contour_inner), ir.resample(cntrs[1], contour_inner))

#calculate normals
nrmls = [wgeo.normals_from_contour_discrete(c) for c in cntrs];

plt.figure(3); plt.clf();
#plt.imshow(img)
for cc,nn in zip(cntrs, nrmls):
  for c,n in zip(cc[:-1], nn):
    cn = c + 3 * n;
    plt.plot([c[0], cn[0]], [c[1], cn[1]], c = 'k')
  plt.scatter(cc[:,0], cc[:,1], c= 'r');
plt.axis('equal')