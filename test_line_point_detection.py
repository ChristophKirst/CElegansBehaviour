# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:25:51 2018

@author: ckirst
"""

import numpy as np

import cv2

import experiment.data as exd

import matplotlib.pyplot as plt

import visualization.plot as cplt

import ClearMap.GUI.DataViewer as dv
import ClearMap.IO as io

#%%

fid = 355000;
fid = 408757;
#fid = 170000;
#fid = 620000;
fid = 530000;
#fid = 529306+10;
#fid = 529306+60;
fid = 518073;
fid = 170000;


raw =  exd.image_data(frame = fid, smooth = None, dtype = 'float32');
gray = exd.image_data(frame = fid, smooth = (5,5), dtype = 'float32');

gray_s = 255*(gray -gray.min())/(gray.max()-gray.min());
#cplt.plot([gray_s], fig = 1)

sigma = 1.5;

import imageprocessing.ridge_detection as rd;
reload(rd);

ridges,normals,points,evals,evecs = rd.detect_ridges(gray_s, sigma, lower = 5.0, return_info = True);

#cplt.plot([gray_s, ridges, gray_s], fig = 10);
#rd.plot(ridges, normals, points, image = gray_s);

import worm.geometry_new as wgn;

plt.clf();
res = wgn.shape_from_image(gray, npoints = 21, ncontour = 300, absolute_threshold=132, smooth_center= 0.0, smooth_left_right=0.0, smooth_head_tail=7.5, sigma = None, verbose = True, center_offset = 0)

plt.subplot(2,3,5);
rd.plot(ridges, normals, points, image = gray_s);


#%%


res = wgn.detect_contour(gray, level = 140)

for r in res:
  plt.plot(*r.T, c = 'm');


#%%

io.writeData('/home/ckirst/test4.tif', gray);


#%% Gaussian kernels for derivatives 

sigma = 1.5;

import imageprocessing.ridge_detection as rd;
reload(rd);

n,p,evals,evecs = rd.ridge_points(gray_s, sigma);
cplt.plot([gray, evals[:,:,0], evals[:,:,1], evals[:,:,0]> 0], fig = 3)
cplt.plot([gray, n[:,:,0], n[:,:,1]], fig = 4)




#%%
reload(rd);

sigma = 3;

plt.figure(3); plt.clf();
ax = plt.subplot(2,3,1);
plt.imshow(gray, interpolation = 'none')
for i,dxy in enumerate([[1,0], [0,1], [2,0], [0,2], [1,1]]):
  plt.subplot(2,3,2+i, sharex = ax, sharey = ax);
  plt.imshow(rd.gaussian_derivative(gray,sigma, dxy[0], dxy[1]), interpolation = 'none')
  plt.title('x=%d, y=%d' % tuple(dxy)); 
  
  

#%%

bil = cv2.bilateralFilter(gray, 20, 3, 3);
th = 138;

cplt.plot([gray, bil, raw, gray > th, bil > th, raw > th], fig = 2)
