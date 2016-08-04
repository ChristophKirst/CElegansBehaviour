# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:16:00 2016

@author: ckirst
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

import skimage.feature as ftr


import experiment as exp;

img = exp.load_img(t = 529204)

#img = exp.load_img(t = 500017)

t = 0;


t = t + 1;
img = exp.load_img(t = 529204 + t)
#img = cv2.medianBlur(img,5);
img = cv2.GaussianBlur(img, (5,5), 1);


#dx = cv2.Sobel(img, sobel_x, 1, 0, 3);
#dy = cv2.Sobel(img, sobel_y, 0, 1, 3);


from scipy import ndimage

sx = ndimage.sobel(img.astype(float), axis=0, mode='reflect')
sy = ndimage.sobel(img.astype(float), axis=1, mode='reflect')
dst = np.hypot(sx, sy)


#dst = cv2.cornerHarris(img,6,3,0.1)
#dst = ftr.corner_shi_tomasi(img);
#e = 20;
#dst = dst[e:-e];
#dst = dst[:, e:-e];

#dst = ftr.corner_moravec(img);
#dst = cv2.Laplacian(img,cv2.CV_64F)
#dst = cv2.Canny(img, 0, 50)


#result is dilated for marking the corners, not important
#dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
#img[dst>0.01*dst.max()]=[0,0,255]

#cv2.imshow('dst',img)
plt.figure(1); plt.clf();
plt.subplot(1,2,1)
plt.imshow(img);
plt.subplot(1,2,2)
plt.imshow(dst)



#if cv2.waitKey(0) & 0xff == 27:
#    cv2.destroyAllWindows()



hess= ftr.hessian_matrix(img)

plt.figure(100); plt.clf();
for i in range(len(hess)):
  plt.subplot(1,len(hess), i +1);
  plt.imshow(hess[i]);
  
  
  
import imageprocessing.active_worm as aw;

img = exp.load_img(t = 500000)
img = cv2.GaussianBlur(img, (5,5), 1);

wm = aw.WormModel();
wm.from_image(img);
phi = wm.phi();
phi = phi.astype(float);
  
import scipy.ndimage as nd 
sx = nd.sobel(img.astype(float), axis=0, mode='reflect');
sy = nd.sobel(img.astype(float), axis=1, mode='reflect');
grad = np.hypot(sx, sy);

reg = 10;

plt.figure(15); plt.clf();
plt.subplot(1,3,1);
plt.imshow(grad);
plt.subplot(1,3,2);
plt.imshow(np.exp(-np.abs(phi) / reg))
plt.subplot(1,3,3);
plt.imshow(np.exp(-np.abs(phi) / reg) * grad)
      