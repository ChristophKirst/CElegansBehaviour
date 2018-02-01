# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 00:43:17 2018

@author: ckirst
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


#%%
#t = t+1;
gray = np.array(blur[t], dtype = 'float32')

corners = cv2.goodFeaturesToTrack(gray,5,0.01,5)
corners = np.int0(corners)

for j,i in enumerate(corners):
    x,y = i.ravel()
    cv2.circle(gray,(x,y),1, 200 + j * 10,-1)

plt.figure(25); plt.clf();
plt.imshow(gray, interpolation = 'none'); plt.show()


#%%
#gray = np.array(blur[t], dtype = 'float32')
evs = cv2.cornerEigenValsAndVecs(gray, blockSize = 3, ksize = 3)

plt.figure(26); plt.clf();
for i in range(6):
  plt.subplot(2,3,i+1);
  plt.imshow(evs[:,:,i], interpolation ='none');
  


for c in corners[:,0]:
  print evs[c[0], c[1], 0:2];

#%%
evs[corners[:,0,0], corners[:,0,1], 0:2]

#%%  
plt.figure(25);
for i in corners[[1]]:
    x,y = i.ravel()
    cv2.circle(gray,(x,y),1,100,-1)
plt.imshow(gray, interpolation = 'none');


#%%



t = 10;
img = np.asarray(blur[t], dtype = 'float32');
gray = img; #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = img.shape[:2]

eigen = cv2.cornerEigenValsAndVecs(gray, 15, 3)
eigen = eigen.reshape(h, w, 3, 2)  # [[e1, e2], v1, v2]
flow = eigen[:,:,2]

vis = img.copy()
vis[:] = (192 + np.uint32(vis)) / 2
d = 3
points =  np.dstack( np.mgrid[d/2:w:d, d/2:h:d] ).reshape(-1, 2)
for x, y in points:
   vx, vy = np.int32(flow[y, x]*d)
   cv2.line(vis, (x-vx, y-vy), (x+vx, y+vy), (0, 0, 0), 1, 4)

plt.figure(17);
plt.subplot(1,2,1);
plt.imshow(img)
plt.subplot(1,2,2);
plt.imshow(vis)
