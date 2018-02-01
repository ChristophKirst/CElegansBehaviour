# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 00:20:26 2018

@author: ckirst
"""

import cv2
import numpy as np

import scripts.process_movie_plot as pmp

#%%

#f_id = 50;
#gray = blur[f_id].copy();
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

fid = 61;
gray = blur[fid];

# find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.05)
#dst = cv2.dilate(dst,None)
dst = dst > 200
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
centroids = centroids[1:];


# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

# Now draw them
res = np.hstack((centroids,corners))
res = corners;
res = np.int0(res)

img = blur[fid].copy();
img[res[:,1],res[:,0]]=255
#img[res[:,3],res[:,2]] = 255

pmp.plot(img)
