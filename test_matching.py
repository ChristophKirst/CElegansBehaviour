# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:42:13 2018

@author: ckirst
"""


import os
import shutil
import glob
import natsort

import numpy as np

import matplotlib.pyplot as plt

import cv2

import ClearMap.GUI.DataViewer as dv

import scripts.process_movie_util as pmu

import worm.geometry as wgeo
reload(wgeo)

#%% check worm data

movie_dir = '/run/media/ckirst/My Book/'

#movie_name = 'CAM207_2017-01-30-172321'
movie_name = 'CAM800_2017-01-30-171215'
#movie_name = 'CAM807_2017-01-30-171140'
#movie_name = 'CAM819_2017-01-30-172249'

region_id = 0;

data_dir = '/home/ckirst/Data/Science/Projects/CElegans/Experiment/Movies/'
data_dir = '/home/ckirst/Movies/'

data_name = '%s_%s_%s.npy' % (movie_name, '%d', '%s');

data_image_file  = os.path.join(data_dir, data_name % (region_id, 'images'));
data_info_file   = os.path.join(data_dir, data_name % (region_id, 'info'));
data_meta_file   = os.path.join(data_dir, data_name % (region_id, 'meta'));



turn_file = os.path.join(movie_dir, 'Tests\turns_000.npy');

#%%

data_image = np.lib.format.open_memmap(data_image_file, mode = 'r');
data_info  = np.load(data_info_file);
data_meta  = np.load(data_meta_file);



#%%
import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 0

i = 400000;

img1 = cv2.GaussianBlur(np.asarray(data_image[i], dtype = float), ksize = (3,3), sigmaX = 0);
img2 = cv2.GaussianBlur(np.asarray(data_image[i+1], dtype = float), ksize = (3,3), sigmaX = 0);

img1 = np.asarray(img1, dtype = 'uint8');
img2 = np.asarray(img2, dtype = 'uint8');

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance or True:
        good.append(m)


#%%
if len(good)>MIN_MATCH_COUNT:
  src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
  dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
  
  M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
  matchesMask = mask.ravel().tolist()
  
  h,w = img1.shape
  pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
  dst = cv2.perspectiveTransform(pts,M)
  
  img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
  #print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
  matchesMask = None
    

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

plt.figure(1); plt.clf();
plt.imshow(img3, 'gray'),plt.show()