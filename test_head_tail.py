# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:15:17 2018

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

data_shape_file = os.path.join(data_dir, data_name % (region_id, 'shapes'));
data_contour_file = os.path.join(data_dir, data_name % (region_id, 'contours'));
data_shape_info_file = os.path.join(data_dir, data_name % (region_id, 'shapes_info'));

#%%

data_image = np.lib.format.open_memmap(data_image_file, mode = 'r');
data_info  = np.load(data_info_file);
data_meta  = np.load(data_meta_file);
data_shape = np.load(data_shape_file);

#%%

failed_id = np.where(data_info['failed'])[0]
print('failed: %d' % len(failed_id));





#%%


s = data_shape_info['success']

#%%

fids = range(500000, 500000+300);


ht = np.array([data_shape[f][:2,[0,-1]].T for f in fids])


plt.figure(16); plt.clf();
plt.scatter(ht[:,0,0], ht[:,0,1], c = 'r')
plt.scatter(ht[:,1,0], ht[:,1,1], c = 'b')

#%%

d = data_image[fids]

dv.plot(d.transpose([1,2,0]));


#%% draw contours on images and thenplot via dv

fids = [9];
fids = range(520000+400, 520000+800);
n = len(fids);
dc = np.zeros((n, data_image.shape[1],  data_image.shape[2]));
for i,f in enumerate(fids):
  img = np.asarray(data_image[f].copy(), dtype = 'float32');
  img = cv2.GaussianBlur(img, ksize = (3,3), sigmaX = 0);
  cnt = data_shape[f][:2].T;
  cnt = np.asarray(np.reshape(cnt, (cnt.shape[0], 1, cnt.shape[1])), dtype = int);
  dc[i] = cv2.polylines(img, [cnt], False, 255, 1);
    

#%%

v = dv.plot(dc.transpose([1,2,0]));


#%% head tial trajectories by distance


h = [data_shape[fids[0]][:2, 0].T];
t = [data_shape[fids[0]][:2, -1].T];

for f in fids[1:]:
  ht = data_shape[f][:2, [0,-1]].T;

  dist = np.linalg.norm(ht-h[-1], axis = 1)
  imin = np.argmin(dist);

  h.append(ht[imin]);
  t.append(ht[(imin+1)%2]);

h = np.array(h);
t = np.array(t);

plt.figure(8);plt.clf();

plt.scatter(*h.T, c = 'w');
plt.scatter(*t.T, c = 'r');


#%% plot onto iomages

n = len(fids);
hi = np.asarray(h, dtype = int);
ti = np.asarray(t, dtype = int);
dht = np.zeros((n, data_image.shape[1],  data_image.shape[2]));
for i,f in enumerate(fids):
  img = np.asarray(data_image[f].copy(), dtype = 'float');
  img[hi[i,1], hi[i,0]] = 255;
  img[ti[i,1], ti[i,0]] = 0;
  dht[i] = img;

dv.plot(dht.transpose([1,2,0]))


#%% differences in shapes

s_prev = data_shape[fids[0]][:2].T;

shape_dist = [];
for f in fids[1:]:
  s = data_shape[f][:2].T;
  shape_dist.append(np.linalg.norm(s_prev-s, axis = 1).sum());
  s_prev = s;
  
plt.figure(18); plt.clf();
plt.plot(shape_dist)


#%% plot full trajectories








#%%

movie_files = pmu.get_movie_files(os.path.join(movie_dir, movie_name));
n_frames = pmu.get_n_frames(movie_files)


#%%

pmu.get_movie_from_frame(movie_files, failed_id[0], n_frames=n_frames)

#%% 

plt.figure(1); plt.clf();

plt.imshow(data_image[263332-37443])

#%%


dv.plot(data_image.transpose([2,1,0]))


#%% curled
263332-37520



#%% turns

i = 337714;
i = 109715;

turn = data_image[i-75:i+150];

dv.plot(turn)

#%%

np.save(os.path.join(data_dir, 'Tests/turn_001.npy'), turn)


#%%


turn = np.load(os.path.join(data_dir, 'Tests/turn_000.npy'))

