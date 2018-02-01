# -*- coding: utf-8 -*-
"""
Kalman filter on head an tail movements to predict future head and tails

@author: ckirst
"""


#%% sample data


import os

import numpy as np

import imageio as iio

import matplotlib.pyplot as plt

import cv2

import ClearMap.GUI.DataViewer as dv


import scripts.process_movie_plot as pmp;
import scripts.process_movie_util as pmu;

reload(pmp); reload(pmu);

import worm.geometry as wgeo
reload(wgeo);

import worm.model as wmod
reload(wmod);

#%% Movie files

movie_dir = '/run/media/ckirst/My Book/'

#movie_name = 'CAM207_2017-01-30-172321'
movie_name = 'CAM800_2017-01-30-171215'
#movie_name = 'CAM807_2017-01-30-171140'
#movie_name = 'CAM819_2017-01-30-172249'

data_dir = '/home/ckirst/Data/Science/Projects/CElegans/Experiment/Movies/'
data_dir = '/home/ckirst/Movies'

data_name = '%s_%s_%s.npy' % (movie_name, '%d', '%s');

region_id = 0;

data_image_file  = os.path.join(data_dir, data_name % (region_id, 'images'));
data_info_file   = os.path.join(data_dir, data_name % (region_id, 'info'));
data_meta_file   = os.path.join(data_dir, data_name % (region_id, 'meta'));

data_shape_file = os.path.join(data_dir, data_name % (region_id, 'shapes'));

data_shape_info_file = os.path.join(data_dir, data_name % (region_id, 'shapes_info'));

#%%

data_images = np.lib.format.open_memmap(data_image_file, mode = 'r+');
data_shapes = np.lib.format.open_memmap(data_shape_file, mode = 'r+');
data_shape_info = np.load(data_shape_info_file);


#%%

roi = [slice(652319, 652637)]; # turn
#roi = [slice(586000, 600000)]; # no shape detection failure

images = data_images[roi];
shapes = data_shapes[roi];
shape_info = data_shape_info[roi];
nt = shapes.shape[0];

#%%


blur = np.zeros(images.shape);
for i in range(nt):
  blur[i] = cv2.GaussianBlur(images[i], ksize = (5,5), sigmaX = 0);
  
dv.plot(blur);



#%% head tail trajectories

head_tails = [];
for i in range(nt):
  center = shapes[i][0:2].T;
  head_tails.append(center[[0,-1]]);
head_tails = np.array(head_tails);
  
#%%
  
plt.figure(1); plt.clf();
plt.scatter(*(head_tails[:,0,:].T), color = 'red')
plt.scatter(*(head_tails[:,1,:].T), color = 'blue')


#%%

plt.figure(2); plt.clf();
ax = plt.subplot(1,1,1);
plt.tight_layout();

for t in range(10): #nt):
  wm = wmod.WormModel(center = shapes[t][0:2].T, width = shapes[t][2]);
  wm.plot(image = blur[t], ax = ax);
  plt.draw(); plt.pause(0.01);


