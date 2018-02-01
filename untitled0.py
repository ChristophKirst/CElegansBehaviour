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

movie_name = 'CAM207_2017-01-30-172321'
#movie_name = 'CAM800_2017-01-30-171215'
#movie_name = 'CAM807_2017-01-30-171140'
#movie_name = 'CAM819_2017-01-30-172249'

region_id = 4;

data_dir = '/home/ckirst/Data/Science/Projects/CElegans/Experiment/Movies/'
data_dir = '/home/ckirst/Movies/'

data_name = '%s_%s_%s.npy' % (movie_name, '%d', '%s');

data_image_file  = os.path.join(data_dir, data_name % (region_id, 'images'));
data_info_file   = os.path.join(data_dir, data_name % (region_id, 'info'));
data_meta_file   = os.path.join(data_dir, data_name % (region_id, 'meta'));

#%%

data_image = np.lib.format.open_memmap(data_image_file, mode = 'r');
data_info  = np.load(data_info_file);
data_meta  = np.load(data_meta_file);

#%%

failed_id = np.where(data_info['failed'])[0]
print('failed: %d' % len(failed_id));


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

