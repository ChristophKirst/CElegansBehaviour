# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:44:37 2018

@author: ckirst
"""

#%% t-sne embedding of images 

import os
import shutil
import glob
import natsort

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

data_image = np.lib.format.open_memmap(data_image_file, mode = 'r');
data_info  = np.load(data_info_file);
data_meta  = np.load(data_meta_file);

#%%


ids = range(170000, 220000, 5);
w = 15;
dim = 200;
project = np.random.rand((2*w+1)*(2*w+1), dim);
#project = (project / np.sum(project, axis = 0));
project = project - np.mean(project, axis = 0);

#%%
features = np.zeros((len(ids), dim));
for j,i in enumerate(ids):
  d = cv2.GaussianBlur(np.asarray(data_image[i], dtype = 'float'), ksize= (3,3), sigmaX = 0);
  dz = d[75-w:75+w+1, 75-w:75+w+1].flatten();
  dz[dz < 135] = 135;
  dz[dz > 145] = 145;
  features[j] = np.dot(dz, project);


features = (features - np.mean(features, axis = 0));
features = (features / np.std(features, axis = 0));

print features.shape


#%%
 
dv.plot(features) 

#%%
import sklearn.manifold as sl;
n_components = 2;
metric = 'euclidean';
tsne = sl.TSNE(n_components=n_components, init = 'pca', random_state = 10, metric = metric)
Y = tsne.fit_transform(features)

#%%


fig = plt.figure(17);
plt.clf();
plt.scatter(Y[:,0], Y[:,1], c = range(len(Y[:,0])), cmap = plt.cm.Spectral);
plt.title("t-SNE")
plt.tight_layout();

ax = plt.gca();

#for k in ids[::len(ids)/20]:
for k in range(0, len(ids), len(ids)/500):
  d = cv2.GaussianBlur(np.asarray(data_image[ids[k]], dtype = 'float'), ksize= (3,3), sigmaX = 0);
  dz = d[75-w:75+w+1, 75-w:75+w+1];
  dz[dz < 135] = 135;
  ax.imshow(dz, extent = (Y[k,0], Y[k,0] + 5, Y[k,1], Y[k,1] + 5));
  
  
    
ax.set_xlim(np.min(Y[:,0]),np.max(Y[:,0])+5)
ax.set_ylim(np.min(Y[:,1]),np.max(Y[:,1])+5)

plt.draw();


#%%
d1 = data_image[ids[1333]][75-w:75+w+1, 75-w:75+w+1];
d2 = data_image[ids[1931]][75-w:75+w+1, 75-w:75+w+1];


d2 =  data_image[ids[2156]]
d2 =  data_image[ids[2205]]


d1.sum()
d2.sum()




dv.plot(d1)
dv.plot(d2)
