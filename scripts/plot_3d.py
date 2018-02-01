# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 02:07:23 2018


3d plotting volumetric data

@author: ckirst
"""
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

#%% Movie files

movie_dir = '/run/media/ckirst/My Book/'

#movie_name = 'CAM207_2017-01-30-172321'
movie_name = 'CAM800_2017-01-30-171215'
#movie_name = 'CAM807_2017-01-30-171140'
#movie_name = 'CAM819_2017-01-30-172249'

data_dir = '/home/ckirst/Data/Science/Projects/CElegans/Experiment/Movies/'
#data_dir = '/home/ckirst/Movies'

data_name = '%s_%s_%s.npy' % (movie_name, '%d', '%s');

region_id = 0;


data_image_file  = os.path.join(data_dir, data_name % (region_id, 'images'));
data_info_file   = os.path.join(data_dir, data_name % (region_id, 'info'));
data_meta_file   = os.path.join(data_dir, data_name % (region_id, 'meta'));

data_shape_file = os.path.join(data_dir, data_name % (region_id, 'shapes'));



#%%

turn = np.load(os.path.join(data_dir, 'Tests/turn_000.npy'))

#dv.plot(turn)
data = -turn;
blur = np.zeros(data.shape);

for i in range(data.shape[0]):
  blur[i] = cv2.GaussianBlur(data[i], ksize = (5,5), sigmaX = 0);

dv.plot(blur>185)

#%%

data0 = blur * (blur>185);
for i in range(data0.shape[0]):
  data0[i][data0[i] > 0] = i +1;

#%%

import vispy as vp;
#from vispy import app, visuals, scene

# build visuals
GraphPlot3D = vp.scene.visuals.create_visual_node(vp.visuals.VolumeVisual)

# build canvas
canvas = vp.scene.SceneCanvas(keys='interactive', title='plot3d', show=True)

# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 0
view.camera.distance = 100
view.camera.elevation = 0
view.camera.azimuth = 0

cc = (np.array(data.shape) // 2);
cc = cc[[2,1,0]]
view.camera.center = cc

#color = colors;

gp = GraphPlot3D(data0, method = 'mip', relative_step_size=1.0, parent=view.scene, cmap = 'hot', threshold = 1, clim = (1.0, 100.0))


#%%

vp.color.get_colormaps()

cm = vp.color.get_colormap('hot')
#cm = vp.color.get_colormap('fire')
cm = vp.color.get_colormap('RdYeBuCy');
my_colors = cm.map(np.linspace(0,1,10));
#my_colors = [[0,0,0,0], [1,1,1,0], [1,0,0,1]];
my_colors[:,-1] = 0.75;
my_colors = np.vstack([np.array([[0,0,0,0]]*1), my_colors])
cma = vp.color.Colormap(my_colors, controls = np.hstack([[0.0], np.linspace(1.0/(len(my_colors)-1), 1.0, len(my_colors)-1)]), interpolation='linear');

gp.cmap = cma;

#%%


import vispy as vp;

cms =  vp.color.get_colormaps()
for k in vp.color.get_colormaps():
  print k
  try:
    gp.cmap = cms[k];
    gp.update();
  except:
    print('some error');
 
   
  try:
      input("Press enter to continue")
  except SyntaxError:
      pass


