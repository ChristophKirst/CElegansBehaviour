# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 20:28:03 2016

@author: ckirst
"""

import experiment.data as exp

try:
  from importlib import reload
except:
  pass

import numpy as np

import matplotlib.pyplot as plt
import ClearMap.GUI.DataViewer as dv

import worm.geometry as wgeo

import worm.geometry_new as wgn
reload(wgn);


import scripts.vgg as vgg
import tensorflow as tf


#%% Test data shapes

fid = 200000;
di = exp.image_data(frame = fid, smooth = True)
dc = exp.contour_data(frame = fid);

plt.figure(1); plt.clf();
plt.imshow(di, cmap = 'gray');
plt.plot(dc[0,:], dc[1,:])


#%%
def make_matrix(content):
  _,sx,sy,si = content.shape;
  sii = int(np.ceil(np.sqrt(si)));
  mat = np.zeros((sii * sx, sii * sy));
  for i in range(sii):
    for j in range(sii):
      k = i * sii + j;
      if k < si:
        ix = i * sx;
        jy = j * sy;
        mat[ix:(ix+sx), jy:(jy+sy)] = content[0,:,:,k];
  return mat;

def to_color(img):
  imgc = img[:,:,np.newaxis];
  imgc = np.concatenate([imgc]*3, axis = 2);
  return imgc;

#%% Setup Deep Network / Tensorflow

img = exp.image_data(frame = 0, smooth = True);
imgc = to_color(img);

batch_size = 100;
shape = (batch_size,) + imgc.shape;
image = tf.placeholder('float', shape=shape);
net, mean_pixel = vgg.net('/home/ckirst/Science/Projects/CElegans/Analysis/MachineVision/imagenet-vgg-verydeep-19', image);

g = tf.Graph()
sess = tf.Session();

#%%

fids = range(300000, 400000, 1000)[:batch_size];
img = [exp.image_data(frame = f, smooth = True) for f in fids];
imgc = [to_color(i) for i in img];

img_pre = np.array([vgg.preprocess(i, mean_pixel) for i in imgc])
features = net['relu5_4'].eval(feed_dict={image: img_pre}, session = sess)
features_flat = np.reshape(features, (batch_size, -1));

centers =[exp.shape_data(frame = f)[:2,:] for f in fids];
thetas = np.array([wgeo.theta_from_center_discrete(c.T)[0] for c in centers]);


#%% Regression

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

model = linear_model.LinearRegression()

model.fit(features_flat, thetas)


#%% Test 

fids_test = range(300500, 400500, 1000)[:batch_size];
img = [exp.image_data(frame = f, smooth = True) for f in fids_test];
imgc = [to_color(i) for i in img];

img_pre = np.array([vgg.preprocess(i, mean_pixel) for i in imgc])
features = net['relu5_4'].eval(feed_dict={image: img_pre}, session = sess)
features_flat = np.reshape(features, (batch_size, -1));

centers =[exp.shape_data(frame = f)[:2,:] for f in fids_test];
thetas = np.array([wgeo.theta_from_center_discrete(c.T)[0] for c in centers]);


thetas_predict =  model.predict(features_flat);


#%%

plt.figure(6); plt.clf();
for t,p in zip(thetas, thetas_predict):
  plt.plot(t,p)


#%%

plt.figure(7); plt.clf();
for t in thetas[:8]:
  plt.plot(t)

#%%

def arrange_plots(n):
  m = int(np.ceil(np.sqrt(n)));
  k = int(np.ceil(1.0*n/m));
  return m,k;

def plot_data(f, fig = 10):
  plt.figure(fig); plt.clf();
  if not hasattr(f, '__len__'):
    f = [f];
  n = len(f);
  m,k = arrange_plots(n);
  
  for j, fi in enumerate(f):
    plt.subplot(m,k,j+1);
    img = exp.image_data(frame = fi, smooth = True);
    center = exp.shape_data(frame = fi);  
    plt.imshow(img, cmap = 'gray');
    plt.plot(center[0,:], center[1,:], c = 'r');
    plt.xlim(0,151); plt.ylim(0, 151); 
    plt.title('%d' % fi);
  
  plt.tight_layout();
  

plot_data(fids_test[:12])

#%%

#%%


layer =  'relu4_2'
layers = ['relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4', 
                  'relu4_1', 'relu4_2', 'relu4_3',  'relu4_4', 'relu5_1',  'relu5_2', 'relu5_3', 'relu5_4'];

img_pre = np.array([vgg.preprocess(imgc, mean_pixel)])

features = {}

g = tf.Graph()
sess = tf.Session();
for i,layer in enumerate(layers):
  features[layer] = net[layer].eval(feed_dict={image: img_pre}, session = sess)

  mat = make_matrix(features[layer] )
  #fplt.plot_array(mat, color = 'viridis', title = layer)
  
  fig = plt.figure(i+100);
  plt.clf();
  plt.imshow(mat, cmap = 'pink', interpolation = 'none');
  plt.tight_layout();
  #plt.title('vgg layer %s' % layer)
  plt.axis('off')
  #fig.savefig('/home/ckirst/vgg_%s.png' % layer, facecolor = 'white')

fig = plt.figure(99);
plt.clf();
layer = 'input'
plt.imshow(img, cmap = 'pink', interpolation = 'none');
plt.tight_layout();
#plt.title('vgg layer %s' % layer)
plt.axis('off')
#fig.savefig('/home/ckirst/vgg_%s.png' % layer, facecolor = 'white')


plt.figure(1); plt.clf();
plt.imshow(img)

#%%

