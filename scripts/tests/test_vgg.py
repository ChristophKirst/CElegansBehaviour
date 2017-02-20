# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 20:28:03 2016

@author: ckirst
"""

import scripts.vgg as vgg

import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np

import analysis.experiment as exp;




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
  
import scripts.plot as fplt;


img = exp.load_img(t = 500000, sigma=1.0);

# make color??
imgc = img[:,:,np.newaxis]
imgc = np.concatenate([imgc]*3, axis = 2)

shape = (1,) + imgc.shape;

image = tf.placeholder('float', shape=shape);


net, mean_pixel = vgg.net('/home/ckirst/Science/Projects/CElegansBehaviour/Analysis/imagenet-vgg-verydeep-19', image);


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
  fig.savefig('/home/ckirst/vgg_%s.png' % layer, facecolor = 'white')

fig = plt.figure(99);
plt.clf();
layer = 'input'
plt.imshow(img, cmap = 'pink', interpolation = 'none');
plt.tight_layout();
#plt.title('vgg layer %s' % layer)
plt.axis('off')
fig.savefig('/home/ckirst/vgg_%s.png' % layer, facecolor = 'white')


plt.figure(1); plt.clf();
plt.imshow(img)
