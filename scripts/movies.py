# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:23:59 2016

@author: ckirst
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import experiment as exp


plt.figure(1); plt.clf();
img = exp.load_img(t=100000);
plt.imshow(img, cmap = 'gray')

#animate movie  
fig, ax = plt.subplots(1,2)

img = exp.load_img(t=100000)[30:120, 30:120];
img2 =exp.load_img(t=100000+1)[30:120, 30:120];
figimg = ax[0].imshow(img, cmap = 'gray', clim = (0,256), interpolation="nearest");
delta = np.asarray(img2, dtype = float) - np.asarray(img, dtype = float)
figimg2 = ax[1].imshow(delta, cmap = 'jet', clim = (-30,60), interpolation="nearest");
plt.show();

for t in range(200000, 300000):
  img = img2;
  img2 = exp.load_img(t=t+1)[30:120, 30:120];
  delta = np.asarray(img2, dtype = float) - np.asarray(img, dtype = float)
  figimg.set_data(img);
  figimg2.set_data(delta);
  
  ax[0].set_title('t=%d' % t);
  time.sleep(0.0001)
  fig.canvas.draw()
  fig.canvas.flush_events()
  
  
  
#animate movie  
fig, ax = plt.subplots(1,3)

wid = 96;

img = exp.load_img(t=100000, wid = wid)[20:130, 20:130];
img2 =exp.load_img(t=100000+1, wid = wid)[20:130, 20:130];
figimg = ax[0].imshow(img, cmap = 'gray', clim = (0,256),  interpolation="nearest");
delta = np.asarray(img2, dtype = float) - np.asarray(img, dtype = float)
figimg2 = ax[1].imshow(delta, cmap = 'jet', clim = (-30,60),  interpolation="nearest");
plt.show();

xy = exp.load(wid = wid);

t0 = 507000;
#line, = ax[2].plot([], [], lw=2, color = 'b', marker = 'o');


for t in range(t0, t0 + 15000, 2):
  img = img2;
  img2 = exp.load_img(t=t+1, wid = wid)[20:130, 20:130];
  delta = np.asarray(img2, dtype = float) - np.asarray(img, dtype = float)
  figimg.set_data(img);
  figimg2.set_data(delta);
  #line.set_data(xy[t0:t,0], xy[t0:t,1]);

  #ax[2].plot(xy[t-1:t,0], xy[t-1:t,1], color = 'gray');    
  ax[2].scatter(xy[t,0], -xy[t,1]);  
  
  ax[0].set_title('t=%d' % t);
  time.sleep(0.0001)
  fig.canvas.draw()
  fig.canvas.flush_events()