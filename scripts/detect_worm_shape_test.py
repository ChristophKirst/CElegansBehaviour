# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 23:13:51 2018

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
import scripts.process_movie_plot as pmp

#% check worm data

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


#%

turn = np.load('/home/ckirst/Science/Projects/CElegans/Experiment/Movies/Tests/turn_000.npy')
#dv.plot(turn)

blur = np.zeros(turn.shape);
for i in range(turn.shape[0]):
  blur[i] = cv2.GaussianBlur(turn[i], ksize = (5,5), sigmaX = 0);

#dv.plot(blur)


#%%

f_id = 19;

img, contours, hierarchy  = cv2.findContours((blur[f_id] < 75).view('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#img, contours, hierarchy  = cv2.findContours((blur[f_id] < 68).view('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#img, contours, hierarchy  = cv2.findContours((blur[f_id] < 65).view('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

img = blur[f_id].copy();
cv2.drawContours(img, contours, -1, 100, 1);

pmp.plot(img)


#%
# detect worm shape

n_contours = len(contours);

#%%

f_id = 80

import worm.geometry as wgeo
reload(wgeo);

plt.clf()
res = wgeo.shape_from_image(blur[f_id], threshold_factor = 0.975, sigma =  None, verbose = True, smooth = 15, smooth_center = 2.5, npoints = 45, ncontour = 80, center_offset = 3)


#%%

import worm.model as wmod

wm = wmod.WormModel(npoints = 21)

f_id = 5;
plt.figure(1); plt.clf();
wm.from_image(blur[f_id], sigma = None, verbose = True, smooth = 10)

plt.figure(19); plt.clf();
plt.subplot(1,2,1);
wm.plot(blur[f_id])
plt.subplot(1,2,2);
plt.plot(wm.width);
plt.draw();


#%%
import worm.geometry as wgeo
cnt = wgeo.detect_contour(blur[f_id], 75)


#%%
smooth = 20;

pts = cnt[0][:-1];

nextra = 20;
ptsa = np.vstack([pts[-nextra:], pts, pts[:nextra+1]]);

cinterp, u = wgeo.splprep(ptsa.T, u = None, s = smooth, per = 0, k = 4) 

u0 = u[nextra];
u1 = u[-nextra-2];

#u0 = 0;
#u1 = 1;
ncontour = 100;
us = np.linspace(u0, u1, ncontour)
#us = u;
x, y = wgeo.splev(us, cinterp, der = 0)

dx, dy = wgeo.splev(us, cinterp, der = 1)
d2x, d2y = wgeo.splev(us, cinterp, der = 2)
k = (dx * d2y - dy * d2x)/np.power(dx**2 + dy**2, 1.5);
kk = np.hstack([k[-nextra:], k, k[:nextra]]);



plt.figure(20); plt.clf();
plt.subplot(1,2,1);
plt.imshow(blur[f_id]);
plt.plot(x,y);


plt.subplot(3,2,2);
plt.plot(dx);
plt.plot(dy);
plt.subplot(3,2,4);
plt.plot(d2x);
plt.plot(d2y);
plt.subplot(3,2,6);
plt.plot(kk)



#%%

x,y = ptsa.T
dx = x[1:]-x[:-1];
dy = y[1:]-y[:-1];
dt = np.sqrt(dx*dx + dy*dy);
t = np.hstack([[0], np.cumsum(dt)]);t


#%%

from scipy.interpolate import UnivariateSpline


pts = cnt[0][:-1];

nextra = 20;
kk = 2 * nextra;;
ptsa = np.vstack([pts[-kk:], pts, pts[:kk]]);
x,y = ptsa.T
dx = x[1:]-x[:-1];
dy = y[1:]-y[:-1];
dt = np.sqrt(dx*dx + dy*dy);
t = np.hstack([[0], np.cumsum(dt)]);

fx = UnivariateSpline(t, x, k=4)
fy = UnivariateSpline(t, y, k=4)


ta = np.linspace(t[nextra], t[-nextra], ncontour);
#ta = np.linspace(0, t[-1], ncontour);



dx = fx.derivative(1)(ta)
ddx = fx.derivative(2)(ta)
dy = fy.derivative(1)(ta)
ddy = fy.derivative(2)(ta)
curvature = (dx * ddy - dy * ddx) / np.power(dx*dx + dy*dy, 3 / 2)

plt.figure(50); plt.clf();
plt.plot(curvature)


#%%
plt.figure(80); plt.plot(cnt[0][:,0], cnt[0][:,1]); plt.gca().xlim = (0,151); plt.ylim = (0,151)
plt.axes().set_aspect('equal')
plt.draw()



#%%


from skimage.morphology import skeletonize

binr = blur[f_id] < 75;
skel = skeletonize(binr);


pmp.plot([binr, np.asarray(binr, dtype = int) + skel])

plt.plot(res[1][:,0], res[1][:,1])


