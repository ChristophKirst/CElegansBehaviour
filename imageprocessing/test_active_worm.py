# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 18:20:55 2016

@author: ckirst
"""

import numpy as np
import matplotlib.pyplot as plt;

import copy
import time

import imageprocessing.active_worm as aw;


### Test simple shape properties 
reload(aw)
#ws = aw.WormModel(theta = 0.1, width = 10 * (1 - np.exp(-0.1 * (21 - np.abs(2* (np.arange(20)+1) - 21)))));
ws = aw.WormModel(theta = 0.2, width = None, length = 80);
#ws.widen(0.5);

plt.figure(1); plt.clf();
ws.plot()
plt.xlim(0, 151);
plt.ylim(0, 151);


cl = ws.center_line()

plt.figure(1); plt.clf();
plt.scatter(cl[:,0], cl[:,1])


poly = ws.polygon()
plt.figure(2); plt.clf();
plt.scatter(poly[:,0], poly[:,1])


mask = ws.mask();
plt.figure(1); plt.clf();
plt.subplot(1,3,1)
plt.imshow(mask);

phi = ws.phi();
plt.subplot(1,3,2)
plt.imshow(phi);

plt.subplot(1,3,3);
plt.imshow(phi < 0)


### Test deformations
reload(aw);
ws = aw.WormModel(theta = 0.1, width = None);

def plot_worms(ws, ws2):
  plt.figure(1); plt.clf();
  ws.plot();
  ws2.plot(ccolor = 'magenta')


ws2 = copy.deepcopy(ws);
ws2.translate([10,15]);
plot_worms(ws, ws2);

ws2 = copy.deepcopy(ws);
ws2.rotate(0.5);
plot_worms(ws, ws2);

ws2 = copy.deepcopy(ws);
ws2.rotate(np.pi/2);
plot_worms(ws, ws2);

reload(aw);
ws = aw.WormModel(theta = 0.1, width = None, npoints = 21);

plt.figure(10); plt.clf();
ws.plot();
plt.xlim(55, 95); plt.ylim(55,95)

ws2 = copy.deepcopy(ws);
ws2.move_forward(-9, straight = True);
plot_worms(ws, ws2);


for s in np.linspace(0, 40, 60):
  print s
  ws2 = copy.deepcopy(ws);
  ws2.move_forward(s);
  plot_worms(ws, ws2);
  fig = plt.gcf();
  fig.canvas.draw()
  fig.canvas.flush_events()
  time.sleep(0.001)


ws2 = copy.deepcopy(ws);
ws2.curve([-1]);
plot_worms(ws, ws2);

ws2 = copy.deepcopy(ws);
ws2.bend(0.5, exponent  = 10, head = True);
plot_worms(ws, ws2);

ws2 = copy.deepcopy(ws);
ws2.bend(0.5, exponent  = 5, head = False);
plot_worms(ws, ws2);


### generate from image 
reload(aw);

import experiment as exp
import scipy.ndimage.filters as filters

# load image
img = exp.load_img(wid = 80, t= 500000);
imgs = filters.gaussian_filter(np.asarray(img, float), 1.0);

ws = aw.WormModel();
ws.from_image(imgs, verbose = True);

plt.figure(1); plt.clf();
ws.plot(image = imgs)


### self intersections

ws = aw.WormModel(theta = np.hstack([np.linspace(0.1, 0.8, 10), np.linspace(0.9, 0.1, 11)]) , length = 150);

mask = ws.mask();
phi = ws.phi();

plt.figure(1); plt.clf();
plt.subplot(1,2,1)
plt.imshow(mask);
plt.subplot(1,2,2)
plt.imshow(phi);
ws.plot()




### test error function
reload(aw);

import experiment as exp
import scipy.ndimage.filters as filters

# load image
img = exp.load_img(wid = 80, t= 500000);
imgs = filters.gaussian_filter(np.asarray(img, float), 1.0);

ws = aw.WormModel();
ws.from_image(img, verbose = True);

print ws.error(image = imgs)
plt.subplot(2,3,5)
ws.plot()
plt.subplot(2,3,2)
ws.plot()
plt.subplot(2,3,3)
ws.plot()


### test error function
reload(aw);

import experiment as exp
import scipy.ndimage.filters as filters

# load image
img = exp.load_img(wid = 80, t= 500000);
imgs = filters.gaussian_filter(np.asarray(img, float), 1.0);

ws = aw.WormModel();
ws.from_image(imgs);

shifts = np.linspace(-20, 20, 11);
errors = np.zeros(shifts.shape[0]);

fig = plt.figure(101); plt.clf();
ax = plt.subplot(1,2,1);
plt.imshow(imgs);
for i,s in enumerate(shifts):
  ws2 = copy.deepcopy(ws);
  ws2.move_forward(s, straight = True);
  errors[i] = ws2.error(image = imgs);
  ws2.plot(image = None, ax = ax);
plt.subplot(1,2,2)
plt.plot(errors)



# curvature

curvatures = np.linspace(-1.5, 1.5, 11);
errors = np.zeros(shifts.shape[0]);

fig = plt.figure(102); plt.clf();
plt.subplot(1,2,1);
plt.imshow(imgs);
for i,s in enumerate(curvatures):
  ws2 = copy.deepcopy(ws);
  ws2.curve([0, s]);
  errors[i] = ws2.error(image = imgs);
  ws2.plot(image = None, ax = fig.axes[0]);
plt.subplot(1,2,2)
plt.plot(errors)


# bends

reload(aw);

img2 = exp.load_img(wid = 80, t= 500000+17);
img2s = filters.gaussian_filter(np.asarray(img2, float), 1.0);

#img2s[img2s < 80] = 0;

ws = aw.WormModel();
ws.from_image(img2s);
ws.widen(0.4);

bends = np.linspace(-0.4, 0.4, 21);
errors = np.zeros(bends.shape[0]);

fig = plt.figure(103); plt.clf();
plt.subplot(1,2,1);
plt.imshow(img2s);
for i,s in enumerate(bends):
  ws2 = copy.deepcopy(ws);
  ws2.bend(s, exponent = 5, head = False);
  errors[i] = ws2.error(image = img2s, epsilon=0.5, border = 10000000, border_epsilon=10, out_vs_in=1);
  fig = plt.figure(103);
  plt.subplot(1,2,1);
  ws2.plot(image = None, ax = fig.axes[0]);
plt.subplot(1,2,2)
plt.plot(errors)


i =errors.argmin()
i = 10
ws2 = copy.deepcopy(ws);
ws2.bend(bends[i], exponent = 4, head = False);

fig = plt.figure(105+i);
plt.subplot(1,2,1)
ws2.plot(image = img2s);
plt.subplot(1,2,2);
plt.imshow(ws2.mask() * img2s)
np.sum(ws2.mask() * img2s)


### optimize worm coordinates
reload(aw); 
t0 = 513466;

img = exp.load_img(wid = 80, t= t0);
imgs = filters.gaussian_filter(np.asarray(img, float), 1.0);
 
img2 = exp.load_img(wid = 80, t= t0+3);
img2s = filters.gaussian_filter(np.asarray(img2, float), 1.0);

#img2s[img2s < 80] = 00;

ws = aw.WormModel();
ws.from_image(imgs, verbose = True);
ws.widen(0.6);

plt.figure(99); plt.clf();
ws.plot(image = imgs)


plt.figure(100); plt.clf();
ax = plt.subplot(1,2,1);
ws.plot(image = img2s, ax = ax)




### optimize movement
reload(aw);

ws = aw.WormModel();
ws.from_image(imgs, verbose = True);
ws.widen(0.6);

shifts = np.linspace(-20, 20, 21);
errors = np.zeros(shifts.shape[0]);

fig = plt.figure(101); plt.clf();
ax = plt.subplot(1,3,1);
plt.imshow(img2s);
for i,s in enumerate(shifts):
  ws2 = copy.deepcopy(ws);
  ws2.move_forward(s, straight = True);
  errors[i] = ws2.error(image = img2s, border = 1000000, epsilon = 15, border_epsilon=15);
  ws2.plot(image = None, ax = ax);
plt.subplot(1,3,2)
plt.plot(errors)

plt.subplot(1,3,3)
ws2 = copy.deepcopy(ws);
ws2.move_forward(shifts[errors.argmin()], straight = True);
ws2.plot(image = img2s);
















#ws2.optimize(image = img2s, options={'gtol': 1e-12, 'disp': True, 'eps': 0.00001}, method = 'CG', nmodes = 0)

#error = ws.error(imgs);
ws2 = copy.deepcopy(ws);
par= ws2.optimize(image = img2s*img2s*img2s*img2s, nmodes=3)

plt.figure(100); plt.clf();
ax = plt.subplot(1,2,1);
ws.plot(image = img2s, ax = ax)
ax = plt.subplot(1,2,2);
ws2.plot(image = img2s, ax = ax)

print par





# bends

bends = np.linspace(-0.5, 0.5, 11);
errors = np.zeros(bends.shape[0]);

fig = plt.figure(102); plt.clf();
plt.subplot(1,2,1);
plt.imshow(img2s);
for i,s in enumerate(bends):
  ws3 = copy.deepcopy(ws2);
  ws3.bend(s, exponent = 5, head = True);
  errors[i] = ws3.error(image = img2s) #, epsilon=2, border = None, border_epsilon=2, out_vs_in=1);
  ws3.plot(image = None, ax = fig.axes[0]);
plt.subplot(1,2,2)
plt.plot(errors)
























img3 = exp.load_img(wid = 80, t= t0+4);
img3s = filters.gaussian_filter(np.asarray(img3, float), 1.0);

ws3 = copy.deepcopy(ws2);
par= ws3.optimize(image = img3s, nmodes=1)


plt.figure(100); plt.clf();
ax = plt.subplot(1,2,1);
ws2.plot(image = img3s, ax = ax)
ax = plt.subplot(1,2,2);
ws3.plot(image = img3s, ax = ax)






#ws2.optimize(image = img2s, options={'gtol': 1e-9, 'norm': 20, 'disp': True, 'eps': 0.005}, method = 'BFGS', nmodes = 0)
#ws2.optimize(image = img2s, options={'ftol' : 0.01, 'gtol': 1e-4, 'xtol': 0.01, 'disp': True, 'eps': 0.1}, method = 'Nelder-Mead', nmodes = 0)
#ws2.optimize(image = img2s, method='TNC', options={'disp': True, 'minfev': 0.0 * error, 'scale': None, 'rescale': -1, 
#                                                   'offset': None, 'gtol': -1, 'eps': .01, 'eta': 0.01, 'maxiter': None, 
#                                                   'maxCGit': 0, 'ftol': -1, 'xtol': 1, 'stepmx': 2, 'accuracy': 0}, nmodes = 0)









t1 = 0;

t1 += 2;
img2 = exp.load_img(wid = 80, t= t0+2+t1);
img2s = filters.gaussian_filter(np.asarray(img2, float), 1.0);

ws3 = copy.deepcopy(ws2);
ws3.optimize(image = img2s)
                                              
plt.figure(100); plt.clf();
ax = plt.subplot(1,2,1);
ws2.plot(image = img2s, ax = ax)
ax = plt.subplot(1,2,2);
ws3.plot(image = img2s, ax = ax)

ws2 = copy.deepcopy(ws3);




plt.figure(100); plt.clf();
plt.subplot(1,2,1);
ws.plot(image = imgs)
plt.subplot(1,2,2);
ws2 = copy.deepcopy(ws);
ws2.bend(0.15, exponent = 10, front = True);
ws2.plot(image = img2s)

# optimize to new image

# load image
img2 = exp.load_img(wid = 80, t= 500000+2);
img2s = filters.gaussian_filter(np.asarray(img2, float), 1.0);

plt.figure(10); plt.clf();
plt.subplot(3,2,1);
plt.imshow(img)
plt.subplot(3,2,2);
plt.imshow(imgs)
plt.subplot(3,2,3);
plt.imshow(img2)
plt.subplot(3,2,4);
plt.imshow(img2s)
plt.subplot(3,2,5);
plt.imshow(img2 - img)
plt.subplot(3,2,6);
plt.imshow(img2s - imgs)


#compare errors

e1 = ws.error(image = imgs)
e2 = ws.error(image = img2s)
print e1,e2


reload(aw)

ws.error(image = img2s)

opt = ws.optimize(image = img2s, options={'gtol': 1e-6, 'disp': True, 'eps':.01})
par = opt['x'];
ws3 = aw.WormShape(nseg = 20, l = par[:ws.nseg], theta =  par[ws.nseg:2*ws.nseg], width = par[2*ws.nseg:3*ws.nseg], x0 = par[-2], y0 = par[-1]);

xyl, xyr, xym = ws3.sides();
plt.figure(4); plt.clf();
plt.imshow(img2s)
plt.scatter(xyl[:,0], xyl[:,1], c = 'green');
plt.scatter(xyr[:,0], xyr[:,1], c = 'red');
plt.scatter(xym[:,0], xym[:,1], c = 'black');