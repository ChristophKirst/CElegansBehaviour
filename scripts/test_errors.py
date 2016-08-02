# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 17:46:04 2016

@author: ckirst
"""

import numpy as np
import matplotlib.pyplot as plt;

import copy

import imageprocessing.active_worm as aw;

import experiment as exp
import scipy.ndimage.filters as filters

from scripts.analyse_wormshape import analyse_shape
from skimage.filters import threshold_otsu

### Test phi error - move


reload(aw);

# load image
img = exp.load_img(wid = 80, t= 513466);
imgs = filters.gaussian_filter(np.asarray(img, float), 1.0);
#imgs[imgs < 60] = 60;
imgs = imgs - imgs.max();

threshold_level = 0.95;

level = threshold_level * threshold_otsu(imgs);

mask = imgs < level;
phi = aw.mask2phi(mask);


ws = aw.WormModel();
ws.from_image(imgs);
ws.widen(0.75);

phiw = ws.phi();
maskw = ws.mask();

plt.figure(1); plt.clf();
plt.subplot(2,3,1)
plt.imshow(imgs);
plt.subplot(2,3,2)
plt.imshow(mask)
plt.subplot(2,3,3)
plt.imshow(np.tanh(phi/2))
plt.subplot(2,3,4)
plt.imshow(maskw);
plt.subplot(2,3,5)
plt.imshow(maskw-mask)
plt.subplot(2,3,6)
epsilon = 5.0;
plt.imshow(np.tanh(phiw/epsilon)-np.tanh(phi/epsilon))



def test_error(func, vmin, vmax, n, nfig = 101):
  vals = np.linspace(vmin, vmax, n);
  errors = np.zeros(vals.shape[0]);

  plt.figure(nfig); plt.clf();
  ax = plt.subplot(1,3,1);
  plt.imshow(imgs);
  for i,v in enumerate(vals):
    ws2 = copy.deepcopy(ws);
    func(ws2,v);
    #errors[i] = ws2.error_center_line(image = imgs, npoints = 2*ws2.npoints+1, overlap = True );
    errors[i] = ws2.error_contour(phi, epsilon = 2.0);
    ws2.plot(image = None, ax = ax);

  plt.subplot(1,3,2)
  plt.plot(errors)

  plt.subplot(1,3,3);
  ws2 = copy.deepcopy(ws);
  func(ws2,vals[errors.argmin()]);
  ws2.plot(image = imgs);
  return ws2


f = lambda w,x: w.widen(x);
ws2 = test_error(f, 0.9, 1.75, 21)


phiw = ws2.phi();
maskw = ws2.mask();

plt.figure(1); plt.clf();
plt.subplot(2,3,1)
plt.imshow(imgs);
plt.subplot(2,3,2)
plt.imshow(mask)
plt.subplot(2,3,3)
plt.imshow(phi)
plt.subplot(2,3,4)
plt.imshow(maskw);
plt.subplot(2,3,5)
plt.imshow(maskw-mask)
plt.subplot(2,3,6)
plt.imshow(phiw-phi)





f = lambda w,x: w.move_forward(x, straight = True);
test_error(f, -10, 10, 21)



### Test center line error - bend
reload(aw);

f = lambda w,x: w.bend(x, head = True);
test_error(f, -0.75, 0.75, 21)



### Test center line error - curve

f = lambda w,x: w.curve([0, x]);
test_error(f, -2, 2, 21)


