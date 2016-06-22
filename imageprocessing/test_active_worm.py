# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 18:20:55 2016

@author: ckirst
"""

import numpy as np
import matplotlib.pyplot as plt;
  
import imageprocessing.active_worm as aw;
reload(aw)


ws = aw.WormShape(theta = 0.1, width = 10 * (1 - np.exp(-0.1 * (21 - np.abs(2* (np.arange(20)+1) - 21)))));

ml = ws.midline()

plt.figure(1); plt.clf();
plt.scatter(ml[:,0], ml[:,1])

xyl, xyr, xym = ws.sides();
plt.figure(1); plt.clf();
plt.scatter(xyl[:,0], xyl[:,1], c = 'green');
plt.scatter(xyr[:,0], xyr[:,1], c = 'red');
plt.scatter(xym[:,0], xym[:,1], c = 'black');


poly = ws.polygon()
plt.figure(2); plt.clf();
plt.scatter(poly[:,0], poly[:,1])


mask = ws.mask();
plt.figure(1); plt.clf();
plt.imshow(mask);

phi = ws.phi();
plt.figure(1); plt.clf();
plt.imshow(phi);


### self intersections

ws = aw.WormShape(theta = np.hstack([np.linspace(0.1, 0.8, 10), np.linspace(0.9, 0.1, 11)]) , l = 4, width = 3 * (1 - np.exp(-0.1 * (21 - np.abs(2* (np.arange(20)+1) - 21)))));

ml = ws.midline()
plt.figure(1); plt.clf();
plt.plot(ml[:,0], ml[:,1])


mask = ws.mask();
plt.figure(1); plt.clf();
plt.imshow(mask);

phi = ws.phi();
plt.figure(1); plt.clf();
plt.imshow(phi);

