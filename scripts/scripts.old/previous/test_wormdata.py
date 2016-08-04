# -*- coding: utf-8 -*-
"""
Created on Sun May 22 23:50:13 2016

@author: ckirst
"""

import numpy as np
import matplotlib.pyplot as plt
import wormdata as wd


#make face path
#random motion
n1 = 400;
n2 = 400;
n = n1+n2;
path = np.zeros((n,2));
path[:n1,:] = np.random.rand(n1,2);
path[n1:n,:] = 0.1 * np.random.rand(n1,2) + np.outer(np.linspace(0,100, n2),[0, 1]);

w = wd.WormData(path, stage = np.zeros(n), label = ('x', 'y'), wid = 1);
w.replaceInvalid();
plt.figure(1); plt.clf();
w.plotTrajectory(size = 50);


nn = 20;
diffs = w.calculateDiffusionParameter(n=nn, parallel = False)

plt.figure(2); plt.clf();
diffsshift = np.zeros_like(diffs);
diffsshift[nn/2:, :] = diffs[:-nn/2,:];
diffsshift[:nn/2. :] = np.NaN;
w.plotTrajectory(colordata = diffsshift[:,0], size = diffsshift[:,-1]* 1000)

plt.figure(4); plt.clf();

for i in range(diffs.shape[1]):
  plt.subplot(diffs.shape[1], 1, i+1)
  plt.plot(diffs[:,i])
  