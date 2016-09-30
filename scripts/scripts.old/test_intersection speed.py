# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 21:29:58 2016

@author: ckirst
"""

# test line segments deparated by nans
import numpy as np
import matplotlib.pyplot as plt;
import interpolation.intersections as ii;
reload(ii)

s = 2 * np.pi* np.linspace(0,1,150);
xy1 = np.vstack([np.cos(s), np.sin(2*s)]).T;
#xy2 = np.vstack([0.5 * s, 0.2* s]).T + [-2,-0.5];
#xy2[40:80,:] = np.nan;
xy2 = np.array([[-2,-0.5],[-0.1, -0.1], [np.nan,np.nan], [0.1, 0.1], [2, 0.5]]);
xy2l = np.array(list(xy2)*10);


from utils.timer import timeit

@timeit
def i1():
  for i in range(10):
    xy0,i,j,di,dj = ii.curve_intersections_discrete(xy1, xy2);

@timeit
def i2():
  xy0,i,j,di,dj = ii.curve_intersections_discrete(xy1, xy2l);


i1();
i2();



xy0,i,j,di,dj = ii.curve_intersections_discrete(xy1, xy2);

plt.figure(1); plt.clf();
plt.plot(xy1[:,0], xy1[:,1]);
plt.plot(xy2[:,0], xy2[:,1]);
plt.scatter(xy0[:,0], xy0[:,1], c = 'm', s = 40);
plt.axis('equal')
  