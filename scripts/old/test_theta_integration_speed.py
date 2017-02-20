# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:21:18 2016

@author: ckirst
"""

## test speed of creating center line from theta, splnie vs discrete versions

import numpy as np
import matplotlib.pyplot as plt
import interpolation.curve as curve;
import interpolation.spline as spline;

import worm.geometry as wgeo;

from utils.timer import timeit;

reload(curve); reload(spline);

#make the data

npoints = 21;
theta = np.linspace(0,1,npoints);
theta = np.sin(theta);

c = spline.Spline(theta);


@timeit
def d():
  return wgeo.center_from_theta(theta);

@timeit
def s():
  return curve.theta_to_curve(c);
  

c1 = d();
c2 = s();


#-> discrete processin 10x faster

plt.figure(1); plt.clf();
plt.plot(c1[:,0], c1[:,1]);
c2.plot();
