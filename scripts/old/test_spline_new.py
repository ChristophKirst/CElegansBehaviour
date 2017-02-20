# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 02:14:56 2016

@author: ckirst
"""

### Splines
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import interpolation.spline as sp;
reload(sp);

s = sp.Spline(nparameter = 10, degree = 3);  

s = sp.Spline(nparameter = 10, npoints = 141, degree = 3);  
x = np.sin(10* s.points) + np.sin(2 * s.points);
#s.set_values(x);
s.parameter
s.values
s.values = x;
s.parameter

p = s.parameter;
tck = splrep(s.points, x, t = s.knots, task = -1, k = s.degree);
xpp = splev(s.points, tck);
xp = s.get_values();

plt.figure(1); plt.clf();
plt.plot(x);
plt.plot(xp);
plt.plot(xpp);

plt.figure(2); plt.clf();
for i in range(s.nparameter):
  pp = np.zeros(s.nparameter);
  pp[i] = 1;
  xp = s.get_values(parameter = pp);
  plt.plot(s.points, xp);
plt.title('basis functions scipy')

# test spline basis

reload(sp);
s = sp.Spline(nparameter = 10, npoints = 141, degree = 3);

bm = s.projection_matrix();
plt.figure(3); plt.clf();
plt.plot(bm);
plt.title('basis functions matrix')


xb = bm.dot(p);
xp = s.get_values(parameter = p);

#xb = np.dot(np.invert(bm), pb);

plt.figure(4); plt.clf();
plt.plot(x);
#plt.plot(0.1* xb);
plt.plot(xb)
plt.plot(xp)

# invertible case of npoints = nparameter
s = sp.Spline(nparameter = 25, npoints = 25, degree = 3);
x = np.sin(10* s.points) + np.sin(2 * s.points);
s.values = x;
p = s.parameter;

bm = s.projection;
pb = np.linalg.inv(bm).dot(x);

plt.figure(5); plt.clf();
plt.plot(p);
plt.plot(pb);

xs = s(s.points, parameter = p);
xp = bm.dot(pb);

plt.figure(6); plt.clf();
plt.plot(x);
plt.plot(xs);
plt.plot(xp);

tck = s.tck();

from utils.timer import timeit
@timeit
def ts():
  return splev(np.linspace(0,1,s.npoints), tck);
@timeit
def tb():
  return s.projection.dot(p);
    
ps = ts();
pb = tb();
np.allclose(pb, ps)

# factor ~10 times faster with basis

# test shifting spline by a value s
s = sp.Spline(nparameter = 25, npoints = 25, degree = 3);
x = np.sin(10* s.points) + np.sin(2 * s.points);
s.set_values(x);
p = s.parameter;  
xnew = s(s.points + 0.1, p);
import copy
s2 = copy.deepcopy(s);
s2.shift(0.05, extrapolation=1);

plt.figure(7); plt.clf();
s.plot();
plt.plot(s.points, xnew);
s2.plot();


# test integration and derivatives
reload(sp)
points = np.linspace(0, 2*np.pi,50);
x = np.sin(points); # + np.sin(2 * points);
s = sp.Spline(values = x, points = points, nparameter = 20);
plt.figure(1); plt.clf();
s.plot();
d = s.derivative()
d.plot()
i = s.integral();
i.plot();
