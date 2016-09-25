# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 04:21:33 2016

@author: ckirst
"""

import numpy as np
import matplotlib.pyplot as plt;
from scipy.interpolate import splprep, splev
import interpolation.spline as sp;
reload(sp)

s = np.linspace(0,1,50);
xy1 = np.vstack([s, np.sin(2 * np.pi * s)]).T;
c1 = sp.Curve(xy1, nparameter = 10);
#c1.uniform();

plt.figure(1); plt.clf();
plt.subplot(1,2,1);
c1.plot()
v = list(c1.values.T);
plt.scatter(v[0], v[1], c = 'r', s= 100)

tck,u = splprep(c1.values.T, s = 0.0);
v = splev(np.linspace(0,1,50), tck);
vs = np.vstack(v).T;

c2 = sp.Curve(vs, nparameter = c1.nparameter);
#t = np.linspace(0,1,20);
#k = 3;
#t = np.r_[[0]*k, t, [1]*k];
tck2,u2 = splprep(vs.T); #t = t, k = k, task = -1, nest = );
plt.scatter(v[0], v[1], c = 'm', s = 70);
c2.plot();
plt.scatter(c2.values[:,0], c2.values[:,1], c = 'k', s = 40)

c3 = c1.copy();
c3.resample_uniform();
#c3.uniform();

c3.plot();
plt.scatter(c3.values[:,0], c3.values[:,1], c = 'g', s = 20)

plt.xlim(0,1)
plt.ylim(0.1)
plt.axis('equal')

plt.subplot(1,2,2);
plt.plot(u)
plt.plot(u2);


c4 = c3.copy();
c4.uniform();

plt.figure(20); plt.clf();
plt.subplot(1,3,1);
plt.plot(c3.basis, 'r');
plt.plot(c4.basis, 'b');
plt.subplot(1,3,2);
plt.plot(c3.parameter);
plt.plot(c4.parameter);
plt.subplot(1,3,3);
c3.plot();
c4.plot();

v = c4.basis.dot(c4.parameter);
plt.plot(v[:,0], v[:,1])


der = c3.derivative();
it = c3.integral();

plt.figure(2); plt.clf();
plt.subplot(1,3,1)
c3.plot(dim=0);
c3.plot(dim=1);
plt.subplot(1,3,2)
der.plot(dim=0);
der.plot(dim=1);
plt.subplot(1,3,3)
it.plot(dim=0);
it.plot(dim=1);



