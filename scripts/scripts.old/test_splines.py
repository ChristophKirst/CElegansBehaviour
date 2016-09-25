# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 21:47:39 2016

@author: ckirst
"""

import numpy as np
from scipy.interpolate import splprep, splev, splrep, splantider
import matplotlib.pyplot as plt

s = np.linspace(0,1,50) * 2 * np.pi;
xy = np.vstack([np.sin(s), np.cos(s)]).T;
xy.shape


x = xy[:,0];
tck1d = splrep(s, x, s = 0);

tck,u = splprep(xy.T, s= 0)

c = np.zeros(tck[1][0].shape[0]+4);
c[:-4] = tck[1][0]
tck1 = (tck[0], c, tck[2])
itck1= splantider(tck1, 1);

c[:-4] = tck[1][1]
tck2 = (tck[0], c, tck[2])
itck2= splantider(tck2, 1);



plt.figure(1); plt.clf();
plt.plot(xy[:,0], xy[:,1])

xys = splev(u, tck);
plt.plot(xys[0], xys[1])


x = xy[:,0];
tck1d = splrep(u, x, s = 0);
xs = splev(u, tck1d)

plt.figure(2); plt.clf();
plt.plot(u,x);
plt.plot(u,xs);


tck[0]
tck1d[0]

tck[1][0].shape
tck1d[1].shape


k = 3;
knots = np.linspace(0,1,)


# test splines:

knots = np.linspace(0,1,50)[1:-1];
#knots = tck[0];
#knots = np.linspace(0,1,50);

knots = np.zeros((numknots + 2*k + 2,), float);
knots[k+1:-k-1] = 


nest = len(knots)


tck,un = splprep(xy.T, u = None, t = knots, task = -1, k = 3, nest = nest);



import numpy as np
from scipy.interpolate import _fitpack, splprep


nn = 20;
u = np.linspace(0,1,nn);
xy = np.vstack([np.sin(u), np.cos(u)]).T;

m = len(u);
w = np.ones(m);

ub = u[0];
ue = u[-1];

k = 3

task = -1;

ipar = 0;

numknots = 7;
knots = np.zeros((numknots + 2*k,), float);
knots[k:-k] = np.linspace(0,1,numknots)

nest = 2*(nn+numknots);

tck,un = splprep(xy.T, u = u, t = knots, task = -1, k = 3, nest = nest);


    t, c, o = _fitpack._parcur(ravel(transpose(x)), w, u, ub, ue, k,
                               task, ipar, s, t, nest, wrk, iwrk, per)




## Thetas



import numpy as np
from interpolation.spline import Spline
import matplotlib.pyplot as plt

s = np.linspace(0,1,20);
x = np.sin(2*np.pi*s);

sp = Spline(values = x, points = s);


plt.figure(1); plt.clf();
plt.plot(s,x);
sp.plot();

sp.from_parameter(sp.parameter - 1.0)
sp.plot();


