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

nn = 100;
pn = 80;

### Construct uniform curves by hand
from scipy.special import ellipeinc
from scipy import optimize

def inverse(fun, x):
  def err(xx):
    return fun(xx)-x;
  return optimize.newton(err, x)


def fun(x):
  return np.sqrt(37)/6.0 * ellipeinc(6*x, 36.0/37.0);


x = np.linspace(0,1,nn);
y = fun(x);
y1 = [inverse(fun, fun(1) * xx) for xx in x];
y2 = [np.sin(6 * inverse(fun, fun(1) * xx)) for xx in x];

s1 = sp.Spline(y1, x, nparameter=pn);
s2 = sp.Spline(y2, x, nparameter=pn);


plt.figure(1); plt.clf();
plt.subplot(1,2,1)
#plt.plot(x, y)
plt.plot(x,y1)
plt.plot(x,y2)
plt.subplot(1,2,2);
s1.plot()
s2.plot()



### Test Curve

s = np.linspace(0,1,nn);
xy1 = np.vstack([s, np.sin(2 * np.pi * s)]).T;
c1 = sp.Curve(xy1, nparameter = pn);

# make uniform by hand
tck,u = splprep(c1.values.T, s = 0.0);
v = splev(np.linspace(0,1,nn), tck);
vs = np.vstack(v).T;
c2 = sp.Curve(vs, nparameter = c1.nparameter);

#resample uniform
c3 = c1.copy();
c3.resample_uniform();

# uniform knots
c4 = c1.copy();
c4.uniform();


plt.figure(2); plt.clf();
plt.subplot(1,4,1);
c1.plot();
plt.axis('equal')
plt.subplot(1,4,2);
c2.plot();
plt.axis('equal')
plt.plot(vs[:,0], vs[:,1])
c1.plot(with_points=False)
plt.subplot(1,4,3);
c3.plot();
plt.axis('equal')
plt.subplot(1,4,4);
c4.plot();
plt.axis('equal')


plt.figure(3); plt.clf();
plt.subplot(1,2,1);
plt.plot(c1.points, c1.basis)
c1.plot()
plt.subplot(1,2,2);
plt.plot(c2.points, c2.basis)
c2.plot(dim=0)
c2.plot(dim=1)
plt.plot(vs[:,0], vs[:,1])
c2.plot()


### Integration and Derivative

cc = c4;

der = cc.derivative();
it = cc.integral();

plt.figure(4); plt.clf();
plt.subplot(1,3,1)
cc.plot(dim=0);
cc.plot(dim=1);
plt.subplot(1,3,2)
der.plot(dim=0);
der.plot(dim=1);
plt.subplot(1,3,3)
it.plot(dim=0);
it.plot(dim=1);

# interesting how a sin function can be such a tough problem !


### Test phi / theta

phi = c4.phi();
theta = c4.theta();
phii = theta.integral();
phid = phi.derivative();

plt.figure(5); plt.clf();
plt.subplot(2,1,1);
phi.plot();
phii.plot();
plt.subplot(2,1,2);
theta.plot();
phid.plot();

### Get curve back from theta

ct = sp.theta_to_curve(theta=phid);

plt.figure(6); plt.clf();
c1.plot();
ct.plot();


reference = 0.5; orientation = phi(reference);
phi2 = theta.integral();
phi0 =  phi2(reference);
phi2.parameter += orientation - phi0; # add a constant = add constant to parameter
phi2.values += orientation - phi0;

plt.figure(6); plt.clf();
phi.plot();
phi2.plot();


points = phi2.points;
phip = phi2(points);
ds = 1.0 /(phi.npoints-1);

tgt = c1.length() * np.vstack([np.cos(phip), np.sin(phip)]).T;

plt.figure(6); plt.clf();
plt.plot(points, tgt[:,0]);
plt.plot(points, tgt[:,1]);
der.plot(dim=0);
der.plot(dim=1);


xy = [0.5,0];
xyc = sp.Curve(tgt, nparameter = pn);
xyci = xyc.integral();
xyc0 = xyci(reference);
xyci.parameter += xy - xyc0;
xyci.values += xy - xyc0;

plt.figure(7); plt.clf();
plt.subplot(1,2,1);
c4.plot();
xyci.plot();
plt.xlim(0,1); plt.ylim(-2,2)
plt.subplot(1,2,2);
c4.plot(dim=0);
c4.plot(dim=1);
xyci.plot(dim=0);
xyci.plot(dim=1);






