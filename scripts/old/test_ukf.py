# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 22:30:17 2016

@author: ckirst
"""

# -*- coding: utf-8 -*-
"""Copyright 2015 Roger R Labbe Jr.
FilterPy library.
http://github.com/rlabbe/filterpy
Documentation at:
https://filterpy.readthedocs.org
Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
This is licensed under an MIT license. See the readme.MD file
for more information.
"""











"""Copyright 2015 Roger R Labbe Jr.
FilterPy library.
http://github.com/rlabbe/filterpy
Documentation at:
https://filterpy.readthedocs.org
Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
from numpy.random import randn



def GetRadar(dt):
    """ Simulate radar range to object at 1K altidue and moving at 100m/s.
    Adds about 5% measurement noise. Returns slant range to the object.
    Call once for each new measurement at dt time from last call.
    """

    if not hasattr (GetRadar, "posp"):
        GetRadar.posp = 0

    if GetRadar.posp  > 10000:
      vel = 0;
    else:
      vel = 100  + .5 * randn()
    alt = 1000 + 10 * randn()
    pos = GetRadar.posp + vel*dt

    v = 0 + pos* 0.05*randn()
    slant_range = math.sqrt (pos**2 + alt**2) + v
    GetRadar.posp = pos

    return slant_range




from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
#from GetRadar import GetRadar
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise



def fx(x, dt):
    """ state transition function for sstate [downrange, vel, altitude]"""
    F = np.array([[1., dt, 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]])

    return np.dot(F, x)


def hx(x):
    """ returns slant range based on downrange distance and altitude"""

    return (x[0]**2 + x[2]**2)**.5


from filterpy.kalman.sigma_points import JulierSigmaPoints as sigmapoints

if __name__ == "__main__":

    dt = 0.05

    points = sigmapoints(n=3, kappa = 0);

    radarUKF = UKF(dim_x=3, dim_z=1, dt=dt, hx = hx, fx = fx, points = points);
    radarUKF.Q *= Q_discrete_white_noise(3, 1, .01)
    radarUKF.R *= 10
    radarUKF.x = np.array([0., 90., 1100.])
    radarUKF.P *= 100.

    t = np.arange(0, 20+dt, dt)
    n = len(t)
    xs = []
    rs = []
    for i in range(n):
        r = GetRadar(dt)
        rs.append(r)
        #rs = r;
        radarUKF.predict();
        radarUKF.update(r)

        xs.append(radarUKF.x)

    xs = np.asarray(xs)

    plt.subplot(411)
    plt.plot(t, xs[:, 0])
    plt.title('distance')

    plt.subplot(412)
    plt.plot(t, xs[:, 1])
    plt.title('velocity')

    plt.subplot(413)
    plt.plot(t, xs[:, 2])
    plt.title('altitude')
    
    plt.subplot(414)
    plt.plot(t, rs)
    plt.title('radar')