# -*- coding: utf-8 -*-
"""
Resampling Module

Module with routines for data and curve resampling
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import numpy as np
from scipy.interpolate import splprep,splrep, splev


def resample_nd(curve, npoints = None, smooth = 0, periodic = False, derivative = 0, order = 5, iterations = 1):
  """Resample n points using n equidistant points along a curve
  
  Arguments:
    points (mxd array): coordinate of the reference points for the curve
    npoints (int): number of resamples equidistant points
    smooth (number): smoothness factor
    periodic (bool): if True assumes the curve is a closed curve
  
  Returns:
    (nxd array): resampled equidistant points
  """
  if npoints is None or npoints is all:
    npoints = curve.shape[0];  
  for i in range(iterations):  
    cinterp, u = splprep(curve.T, u = None, s = smooth, per = periodic, k = order);
    us = np.linspace(u.min(), u.max(), npoints)
    curve = np.vstack(splev(us, cinterp, der = derivative)).T;
  return curve;


def resample_1d(data, npoints = None, smooth = 0, periodic = False, derivative = 0, order = 5, iterations = 0):
  """Resample 1d data using n equidistant points
  
  Arguments:
    data (array): data points
    npoints (int): number of points in equidistant resampling
    smooth (number): smoothness factor
    periodic (bool): if True assumes the curve is a closed curve
  
  Returns:
    (array): resampled data points
  """
  if npoints is None or npoints is None:
    npoints = data.shape[0];

  u0 = np.linspace(0, 1, data.shape[0]);
  us = np.linspace(0, 1, npoints);
  for i in range(iterations):
    dinterp = splrep(u0, data, s = smooth, per = periodic, k = 5);
    data = splev(us, dinterp, der = derivative);
  return data;


def resample(curve, npoints = None, smooth = 0, periodic = False, derivative = 0, order = 5, iterations = 1):
  """Resample n points using n equidistant points along a curve
  
  Arguments:
    points (mxd array): coordinate of the reference points for the curve
    npoints (int): number of resamples equidistant points
    smooth (number): smoothness factor
    periodic (bool): if True assumes the curve is a closed curve
  
  Returns:
    (nxd array): resampled equidistant points
  """
  if curve.ndim > 1:
    return resample_nd(curve, npoints, smooth = smooth, periodic = periodic, derivative = derivative, order = order, iterations = iterations);
  else:
    return resample_1d(curve, npoints, smooth = smooth, periodic = periodic, derivative = derivative, order = order, iterations = iterations);



def test():
  import numpy as np
  import matplotlib.pyplot as plt
  import interpolation.resampling as res
  reload(res)
  
  curve = np.linspace(0,10,50);
  curve = np.vstack([curve, np.sin(curve)]).T;
  
  rcurve = res.resample(curve, npoints = 150, smooth = 0);
  plt.figure(1); plt.clf();
  plt.plot(rcurve[:,0], rcurve[:,1], 'red');
  plt.plot(curve[:,0], curve[:,1], 'blue');
  
  curve1d = np.sin(np.linspace(0,1,50) * 2 * np.pi);
  rcurve1d = res.resample(curve1d, npoints = 150, smooth = 0);
  plt.figure(2); plt.clf();
  plt.plot(curve1d);
  plt.plot(rcurve1d);
  
  
if __name__ == "__main__":
  test();