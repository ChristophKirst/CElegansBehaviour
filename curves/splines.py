# -*- coding: utf-8 -*-
"""
Spline module to handle spline functions

This module provides a basic class to handle curves represented as splines,
in particular mappings between spline parameter and curve spaces
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'


import numpy as np

from scipy.interpolate import splrep, splev

from curves import Curve;

class Spline(Curve):
  """Class for handling spline interpolation and conversion between bspline coefficients and splines"""
  
  def __init__(self, points = None, values = None, parameter = None, nparameter = 10, npoints = all, degree = 3, knots = None):
    """Constructor for Spline
    
    Arguments:
      values (array): the values of the curve, if None flat curve
      points (array): samples points, if none linspace(0,1,nsamples)
      parameter (array): parameter of the spline, if None determined form values
      nparameter (int):  number of parameter for the spline
      npoints (int or all): number of sample points, if all len(values) or len(samples) or nparameter
      degree (int): degree of the spline
      knots (array or None): knot vector (excluding boundary points), if None linspace(0,1,nbasis-2*degree)[1:-1]
    
    Note:
      the splrep interpolation function takes a knot vecor for the inner points only
      In addition there are degree + 1 zero coeffcients in the final spline coeffciencts. 
      Thus the number of parameter is given by:
      nparameter = len(knots) + 2 + 2 * degree - (degree + 1) = len(knots) + degree + 1
      
      For speed the basis matrices that project from and in case nparameter = nsample to the parameter are precalucalted
      and used instead of the standard interpolating routines
    """
    
    self._degree = int(degree);

    if values is not None:
      values = np.array(values, dtype = float);
      npoints = values.shape[0];
    
    if points is None:   
      if npoints is None:
        if nparameter is None:
          raise RuntimeError('cannot determine number of samples!');
        npoints = nparameter;
      self._points = np.linspace(0,1,npoints);
    else:
      self._points = np.array(points, dtype = float);
    
    self._npoints = self._points.shape[0];
    
    if parameter is None:
      if knots is None:
        if nparameter is None:
          raise RuntimeError('parameter, knot vector or number of parameter needs to be defined!');
        self._knots = np.linspace(0,1, nparameter - degree - 1 + 2)[1:-1];
        self._nparameter = int(nparameter);
      else:
        self._knots = np.array(knots, dtype = float);
        self._nparameter = int(knots.shape[0] + 1 + degree);
        
      if values is None:
        self._parameter = np.zeros(self.nparameter);
      else:
        self._basis_inv = None;
        self.parameter_from_values(values);
        
    else:
      self._parameter = np.array(parameter, dtype = float);
      self._nparameter = self._parameter.shape[0];
      
      if knots is None:
        self._knots = np.linspace(0,1, self._nparameter - self._degree - 1 + 2)[1:-1];
      else:
        self._knots = np.array(knots, dtype = float);
        if self._knots.shape[0] != int(knots.shape[0] + 1 + degree):
          raise RuntimeError('parameter and knots have inconsistent dimensions %d + 1 + %d != %d' % (self.knots.shape[0], self.degree, self.nparameter)); 
    
    # get all knots
    y = np.linspace(0,1,self._npoints);
    tck = splrep(self._points, y, t = self._knots, task = -1, k = self._degree);
    self._knots_all = tck[0];
    
    # calcualte basis matrix
    self._basis = self.basis_matrix();
    
    if self._npoints == self._nparameter:
      self._basis_inv = np.linalg.inv(self._basis);
    else:
      self._basis_inv = None;
    
  
  def from_values(self, values, points = None):
    """Calcualte the bspline parameter for the data points y

    Arguments:
      values (array or None): values of data points, if None return the internal parameter
      points (array or None): sample points, if None use internal sample pints
      
    Returns
      array: the bspline parameter
    """
    if points is None and self._basis_inv is not None: # use matrix multiplication if possible
      self._parameter = self._basis_inv.dot(values);
    else:
      if points is None:
        points = self._points;
      tck = splrep(self._points, values, t = self._knots, task = -1, k = self._degree);
      self._parameter = tck[1][:self._nparameter];
    return self._parameter; 
  
  def from_parameter(self, parameter):
    """Change parameter of the spline
    
    Arguments:
      parameter (array): the new parameter
    """
    self._parameter = parameter;
    return self._parameter;
  
  def values(self, points = None, derivative = 0, extrapolation = 1):
    """Calculates the values of the curve along the sample points
    
    Arguments:
      parameter (array or None): the bspline parameter, if None use internal parameter
      derivative (int): the order of the derivative
      extrapolation (int):  0=extrapolated value, 1=return 0, 2=raise a ValueError, 3=boundary value

    Returns:
      array: the values of the spline at the sample points
    """
    
    if derivative == 0 and points is None: # fast matrix version
      return self._basis.dot(self._parameter);
    else: # full interpolation
      if points is None:
        points = self._points;
      pp = np.pad(self._parameter, (0,self._degree+1), 'constant');
      tck = (self._knots_all, pp, self._degree);
      return splev(points, tck, der = derivative, ext = extrapolation);  
  

  
  def __call__(self, points = None, parameter = None, derivative = 0, extrapolation = 1):
    """Calculates the values of the curve along the sample points
    
    Arguments:
      samples (array or None): the sample points for the values, if None use internal samples points
      parameter (array or None): the bspline parameter, if None use internal parameter
      derivative (int): the order of the derivative
      extrapolation (int):  0=extrapolated value, 1=return 0, 2=raise a ValueError, 3=boundary value

    Returns:
      array: the values of the spline at the sample points
    """
    
    if parameter is None:
      parameter = self._parameter;
    
    if points is None and derivative == 0: # fast matrix version
      return self._basis.dot(parameter);
    else: # full interpolation
      if points is None:
        points = self._points
      pp = np.pad(parameter, (0,self._degree+1), 'constant');
      tck = (self._knots_all, pp, self._degree);
      return splev(points, tck, der = derivative, ext = extrapolation);


  def basis_matrix(self, derivative = 0):
    """Basis matrix to calucalte spline coefficients via matrix product""" 
  
    basis = np.zeros((self._nparameter, self._npoints));
    for i in range(self._nparameter):
      p = np.zeros(self._nparameter);
      p[i] = 1;
      pp = np.pad(p, (0,self._degree+1), 'constant');
      tck = (self.knots_all, pp, self._degree);
      basis[i] = splev(self._points, tck, der = derivative, ext = 1);
    
    return basis.T;
    
    
#  def _coxDeBoor(self, u, k, d, knots):
#      """Spline basis utility to calcualte spline basis at sample points"""
#      if (d == 0):
#          if (knots[k] <= u and u < knots[k+1]):
#              return 1
#          return 0
#
#      den1 = knots[k+d] - knots[k]
#      den2 = knots[k+d+1] - knots[k+1]
#      eq1  = 0;
#      eq2  = 0;
#
#      if den1 > 0:
#          eq1 = ((u-knots[k]) / den1) * self._coxDeBoor(u,k,(d-1), knots)
#      if den2 > 0:
#          eq2 = ((knots[k+d+1]-u) / den2) * self._coxDeBoor(u,(k+1),(d-1), knots)
#
#      return eq1 + eq2;  
#  def basis_matrix(self):
#    """Basis matrix to calucalte spline coefficients via matrix product""" 
#    
#    nbasis = len(self.knots_all);
#    nbasis = self.ncoefficients;
#    basis = np.zeros((nbasis, self.nsamples));
#    for s in range(self.nsamples):
#      for k in range(nbasis):
#        basis[k,s] = self._coxDeBoor(self.s[s], k, self.degree, self.knots_all);
#    
#    return basis;
    
    
def test():
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.interpolate import splrep, splev
  import curves.splines as sp;
  reload(sp);

  s = sp.Spline(nparameter = 10, npoints = 141, degree = 3);
  
  x = np.sin(10* s.samples) + np.sin(2 * s.samples);
  p = s.parameter_from_values(x)
  tck = splrep(s.samples, x, t = s.knots, task = -1, k = s.degree);
  xpp = splev(s.samples, tck);
  xp = s.values_from_parameter(p);
  
  plt.figure(1); plt.clf();
  plt.plot(x);
  plt.plot(xp);
  plt.plot(xpp);
  
  plt.figure(2); plt.clf();
  for i in range(s.nparameter):
    pp = np.zeros(s.nparameter);
    pp[i] = 1;
    xp = s.values_from_parameter(pp);
    plt.plot(s.samples, xp);
  plt.title('basis functions scipy')
  
  # test spline basis
  
  reload(sp);
  s = sp.Spline(nparameter = 10, nsamples = 141, degree = 3);

  bm = s.basis_matrix();
  plt.figure(3); plt.clf();
  plt.plot(bm);
  plt.title('basis functions matrix')
  
  xb = bm.dot(p);
  xp = s.values_from_parameter(p);
  
  #xb = np.dot(np.invert(bm), pb);

  plt.figure(4); plt.clf();
  plt.subplot(1,2,2);
  plt.plot(x);
  #plt.plot(0.1* xb);
  plt.plot(xb)
  plt.plot(xp)
  
  # case of nsmaples = ncoefficients
  s = sp.Spline(nparameter = 25, nsamples = 25, degree = 3);
  x = np.sin(10* s.samples) + np.sin(2 * s.samples);
  p = s.parameter_from_values(x);
  
  bm = s.basis;
  pb = np.linalg.inv(bm).dot(x);
  
  plt.figure(5); plt.clf();
  plt.plot(p);
  plt.plot(pb);
  
  xs = s(s.samples, p);
  xp = bm.dot(pb);
  
  plt.figure(6); plt.clf();
  plt.plot(x);
  plt.plot(xs);
  plt.plot(xp);
  
  from utils.timer import timeit
  @timeit
  def ts():
    return s(s.samples, p);    
  @timeit
  def tb():
    return s.values_from_parameter(p);
      
  ps = ts();
  pb = tb();
  np.allclose(pb, ps)
  
  # factor ~10 times faster with basis
  
  # test shifting spline by a value s
  s = sp.Spline(nparameter = 25, nsamples = 25, degree = 3);
  x = np.sin(10* s.samples) + np.sin(2 * s.samples);
  p = s.parameter_from_values(x);  
  xnew = s(s.samples + 0.05, p);
  
  plt.figure(7); plt.clf();
  plt.plot(x);
  plt.plot(xnew);
  

if __name__ == "__main__ ":
  test();