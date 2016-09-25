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
import matplotlib.pyplot as plt
import copy

from scipy.interpolate import splrep, splev, splint, splder, splantider, sproot, splprep;

from interpolation.intersections import curve_intersections_discrete;

#from interpolation.curve import Curve;
class Spline:
  """Class for handling 1d spline interpolation and conversion between bspline coefficients and splines"""
  
  def __init__(self, values = None, points = None, parameter = None, npoints = None, nparameter = None, degree = 3, knots = None, tck = None):
    """Constructor for Spline
    
    Arguments:
      values (array): the values from which to construct the spline
      points (array): samples points, if none linspace(0,1,nsamples)
      parameter (array): parameter of the spline, if None determine from values
      nparameter (int):  number of parameter for the spline, if None determined from len(parameter), len(values) or len(points)
      npoints (int): number of sample points, if None len(values) or len(points) or nparameter
      degree (int): degree of the spline
      knots (array): knot vector (excluding boundary points), if None linspace(points[0],points[-1],nparameter-2*degree)[1:-1]
      tck (tuple): intialize form tck tuple returned by splrep or splprep
    
    Note:
      the splrep interpolation function takes a knot vecor for the inner points only
      In addition there are degree + 1 zero coeffcients in the final spline coefficients. 
      Thus the number of parameter is given by:
      nparameter = len(knots) + 2 + 2 * degree - (degree + 1) = len(knots) + degree + 1
      npoints = len(points) = len(knots) + 2
      
      For speed the basis matrices that project from and in case it is invertible to the parameter are precalculated
      and used instead of the standard interpolating routines
    """
    
    if isinstance(values, Spline):
      self = copy.deepcopy(values);
      return;
    
    if tck is not None:
      self.from_tck(tck, points = points, npoints = npoints);
      return;
    
    self.degree = int(degree);

    # initialize sample points
    if values is not None:
      values = np.array(values, dtype = float);
      npoints = values.shape[0];
    
    if points is None:   
      if npoints is None:
        if nparameter is None:
          raise RuntimeError('cannot determine the number of smaple points, values, points, number of points or number of parameter needs to be defined!');
        npoints = nparameter;
      self.points = np.linspace(0,1,npoints);
    else:
      self.points = np.array(points, dtype = float);
    self.npoints = self.points.shape[0];
    
    # initialize number of parameter and knots
    if parameter is None:
      if knots is None:
        if nparameter is None:
          nparameter = self.npoints;
        self.knots = np.linspace(self.points[0], self.points[-1], nparameter - degree + 1)[1:-1];
        self.nparameter = int(nparameter);
      else:
        self.knots = np.array(knots, dtype = float);
        self.nparameter = int(knots.shape[0] + 1 + degree);
    
    else:
      self.parameter = np.array(parameter, dtype = float);
      self.nparameter = self.parameter.shape[0];
      
      if knots is None:
        self.knots = np.linspace(self.points[0], self.points[-1], self.nparameter - self.degree + 1)[1:-1];
      else:
        self.knots = np.array(knots, dtype = float);
        if self.knots.shape[0] != int(knots.shape[0] + 1 + degree):
          raise RuntimeError('parameter and knots have inconsistent dimensions %d + 1 + %d != %d' % (self.knots.shape[0], self.degree, self.nparameter)); 
    
    self.initialize_all_knots();
    
    # calcualte basis matrix
    self.initialize_basis();       
    
    # initialize parameter and values
    if parameter is None: 
      if values is None:
        self.from_parameter(np.zeros(self.nparameter));
      else:
        self.from_values(values);
    else:
      if values is not None:
        raise RuntimeError('both parameter and values are specified, only one can be specified!');
      self.from_parameter(parameter);  
    
  
  def initialize_all_knots(self):
    """Initializes all knots"""
    y = np.zeros(self.npoints);
    tck = splrep(self.points, y, t = self.knots, task = -1, k = self.degree);
    self.knots_all = tck[0];
  
  
  def basis_matrix(self, derivative = 0):
    """Basis matrix to calucalte spline coefficients via matrix product""" 
  
    basis = np.zeros((self.nparameter, self.npoints));
    for i in range(self.nparameter):
      p = np.zeros(self.nparameter);
      p[i] = 1;
      pp = np.pad(p, (0,self.degree+1), 'constant'); # splev wants parameter
      tck = (self.knots_all, pp, self.degree);
      basis[i] = splev(self.points, tck, der = derivative, ext = 1);
    
    return basis.T;
    
  
  def initialize_basis(self):
    """Initializes the basis matrices"""
    self.basis = self.basis_matrix();
    if self.npoints == self.nparameter:
      try:
        self.basis_inv = np.linalg.inv(self.basis);
      except:
        self.basis_inv = None;
    else:
      self.basis_inv = None;
      
  def copy(self):
    """Deep copy the Spline"""
    return copy.deepcopy(self);
  
  
  def resample(self, points = None, npoints = None):
    """Resample the curve along new points or number of points
    
    Arguments:
      points (array): sample points
      npoints (int): number of sample points
    """
    tck = self.tck();
    self.from_tck(tck, points = points, npoints = npoints);  
  
  
  def from_values(self, values):
    """Calcualte the bspline parameter for the data points y

    Arguments:
      values (array or None): values of data points, if None return the internal parameter
      
    Returns
      array: the bspline parameter
    """
    if self.basis_inv is not None: # use matrix multiplication if possible
      self.parameter = self.basis_inv.dot(values);
      #self.values = self.basis.dot(self.parameter);
      self.values = values;
    else:
      tck = splrep(self.points, values, t = self.knots, task = -1, k = self.degree);
      self.parameter = tck[1][:self.nparameter];
      self.values = self.basis.dot(self.parameter);
  
  
  def from_parameter(self, parameter):
    """Change parameter of the spline
    
    Arguments:
      parameter (array): the new parameter
    """
    self.parameter = parameter;
    self.values = self.basis.dot(self.parameter);
  
    
  def from_tck(self, tck, points = None, npoints = None):
    """Change spline parameter and knot structure using a tck object returned by splrep
    
    Arguments:
        tck (tuple): t,c,k tuple returned by splrep
    """
        
    self.degree = tck[2];    
    self.knots_all = tck[0];
    self.knots = tck[0][self.degree+1:-self.degree-1];
    self.nparameter = self.knots.shape[0] + self.degree + 1;
    self.parameter = tck[1][:self.nparameter];
    if points is None:
      if npoints is None or npoints is all:
        self.points = np.linspace(0,1,self.nparameter);
      else:
        self.points = np.array(points, dtype = float);
    else:
      self.points = np.array(points, dtype = float);
    self.npoints = self.points.shape[0];
    self.initialize_basis();
    self.values = self.basis.dot(self.parameter);
  
    
  def tck(self):
    """Returns a tck tuple for use with spline functions
    
    Note:
      This returns a a tck tuple as splrep
    """  
    p = np.pad(self.parameter, (0,self.degree+1), 'constant');
    return (self.knots_all, p, self.degree);
  
  
  def tck_ndim(self):
    """Returns a tck tuple for use with spline functions
    
    Note:
      This returns a a tck tuple as splprep
    """  
    return (self.knots_all, [self.parameter], self.degree);
    
  
  def to_curve(self):
    """Converts the 1d spline to a 1d curve"""
    return Curve(tck = self.tck_ndim(), points = self.points);
  
  
  def get_values(self, parameter = None, points = None, derivative = 0, extrapolation = 1):
    """Calculates the values of the curve along the sample points
    
    Arguments:
      parameter (array or None): the bspline parameter, if None use internal parameter
      points (array or None): the sample points for the values, if None use internal samples points
      derivative (int): the order of the derivative
      extrapolation (int):  0=extrapolated value, 1=return 0, 2=raise a ValueError, 3=boundary value

    Returns:
      array: the values of the spline at the sample points
    """
    if points is all:
      points = None;
    
    if parameter is None:
       if derivative == 0 and points is None: # fast matrix version
         return self.values;
       else:
         parameter = self.parameter;
    
    if derivative == 0 and points is None:
      return self.basis.dot(parameter);
    else: # full interpolation
      if points is None:
        points = self.points;
      pp = np.pad(parameter, (0,self.degree+1), 'constant');
      tck = (self.knots_all, pp, self.degree);
      return splev(points, tck, der = derivative, ext = extrapolation);  
  
    
  def __call__(self, points = None, parameter = None, derivative = 0, extrapolation = 1):
    """Calculates the values of the curve along the sample points
    
    Arguments:
      points (array or None): the sample points for the values, if None use internal samples points
      parameter (array or None): the bspline parameter, if None use internal parameter
      derivative (int): the order of the derivative
      extrapolation (int):  0=extrapolated value, 1=return 0, 2=raise a ValueError, 3=boundary value

    Returns:
      array: the values of the spline at the sample points
    """
    return self.get_values(parameter = parameter, points = points, derivative = derivative, extrapolation = extrapolation);\
    
  
  def integrate(self, lo, up, function = None):
    """Integrate between lo and up of the values
    
    Arguments:
      lo,up (float or array): lower/upper bound of integration
      function (function or None): function applied to va;lues before integration
    
    Returns:
      float: the integral
    """
    
    if function is None:
      tck = self.tck();
    else:
      values = function(self.values);
      tck = splrep(self.points, values, t = self.knots, task = -1, k = self.degree);
    
    if isinstance(lo, np.ndarray):
      return np.array([splint(lo[i],up[i],tck) for i in range(len(lo))]);
    else:
      return splint(lo,up,tck);
    
 
  def derivative(self, n = 1):
    """Returns derivative as a spline
    
    Arguments:
      n (int): order of the derivative
      
    Returns:
      Spline: the derivative of the spline with degree - n 
    """
    
    tck = splder(self.tck(), n);
    return Spline(tck = tck, points = self.points);
  
        
  def integral(self, n = 1):
    """Returns integral / anti-derivative as a spline
    
    Arguments:
      n (int): order of the integral
      
    Returns:
      Spline: the integral of the spline with degree + n  
    """
    
    tck = splantider(self.tck(), n);
    return Spline(tck = tck, points = self.points);
  
  
  def roots(self):
    """ Return the zeros of the spline.

    Note: 
      Only cubic splines are supported.
    """
    
    if self.degree == 3:
      z,m,ier = sproot(*self.tck(), mest = 10);
      if not ier == 0:
        raise RuntimeError("Error code returned by spalde: %s" % ier)
      return z[:m]
    raise RuntimeError('finding roots unsupported for non-cubic splines')
  
  
  def shift(self, shift, extrapolation = 1):
    """Shift spline
    
    Arguments:
      shift (float): shift
      extrapolation (int): 0=extrapolated value, 1=return 0, 2=raise a ValueError, 3=boundary value
    """
    
    values = self(self.points + shift, extrapolation = extrapolation);
    self.from_values(values);
  
  
  def plot(self, *args, **kwargs):
    """Plot the Spline"""
    plt.plot(self.points, self.values, *args, **kwargs);
    




#from interpolation.curve import Curve;
class Curve:
  """Class for handling multi-dimensional spline interpolation and conversion between bspline coefficients and splines"""
  
  def __init__(self, values = None, points = None, parameter = None, npoints = None, nparameter = None, degree = 3, ndim = None, knots = None, tck = None):
    """Constructor for Curve
    
    Arguments:
      values (dxn array): the values of the curve
      points (array): sample points along the curve parameter, if none linspace(0,1,nsamples)
      parameter (array): parameter of the spline coefficients for each dimension, if None determine from values
      npoints (int): number of sample points alsong the curve, if None len(values) or len(points) or nparameter
      nparameter (int):  number of spline coefficients in each dimension, if None determined from len(parameter), len(values) or len(points)
      degree (int): degree of the spline
      knots (array): knot vector (excluding boundary points), if None linspace(points[0],points[-1],nparameter-2*degree)[1:-1]
      tck (tuple): intialize form tck tuple returned by splprep
    
    Note:
      The splrep interpolation function takes a knot vecor for the inner points only
      In addition there are degree + 1 zero coeffcients in the final spline coefficients. 
      Thus the number of parameter is given by:
      nparameter = len(knots) + 2 + 2 * degree - (degree + 1) = len(knots) + degree + 1
      npoints = len(points) = len(knots) + 2
      
      For speed the basis matrices that project from and in case it is invertible to the parameter are precalculated
      and used instead of the standard interpolating routines
    """
    
    if tck is not None:
      self.from_tck(tck, points = points, npoints = npoints);
      return;
    
    self.degree = int(degree);
    
    if ndim is None:
      ndim = 1;
    self.ndim = int(ndim);    
    
    # initialize sample points
    if values is not None:
      values = np.array(values, dtype = float);
      if values.ndim == 1:
        values = values[:, np.newaxis];
      npoints = values.shape[0];
      self.ndim = values.shape[1];
    
    if points is None:   
      if npoints is None:
        if nparameter is None:
          raise RuntimeError('cannot determine the number of smaple points, values, points, number of points or number of parameter needs to be defined!');
        npoints = nparameter;
      self.points = np.linspace(0,1,npoints);
    else:
      self.points = np.array(points, dtype = float);
    self.npoints = self.points.shape[0];
    
    #initialize number of parameter and knots
    if parameter is None:
      if knots is None:
        if nparameter is None:
          #raise RuntimeError('parameter, knot vector or number of parameter needs to be defined!');
          nparameter = self.npoints;
        self.knots = np.linspace(self.points[0], self.points[-1], nparameter - degree + 1)[1:-1];
        self.nparameter = int(nparameter);
      else:
        self.knots = np.array(knots, dtype = float);
        self.nparameter = int(knots.shape[0] + 1 + degree);
      self.parameter = None;
    else:
      self.parameter = np.array(parameter, dtype = float);
      if self.parameter.ndim == 1:
        self.parameter = self.parameter[:,np.newaxis];
      self.ndim = self.parameter.shape[1];
      self.nparameter = self.parameter.shape[0];
      
      if knots is None:
        self.knots = np.linspace(self.points[0], self.points[-1], self.nparameter - self.degree + 1)[1:-1];
      else:
        self.knots = np.array(knots, dtype = float);
        if self.knots.shape[0] != int(knots.shape[0] + 1 + degree):
          raise RuntimeError('parameter and knots have inconsistent dimensions %d + 1 + %d != %d' % (self.knots.shape[0], self.degree, self.nparameter)); 
    
    # get all knots
    self.initialize_all_knots();
    
    # calcualte basis matrix
    self.initialize_basis();       
    
    if self.parameter is None:
      if values is None:
        self.from_parameter(np.zeros((self.nparameter, self.ndim)));
      else:
        self.from_values(values);
    else:
      if values is not None:
        raise RuntimeError('initilializing with parameter and values, only specify one of both!');
      self.values = np.zeros((self.npoints, self.ndim));
      for d in range(self.ndim):
        self.values[:,d] = self.basis.dot(self.parameter[:,d]);
    
  
  def initialize_all_knots(self):
    """Initializes all knots"""
    y = np.zeros(self.npoints);
    tck = splrep(self.points, y, t = self.knots, task = -1, k = self.degree);
    self.knots_all = tck[0];
  
  
  def basis_matrix(self, derivative = 0):
    """Basis matrix to calucalte spline coefficients via matrix product""" 
    
    basis = np.zeros((self.nparameter, self.npoints));
    for i in range(self.nparameter):
      p = np.zeros(self.nparameter);
      p[i] = 1;
      pp = np.pad(p, (0,self.degree+1), 'constant');
      tck = (self.knots_all, pp, self.degree);
      basis[i] = splev(self.points, tck, der = derivative, ext = 1);
    
    return basis.T;
    
  
  def initialize_basis(self):
    """Initializes the basis matrices"""
    self.basis = self.basis_matrix();
    if self.npoints == self.nparameter:
      try:
        self.basis_inv = np.linalg.inv(self.basis);
      except:
        self.basis_inv = None;
    else:
      self.basis_inv = None;
      
  
  def copy(self):
    """Deep copy the Curve"""
    return copy.deepcopy(self);
  
  
  def resample(self, points = None, npoints = None):
    """Resample the curve along new points or number of points
    
    Arguments:
      points (array): sample points
      npoints (int): number of sample points
    """
    tck = self.tck();
    self.from_tck(tck, points = points, npoints = npoints);
  
    
  def resample_uniform(self, npoints = None):
    """Resample the curve such that the distances between the points are uniformly spaced"""
    
    tck,u = splprep(self.values.T, s = 0);
    
    self.degree = tck[2]; 
    self.ndim = len(tck[1]);
    self.knots_all = tck[0];
    self.knots = tck[0][self.degree+1:-self.degree-1];
    self.nparameter = self.knots.shape[0] + self.degree + 1;
    
    self.parameter = np.zeros((self.nparameter, self.ndim));
    for i,cc in enumerate(tck[1]):    
      self.parameter[:,i] = cc[:self.nparameter];
  
    if npoints is None:
      npoints = self.npoints;
    else:
      self.npoints = npoints;  
    self.points = np.linspace(0,1, npoints);
    
    values = splev(self.points, tck, der = 0, ext = 1);  
    self.values = np.vstack(values).T;
    
    self.initialize_basis();
  
  
  def reknot(self, knots = None, nparameter = None):
    """Change to the specified knot vector specifications
    
    Arguments:
      knots (array or None): the inner knot vectors, if None use linspace(0,1,nparameter-degree-1 + 2)[1:-1]
      nparameter (int or None): number of parameter of the curve in each dimension
    
    Note:
      In order to compare parameters of curves the knot vectors have to be the same.
      nparameter = len(knots) + degree + 1
    """
    
    values = self.values;

    if knots is None and nparameter is None:
      return;
    
    if knots is None:
      self.knots = np.linspace(self.points[0], self.points[-1], nparameter - self.degree + 1)[1:-1];
      self.nparameter = int(nparameter);
    else:
      self.knots = np.array(knots, dtype = float);
      self.nparameter = int(knots.shape[0] + 1 + self.degree);

    self.initialize_all_knots();
    self.initialize_basis();
    self.from_values(values);
    
  
  def reknot_uniform(self):
    """Change to uniformly spaced knots"""
    self.reknot(nparameter = self.nparameter);
  
  
  def uniform(self):
    """Make sample points and knots uniformly spaced"""
    self.resample_uniform();
    self.reknot(nparameter = self.nparameter);
    
  
  def from_values(self, values, dim = None):
    """Calcualte the bspline parameter for the data points y

    Arguments:
      values (array): values of data points
      dim (int or None): the dimension at which to change the curve, if None change dimension to the dim of values
      
    Returns
      array: the bspline parameter
    """

    if values.ndim == 1:
        values = values[:,np.newaxis];
    
    if dim is None:
      vdims = range(values.shape[1]);
      pdims = range(values.shape[1]);
      self.ndim = values.shape[1];
      self.parameter = np.zeros((self.nparameter, self.ndim));
      self.values    = np.zeros((self.npoints, self.ndim));
    else:
      vdims = range(values.shape[1]);
      pdims = np.array(dim, dtype = int);
      if len(vdims) != len(pdims):
        raise RuntimeError('inconsistent number of dimensions of values and dimension parameter!');
      
    if self.basis_inv is not None: # use matrix multiplication if possible
      for v,p in zip(vdims, pdims):
        self.parameter[:,p] = self.basis_inv.dot(values[:,v]);
        #self.values = self.basis.dot(self.parameter);
        self.values[:,p] = values[:,v];
    else:
      #tck,u = splprep(values, u = self.points, t = self.knots, task = -1, k = self.degree, s = 0); # splprep crashes due to erros in fitpack
      #for d in range(self.ndim):
      #  self.parameter[d] = tck[1][d];
      #  self.values[d] = self.basis.dot(self.parameter[d]);
      for v,p in zip(vdims, pdims):
        tck = splrep(self.points, values[:,v], t = self.knots, task = -1, k = self.degree);
        self.parameter[:,p] = tck[1][:self.nparameter];
        self.values[:,p] = self.basis.dot(self.parameter[:,p]);

  
  def from_parameter(self, parameter, dim = None):
    """Change parameter of the spline
    
    Arguments:
      parameter (array): the new parameter
      dim (int or None): the dimension at which to change the curve, if None change dimension to the dim of values
    """
    if parameter.ndim == 1:
        parameter = parameter[:, np.newaxis];
    
    if dim is None:
      vdims = range(parameter.shape[1]);
      pdims = range(parameter.shape[1]);
      self.ndim = parameter.shape[1];
      self.parameter = np.zeros((self.nparameter, self.ndim));
      self.values    = np.zeros((self.npoints, self.ndim));
    else:
      vdims = range(parameter.shape[1]);
      pdims = np.array(dim, dtype = int);
      if len(vdims) != len(pdims):
        raise RuntimeError('inconsistent number of dimensions of values and dimension parameter!');    
    
    for v,p in zip(vdims, pdims):
      self.parameter[:,p] = parameter[:,v];
      self.values[:,p] = self.basis.dot(self.parameter[:,p]);
  
    
  def from_tck(self, tck, points = None, npoints = None):
    """Change spline parameter and knot structure using a tck object returned by splprep or splrep
    
    Arguments:
        tck (tuple): t,c,k tuple returned by splprep
    """
        
    self.degree = tck[2];    
    self.knots_all = tck[0];
    self.knots = tck[0][self.degree+1:-self.degree-1];
    self.nparameter = self.knots.shape[0] + self.degree + 1;
    
    c = tck[1];
    if isinstance(c, list):
      self.ndim = len(tck[1]);
    elif isinstance(c, np.ndarray):
      if c.ndim > 1:
        self.ndim = c.shape[0];
      else:
        self.ndim = 1;
        c = [c];
    
    self.parameter = np.zeros((self.nparameter, self.ndim));
    for i,cc in enumerate(c):    
      self.parameter[:,i] = cc[:self.nparameter];
    
    if points is None:
      if npoints is None or npoints is all:
        self.points = np.linspace(0,1,self.nparameter);
      else:
        self.points = np.linspace(0,1,npoints);
    else:
      self.points = np.array(points, dtype = float);
    self.npoints = self.points.shape[0];
    
    self.initialize_basis();
    
    self.values = np.zeros((self.npoints, self.ndim));
    for d in range(self.ndim):
      self.values[:,d] = self.basis.dot(self.parameter[:,d]);
  
    
  def tck(self, dim = None):
    """Returns a tck tuple for use with spline functions
    
    Arguments:
      dim (None, list or int): dimensions for which to return tck tuple, if None return for all
      
    Returns:
      tuple: tck couple as returned by splprep
    """
    
    if dim is None:
      return (self.knots_all, self.parameter.T, self.degree);
    else:
      if isinstance(dim,int):
        dim = [dim];
      return (self.knots_all, self.parameter[:,dim].T, self.degree);
  
  
  def tck_1d(self, dim):
    """Returns tck tuple for use with 1d spline functions
    
    Arguments:
      dim (int): dimension for which to return tck tuple
      
    Returns:
      tuple: tck couple as retyurned by splrep
    """
    p = np.pad(self.parameter[:,dim], (0,self.degree+1), 'constant');
    return (self.knots_all, p, self.degree);
  
  
  def to_spline(self, dim):
    """Returns a Spline for a specific dimension
    
    Arguments:
      dim (int): dimension for which to return Spline
      
    Returns:
      Spline: Spline along the indicated dimension
    """
    return Spline(tck = self.tck_1d(dim), points = self.points);
  
  
  def to_splines(self, dim = None):
    """Returns list of splines for each dimension
    
    Arguments:
      dim (None, list or int): dimensions for which to return Splines
    
    Returns:
      list: list of Splines along the indicated dimension
    """
    if dim is None:
      dim = range(self.ndim);
    if isinstance(dim,int):
      dim = [dim];
    
    return [self.to_spline(d) for d in dim];
  
  
  def get_values(self, parameter = None, points = None, dim = None, derivative = 0, extrapolation = 1):
    """Calculates the values of the curve along the sample points
    
    Arguments:
      parameter (array or None): the bspline parameter, if None use internal parameter
      points (array or None): the sample points for the values, if None use internal samples points
      dim (None, list or int): dimensions for which to return values
      derivative (int): the order of the derivative
      extrapolation (int):  0=extrapolated value, 1=return 0, 2=raise a ValueError, 3=boundary value

    Returns:
      array: the values of the spline at the sample points
    """
    
    if dim is None:
      dim = range(self.ndim);
    if isinstance(dim,int):
      dim = [dim];    
    
    if points is all:
      points = None;
    
    if parameter is None:
       if derivative == 0 and points is None: # nothing to do, simply return values
         return self.values[:,dim];
       else:
         parameter = self.parameter[:,dim];
    
    if derivative == 0 and points is None: # evalute on different parameter set
      values = np.zeros((self.npoints, parameter.shape[1]));
      for i in range(parameter.shape[1]):
        values[:,i] = self.basis.dot(parameter[:,i]);
      return values;
    
    else: # full interpolation on different smaple points or derivative
      if points is None:
        points = self.points;
      #pp = np.pad(parameter, (0,self.degree+1), 'constant');
      tck = (self.knots_all, parameter.T, self.degree);
      values = splev(points, tck, der = derivative, ext = extrapolation);  
      return np.vstack(values).T
  
    
  def __call__(self, points = None, parameter = None, dim = None, derivative = 0, extrapolation = 1):
    """Calculates the values of the curve along the sample points
    
    Arguments:
      points (array or None): the sample points for the values, if None use internal samples points
      parameter (array or None): the bspline parameter, if None use internal parameter
      dim (None, list or int): dimensions for which to return values
      derivative (int): the order of the derivative
      extrapolation (int):  0=extrapolated value, 1=return 0, 2=raise a ValueError, 3=boundary value

    Returns:
      array: the values of the spline at the sample points
    """
    return self.get_values(parameter = parameter, points = points, derivative = derivative, extrapolation = extrapolation);
  
 
  def derivative(self, n = 1, dim = None):
    """Returns derivative as a spline
    
    Arguments:
      n (int): order of the derivative
      
    Returns:
      Spline: the derivative of the spline with degree - n 
    """
    
    if n < 0:
        return self.integral(-n, dim = dim);    
    
    if dim is None:
      dim = range(self.ndim);
    if isinstance(dim,int):
      dim = [dim];       
    
    t, c, k = self.tck();
    c = list(c);
    if n > k:
        raise ValueError(("Order of derivative (n = %r) must be <= order of spline (k = %r)") % (n, k));

    with np.errstate(invalid='raise', divide='raise'):
      try:
        for j in range(n):
          dt = t[k+1:-1] - t[1:-k-1]
          for d in dim:
            c[d] = (c[d][1:] - c[d][:-1]) * k / dt;
          t = t[1:-1];
          k -= 1;
      
      except FloatingPointError:
          raise ValueError(("The spline has internal repeated knots and is not differentiable %d times") % n);
    tck = (t, [c[d] for d in dim], k);
    return tck;
    #for d in dim:
    #  tck = splder(self.tck_1d(d), n);
    #  c.append(tck[1][:self.nparameter - n]);
    #tck = (tck[0], c, tck[2]);
    return Curve(tck = tck, points = self.points);
  
        
  def integral(self, n = 1, dim = None):
    """Returns integral / anti-derivative as a spline
    
    Arguments:
      n (int): order of the integral
      
    Returns:
      Spline: the integral of the spline with degree + n  
    """

    if n < 0:
        return self.derivative(-n, dim = dim);

    if dim is None:
      dim = range(self.ndim);
    if isinstance(dim,int):
      dim = [dim];  

    t, c, k = self.tck();
    c = list(c);
    for j in range(n):
        dt = t[k+1:] - t[:-k-1]
        for d in dim:
          c[d] = np.cumsum(c[d] * dt) / (k + 1);
          c[d] = np.r_[0, c[d]];
        
        # New knots
        t = np.r_[t[0], t, t[-1]]
        k += 1
    tck = (t, [c[d] for d in dim], k);

    #c = [];    
    #for d in dim:
    #  tck = splantider(self.tck_1d(d), n);
    #  c.append(tck[1][:self.nparameter + n]);
    #tck = (tck[0], c, tck[2]);
    
    return Curve(tck = tck, points = self.points);
    
  
  def tanget(self):
    """Returns a cruve representing the tanget vector along the curve"""
    der = self.derivative();
    #now normalise
    tgs = der(self.points);
    nrm = np.linalg.norm(tgs, axis = 1);
    tgs = (tgs.T/nrm).T;
    return Curve(values = tgs, points = self.points, knots = self.knots, degree = self.degree);
  
  
  def normal(self):
    """Returns a curve representing the normal vectors along a 2d curve"""  
    if self.ndim != 2:
      raise RuntimeError('normals for non 2d curves not implemented yet');
    der = self.derivative();
    #now normalise
    nrmls = der(self.points);
    nrmls = nrmls[:,[1,0]];
    nrm = np.linalg.norm(nrmls, axis = 1);
    nrmls = (nrmls.T/nrm).T;
    return Curve(values = nrmls, points = self.points, knots = self.knots, degree = self.degree);
  
  
  def phi(self):
    """Returns a Spline representing the tangent angle along a 2d curve"""
    if self.ndim != 2:
      raise RuntimeError('tanget angle can only be computed for 2d curves');
    
    #get the tangents  and tangent angles
    tgs = splev(self.points, self.tck(), der = 1);
    tgs = np.vstack(tgs).T;
    phi = np.arctan2(tgs[:,1], tgs[:,0]);
    phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi;
    #return Spline(phi, points = self.points, knots = self.knots, degree = self.degree - 1);
    tck = splrep(self.points, phi, s = 0.0, k = self.degree + 1);
    return Spline(tck = tck);
  
  
  def theta(self,  with_xy = False, with_orientation = False, reference = 0.5):
    """Returns a Spline representing the derivative of the tangent angle along a 2d curve
    
    Arguments:
      with_position (bool): if True also return absolute position
      with_orientation (bool): if True also return absolute orientation of the curve
      refernece (float): reference point for absolute position and orientation
    """
    if self.ndim != 2:
      raise RuntimeError('tanget angle can only be computed for 2d curves');
    
    #get the tangents  and tangent angles    
    tgs = splev(self.points, self.tck(), der = 1);
    tgs = np.vstack(tgs).T;
    phi = np.arctan2(tgs[:,1],tgs[:,0]);
    phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi;
    #phi = Spline(phi, points = self.points, knots = self.knots, degree = self.degree + 1);
    tck = splrep(self.points, phi, s = 0.0, k = self.degree + 1);
    phi = Spline(tck = tck);
    
    orientation = phi(reference);
    theta = phi.derivative();
    
    if with_xy:
      if with_orientation:
        return  theta, self(reference), orientation
      else:
        return  theta, self(reference)
    else:
      if with_orientation:
        return  theta, orientation
      else:
        return  theta
  
  
  def intersections_with_line(self, point0, point1, with_points = True, with_xy = False, nsamples = None, robust = True):
    """Find positions along the spline that intersect with the line segment between two points
    
    Arguments:
      point1,point2 (2 arrays): points defining the line segment
      with_points (bool): if true return the position of the intersection for the parametrizations of the curve
      with_xy (bool): if true also return the intersection points
      nsamples (int or None): if not None, use this number of intermediate discrete sample points
      robust (bool): robust but slower computation
    
    Returns:
       n array: position of the intersection point on this curve
       n array: position of the intersection point for intersected curve
       nx2 array: points of intersection
    """
    return self.intersections(np.vstack([point0, point1]), with_points=with_points, with_xy=with_xy, nsamples=nsamples, robust=robust);
      
    
  def intersections(self, curve, with_points = True, with_xy = False, nsamples = None, robust = True):
    """Find the intersection between two 2d curves based on discrete sampling of the curve
    
    Arguments:
      curve (Curve): curve to intersect with
      with_points (bool): if true return the position of the intersection for the parametrizations of the curve
      with_xy (bool): if true also return the intersection points
      nsamples (int or None): if not None, use this number of intermediate discrete sample points
      robust (bool): robust but slower computation
    
    Returns:
       n array: position of the intersection point on this curve
       n array: position of the intersection point for intersected curve
       nx2 array: points of intersection
    """
    
    if self.ndim != 2:
      raise RuntimeError('tanget angle can only be computed for 2d curves');
    
    if nsamples is None:
      points = self.points;
    else:
      points = np.linspace(0,1,nsamples);
    
    if isinstance(curve, Curve):
      curve2 = curve(points);
    else:
      curve2 = curve;
    
    curve1 = self(points);
    xy, i, j, di, dj = curve_intersections_discrete(curve1, curve2);
    
    if with_points:
      n1,n2 = curve1.shape[0], curve2.shape[0];
      p1 = points[i] + di/(n1-1);
      p2 = points[j] + dj/(n2-1);
      
      if with_xy:
        return xy, p1, p2;
      else:
        return p1,p2;
    else:
        return xy;
  
    
   
  def plot(self, dim = None, **kwargs):
    """Plot the Curve"""
    
    if self.ndim == 1:
      plt.plot(self.points, self.values[:,0], **kwargs);
    else:
      if dim is None:
        dim = [0,1];
      else:
        dim = np.array([dim], dtype = int);
      if len(dim) > 1:
        plt.plot(self.values[:,dim[0]], self.values[:,dim[1]], **kwargs);
      else:
        plt.plot(self.points, self.values[:,dim[0]], **kwargs);
   
    


def theta_to_curve(theta, xy = [0,0], orientation = 0, reference = 0.5, npoints = None):
  """Construct a 2d curve from the center bending angle
  
  Arguments:
    theta (Spline): center bending angles as spline
    xy (2 array): position as reference point;
    orientation (float): orientation at reference point
    reference (float): reference position along curve
    nsamples (int or None): number of intermediate sample points, if None use npoints
  """
  phi = theta.integral();
  phi0 =  phi(reference);
  phi.parameter += orientation - phi0; # add a constant = add constant to parameter
  phi.values += orientation - phi0;
  
  if npoints is None or npoints is all:
    points = theta.points;
  else:
    points = np.linspace(0,1,npoints);
  
  phi = phi(points);
  tgt = np.vstack([np.cos(phi), np.sin(phi)]).T;
  tgt = Curve(tgt, points = points);
  xyc = tgt.integral();
  xyc0 = xyc(reference);
  xyc.parameter += xy - xyc0;
  xyc.values += xy - xyc0;
  
  return xyc;



def test():
  
  ### Splines
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.interpolate import splrep, splev
  import interpolation.spline as sp;
  reload(sp);

  s = sp.Spline(nparameter = 10, npoints = 141, degree = 3);  
  x = np.sin(10* s.points) + np.sin(2 * s.points);
  s.from_values(x)
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
    xp = s.get_values(pp);
    plt.plot(s.points, xp);
  plt.title('basis functions scipy')
  
  # test spline basis
  
  reload(sp);
  s = sp.Spline(nparameter = 10, npoints = 141, degree = 3);

  bm = s.basis_matrix();
  plt.figure(3); plt.clf();
  plt.plot(bm);
  plt.title('basis functions matrix')
  

  xb = bm.dot(p);
  xp = s.get_values(p);
  
  #xb = np.dot(np.invert(bm), pb);

  plt.figure(4); plt.clf();
  plt.plot(x);
  #plt.plot(0.1* xb);
  plt.plot(xb)
  plt.plot(xp)
  
  # invertible case of npoints = nparameter
  s = sp.Spline(nparameter = 25, npoints = 25, degree = 3);
  x = np.sin(10* s.points) + np.sin(2 * s.points);
  s.from_values(x);
  p = s.parameter;
  
  bm = s.basis;
  pb = np.linalg.inv(bm).dot(x);
  
  plt.figure(5); plt.clf();
  plt.plot(p);
  plt.plot(pb);
  
  xs = s(s.points, p);
  xp = bm.dot(pb);
  
  plt.figure(6); plt.clf();
  plt.plot(x);
  plt.plot(xs);
  plt.plot(xp);
  
  from utils.timer import timeit
  @timeit
  def ts():
    return s(s.points, p);    
  @timeit
  def tb():
    return s.get_values(p);
      
  ps = ts();
  pb = tb();
  np.allclose(pb, ps)
  
  # factor ~10 times faster with basis
  
  # test shifting spline by a value s
  s = sp.Spline(nparameter = 25, npoints = 25, degree = 3);
  x = np.sin(10* s.points) + np.sin(2 * s.points);
  p = s.from_values(x);  
  xnew = s(s.points + 0.05, p);
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


  ### Curves
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.interpolate import splrep, splev, splprep
  import interpolation.spline as sp;
  reload(sp);

  s = np.linspace(0,1,50) * 2 * np.pi;
  xy = np.vstack([np.sin(2*s), np.cos(s)]).T;

  plt.figure(10); plt.clf();
  plt.plot(xy[:,0], xy[:,1])

  reload(sp)
  c = sp.Curve(xy, nparameter = 20);

  c.plot();
  
  sp = c.to_spline(0)
  sp.plot()
  
  sp = c.to_spline(1)
  sp.plot()

  c.parameter.shape
  
  tck = c.tck();
  pts = splev(np.linspace(0.3, 0.5, 10), tck)
  pts = np.vstack(pts).T;
  spts= c(np.linspace(0.3, 0.5, 10))
  np.allclose(pts, spts)
  
  
  # curve intersections
  import numpy as np
  import matplotlib.pyplot as plt;
  import interpolation.spline as sp;
  reload(sp)
  
  s = 2 * np.pi* np.linspace(0,1,50);
  xy1 = np.vstack([np.cos(s), np.sin(2*s)]).T;
  xy2 = np.vstack([0.5 * s, 0.2* s]).T + [-2,-0.5];
  c1 = sp.Curve(xy1);
  c2 = sp.Curve(xy2);
  
  xy0,p1,p2 = c1.intersections(c2, with_xy = True);
  
  plt.figure(1); plt.clf();
  c1.plot();
  c2.plot();
  plt.scatter(xy0[:,0], xy0[:,1], c = 'm', s = 40);
  xy0s = c1(p1);
  plt.scatter(xy0s[:,0], xy0s[:,1], c = 'k', s = 40);
  plt.axis('equal')
  xys = c1();
  plt.scatter(xys[:,0], xys[:,1], c = 'r', s = 40);
  plt.axis('equal') 
  
  # intersection with line segment
  pt0 = [-1.5, -0.5];
  pt1 = [1.5, 1]
  xy0,p1,p2 = c1.intersections_with_line(pt0, pt1, with_xy = True);
  
  plt.figure(1); plt.clf();
  c1.plot();
  xyp = np.vstack([pt0,pt1]);
  plt.plot(xyp[:,0], xyp[:,1])
  plt.scatter(xy0[:,0], xy0[:,1], c = 'm', s = 40);
  plt.axis('equal')
  
  
  # intersections with resampling
  
  s = 2 * np.pi* np.linspace(0,1,25);
  xy1 = np.vstack([np.cos(s), np.sin(2*s)]).T;
  xy2 = np.vstack([0.5 * s, 0.2* s]).T + [-2,-0.5];
  c1 = sp.Curve(xy1);
  c2 = sp.Curve(xy2);
  
  xy0,p1,p2 = c1.intersections(c2, with_xy = True);
  xy0r,p1r,p2r = c1.intersections(c2, with_xy = True, nsamples = 50);
  
  
  plt.figure(1); plt.clf();
  c1.plot();
  c2.plot();
  plt.scatter(xy0[:,0], xy0[:,1], c = 'm', s = 40);
  xy0s = c1(p1);
  plt.scatter(xy0s[:,0], xy0s[:,1], c = 'k', s = 40);
  plt.axis('equal')
  xys = c1(p1r);
  plt.scatter(xys[:,0], xys[:,1], c = 'r', s = 40);
  plt.axis('equal') 
  c1.resample(npoints = 50);
  c1.plot()
  
  
  ### Resampling
  reload(sp);
  s = 2 * np.pi* np.linspace(0,1,25);
  xy1 = np.vstack([np.cos(s), np.sin(2*s)]).T;
  c1 = sp.Curve(xy1);
  c2 = c1.copy();
  c2.resample(npoints = 150);
  c2.values.shape
  
  plt.figure(2); plt.clf();
  c1.plot();
  c2.plot()
  
  ### Uniform Sampling and knotting
  import numpy as np
  import matplotlib.pyplot as plt;
  from scipy.interpolate import splprep
  import interpolation.spline as sp;
  reload(sp)
  
  s = 2 * np.pi* np.linspace(0,1,50);
  xy1 = np.vstack([np.cos(s), np.sin(2*s)]).T;
  c1 = sp.Curve(xy1);
  
  tck = c1.tck();
  tck1,u = splprep(c1.values.T, s = 0)
  c2 = sp.Curve(tck = tck1, points = u);
  c2.resample(npoints = c2.npoints);
  
  #c2 = c1.copy();
  #c2.resample_uniform();
  c3 = c1.copy();
  c3.resample_uniform();
  
  c4 = c2.copy();
  c4.reknot_uniform();
  
  plt.figure(1); plt.clf();
  #plt.subplot(1,2,1);
  c1.plot();
  xy = c1.values;
  plt.scatter(xy[:,0], xy[:,1]);
  #plt.subplot(1,2,2);
  c2.plot();
  xy = c2.values;
  plt.scatter(xy[:,0], xy[:,1]);
  c3.plot();
  xy = c3.values;
  plt.scatter(xy[:,0], xy[:,1]);  
  c4.plot();
  xy = c4.values;
  plt.scatter(xy[:,0], xy[:,1]);  
  
  
  plt.figure(2); plt.clf();
  #plt.plot(c1.points);
  #plt.plot(c2.points);
  plt.plot(np.diff(c1.knots)- np.mean(np.diff(c1.knots)), '+');
  plt.plot(np.diff(c2.knots)- np.mean(np.diff(c1.knots)), 'o');
  plt.plot(np.diff(c3.knots)- np.mean(np.diff(c1.knots)), '.');
  plt.plot(np.diff(c4.knots)- np.mean(np.diff(c1.knots)), 'x');
  
  
  
  ### Theta
  import numpy as np
  import matplotlib.pyplot as plt;
  from scipy.interpolate import splprep, splev, splrep
  import interpolation.spline as sp;
  reload(sp)
  
  
  s = np.linspace(0,1,150);
  xy1 = np.vstack([s, np.sin(2 * np.pi * s)]).T;
  c1 = sp.Curve(xy1, nparameter = 20);
  c1.uniform();
  c1.nparameter  
  
  tck = c1.derivative();
  
  
  plt.figure(78); plt.clf();
  plt.subplot(1,2,1);
  c1.plot(dim=0);
  c1.plot(dim=1);
  c1.plot();
  plt.plot(xy1[:,0], xy1[:,1])
  
  plt.subplot(1,2,2);
  dc.plot(dim=0);
  dc.plot(dim=1);
  
  #plt.plot(xy1[:,0], xy1[:,1])
  
  
  
  #get the tangents  and tangent angles
  #tgs = splev(c1.points, c1.tck(), der = 1);
  #tgs = np.vstack(tgs).T;
  dc = c1.derivative();
  tgs = dc.values;
  phi = np.arctan2(tgs[:,1], tgs[:,0]);
  phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi;
  
  plt.figure(5); plt.clf();
  plt.plot(phi);
  
  #return Spline(phi, points = self.points, knots = self.knots, degree = self.degree - 1);
  tck = splrep(c1.points, phi, s = 0.0, k = c1.degree + 1);
  phi = Spline(tck = tck);  
  
  
  
  
  
  theta, xy, o = c1.theta(with_xy=True, with_orientation= True)
  phi = c1.phi();
  theta2 = phi.derivative();
  
  plt.figure(1); plt.clf();
  plt.subplot(1,3,1);
  c1.plot();
  plt.subplot(1,3,2);
  phi.plot();
  plt.subplot(1,3,3);
  theta.plot(); 
  theta2.plot()
  
  c2 = sp.theta_to_curve(theta);
  
  plt.figure(21); plt.clf();
  c1.plot();
  c2.plot();
  
  orientation = o; reference = 0.5;
  phi = theta.integral();
  p0 = phi(reference);
  phi.parameter += orientation - p0; # add a constant = add constant to parameter
  phi.values += orientation - p0;
  
  phi2 = c1.phi();  
  
  plt.figure(23); plt.clf()
  phi.plot()
  phi2.plot()
  
  points = phi.points;
  phiv = phi(points);
  dt = 1.0 / (points.shape[0] - 1);
  #dt = 1;
  tgtv = dt * np.vstack([np.cos(phiv), np.sin(phiv)]).T;
  xycv = np.cumsum(tgtv, axis = 0)  * np.pi;
  
  plt.figure(100); plt.clf();
  plt.plot(xycv[:,0], xycv[:,1])
  c1.plot();
  
  tgt = sp.Curve(tgtv, points = points);
  xyc = tgt.integral();
  #xyc0 = xyc(reference);
  #xyc.parameter += xy - xyc0;
  #xyc.values += xy - xyc0;
  
  plt.figure(24); plt.clf();
  xyc.plot()
  
  plt.figure(25); plt.clf();
  tgt.plot(dim=0);
  tgt.plot(dim=1)
  
  ### Calculus
  import numpy as np
  import matplotlib.pyplot as plt;
  from scipy.interpolate import splprep, splev, splrep
  import interpolation.spline as sp;
  reload(sp)
  
  
  s = np.linspace(0,2*np.pi,150);
  y = np.sin(s);
  
  s = sp.Spline(values = y, points = s, nparameter=20);
  
  si = s.integral();
  sd = s.derivative();
  
  plt.figure(1); plt.clf();
  s.plot(); si.plot(); sd.plot();

  
  # for curve
  s = np.linspace(0,2*np.pi,130);
  xy = np.vstack([s, np.sin(s)]).T;  
  
  c = sp.Curve(xy, nparameter = 20);
  c.uniform();
  
  ci = c.integral();
  cd = c.derivative();
  
  plt.figure(2); plt.clf();
  plt.subplot(1,3,1);
  c.plot(dim = 0);
  ci.plot(dim = 0);
  cd.plot(dim = 0);
  plt.subplot(1,3,2);
  c.plot(dim = 1);
  ci.plot(dim = 1);
  cd.plot(dim = 1);
  plt.subplot(1,3,3);
  c.plot();
  plt.scatter(c.values[:,0], c.values[:,1], c= 'm', s = 40)
  plt.plot(xy[:,0], xy[:,1])
  plt.axis('equal')
  
  
  
  

if __name__ == "__main__ ":
  test();
  
  
# Phython Cox de Bor Basis
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