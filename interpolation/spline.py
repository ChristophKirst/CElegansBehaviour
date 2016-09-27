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
class Spline(object):
  """Class for handling 1d spline interpolation and conversion between bspline coefficients and splines"""
  
  def __init__(self, values = None, points = None, parameter = None, npoints = None, nparameter = None, degree = 3, knots = None, tck = None, with_projection = False):
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
      with_projection (bool): if True calculate values via projection matrix to speed up calculation
    
    Note:
      The spline is fully defined by the knot vector and the spline coefficients.
      In addition the sample points and projection matrices if requested are cached for speed/convenience
      
      The number of free parameter given the knot vector (excluding the boundary points) is given by:
      nparameter = len(knots) + 2 + 2 * degree - (degree + 1) = len(knots) + degree + 1
    """
    
    # caches
    self._npoints = None;
    self._points = None;
    self._values = None;
    self._projection = None;
    self._projection_inverse = None;
    
    # setup
    self.with_projection = with_projection;
    
    if tck is not None:
      if points is not None:
        self.from_tck(tck, points = points);
      else:
        self.from_tck(tck, points = npoints);
      return;
    
    self.degree = int(degree);

    # initialize sample points
    if values is not None:
      values = np.array(values, dtype = float);
      npointsi = values.shape[0];
    else:
      npointsi = npoints;
    
    #initialize input values
    if parameter is not None:
      parameter = np.array(parameter, dtype = float);
      nparameter = parameter.shape[0];
    
    if knots is not None:
      knots = np.array(knots, dtype = float);
      nparameter = knots.shape[0] + self.degree + 1;
    
    if points is None:   
      if npointsi is None:
        if nparameter is None:
          raise ValueError('cannot determine the number of sample points, values, points, number of points or number of parameter needed to define Spline!');
        npointsi = nparameter;
      pointsi = np.linspace(0,1,npointsi);
    else:
      pointsi = np.array(points, dtype = float);
    npointsi = pointsi.shape[0];
    
    # initialize number of parameter / knots
    if parameter is None:
      if knots is None:
        if nparameter is None:
          nparameter = npointsi;
        knots = np.linspace(pointsi[0], pointsi[-1], nparameter - self.degree + 1)[1:-1];
    else:      
      if knots is None:
        knots = np.linspace(pointsi[0], pointsi[-1], nparameter - self.degree + 1)[1:-1];
      else:
        if nparameter != knots.shape[0] + 1 + self.degree:
          raise ValueError('parameter and knots have inconsistent dimensions %d + 1 + %d != %d' % (knots.shape[0], self.degree, nparameter)); 
    
    #update the knots
    self.set_knots(knots, pointsi[[0,-1]], update_parameter = False);
    
    # initialize parameter and cache values
    if parameter is None:
      if values is None:
        self.set_parameter(np.zeros(nparameter));
        if npoints is not None or points is not None:
          self.set_points(pointsi);
      else:
        self.set_values(values, points = pointsi);
    else:
      if values is not None:
        raise RuntimeWarning('both parameter and values are specified, using parameter to define spline!');
      self.set_parameter(parameter);
      if npoints is not None or points is not None:
        self.set_points(pointsi);
    
  
  def copy(self):
    """Deep copy the Spline"""
    return copy.deepcopy(self);
  
  ############################################################################
  ### Properties 
  
  @property
  def points(self):
    """Sample points of the Spline"""
    return self._points;

  @points.setter
  def points(self, points):
    self.set_points(points);
  
  
  @property
  def npoints(self):
    """Number of sample points of the Spline"""
    return self._npoints;
  
  @npoints.setter
  def npoints(self, npoints):
    self.set_points(npoints);
  
  
  @property
  def values(self):
    """Values of the Spline along the sample points"""
    if self._values is None:
      self._values = self.get_values();
    return self._values;
  
  @values.setter
  def values(self, values):
    if values is None:
      self._values = None;
    else:
      self.set_values(values);
    
  
  @property
  def parameter(self):
    """Parameter of the Spline along the sample points"""
    return self._parameter;
  
  @parameter.setter
  def parameter(self, parameter):
    self.set_parameter(parameter);
  
  
  @property
  def nparameter(self):
    """Number of parameter of the Spline"""
    return self._nparameter;
  
  @nparameter.setter
  def nparameter(self, nparameter):
    self.set_knots(nparameter = nparameter);
  
  
  @property
  def knots(self):
    """Inner knot vector of the Spline"""
    return self._knots;
  
  @knots.setter
  def knots(self, knots):
    if isinstance(knots, int):
      self.set_knots(nparameter = knots + self.degree + 1);
    else:
      self.set_knots(knots = knots);
      
      
  @property
  def knots_all(self):
    """Full knot vector of the Spline"""
    if self._knots_all is None:
      self._knots_all = self.knots
    return self._knots_all;
  
  @knots_all.setter
  def knots_all(self, knots):
    if isinstance(knots, int):
      self.set_knots(nparameter = knots - 2 * self.degree - 2);
    else:
      k = self.degree;
      self.set_knots(knots = knots[k+1:-k-1], points = knots[[0,-1]]);
  
  
  @property
  def projection(self):
    """The projection matrix from parameter to value space"""
    if self._projection is None:
      self._projection = self.projection_matrix();
    return self._projection;
  
  @projection.setter
  def projection(self, projection):
    if projection is None:
      self._projection = None;
    else:
      raise RuntimeError('projection matrix cannot be set!');
  
  
  @property
  def projection_inverse(self):
    """The inverse projection matrix from parameter to value space"""
    if self._projection_inverse is None:
      self._projection_inverse = self.projection_matrix_inverse();        
    return self._projection_inverse;
  
  @projection_inverse.setter
  def projection_inverse(self, projection_inverse):
    if projection_inverse is None:
      self._projection_inverse = None;
    else:
      raise RuntimeError('inverse projection matrix cannot be set!');
  
  
  ############################################################################
  ### Spline Setter
    
  def set_points(self, points):
    """Set the sample points for the spline
    
    Arguments:
      points (array, int or None): set sample points, number of sample points or if None delete points cache;
    """     
    if points is self._points:
      return;
    elif points is None: # delte points
      self._points = None;
      self._npoints = None;
    elif isinstance(points, int):
      if self._points is None:
        self._points = np.linspace(0,1,points);
      else:
        self._points = np.linspace(points[0],points[-1],points);
      self._npoints = points;
    else:
      self._points = np.array(points, dtype = float);
      self._npoints = self._points.shape[0];
    
    # values and projection matrix will need to be updated after change of the sample points
    self._values = None;
    self._projection = None;
    self._projection_inverse = None;
  
  
  def set_points_uniform(self, npoints = None):
    """Set uniformly spaced sample points
    
    Arguments:
      npoints (int or None): number of sample points
    """
    if npoints is None:
      if self._npoints is None:
        raise ValueError('cannot determine number of points!');
      npoints = self._npoints;
    self.set_points(npoints);
  
  
  def set_knots(self, knots = None, points = None, nparameter = None, update_parameter = True):
    """Change the knot vector
    
    Arguments:
      knots (array, int or None): the inner knot vector or number of knots
      points (array, int or None): sample points
      nparameter (int or None): number of parameter if knots is not given
      update_parameter (bool): if True update parameter according to the new knots
    """
    k = self.degree;
    
    if points is not None:
      if len(points) > 2:
        self.set_points(points);
        points = self._points;
    elif points is None:
      if self._points is None:
        points = [0,1];
      else:
        points = self._points
        
    if knots is None:
      if nparameter is None:
        raise ValueError('to set knots sepcify knot vector or number of parameter!');      
      knots = np.linspace(points[0], points[-1], nparameter - k + 1)[1:-1];
    else:
      knots = np.array(knots, dtype = float);
    nparameter = knots.shape[0] + 1 + k;

    # sample old values in order to update new parameter    
    if update_parameter:
      if len(points) == 2: # no internal points and only end points given -> use minimal number of points
        points = np.linspace(points[0], points[1], nparameter);
      values = self.get_values(points);
    
    # set the new knot values
    self._nparameter = nparameter;
    self._knots = knots;
    self._knots_all = np.zeros(self._knots.shape[0] + 2 * k + 2);
    self._knots_all[k+1:-k-1] = self._knots;
    self._knots_all[:k+1] = points[0];
    self._knots_all[-k-1:]= points[-1];
    if not np.all(np.diff(self.knots_all) >= 0):
      raise ValueError('inner knot vector and sample point specification inconsistent!');  
    #y = np.zeros(self.npoints);
    #tck = splrep(self.points, y, t = self.knots, task = -1, k = self.degree);
    #self.knots_all = tck[0];
    
    # values and projection matrix will need to be updated after change of the knots
    self._values = None;
    self._projection = None;
    self._projection_inverse = None;  
    
    # update parameter due to knot vector change 
    if update_parameter:
      self.set_values(values, points);
  
  
  def set_knots_uniform(self, nparameter = None):
    """Reknot to uniformly spaced knots"""
    if nparameter is None:
      nparameter = self.nparameter;
    self.set_knots(nparameter = nparameter);
    
  
  def uniform(self, npoints = None):
    """Make sample points and knots uniformly spaced"""
    self.set_points_uniform(npoints = npoints);
    self.set_knots_uniform();
  
  
  def set_values(self, values, points = None):
    """Calculate the bspline parameter for the given values and sample points 

    Arguments:
      values (array): values of data points
      points (array or None): sample points for the data values, if None use internal points or linspace(0,1,values.shape[0])
    """
    values = np.array(values, dtype = float);
    
    #set points
    if points is None:
      if self._points is None:
        self.set_points(values.shape[0]);
    else:
      self.set_points(points);
    
    if values.shape[0] != self._points.shape[0]:
      raise ValueError('number of values %d mismatch number of points %d' % (values.shape[0], self._points.shape[0]));
    
    #set parameter from values
    if self.with_projection and self.projection_inverse is not False:
      self._parameter = self._projection_inverse.dot(values); 
      self._values = values;  
    else:
      tck = splrep(self._points, values, t = self._knots, task = -1, k = self.degree);
      self._parameter = tck[1][:self._nparameter];
      # values changed
      self._values = None;
      #self._values = values; #fast but imprecise as values on spline due approximation might differ!
  
  
  def set_parameter(self, parameter):
    """Change parameter of the spline
    
    Arguments:
      parameter (array): the new parameter values
    """
    parameter = np.array(parameter, dtype = float);
    if parameter.shape[0] != self._nparameter:
      raise ValueError('length of passed parameter %d mismatch number of internal parameter %d' % (parameter.shape[0], self._nparameter));
    
    self._parameter = parameter;
    
    # parameter changed  -> values changed
    self._values = None;
  
    
  def from_tck(self, tck, points = None):
    """Set spline parameter and knot structure using a tck object returned by splrep
    
    Arguments:
        tck (tuple): t,c,k tuple as returned by splrep
        points (int, array or None): optional sample points pecification
    """
    t,c,k = tck;
    self.degree = k;

    # set knots
    self._knots = t[k+1:-k-1];  
    self._knots_all = t;
    self._nparameter = self._knots.shape[0] + k + 1;
    
    # set parameter
    self._parameter = c[:self._nparameter];
    
    #set points
    if points is not None:
      self.set_points(points);
    
    # values and projection matrix will need to be updated after change of the knots
    self._values = None;
    self._projection = None;
    self._projection_inverse = None;  
     
   
  ############################################################################
  ### Spline getter   
  
  
  def get_points(self, points = None, error = None):
    """Tries to define the sample points with given information
    
    Arguments:
      points (int, array or None): points information
      error (string orNone): if not None generate an error if points could not be defined from point
    
    Returns:
      array or None: the points
    """
    if points is None:
      if self._points is None:
        if error is not None:
          raise ValueError(error);
      return self._points;
    elif isinstance(points, int):
      if self._points is None:
        return np.linspace(0,1,points);
      else:
        return np.linspace(self._points[0],self._points[1],points);
    elif isinstance(points, np.ndarray):
      return points;
    else:
      if error is not None:
        raise ValueError(error);
      else:
        return None;  
  
  
  def get_values(self, points = None, parameter = None, derivative = 0, extrapolation = 1):
    """Calculates the values of the curve along the sample points
    
    Arguments:
      points (array or None): the sample points for the values, if None use internal samples points
      parameter (array or None): the bspline parameter, if None use internal parameter
      derivative (int): the order of the derivative
      extrapolation (int):  0=extrapolated value, 1=return 0, 2=raise a ValueError, 3=boundary value

    Returns:
      array: the values of the spline at the sample points
    """

    points = self.get_points(points, error = 'sample points need to be specified for the calculation of the values of this spline!')      

    if parameter is None:
      parameter = self._parameter;
    
    if points is self._points and derivative == 0:
      if parameter is self._parameter and self._values is not None: #cached version
        return self._values;
      if self.with_projection:
        return self.projection.dot(parameter);
    
    # full interpolation
    pp = np.pad(parameter, (0,self.degree+1), 'constant');
    tck = (self._knots_all, pp, self.degree);
    return splev(points, tck, der = derivative, ext = extrapolation);  
  
  
  def get_points_and_values(self, points = None, parameter = None, derivative = 0, extrapolation = 1, error = None):
    """Calculates the values of the curve along the sample points
    
    Arguments:
      points (array or None): the sample points for the values, if None use internal samples points
      parameter (array or None): the bspline parameter, if None use internal parameter
      derivative (int): the order of the derivative
      extrapolation (int):  0=extrapolated value, 1=return 0, 2=raise a ValueError, 3=boundary value

    Returns:
      array: tha sample points
      array: the values of the spline at the sample points
    """
    points = self.get_points(points, error = error)      

    if parameter is None:
      parameter = self._parameter;
    
    if points is self._points and derivative == 0:
      if parameter is self._parameter and self._values is not None: #cached version
        return points, self._values;
      if self.with_projection:
        return points, self.projection.dot(parameter);
    
    # full interpolation
    pp = np.pad(parameter, (0,self.degree+1), 'constant');
    tck = (self._knots_all, pp, self.degree);
    return points, splev(points, tck, der = derivative, ext = extrapolation);  

    
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
    return self.get_values(points = points, parameter = parameter, derivative = derivative, extrapolation = extrapolation);
  
  
  def projection_matrix(self, points = None, derivative = 0, extrapoltaion = 1):
    """Projection matrix to calucalte spline coefficients via dot product
    
    Arguments:
      points (array or None): projection matrix for this set of sample points (if None use internal points)
      derivative (int): if n>0 the projection matrix of the n-th derivative is returned
      extrapolation (int): projection matrix with 0=extrapolated value, 1=return 0, 3=boundary value
    """
    points = self.get_points(points, error = 'sample points need to be specified for the calculation of the projection matrix!')
    
    projection = np.zeros((self._nparameter, points.shape[0]));
    for i in range(self._nparameter):
      p = np.zeros(self._nparameter);
      p[i] = 1;
      pp = np.pad(p, (0,self.degree+1), 'constant'); # splev wants parameter
      tck = (self.knots_all, pp, self.degree);
      projection[i] = splev(points, tck, der = derivative, ext = extrapoltaion);
    return projection.T; 


  def projection_matrix_inverse(self, points = None, derivative = 0, extrapoltaion = 1):
    """Projection matrix to calucalte spline coefficients via dot product
    
    Arguments:
      points (array or None): projection matrix for this set of sample points (if None use internal points)
      derivative (int): if n>0 the projection matrix of the n-th derivative is returned
      extrapolation (int): projection matrix with 0=extrapolated value, 1=return 0, 3=boundary value
    """
    if (points is None or points is self._points) and derivative == 0 and extrapoltaion == 1:
      projection = self.projection;
    else:
      projection = self.projection_matrix(points = points, derivative = derivative, extrapoltaion = extrapoltaion);
    
    try:
      if projection.shape[0] != projection.shape[1]:
        return False;
      projection_inverse = np.linalg.inv(projection);
    except:
      return False;
    
    return projection_inverse;
  
    
  def tck(self):
    """Returns a tck tuple for use with spline functions
    
    Note:
      This returns a tck tuple as splrep
    """  
    k = self.degree;
    p = np.pad(self._parameter, (0,k+1), 'constant');
    return (self._knots_all, p, k);
  
  
  def tck_ndim(self):
    """Returns a tck tuple for use with multi-dimensional spline functions
    
    Note:
      Returns a tck tuple for use with splprep
    """  
    return (self._knots_all, [self._parameter], self.degree);
    
  
  def to_curve(self):
    """Converts the 1d spline to a 1d curve"""
    return Curve(tck = self.tck_ndim(), points = self.points);
  
  
  ############################################################################
  ### Spline functionality     
  
  def integrate(self, lo, up, points = None, function = None):
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
      points = self._define_points(points, error = 'points need to be specified for calculation of integral of function of spline!');
      values = function(self.get_values(points = points));
      tck = splrep(points, values, t = self._knots, task = -1, k = self.degree);
    
    if isinstance(lo, np.ndarray):
      if len(up) != len(lo):
        raise ValueError('lower and upper bounds expected to have same shape!');
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
    if self.degree != 3:
      raise RuntimeError('finding roots unsupported for non-cubic splines')
    z,m,ier = sproot(*self.tck(), mest = 10);
    if not ier == 0:
      raise RuntimeError("Error code returned by spalde: %s" % ier)
    return z[:m];

  
  def shift(self, shift, points = None, extrapolation = 1):
    """Shift spline
    
    Arguments:
      shift (float): shift
      extrapolation (int): 0=extrapolated value, 1=return 0, 2=raise a ValueError, 3=boundary value
    """
    points = self.get_points(points, error = 'points need to be specified in order to shift spline!');
    values = self.get_values(points + shift, extrapolation = extrapolation);
    self.set_values(values, points);
  
  
  def add(self, value):
    """Adds a constant value to spline
    
    Arguments:
      value: add this value to the spline
    """
    self._parameter += value;
    if self._values is not None:
      self._values += value;
  
      
  def multiply(self, factor):
    """Multiply spline by a factor
            
    Arguments:
      factor: scale pline by this factor
    """
    self._parameter *= factor;
    if self._values is not None:
      self._values *= factor;
  
  ############################################################################
  ### Visualization
    
  def plot(self, points = None, **kwargs):
    """Plot the Spline"""
    points, values = self.get_points_and_values(points, error = 'points need to be specified in order to plot spline!');
    plt.plot(points, values, **kwargs);






class Curve(Spline):
  """Class for handling multi-dimensional spline interpolation and conversion between bspline coefficients and splines"""
  
  def __init__(self, values = None, points = None, parameter = None, npoints = None, nparameter = None, ndim = None, degree = 3, knots = None, tck = None, with_projection = True):
    """Constructor for Curve
    
    Arguments:
      values (dxn array): the values of the curve
      points (array): sample points along the curve parameter, if none linspace(0,1,nsamples)
      parameter (array): parameter of the spline coefficients for each dimension, if None determine from values
      npoints (int): number of sample points alsong the curve, if None len(values) or len(points) or nparameter
      nparameter (int):  number of spline coefficients in each dimension, if None determined from len(parameter), len(values) or len(points)
      ndim (int): dimension of the values
      degree (int): degree of the spline
      knots (array): knot vector (excluding boundary points), if None linspace(points[0],points[-1],nparameter-2*degree)[1:-1]
      tck (tuple): intialize form tck tuple returned by splprep
      with_projection (bool): if True calculate values via projection matrix to speed up calculation
    
    Note:
      The spline is fully defined by the knot vector and the spline coefficients for each dimension.
      In addition the sample points and projection matrices if requested are cached for speed/convenience
      
      The number of free parameter given the knot vector (excluding the boundary points) is given by:
      nparameter = len(knots) + 2 + 2 * degree - (degree + 1) = len(knots) + degree + 1
      for each dimension, the total number of parameter is ndim * nparameter
    """
    
    # caches
    self._npoints = None;
    self._points = None;
    self._values = None;
    self._projection = None;
    self._projection_inverse = None;
    
    # setup
    self.with_projection = with_projection;    
    
    if tck is not None:
      if points is not None:
        self.from_tck(tck, points = points);
      else:
        self.from_tck(tck, points = npoints);
      return;
    
    self.degree = int(degree);
    
    if ndim is None:
      ndim = 1;
    self.ndim = int(ndim);    
    
    #initialize input values
    if values is not None:
      values = np.array(values, dtype = float);
      if values.ndim == 1:
        values = values[:, np.newaxis];
      npointsi = values.shape[0];
      self.ndim = values.shape[1];
      
    if parameter is not None:
      parameter = np.array(parameter, dtype = float);
      if parameter.ndim == 1:
        parameter = parameter[:, np.newaxis];
      nparameter = parameter.shape[0];
      self.ndim = values.shape[1];

    if knots is not None:
      knots = np.array(knots, dtype = float);
      nparameter = knots.shape[0] + self.degree + 1;      
    
    if points is None:   
      if npointsi is None:
        if nparameter is None:
          raise RuntimeError('cannot determine the number of sample points, values, points, number of points or number of parameter needs to be defined!');
        npointsi = nparameter;
      pointsi = np.linspace(0,1,npoints);
    else:
      pointsi = np.array(points, dtype = float);
    npointsi = pointsi.shape[0];
    
    #initialize number of parameter and knots
    if parameter is None:
      if knots is None:
        if nparameter is None:
          #raise RuntimeError('parameter, knot vector or number of parameter needs to be defined!');
          nparameter = npointsi;
        knots = np.linspace(pointsi[0], pointsi[-1], nparameter - self.degree + 1)[1:-1];
    else:      
      if knots is None:
        knots = np.linspace(pointsi[0], pointsi[-1], nparameter - self.degree + 1)[1:-1];
      else:
        if nparameter != knots.shape[0] + 1 + self.degree:
          raise ValueError('parameter and knots have inconsistent dimensions %d + 1 + %d != %d' % (knots.shape[0], self.degree, nparameter)); 
    
    #update the knots
    self.set_knots(knots, pointsi[[0,-1]], update_parameter = False);
    
    # calcualte basis matrix
    self.initialize_basis();       
    
    # initialize parameter and cache values
    if parameter is None:
      if values is None:
        self.set_parameter(np.zeros((nparameter, self.ndim)));
        if npoints is not None or points is not None:
          self.set_points(pointsi);
      else:
        self.set_values(values, points = pointsi);
    else:
      if values is not None:
        raise RuntimeWarning('both parameter and values are specified, using parameter to define spline!');
      self.set_parameter(parameter);
      if npoints is not None or points is not None:
        self.set_points(pointsi);
    
  
  ############################################################################
  ### Spline Setter
  
      
  def set_points_uniform_length(self, npoints = None):
    """Resample the curves base points such that the distances between the curve vectors are uniform
    
    Arguments:
      npoints (int or None): number of sample points    
    """
    if npoints is None:
      if self._npoints is None:
        raise ValueError('cannot determine number of points for uniform sampling!');
      npoints = self._npoints; 
    
    tck,u = splprep(self.values.T, s = 0);
    points = np.linspace(0,1,npoints)
    values = splev(points, tck, der = 0, ext = 1);  
    self.set_values(np.vstack(values).T, points);    
  
  
  def uniform(self, npoints = None):
    """Make sample points and knots uniformly spaced"""
    self.set_points_uniform_length(npoints = npoints);
    self.set_knots_uniform();
    
  
  def set_values(self, values, points = None, dimension = None):
    """Calcualte the bspline parameter for the data points y

    Arguments:
      values (array): values of data points
      points (array or None): sample points for the data values, if None use internal points or linspace(0,1,values.shape[0])
      dimension (int, list or None): the dimension(s) at which to change the curve, if None change dimension to values.shape[0]
    """
    values = np.array(values, dtype = float);
    if values.ndim == 1:
        values = values[:,np.newaxis];
    vdims = range(values.shape[1]);
    
    # determine the dimensions at which to change curve
    if dimension is None:
      pdims = range(values.shape[1]);
      self.ndim = values.shape[1];
    else:
      pdims = np.array(dimension, dtype = int);
    
    if len(vdims) != len(pdims) or len(pdims) != values.shape[1] or max(pdims) > self.ndims:
      raise RuntimeError('inconsistent number of dimensions %d, values %d and parameter %d and curve %d' % (values.shape[1], len(vdims), len(pdims), self.ndims));
    
    #set points
    if points is None:
      if self._points is None:
        self.set_points(values.shape[0]);
    else:
      self.set_points(points);
    
    if values.shape[0] != self._points.shape[0]:
      raise ValueError('number of values %d mismatch number of points %d' % (values.shape[0], self._points.shape[0]));
    
    
    #set parameter from values
    if self.with_projection and self.projection_inverse is not False:
      self._parameter[:,pdims] = self._projection_inverse.dot(values);
      self._values[:,pdims] = values;
    else:
      #tck,u = splprep(values, u = self.points, t = self.knots, task = -1, k = self.degree, s = 0); # splprep crashes due to erros in fitpack
      #for d in range(self.ndim):
      #  self.parameter[d] = tck[1][d];
      #  self.values[d] = self.basis.dot(self.parameter[d]);
      for v,p in zip(vdims, pdims):
        tck = splrep(self.points, values[:,v], t = self.knots, task = -1, k = self.degree);
        self.parameter[:,p] = tck[1][:self.nparameter];
      
      # values will change
      self._values = None;
      #self._values = values; #fast but imprecise as values of spline due approximation might differ!    
  
  
  def set_parameter(self, parameter, dimension = None):
    """Change parameter of the spline
    
    Arguments:
      parameter (array): the new parameter
      dimension (int, list or None): the dimension(s) at which to change the curve, if None change dimension to the dim of values
    """
    parameter = np.array(parameter, dtype = float);
    if parameter.ndim == 1:
        parameter = parameter[:, np.newaxis];
    
    if dimension is None:
      pdims = range(parameter.shape[1]);
      self.ndim = parameter.shape[1];
    else:
      pdims = np.array(dimension, dtype = int);  
      
    if len(pdims) != parameter.shape[0] or max(pdims) > self.ndims:
      raise RuntimeError('inconsistent number of dimensions %d, parameter %d and curve %d' % (parameter.shape[0], len(pdims), self.ndims));
    
    self._parameter[:,pdims] = parameter;
    
    #values need to be updated
    self._values = None;
  
    
  def from_tck(self, tck, points = None):
    """Change spline parameter and knot structure using a tck object returned by splprep or splrep
    
    Arguments:
        tck (tuple): t,c,k tuple returned by splprep
        points (int, array or None): optional sample points pecification
    """
    t,c,k = tck;
    self.degree = k;

    # set knots
    self._knots = t[k+1:-k-1];  
    self._knots_all = t;
    self._nparameter = self._knots.shape[0] + k + 1;
    
    #set parameter
    if isinstance(c, list):
      c = np.vstack(c);
    elif isinstance(c, np.ndarray) and c.ndim == 1:
      c = np.vstack([c]);
    c = c[:,:self._nparameter].T;
    
    self.ndim = c.shape[1];    
    self._parameter = c[:];
    
    #set points    
    if points is not None:
      self.set_points(points);
    
    # values and projection matrix will need to be updated after change of the knots
    self._values = None;
    self._projection = None;
    self._projection_inverse = None;  
      
  
  ############################################################################
  ### Spline getter   

  def get_values(self, points = None, parameter = None, dimension = None, derivative = 0, extrapolation = 1):
    """Calculates the values of the curve along the sample points
    
    Arguments:
      points (array or None): the sample points for the values, if None use internal samples points
      parameter (array or None): the bspline parameter, if None use internal parameter
      dimensions (None, list or int): the dimension(s) for which to return values
      derivative (int): the order of the derivative
      extrapolation (int):  0=extrapolated value, 1=return 0, 2=raise a ValueError, 3=boundary value

    Returns:
      array: the values of the spline at the sample points
    """
    
    if dimension is None:
      dimension = range(self.ndim);
    if isinstance(dimension,int):
      dimension = [dimension];   
    
    points = self.get_points(points, error = 'sample points need to be specified for the calculation of the values of this curve!')   

    if parameter is None:
      parameter = self._parameter[:,dimension];
    
    if points is self._points and derivative == 0:
      if parameter is self._parameter and self._values is not None: #cached version
        return self._values[:,dimension];
      if self.with_projection:
        return self.projection.dot(parameter);
    
    # full interpolation
    tck = (self._knots_all, parameter.T, self.degree);
    values = splev(points, tck, der = derivative, ext = extrapolation);  
    return np.vstack(values).T
    
    
  def get_points_and_values(self, points = None, parameter = None, dimension = None, derivative = 0, extrapolation = 1):
    """Calculates the values of the curve along the sample points
    
    Arguments:
      points (array or None): the sample points for the values, if None use internal samples points
      parameter (array or None): the bspline parameter, if None use internal parameter
      dimensions (None, list or int): the dimension(s) for which to return values
      derivative (int): the order of the derivative
      extrapolation (int):  0=extrapolated value, 1=return 0, 2=raise a ValueError, 3=boundary value

    Returns:
      array: tha sample points
      array: the values of the spline at the sample points
    """
    
    if dimension is None:
      dimension = range(self.ndim);
    if isinstance(dimension,int):
      dimension = [dimension];   
    
    points = self.get_points(points, error = 'sample points need to be specified for the calculation of the values of this curve!')   

    if parameter is None:
      parameter = self._parameter[:,dimension];
    
    if points is self._points and derivative == 0:
      if parameter is self._parameter and self._values is not None: #cached version
        return points, self._values[:,dimension];
      if self.with_projection:
        return points, self.projection.dot(parameter);
    
    # full interpolation
    tck = (self._knots_all, parameter.T, self.degree);
    values = splev(points, tck, der = derivative, ext = extrapolation);  
    return points, np.vstack(values).T    
    
    
  def __call__(self,  points = None, parameter = None, dimension = None, derivative = 0, extrapolation = 1):
    """Calculates the values of the curve along the sample points
    
    Arguments:
      points (array or None): the sample points for the values, if None use internal samples points
      parameter (array or None): the bspline parameter, if None use internal parameter
      dimensions (None, list or int): the dimension(s) for which to return values
      derivative (int): the order of the derivative
      extrapolation (int):  0=extrapolated value, 1=return 0, 2=raise a ValueError, 3=boundary value

    Returns:
      array: the values of the spline at the sample points
    """
    return self.get_values(parameter = parameter, points = points, dimension = dimension, derivative = derivative, extrapolation = extrapolation); 
  
  
  def tck(self, dimension = None):
    """Returns a tck tuple for use with spline functions
    
    Arguments:
      dimension (None, list or int): dimension(s) for which to return tck tuple, if None return for all
      
    Returns:
      tuple: tck couple as returned by splprep
    """
    if dimension is None:
      return (self._knots_all, self._parameter.T, self.degree);
    else:
      if isinstance(dimension,int):
        dimension = [dimension];
      return (self._knots_all, self._parameter[:,dimension].T, self.degree);
  
  
  def tck_1d(self, dimension = 0):
    """Returns tck tuple for use with 1d spline functions
    
    Arguments:
      dimension (int): dimension for which to return tck tuple
      
    Returns:
      tuple: tck couple as retyurned by splrep
    """
    k = self.degree;
    p = np.pad(self._parameter[:,dimension], (0,k+1), 'constant');
    return (self._knots_all, p, k);
  
  
  def to_spline(self, dimension = 0):
    """Returns a Spline for a specific dimension
    
    Arguments:
      dimension (int): dimension for which to return Spline
      
    Returns:
      Spline: Spline along the indicated dimension
    """
    return Spline(tck = self.tck_1d(dimension), points = self.points);
  
  
  def to_splines(self, dimension = None):
    """Returns list of splines for each dimension
    
    Arguments:
      dimension (None, list or int): dimensions for which to return Splines
    
    Returns:
      list: list of Splines along the indicated dimension
    """
    if dimension is None:
      dimension = range(self.ndim);
    if isinstance(dimension,int):
      dimension = [dimension];
    
    return [self.to_spline(d) for d in dimension];
  
  
  ############################################################################
  ### Spline functionality       

  
  def length(self, points = None) :
    """Returns length of the curve estimated from line segments"""
    d = np.diff(self.get_values(points = points), axis = 0);
    return np.sum(np.sqrt(np.sum(d*d, axis = 1)));
  
    
  def displace(self, value):
    """Shift curve by a vector
        
    Arguments:
      value: displace curve by this vector 
    """
    self._parameter += value;
    if self._values is not None:
      self._values += value;
  
  
  def scale(self, factor):
    """Scale cruve by a factor
            
    Arguments:
      factor: scale curve by this factor
    """
    self._parameter *= factor;
    if self._values is not None:
      self._values *= factor;
  
  
  def derivative(self, n = 1, dimension = None):
    """Returns derivative of the coordinates as a curve
    
    Arguments:
      n (int): order of the derivative
      dimension (None, list or int): dimension(s) for which to return tck tuple, if None return all
      
    Returns:
      Curve: the derivative of the curve with degree - n 
    """
    if n < 0:
      return self.integral(-n, dimension = dimension);    
    
    if dimension is None:
      dimension = range(self.ndim);
    if isinstance(dimension,int):
      dimension = [dimension];       
    
    #tck of the derivative
    t, c, k = self.tck();
    c = list(c);
    if n > k:
        raise ValueError(("Order of derivative (n = %r) must be <= order of spline (k = %r)") % (n, k));
    with np.errstate(invalid='raise', divide='raise'):
      try:
        for j in range(n):
          dt = t[k+1:-1] - t[1:-k-1]
          for d in dimension:
            c[d] = (c[d][1:] - c[d][:-1]) * k / dt;
          t = t[1:-1];
          k -= 1;
      
      except FloatingPointError:
          raise ValueError(("The spline has internal repeated knots and is not differentiable %d times") % n);
    tck = (t, [c[d] for d in dimension], k);
    
    ##alternative
    #c = [];
    #for d in dim:
    #  tck = splder(self.tck_1d(d), n);
    #  c.append(tck[1][:self.nparameter - n]);
    #tck = (tck[0], c, tck[2]);
    
    return Curve(tck = tck, points = self.points);
  
        
  def integral(self, n = 1, dimension = None):
    """Returns integral / anti-derivative of the coordinates as a curve
    
    Arguments:
      n (int): order of the integral
      dimension (None, list or int): dimension(s) for which to return tck tuple, if None return all
      
    Returns:
      Curve: the integral of the curve with degree + n  
    """
    if n < 0:
        return self.derivative(-n, dimension = dimension);

    if dimension is None:
      dimension = range(self.ndim);
    if isinstance(dimension,int):
      dimension = [dimension];  

    #tck for derivative
    t, c, k = self.tck();
    c = list(c);
    for j in range(n):
        dt = t[k+1:] - t[:-k-1]
        for d in dimension:
          c[d] = np.cumsum(c[d] * dt) / (k + 1);
          c[d] = np.r_[0, c[d]];
        t = np.r_[t[0], t, t[-1]]
        k += 1
    tck = (t, [c[d] for d in dimension], k);
    
    ##alternative
    #c = [];    
    #for d in dim:
    #  tck = splantider(self.tck_1d(d), n);
    #  c.append(tck[1][:self.nparameter + n]);
    #tck = (tck[0], c, tck[2]);
    
    return Curve(tck = tck, points = self.points);
    
  
  def tanget(self, points = None):
    """Returns a cruve representing the tanget vector along the curve
    
    Arguments:
      points (int, array or None): sample points used to determine the tangets
        
    Returns:
      Curve: curve representing the tangents
    """
    der = self.derivative();
    #now normalise
    points = self.get_points(points, error = 'cannot determine sample points needed for the calculation of the tangets');
    tgs = der(points);
    nrm = np.linalg.norm(tgs, axis = 1);
    tgs = (tgs.T/nrm).T;
    return Curve(values = tgs, points = points, knots = self._knots, degree = self.degree);
  
  
  def normal(self, points = None):
    """Returns a curve representing the normal vectors along a 2d curve
        
    Arguments:
      points (int, array or None): sample points used to determine the normals
    
    Returns:
      Curve: curve representing the normals
    """
    if self.ndim != 2:
      raise RuntimeError('normals for non 2d curves not implemented yet');
    der = self.derivative();
    #now normalise
    points = self.get_points(points, error = 'cannot determine sample points needed for the calculation of the normals');
    nrmls = der(points);
    nrmls = nrmls[:,[1,0]];
    nrm = np.linalg.norm(nrmls, axis = 1);
    nrmls = (nrmls.T/nrm).T;
    return Curve(values = nrmls, points = points, knots = self._knots, degree = self.degree);
  
  
  def phi(self, points = None):
    """Returns a Spline representing the tangent angle along a 2d curve
    
    Arguments:
      points (int, array or None): sample points used to determine the phi
    
    Returns:
      Spline: spline of phi
    """
    if self.ndim != 2:
      raise RuntimeError('phi angle can only be computed for 2d curves');
    
    points = self.get_points(points, error = 'cannot determine sample points needed for the calculation of phi');    
    
    #get the tangents  and tangent angles
    tgs = splev(points, self.tck(), der = 1);
    tgs = np.vstack(tgs).T;
    phi = np.arctan2(tgs[:,1], tgs[:,0]);
    phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi;
    #return Spline(phi, points = self.points, knots = self.knots, degree = self.degree - 1);
    tck = splrep(points, phi, s = 0.0, k = self.degree + 1);
    return Spline(tck = tck, points = points);
  
  
  def theta(self, points = None, with_xy = True, with_length = True, with_orientation = True, reference = 0.5):
    """Returns a Spline representing the derivative of the tangent angle along a 2d curve
    
    Arguments:
      points (int, array or None): sample points used to determine theta
      with_lenth (bool): if True also return length of the curve
      with_position (bool): if True also return absolute position
      with_orientation (bool): if True also return absolute orientation of the curve
      reference (float): reference point for absolute position and orientation
    
    Returns:
      Spline: spline of theta
      
    Note:
      To fully reconstruct the curve, the center point, length and orientation is needed.
    """
    if self.ndim != 2:
      raise RuntimeError('theta angle can only be computed for 2d curves');
      
    points = self.get_points(points, error = 'cannot determine sample points needed for the calculation of theta');    
    
    #get the tangents  and tangent angles    
    tgs = splev(points, self.tck(), der = 1);
    tgs = np.vstack(tgs).T;
    phi = np.arctan2(tgs[:,1],tgs[:,0]);
    phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi;
    #phi = Spline(phi, points = self.points, knots = self.knots, degree = self.degree + 1);
    tck = splrep(points, phi, s = 0.0, k = self.degree + 1);
    phi = Spline(tck = tck);
    
    orientation = phi(reference);
    theta = phi.derivative();

    rtr = [theta];
    if with_xy:
      rtr.append(self(reference));      
    if with_length:
      rtr.append(self.length());       
    if with_orientation:
      rtr.append(orientation);
    return tuple(rtr);
  
  
  def intersections_with_line(self, point0, point1, with_points = True, with_xy = False, points = None, robust = True):
    """Find positions along the spline that intersect with the line segment between two points
    
    Arguments:
      point1,point2 (2 arrays): points defining the line segment
      with_points (bool): if true return the position of the intersection for the parametrizations of the curve
      with_xy (bool): if true also return the intersection points
      points (int, array or None): sample points used to determine the intersections
      robust (bool): robust but slower computation
    
    Returns:
       n array: position of the intersection point on this curve
       n array: position of the intersection point for intersected curve
       nx2 array: points of intersection
    """
    return self.intersections(np.vstack([point0, point1]), with_points=with_points, with_xy=with_xy, nsamples=nsamples, robust=robust);
      
    
  def intersections(self, curve, with_points = True, with_xy = False, points = None, robust = True):
    """Find the intersection between two 2d curves based on discrete sampling of the curve
    
    Arguments:
      curve (Curve): curve to intersect with
      with_points (bool): if true return the position of the intersection for the parametrizations of the curve
      with_xy (bool): if true also return the intersection points
      p (int or None): if not None, use this number of intermediate discrete sample points
      points (int, array or None): sample points used to determine the intersections
      robust (bool): robust but slower computation
    
    Returns:
       n array: position of the intersection point on this curve
       n array: position of the intersection point for intersected curve
       nx2 array: points of intersection
    """
    
    if self.ndim != 2:
      raise RuntimeError('tanget angle can only be computed for 2d curves');
    
    points = self.get_points(points, error = 'cannot determine sample points needed for the calculation of the intersections')
    
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
        return p1, p2;
    else:
        return xy;
  
        
  ############################################################################
  ### Visualization
  
  
  def plot(self, points = None, dimension = None, with_points = True, **kwargs):
    """Plot the Curve
    
    Arugments:    
      dimension (int, tuple or None): dimensins which should be plotted
      with_points (bool): plot curve indicating points
    """
    
    if self.ndim == 1:
      x,y = self.get_points_and_values(points = points, dimension = 0);
    else:
      if dimension is None:
        dimension = [0,1];
      else:
        dimension = np.array([dimension], dtype = int);
      if len(dimension) > 1:
        x = self.get_values(points = points, dimension = dimension);
        x,y = tuple(x.T);
      else:
        x,y = self.get_points_and_values(points = points, dimension = dimension[0]);
      
    plt.plot(x,y, **kwargs);
    if with_points:
      plt.scatter(x,y, s = 60);
   
    


def theta_to_curve(theta, length = 1, xy = [0,0], orientation = 0, reference = 0.5, npoints = None):
  """Construct a 2d curve from the center bending angle
  
  Arguments:
    theta (Spline): center bending angles as spline
    xy (2 array): position as reference point;
    orientation (float): orientation at reference point
    reference (float): reference position along curve
    nsamples (int or None): number of intermediate sample points, if None use npoints
  """
  points = theta.get_points(npoints, error = 'to convert theta into a curve number of points needs to be specified');  

  #integrate  
  phi = theta.integral();
  phi.add(orientation - phi(reference));
  
  #sample tangets
  phi = phi(points);
  tgt = Curve(length * np.vstack([np.cos(phi), np.sin(phi)]).T, points = points, knots = phi.knots);
  
  #integrate tangets to curve
  xyc = tgt.integral();
  xyc.add(xy - xyc(reference));
  
  return xyc;


def test():
  
  ### Splines
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.interpolate import splrep, splev
  import interpolation.spline as sp;
  reload(sp);

  s = sp.Spline(nparameter = 10, degree = 3);  

  s = sp.Spline(nparameter = 10, npoints = 141, degree = 3);  
  x = np.sin(10* s.points) + np.sin(2 * s.points);
  s.values = x;
  
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
  #from scipy.interpolate import splprep
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