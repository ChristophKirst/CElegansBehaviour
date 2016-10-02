# -*- coding: utf-8 -*-
"""
Discrete function

This module provides a basic class to handle functions represented as discrete points
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'


import numpy as np
import matplotlib.pyplot as plt
import copy

from scipy.interpolate import splrep, splev, splint, splder, splantider, sproot;


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
        self._points = np.linspace(self._points [0],self._points [-1],points);
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
    
  
  def set_parameter_from_values(self, values, points = None):
    """Change the parameter to represent the values and resmaple on current sample points if necessary
    
    Arguments: 
      values (array): values
      points (array or None): sample points for the values
    """
    
    if points is None:
      self.set_values(values);
    else:
      if points is self._points or (self._points is not None and self._points.shape[0] == points.shape[0] and np.allclose(points, self._points)):
        self.set_values(values);
      else:
        tck = splrep(points, values, t = self._knots, task = -1, k = self.degree);
        self._parameter = tck[1][:self._nparameter];
        # values changed
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
    elif isinstance(points, float):
      return np.array([points]);
    elif isinstance(points, np.ndarray):
      return points;
    else:
      if error is not None:
        raise ValueError(error);
      else:
        return None;  
  
  
  def get_values(self, points = None, parameter = None, derivative = 0, extrapolation = 1):
    """Calculates the values of the spline along the sample points
    
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
    """Calculates the values of the spline along the sample points
    
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
    """Calculates the values of the spline along the sample points
    
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
    
  
  #def to_curve(self):
  #  """Converts the 1d spline to a 1d curve"""
  #  return Curve(tck = self.tck_ndim(), points = self.points);
  
  
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
      points = self.get_points(points = points, error = 'points need to be specified in order to integrate a function along the spline!')
      values = function(self.get_values(points = points));
      #tck = splrep(points, values, t = self._knots, task = -1, k = self.degree);
      tck = splrep(points, values, k = self.degree);
      
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