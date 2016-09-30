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


from interpolation.spline import Spline

from scipy.interpolate import splrep, splev, splprep;

from interpolation.intersections import curve_intersections_discrete;

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
    else:
      npointsi = npoints;
      
      
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
      pointsi = np.linspace(0,1,npointsi);
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
    
    # initialize parameter and cache values
    if parameter is None:
      if values is None:
        self.set_parameter(np.zeros((nparameter, self.ndim)));
        if npoints is not None or points is not None:
          self.set_points(pointsi);
      else:
        self._parameter = np.zeros((nparameter, self.ndim));
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
    
    if len(vdims) != len(pdims) or len(pdims) != values.shape[1] or max(pdims) > self.ndim:
      raise RuntimeError('inconsistent number of dimensions %d, values %d and parameter %d and curve %d' % (values.shape[1], len(vdims), len(pdims), self.ndim));
    
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
      #self._values[:,pdims] = values;
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
      
    if len(pdims) != parameter.shape[0] or max(pdims) > self.ndim:
      raise RuntimeError('inconsistent number of dimensions %d, parameter %d and curve %d' % (parameter.shape[0], len(pdims), self.ndim));
    
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

  
  def tanget(self, points = None, normalize = True):
    """Returns a cruve representing the tanget vector along the curve
    
    Arguments:
      points (int, array or None): sample points used to determine the tangets
      normalize (bool): if true normalize tangent vectors to unit length
        
    Returns:
      Curve: curve representing the tangents
    """
    
    der = self.derivative();
    if normalize:
      #now normalise
      points = self.get_points(points, error = 'cannot determine sample points needed for the calculation of the tangets');
      tgs = der(points);
      nrm = np.linalg.norm(tgs, axis = 1);
      tgs = (tgs.T/nrm).T;
      return Curve(values = tgs, points = points, knots = self._knots, degree = self.degree);
    else:
      return der;
  
  
  def normal(self, points = None, normalize = True):
    """Returns a curve representing the normal vectors along a 2d curve
        
    Arguments:
      points (int, array or None): sample points used to determine the normals
      normalize (bool): if true normalize normal vectors to unit length
    
    Returns:
      Curve: curve representing the normals
    """
    if self.ndim != 2:
      raise RuntimeError('normals for non 2d curves not implemented yet');
    der = self.derivative();
    
    if not normalize:
      der.parameter[:,[1,0]];
      der.scale([-1,1]);
      return der;
    else:
      #normalise
      points = self.get_points(points, error = 'cannot determine sample points needed for the calculation of the normals');
      nrmls = der(points);
      nrmls = nrmls[:,[1,0]];
      nrmls = [-1,1] * nrmls;
      nrm = np.linalg.norm(nrmls, axis = 1);
      nrmls = (nrmls.T/nrm).T;
      return Curve(values = nrmls, points = points, knots = self._knots, degree = self.degree);
  
  
  def normals(self, points = None, normalize = True):
    """Returns a normals along the smaple points
        
    Arguments:
      points (int, array or None): sample points used to determine the normals
      normalize (bool): if true normalize normal vectors to unit length
    
    Returns:
      Curve: curve representing the normals
    """
    if self.ndim != 2:
      raise RuntimeError('normals for non 2d curves not implemented yet');
    der = self.derivative();    
    
    #normalise
    points = self.get_points(points, error = 'cannot determine sample points needed for the calculation of the normals');
    nrmls = der(points);
    nrmls = nrmls[:,[1,0]];
    nrmls = [-1,1] * nrmls;
    nrm = np.linalg.norm(nrmls, axis = 1);
    return (nrmls.T/nrm).T;

  
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
    return self.intersections(np.vstack([point0, point1]), with_points=with_points, with_xy=with_xy, points=points, robust=robust);
      
    
  def intersections(self, curve, with_xy = False, with_indices = True, with_points = True, points = None, robust = True):
    """Find the intersection between two 2d curves based on discrete sampling of the curve
    
    Arguments:
      curve (Curve): curve to intersect with
      with_points (bool): if true return the position of the intersection for the parametrizations of the curve
      with_xy (bool): if True also return the intersection points on the contour
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
    
    res = [];
    if with_xy:
      res.append(xy);
    if with_indices:
      res.append(i);
      res.append(j);
    if with_points:
      n1,n2 = curve1.shape[0], curve2.shape[0];
      res.append(points[i] + di/(n1-1));
      res.append(points[j] + dj/(n2-1));
    
    return tuple(res);
  
        
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
   
    


def theta_to_curve(theta, length = 1, xy = [0,0], orientation = 0, reference = 0.5, points = None):
  """Construct a 2d curve from the center bending angle
  
  Arguments:
    theta (Spline): center bending angles as spline
    xy (2 array): position as reference point;
    orientation (float): orientation at reference point
    reference (float): reference position along curve
    points (int, array or None): intermediate sample points, if None use points in theta
  """
  points = theta.get_points(points, error = 'to convert theta into a curve sample points or number of sample points need to be specified');  

  #integrate  
  phi = theta.integral();
  phi.add(orientation - phi(reference));
  
  #sample tangets
  knots = phi.knots;
  phi = phi(points);
  tgt = Curve(length * np.vstack([np.cos(phi), np.sin(phi)]).T, points = points, knots = knots);
  
  #integrate tangets to curve
  xyc = tgt.integral();
  xyc.add(xy - xyc(reference));
  
  return xyc;



def test():
  """Test Curve class"""
  
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
  from scipy.interpolate import splprep, splev #, splrep
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
  
  
  dc = c1.derivative();
  plt.subplot(1,2,2);
  dc.plot(dim=0);
  dc.plot(dim=1);
  
  #plt.plot(xy1[:,0], xy1[:,1])
  
  #get the tangents  and tangent angles
  #tgs = splev(c1.points, c1.tck(), der = 1);
  #tgs = np.vstack(tgs).T;

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
  