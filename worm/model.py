# -*- coding: utf-8 -*-
"""
Module to model Worm shapes and movements

The WormModel class models the main features of the worm shape and
does inference of the worm shape from images or movies

The worm is parameterized by:

  * xy          - position of the worm (the center of the mid line)
  * orientation - absolute orientation of the worm
  * length      - length of the center line
  * theta       - parameters for the bending of the center line
  * width       - parameters width profile along that center line

The bending and width profiles are parameterized via specialized classes,
handling e.g. b spline calculations

Note:
  The number of variable parameter during tracking / shape detection is the number of control points
  for the bending (len(b)) + for the position (2) + actual velocity (1) and optionally speeds for
  additional bending modes (k)
  
  The bend of the center line is parameterized by a function
  .. math::
    \\theta(s), s \\in [0,1]
  
  The angle of the tangets along the center line is the given by
  
  .. math:
    \\phi(s) = \\int_0.5^s \\theta(s) ds + \\alpha
    
  where :math:`\\alpha`  is the overall orientation of the worm.
  
  The worms center line is
  
  .. math:
    x(s) = \\int_0.5^s \\cos(\\phi(s)) ds + x_0;
    y(s) = \\int_0.5^s \\sin(\\phi(s)) ds + y_0;
    
  where :math:`x_0,y_0` is the center position of the worm.
  
  In this way the parameter for the worm pose are independent of the position and
  orientation
    
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import numpy as np
import matplotlib.pyplot as plt
#import copy
import cv2
#import scipy.ndimage as nd
#from scipy.spatial.distance import cdist
from scipy.interpolate import splrep, splprep, splev

import worm.geometry as wormgeo
from interpolation.curve import Curve
from interpolation.spline import Spline
from interpolation.resampling import resample as resample_curve


from imageprocessing.masking import mask_to_phi #, curvature_from_phi
#from utils.utils import isnumber, eps


class WormModel(object):
  """Class modeling the shape and posture and motion of a worm"""
  
  def __init__(self, length = 40, xy = [75, 75], orientation = 0,
               theta = None, width = None, npoints = 21):
    """Constructor of WormModel
    
    Arguments:
      length (number): length of the center line of the worm [in pixel]
      xy (2 array): position of the center of the center line [in pixel]
      theta (array, Curve or None): center bending angle (if None non bend worm)
      width (array, Curve or None): width profile (if none use default profile)
      npoints (int): number of sample points if theta not defined
      
    Note:
      The number of samples form the theta curve is nsamples - 2
    """
    
    self.length = float(length);
    self.xy = np.array(xy, dtype = float);
    self.orientation = float(orientation);
    self.npoints = int(npoints);
    
    
    if theta is None:
      self.theta = Spline(npoints = self.npoints - 2, nparameter = 10, degree = 3);
    elif isinstance(theta, Curve):
      self.theta = theta;
    else:
      self.theta = Spline(values = theta, npoints = 21, nparameter = 10, degree = 3);
    
    self.npoints = self.theta.npoints + 2;    
    
    if width is None:
      width = self.default_width();
      self.width = Spline(values = width, npoints = self.npoints, nparameter = 10, degree = 3);
    elif isinstance(width, Curve):
      self.width = width;
    else:
      self.width = Spline(values = width, nparameter = 10, degree = 3);
    
    assert self.npoints == self.width.npoints;      

    self.nparameter = self.theta.nparameter + 3; #bend, position and orientation
    self.nparameter_full = self.nparameter + 1 + self.width.nparameter + 1; # length + width profile + speed [theta speeds / turning speeds]
    
    self.speed = 0.0;
    
  
  ############################################################################
  ### Properties
    
  def get_parameter(self, full = False):
    """Parameter of the worm shape"""
    if full:
      return np.hstack([self.theta.parameter, self.xy, self.orientation, self.length, self.width.parameter, self.speed]) #self.speed, self.w, self.theta_speed
    else:
      return np.hstack([self.theta.parameter, self.xy, self.orientation]) #self.speed, self.w, self.theta_speed

  
  def set_parameter(self, parameter):
    """Set the parameter"""    
    istr = 0; iend = self.bending.nparameter;
    self.bending.from_parameter(parameter[istr:iend]);
    istr, iend = iend, iend + 2;
    self.xy = parameter[istr,iend];
    istr, iend = iend, iend + 1;
    self.orientation = parameter[istr,iend];      
    if len(parameter) > self.nparameter:
      istr, iend = iend, iend + 1;
      self.length = parameter[istr,iend];
      istr, iend = iend, iend + self.width.nparameter;
      self.width.from_parameter(parameter[istr:iend]);
      istr, iend = iend, iend + 1;
      self.speed = parameter[istr:iend];
    self.update_phi();
  
  
  ############################################################################
  ### Constructors
  
  def default_width(self):
    """Initial guess for worm width
    
    Note: fit from data, adjust to data / age etc
    """
    def w(x):
      a = 9.56 * 0.5;
      b = 0.351;
      return a * np.power(x,b)*np.power(1-x, b) * np.power(0.5, -2*b);
    
    self.width = w(np.linspace(0, 1, self.npoints));
    return self.width;


  def from_shape(self, left, right, center = None, width = None, nsamples = all, nneighbours = all):
    """Initialize worm model from the left and right border lines
    
    Arguments:
      left (nx2 array): points of left border line
      right (nx2 array): points of right border line
      center (nx2 array): optional center line
      width (n array): optional width of the worm
      nsamples (int): number of intermediate sample point
    """
    
    # get center line and width profile
    if width is None:
      with_width = True;
    else:
      with_width = False;
    
    if center is None or width is None:
      cw = wormgeo.center_from_sides(left, right, with_width = with_width, 
                                     nsamples = nsamples, npoints = nsamples, resample = False, 
                                     smooth=0, nneighbours=nneighbours, 
                                     method='projection');
    if width is None:
      center, width = cw;
    else:
      center = cw;
    
    theta, orientation, xy, length = wormgeo.theta_from_center(center, npoints = self.theta.npoints, 
                                                               nsamples = nsamples, resample=False, smooth=0);
    
    self.theta.from_values(theta);
    self.width.from_values(width);
    self.xy = xy;
    self.orientation = orientation;
    self.length = length;
  
  
  def from_center(self, center, width = None, nsamples = all):
    """Initialize worm model from center line and width
    
    Arguments:
      center_line (nx2 array): points of center line
      width (array or None): width of worm at reference points, if None use initial guess
      nsamples (int or all): number of intermediate samples points
    """
    
    theta, orientation, xy, length = wormgeo.theta_from_center(center, npoints = self.theta.npoints, 
                                                               nsamples = nsamples, resample=False, smooth=0);
    
    self.theta.from_values(theta);
    self.xy = xy;
    self.orientation = orientation;
    self.length = length;    
        
    if width is not None:
      self.width.from_values(width);

    
  def from_image(self, image, sigma = 1, absolute_threshold = None, threshold_factor = 0.95, 
                       npoints_contour = 100, delta = 0.3, smooth = 1.0,
                       verbose = False, save = None):
    
    success, center, left, right, width = wormgeo.shape_from_image(image, 
                             sigma = sigma, absolute_threshold = absolute_threshold,
                             threshold_factor = threshold_factor, npoints_contour = npoints_contour, 
                             delta = delta, smooth = smooth,
                             npoints = self.npoints, 
                             verbose = verbose, save = save);
    
    if success:
      #self.from_lines(shape[1], shape[2], shape[3]);
      self.from_center(center, width);
    else:
      raise RuntimeWarning('failed inferring worm from image');
  
  ############################################################################
  ### Shape Properties 
      
  def center(self, npoints = all, with_normals = False):
    """Returns center line of the worm
    
    Arguments:
      npoints (int or all): number of sample points
      
    Returns:
      array (nx2): points along center line
    """
    
    if npoints is all:
      points = all;
    else:
      points = np.linspace(0,1,npoints);
    
    theta = self.theta.get_values(points = points);
    return wormgeo.center_from_theta(theta, orientation = self.orientation,
                                     xy = self.xy, length = self.length,
                                     npoints = npoints, nsamples = all, resample = False,
                                     smooth = 0, with_normals = with_normals);
 
  
  def shape(self, npoints = all, with_center = False, with_normals = False):
    """Returns left and right side and center line of the worm
    
    Arguments:
      npoints (int or all): number of sample points
      
    Returns:
      array (nx2): points along left side
      array (nx2): points along right side
      array (nx2): points along center line
    """
    
    if npoints is all:
      points = all;
    else:
      points = np.linspace(0,1,npoints);    
    
    theta = self.theta.get_values(points = points);
    width = self.width.get_values(points = points);
    
    centerleftrightnormals = wormgeo.shape_from_theta(theta, width, orientation = self.orientation, 
                             xy = self.xy, length = self.length,
                             npoints = npoints, nsamples = all,
                             resample=False, smooth=0, with_normals=with_normals);
    if with_normals:
      center, left, right, normals = centerleftrightnormals;
      if with_center:
        return left, right, center, normals;
      else:
        return left, right, normals;
    else:
      center, left, right = centerleftrightnormals;
      if with_center:
        return left, right, center;
      else:
        return left, right;
   
  
  def polygon(self, npoints = all):
    """Returns polygon for the worm outline
    
    Arguments:
      npoints (int or all): number of points along one side of the worm
    
    Returns:
      array (2xm): reference points on the polygon
    """
    
    left, right = self.shape();
    poly = np.vstack([left, right[::-1,:]]);
    
    if npoints is not all and npoints != self.npoints:
      poly = resample_curve(poly, npoints);
    
    return poly;
  
  
  def mask(self, size = (151, 151)):
    """Returns a binary mask for the worm shape
    
    Arguments:
      size (tuple ro array): size of the mask
    
    Returns:
      array: mask of worm shape
    """
    
    mask = np.zeros(tuple(size));
    left, right = self.sides();
    
    for i in range(self.npoints-1):
      poly = np.array([left[i,:], right[i,:], right[i+1,:], left[i+1,:]], dtype = np.int32)
      cv2.fillPoly(mask, [poly], 1);
    
    return np.asarray(mask, dtype = bool)
  
  
  def phi(self, size = (151, 151)):
    """Returns implicit contour representation of the worm shape
    
    Arguments:
      size (tuple ro array): size of the contour representation
    
    Returns:
      array:  contour representation of the worm
      
    Note: 
      worm border is given by phi==0
    """
    
    return mask_to_phi(self.mask(size = size));  
    
  
  def position(self):
    return self.xy;
      
  def head(self):
    center = self.center();
    return center[0];
  
  def tail(self):
    center = self.center();
    return center[-1];
    
  def head_tail(self):
    center = self.center();
    return center[[0,-1],:];
  
  def measure(self):
    """Measure some properties at the same time"""
   
    if self.valid:
      cl = self.center();
      n2 = (len(cl)-1)//2;
      
      # positions
      pos_head = cl[0];
      pos_tail = cl[1];
      pos_center = cl[n2];
      
  
      # head tail distance:
      dist_head_tail = np.linalg.norm(pos_head-pos_tail)
      dist_head_center = np.linalg.norm(pos_head-pos_center)
      dist_tail_center = np.linalg.norm(pos_tail-pos_center)
    
      #average curvature
      xymintp, u = splprep(cl.T, u = None, s = 1.0, per = 0);
      dcx, dcy = splev(u, xymintp, der = 1)
      d2cx, d2cy = splev(u, xymintp, der = 2)
      ck = (dcx * d2cy - dcy * d2cx)/np.power(dcx**2 + dcy**2, 1.5);
      curvature_mean = np.mean(ck);
      curvature_variation = np.sum(np.abs(ck))
      
      curled = False;
      
    else:
      pos_head = np.array([0,0]);
      pos_tail = pos_head;
      pos_center = pos_head;
      
  
      # head tail distance:
      dist_head_tail = 0.0
      dist_head_center = 0.0
      dist_tail_center = 0.0
    
      #average curvature
      curvature_mean = 0.0;
      curvature_variation = 0.0
      
      curled = True;
     
    data = np.hstack([pos_head, pos_tail, pos_center, dist_head_tail, dist_head_center, dist_tail_center, curvature_mean, curvature_variation, curled]);
    return data
  
  
  ############################################################################
  ### Worm shape deformations, Worm motions
  
  def translate(self, xy):
    """Translate worm
    
    Arguments:
      xy (tuple): translation vector
    """
    self.xy += np.array(xy);
    
  
  def rotate(self, angle):
    """Rotate worm around center point
    
    Arguments:
      ang (tuple): rotation angle in rad
    """
    self.orientation += angle;
  
  
#  def rotate_around_point(self, angle, point):
#    """Rotate worm around a specified point
#    
#    Arguments:
#      ang (tuple): rotation angle
#      point (tuple): center point for rotation
#    """
    #todo: easier to just rotate the worm and the position
#    cline = self.center();
#    point = np.array(point);
#    cline = np.dot(cline-point,[[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])+point;
#    self.from_center(cline, self.width);
  
  
  def move_forward(self, distance, smooth = 1.0, straight = True):
    """Move worm peristaltically forward
    
    Arguments:
      distance (number): distance to move forward in units of the worm length
      smooth (number): smoothness of the interpolation
      straight (bool): if True extrapolated points move straight
      
    Note:
      The head is first point in center line and postive distances will move the
      worm in this direction.
    """
    
    #integrated angles
    phii = self.theta.integral();
    phi = phii(np.linspace(0, 1, self.npoints)) - phii(0.5) + self.orientation;
    sp = Spline(values = phi);
    
    #xy shift
    dx =  self.length * sp.integrate(0.5, 0.5 + distance, function = np.cos);
    dy =  self.length * sp.integrate(0.5, 0.5 + distance, function = np.sin);
    self.xy += [dx,dy];

    #alpha = self.theta.integrate(0.5, 0.5 + distance);
    alpha = float(phii(0.5+distance) - phii(0.5));
    self.orientation += alpha;

    #shift theta
    self.theta.shift(distance);
  
      
  def stretch(self, factor):
    """Change length of the worm
    
    Arguments:
      factor (number): factor by which to scale the worm length
    """
    self.length *= factor;
  
  
  def scale(self, factor):
    """Scale length and width of the worm
    
    Arguments:
      factor (number): factor by which to scale the worm
    """
    self.length *= factor;
    self.width.from_values(self.width.values * factor);
  
    
  def widen(self, factor):
    """Change wodth of the worm
    
    Arguments:
      factor (number): factor by which to scale the worm width
    """
    self.width.from_values(self.width.values * factor);
    
  
#  def curve(self, mode_amplitudes):
#    """Change curvature properties of the worm
#    
#    Arguments:
#      mode_amplitudes (number or array): additional power in the first fourier modes of the worms angles
#    """
#    #changes curvature by the mode amplitudes;
#    #print self.theta.shape
#    #cos = np.cos(self.theta); -> ok to use theta directly 
#    t = np.fft.rfft(self.theta);
#    #print t.shape;
#    mode_amplitudes = np.array(mode_amplitudes);
#    t[:mode_amplitudes.shape[0]] += mode_amplitudes;
#    #print t.shape;
#    self.theta = np.fft.irfft(t, n = self.npoints-2);
#    #print self.theta.shape
  
  
#  def bend(self, bend, exponent = 4, head = True,):
#    """Change curvature properties of the worm
#    
#    Arguments:
#      bend (number): bending amplitude
#      exponent (number): expoential modulation of the bending
#      head (bool): if True bend head side otherwise tail side
#    """
#    #head tail bend profile
#    n2 = self.center_index();
#    
#    if head:
#      self.theta[:n2-1] += bend * np.exp(-exponent * np.linspace(0,1,n2-1));
#    else:
#      self.theta[n2:] += bend * np.exp(-exponent * np.linspace(1,0,n2-1));
#
#  def invert(self):
#    """Invert head and tail"""
#    
#    #width = self.width.values;;
#    #cl = self.center();
#    #self.from_center(cl[::-1,:], width = width[::-1]);
#    pass

  ############################################################################
  ### Visualization
  
  def plot(self, image = None, npoints = all, ccolor = 'black', lcolor = 'green', rcolor = 'red', ax = None):
    xyl, xyr, xym = self.shape(npoints = npoints, with_center = True);
    if ax is None:
      ax = plt.gca();
    
    if image is not None:
      ax.imshow(image);
    ax.plot(xyl[:,0], xyl[:,1], lcolor);
    ax.scatter(xyl[:,0], xyl[:,1], c = lcolor);
    ax.plot(xyr[:,0], xyr[:,1], rcolor);
    ax.scatter(xyr[:,0], xyr[:,1], c = rcolor);
    ax.plot(xym[:,0], xym[:,1], ccolor);
    ax.scatter(xym[:,0], xym[:,1], c = ccolor);




### Tests


def test():
  import matplotlib.pyplot as plt
  import worm.model as wm;
  reload(wm);
  ws = wm.WormModel();
  
  
  
  
  
  mask = ws.mask();
  
  plt.figure(1); plt.clf();
  plt.imshow(mask);
  
  
  


if __name__ == "__main__":
  test()  
