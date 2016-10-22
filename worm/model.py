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
import copy;
import matplotlib.pyplot as plt
#import copy
import cv2
#import scipy.ndimage as nd
#from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev #,splrep,

import worm.geometry as wormgeo

from interpolation.spline import Spline
from interpolation.curve import Curve
from interpolation.resampling import resample as resample_curve


from imageprocessing.masking import mask_to_phi #, curvature_from_phi
#from utils.utils import isnumber, eps


class WormModel(object):
  """Class modeling the shape and posture and motion of a worm"""
  
  def __init__(self, orientation = 0, xy = [75, 75], length = 50,
               theta = None, width = None, npoints = 21):
    """Constructor of WormModel
    
    Arguments:
      orienation (float): absolute orienation of the worm
      xy (2 array): position of the center of the center line [in pixel]
      length (number): length of the center line of the worm [in pixel]
      theta (array, Curve or None): center bending angle (if None non bend worm)
      width (array, Curve or None): width profile (if none use default profile)
      npoints (int): number of sample points along the center line if theta not defined
      nparameter (int): number of parameter for the center and width splines if they are not defined
      
    Note:
      The number of sample points along center line npoints is 2 points larger than the 
      number of sample points along theta: npoints = theta.npoints + 2
      
      For speed the worm model assumes equally spaced sample points for theta and width
    """
    
    self.length = float(length);
    self.xy = np.array(xy, dtype = float);
    self.orientation = float(orientation);
    self.speed = 0.0;
    
    # Theta 
    if theta is None:
      self.theta = np.zeros(npoints-2);
    elif isinstance(theta, Spline):
      self.theta = theta.values;
    else:
      self.theta = np.array(theta, dtype = float);
    self.npoints = self.theta.shape[0] + 2;

    # Width
    if width is None:
      self.width = wormgeo.default_width(self.npoints);
    elif isinstance(width, Spline):
      self.width = width.values;
    else:
      self.width = np.array(width, dtype = float);
      
    if self.npoints != self.width.shape[0]:
      raise ValueError('number of sample points along center line %d does not match sample points for width %d' % (self.npoints, self.width.shape[0]) );

    # Parameter dimensnios
    self.nparameter = self.npoints - 2 + 1 + 2 + 1; #free parameter: theta, orientation (1), position (2) and and speed (1)
    self.nparameter_total = self.nparameter + 1 + self.npoints ; # length + width profile
    self.orientation_index = self.nparameter - 4;
    self.ndistances = self.npoints * 2 - 2; # side line and head and tail distances to contour
  
  
  def copy(self):
    return copy.deepcopy(self);
  
  ############################################################################
  ### Parameter interface
    
  def get_parameter(self, full = False):
    """Parameter of the worm shape"""
    par = np.hstack([self.theta, self.orientation, self.xy, self.speed]);
    if full:
      return np.hstack([par, self.length, self.width]); #self.speed, self.w, self.theta_speed
    else:
      return par;

  
  def set_parameter(self, parameter):
    """Set the parameter"""
    istr = 0; iend = self.theta.shape[0];
    self.theta = parameter[istr:iend];
    istr, iend = iend, iend + 1;
    self.orientation = parameter[istr:iend];
    istr, iend = iend, iend + 2;
    self.xy = parameter[istr:iend];
    istr, iend = iend, iend + 1;
    self.speed = parameter[istr:iend];
    if len(parameter) > self.nparameter:
      istr, iend = iend, iend + 1;
      self.length = parameter[istr:iend];
      istr, iend = iend, iend + self.width.nparameter;
      self.width = parameter[istr:iend];
  
  
  def parameter_difference(self, parameter1, parameter2 = None):
    """Residual between two parameter sets
    
    Note:
      The orientation variable is 2 pi periodic and thus needs special care when calculating distances in parameter space
    """
    if parameter2 is None:
      parameter2 = self.get_parameter();
    
    diff = parameter1- parameter2;
    oid = self.orientation_index;
    if diff[oid] > np.pi:
      diff[oid] -= 2*np.pi;
    if diff[oid] <= -np.pi:
      diff[oid] += 2*np.pi;
    return diff;
  
  ############################################################################
  ### Constructors
  
  def from_shape(self, left, right, center = None, width = None, nsamples = all, nneighbours = all):
    """Initialize worm model from the left and right border lines
    
    Arguments:
      left (nx2 array): points of left border line
      right (nx2 array): points of right border line
      center (nx2 array): optional center line
      width (n array): optional width of the worm
      nsamples (int): number of intermediate sample points
    """
    
    # get center line and width profile
    if width is None:
      with_width = True;
    else:
      with_width = False;
    
    if center is None or width is None:
      cw = wormgeo.center_from_sides_discrete(left, right, with_width = with_width, 
                                     nsamples = nsamples, npoints = nsamples, resample = False, 
                                     smooth=0, nneighbours=nneighbours, 
                                     method='projection');
    if width is None:
      center, width = cw;
    else:
      center = cw;
    
    npoints =  self.theta.npoints;
    theta, orientation, xy, length = wormgeo.theta_from_center_discrete(center, npoints = npoints, 
                                                               nsamples = nsamples, resample=False, smooth=0);
    
    # non uniform version
    #oints = np.linspace(0,1,npoints-2);
    #self.theta.set_parameter_from_values(theta, points = points);
    #points = np.linspace(0,1,nsamples);
    #self.width.set_parameter_from_values(width, points = points);
    
    #theta and width uniform
    self.theta = theta;
    self.width = width; 
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
    
    theta, orientation, xy, length = wormgeo.theta_from_center_discrete(center, npoints = self.npoints, 
                                                                        nsamples = nsamples, resample=False, smooth=0);
    
    #points = np.linspace(0,1,self.npoints-2);
    #self.theta.set_parameter_from_values(theta, points = points);    
    self.theta = theta;
    self.xy = xy;
    self.orientation = orientation;
    self.length = length;    
    if width is not None:
      #points = np.linspace(0,1,width.shape[0]);
      #self.width.set_parameter_from_values(width, points = points);
      self.width = width;

    
  def from_image(self, image, sigma = 1, absolute_threshold = None, threshold_factor = 0.95, 
                       ncontour = 100, delta = 0.3, smooth = 1.0, nneighbours = 4,
                       verbose = False, save = None):
    
    success, center, left, right, width = wormgeo.shape_from_image(image, 
                             sigma = sigma, absolute_threshold = absolute_threshold,
                             threshold_factor = threshold_factor, ncontour = ncontour, 
                             delta = delta, smooth = smooth, nneighbours = nneighbours,
                             npoints = self.npoints, 
                             verbose = verbose, save = save);
    
    if success:
      #self.from_lines(shape[1], shape[2], shape[3]);
      self.from_center(center, width);
    else:
      raise RuntimeWarning('failed inferring worm from image');
      
      
  def distance_to_contour(self, contour,  search_radius=[5,20], min_alignment = 0, match_head_tail = None, with_points = False, verbose = False):
    """Match worm shape to contour and return distances
    
    Arguments:
      left,right (nx2 array): shape 
      normals (nx2 array): normals for the shape
      contour (Curve): the contour curve
      search_radius (float or array): the search radius to check for points
      min_alignment (float or None): if not None also ensure the normals are aligned
      with_points (bool): if True also return intersection points coordinates
      verbose (bool): plot results    
    
    Returns:
      array: distances between the worm shape sample points and the nearest contour points, occluded points will be set to NaNs.
    
    Note:
      The number of distances is 2 * (npoints-2) + 2 = 2*npoints - 2
      as the head and tail are only counted once.
      
      It might be useful to average distances of corresponding left and right points
    """
    
    left, right, normals = self.shape(with_normals = True);
    res = wormgeo.distance_shape_to_contour_discrete(left,right,normals,contour,
                                                     search_radius=search_radius, min_alignment=min_alignment, match_head_tail=match_head_tail,
                                                     verbose = verbose);
    if match_head_tail is not None:
      distances_left, intersection_pts_left, distances_right, intersection_pts_right, distance_head, head_match, distance_tail, tail_match = res;
      distances = np.hstack([distance_head, distances_left, distances_right, distance_tail]);
      
      if with_points:
        if head_match is None:
          head_pos = np.array([np.nan, np.nan]);
        else:
          head_pos = match_head_tail[head_match];
        if tail_match is None:
          tail_pos = np.array([np.nan, np.nan]);
        else:
          tail_pos = match_head_tail[tail_match]; 
        intersecs = np.vstack([head_pos, intersection_pts_left, intersection_pts_right, tail_pos]).T;
        return (distances, intersecs);
      else:
        return distances;
      
    else:
      distances_left, intersection_pts_left, distances_right, intersection_pts_right = res;
      distances = np.hstack([distances_left[:-1], distances_right[1:]]);
      if with_points:
        return (distances, np.vstack([intersection_pts_left[:-1], intersection_pts_right[1:]]));
      else:
        return distances;
  
  
  ############################################################################
  ### Shape Properties 
      
  def center(self, with_normals = False):
    """Returns center line of the worm
    
    Arguments:
      with_normals (bool): if true also return normals along center points
      
    Returns:
      array (nx2): points along center line
      array (nx2): normals along center points
    """
    
    return wormgeo.center_from_theta_discrete(self.theta, orientation = self.orientation,
                                              xy = self.xy, length = self.length,
                                              npoints = all, nsamples = all, resample = False,
                                              smooth = 0, with_normals = with_normals);
 
  
  def shape(self, with_center = False, with_normals = False, with_width = False):
    """Returns left and right side and center line of the worm
    
    Arguments:
      with_center (bool): if true also return center points
      with_normals (bool): if true also return normals along center points
      with_width (bool): if true also return width profile
      
    Returns:
      array (nx2): points along left side
      array (nx2): points along right side
      array (nx2): points along center line
      array (nx2): normals along center point
      array (n): width profile
    """
    
    #calcualte shape
    centerleftrightnormals = wormgeo.shape_from_theta_discrete(self.theta, self.width, 
                               orientation = self.orientation, xy = self.xy, length = self.length,
                               npoints = all, nsamples = all, resample=False,
                               with_normals=with_normals);
    
    #assemble returns
    if with_normals:
      left, right, center, normals = centerleftrightnormals;
    else:
      left, right, center = centerleftrightnormals;
      normals = None;
    res = [left, right];
    if with_center:
      res.append(center);
    if with_normals:
      res.append(normals);
    if with_width:
      res.append(self.width);
    return tuple(res);
  
  
  def polygon(self, npoints = all):
    """Returns polygon for the worm outline
    
    Arguments:
      npoints (int or all): number of points along one side of the worm
    
    Returns:
      array (2xm): reference points on the polygon
    """
    
    left, right = self.shape();
    poly = np.vstack([left, right[::-1,:]]);
    
    if npoints is not all:
      poly = resample_curve(poly, npoints);
    
    return poly;
  
  
  def contour(self):
    """Returns contour of the worm as Curve
    
    Arguments:
      npoints (int or None): number of sample points along one side of the worm
    
    Returns:
      Curve: curve of the contour
    """
    
    left, right = self.shape();
    poly = np.vstack([left, right[::-1,:]]);
    return Curve(poly);
  
  
  def mask(self, size = (151, 151)):
    """Returns a binary mask for the worm shape
    
    Arguments:
      size (tuple ro array): size of the mask
    
    Returns:
      array: mask of worm shape
    """
    
    mask = np.zeros(tuple(size));
    left, right = self.shape();
    
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
  
  
  def head(self):
    center = self.center();
    return center[0];
  
  def tail(self):
    center = self.center();
    return center[-1];
    
  def head_tail(self):
    center = self.center();
    return center[[0,-1],:];
  
  
  def self_occlusions(self, margin = 0.01, with_bools = True, with_index = False, with_points = False):
    """Returns points of the shape that are occulded by the worm itself
    
    Arguments:
      margin (float): margin by which to reduce width in order to detect real insiders (small fraction of the width)
      with_bools (bool): if True return two array's indicating which points are not occluded
      with_index (bool): if True return two arrays with indices of which points are occluded
      with_points (bool): if True return points on left and right curves that are occluded
      
    Returns
      array, array: left,right array of bools indicating valid non-occluded points
      array, array: left, right indices of occluded points
      array, array: left,right occluded points
    """
    left, right = self.shape();
    return wormgeo.self_occlusions_from_shape_discrete(left, right, margin = margin,  with_bools = with_bools, with_index = with_index, with_points = with_points);
  
  
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
  
  
  def move_forward(self, distance, with_orientation = True, with_xy = True, straight = True):
    """Move worm peristaltically forward
    
    Arguments:
      distance (number): distance to move forward in units of the worm length
      straight (bool): move straight forward for extrapolated points
      
    Note:
      The head is first point in center line and postive distances will move the
      worm in this direction.
    """
    
    theta, orientation, xy, length = wormgeo.move_forward_discrete(distance, self.theta, self.orientation, self.xy, self.length, straight = straight);
    self.theta = theta;
    
    if with_xy:
      self.xy = xy;      
    if with_orientation:
      self.orientation = orientation;
  
      
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
    self.width.multiply(factor);
  
    
  def widen(self, factor):
    """Change wodth of the worm
    
    Arguments:
      factor (number): factor by which to scale the worm width
    """
    self.width.multiply(factor);
    
  
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
  
  def plot(self, image = None, color = None, ccolor = 'black', lcolor = 'green', rcolor = 'red', ax = None):
    xyl, xyr, xym = self.shape(with_center = True);
    if ax is None:
      ax = plt.gca();
    if color is not None:
      ccolor = color; lcolor = color; rcolor = color;
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
  import numpy as np
  import matplotlib.pyplot as plt
  import worm.model as wm;
  reload(wm);
  w = wm.WormModel();
  
  mask = w.mask();
  
  plt.figure(1); plt.clf();
  plt.subplot(1,2,1);
  w.plot();
  plt.subplot(1,2,2);
  plt.imshow(mask);
  
   
  w.theta.add(np.pi);
  plt.figure(2); plt.clf();
  w.plot();
  plt.axis('equal')


if __name__ == "__main__":
  test()  
