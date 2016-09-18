# -*- coding: utf-8 -*-
"""
Module to model Worm shapes and movements

The WormModel class models the main features of the worm shape and
does inference of the worm shape from images or movies

The worm is parameterized by:

  * xy          - position of the worm (the center of the mid line)
  * orientation - absolute orientation of the worm
  * length      - length of the center line
  * bending     - parameters for the bending of the center line
  * width       - parameters width profile along that center line

The bending and width profiles are parameterized via specialized classes,
handling e.g. b spline calculations

Note:
  The number of variable parameter during tracking / shape detection is the number of control points
  for the bending (len(b)) + for the position (2) + actual velocity (1) and optionally speeds for
  additional bending modes (k)
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import numpy as np
import matplotlib.pyplot as plt

import copy

import cv2
import scipy.ndimage as nd

from scipy.spatial.distance import cdist
from scipy.interpolate import splrep, splprep, splev

import worm.shape as shape

from utils.utils import isnumber, eps

from signalprocessing.splines import Spline
from signalprocessing.resampling import resample_curve, resample_data

from imageprocessing.masking import mask_to_phi, curvature_from_phi



class WormModel:
  """Class modeling the shape and posture of a worm"""
  
  def __init__(self, length = 40, xy = [75, 75], orientation = 0,
               bending = None, width = None):
    """Constructor of WormModel
    
    Arguments:
      length (number): length of the center line of the worm [in pixel]
      xy (2 array): position of the center of the center line [in pixel]
      theta (array or None): center bending angle (if None non bend worm)
      width (array or None): width profile (if none use default profile)
      theta_model (array or None): the model for the center bending angle curve (if None non use Spline)
      width_model (array or None): the model for the width profile curve (if none use Spline)
      
    Note:
      The number of samples form the theta curve is nsamples - 2
    """
    
    self.length = float(length);
    self.xy = np.array(xy, dtype = float);
    
    if theta is None:
      self.theta_model = Spline(values = theta, nsample = 21, nparameter = 10, degree = 3);
    else:
      self.theta_model = theta_model;
    
    self.nsamples = self.theta_model.nsamples;    
    
    if width_model is None:
      if width is None:
        width = self.defaul_width();
      self.width_model = Spline(values = width, nparameter = 10, degree = 3);
    else:
      self.width_model = width_model;
    
    assert self.nsamples == self.width_model.nsamples;
    
    self.valid = valid;
  
  
  ############################################################################
  ### Properties
   
  def nparameter(self, full = False):
    """Number of parameter"""
    
    if full: # length + width profile + speed [theta speeds / turning speeds]
      n = 1 + self.width_model.nparameter + 1; # + self.theta_model.nparameter;
    else:
      n = 0;
    
    return n + self.theta_model.nparameter + 3; #theta and position and orientation
   
  
  def parameter(self, full = False):
    """Parameter of the worm shape"""
    
    if full:
      return np.hstack([self.theta_model.parameter, self.xy, self.orientation, self.length, self.width_model.parameter, self.speed]) #self.speed, self.w, self.b_speed
    else:
      return np.hstack([self.theta_model.parameter, self.xy, self.orientation]) #self.speed, self.w, self.b_speed
  
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
    
    self.width = w(np.linspace(0, 1, self.nsamples));
    return self.width;


  def from_parameter(self, parameter):
    """Sets the parameter from parameter array
    
    Arguments:
      parameter (array): parameter as returned by parameter function
    
    Note:
      len(parameter) should be as returned by nparameter function
    """
    
    self.b = np.array(parameter[:self.ncoeffcients], dtype = float);
    self.xy = np.array(parameter[self.ncoeffcients:self.ncoeffcients+2], dtype = float);
    #self.speed = float(parameter[self.ncoeffcients+2]);
    #self.w = np.array(parameter[self.ncoeffcients+3:2*self.ncoeffcients+3], dtype = float);
    #self.b_speed = np.array(parameter[2*self.ncoeffcients+3:3*self.ncoeffcients+3], dtype = float);
    
  
  def from_lines(self, left_line, right_line, center_line = None):
    """Initialize worm model from the left and right border lines
    
    Arguments:
      left_line (nx2 array): points of left border line
      right_line (nx2 array): points of right border line
      center_line (nx2 array): optional center line
    """
    
    # get center line and width profile
    if center_line is None:
      
    
    
    # construct from center line and width
    self.from_center_line(center_line, width);
    
    self.npoints = center_line.shape[0];
    if self.npoints % 2 != 1:
      raise RuntimeWarning('WormShape: number of line points expected to be odd, resmapling points!')
      self.npoints += 1;
    n = self.npoints;
    n2 = self.center_index();
    
    # resample points equidistantly 
    center_line = resample_curve(center_line, n);
    left_line = resample_curve(left_line, n);
    right_line = resample_curve(right_line, n);
    
    # position
    self.position = center_line[n2,:];

    # orientation
    t0 = np.array([0,1], dtype = float); #vertical reference
    t1 = center_line[n2,:]-center_line[n2-1,:];
    self.orientation = np.arctan2(t0[0], t0[1]) - np.arctan2(t1[0], t1[1])
    
    # length
    self.length = np.linalg.norm(t1);
    
    # angles and width
    self.theta = np.zeros(n - 2);
    self.width = np.zeros(n);

    #self.l[n2-1] = np.linalg.norm(t1);   
    c = center_line[n2,:];
    distsl = cdist([c], left_line);
    distsr = cdist([c], right_line);
    self.width[n2] = distsr.min() + distsl.min();
    
    t0 = t1;
    for i in range(n2):
      t1 = center_line[i+1+n2,:]-center_line[i+n2,:];
      #self.l[n2+i] = np.linalg.norm(t1);
      
      self.theta[n2+i-1] = np.arctan2(t0[0], t0[1]) - np.arctan2(t1[0], t1[1])
      
      c = center_line[i+1+n2,:]; # + center_line[i+n2,:]) / 2;
      distsl = cdist([c], left_line);
      distsr = cdist([c], right_line);
      self.width[n2+i+1] = distsr.min() + distsl.min();
      
      self.length += np.linalg.norm(t1);
      t0 = t1; 
    
    t1 = center_line[n2,:]-center_line[n2-1,:];
    for i in range(n2-1):
      t0 = center_line[n2-1-i,:]-center_line[n2-2-i,:];
      #self.l[n2-2-i] = np.linalg.norm(t0);
      self.theta[n2-2-i] = np.arctan2(t0[0], t0[1]) - np.arctan2(t1[0], t1[1])
      
      c = center_line[n2-1-i,:]; # + center_line[n2-2-i,:]) / 2;
      distsl = cdist([c], left_line);
      distsr = cdist([c], right_line);
      self.width[n2-1-i] = distsr.min() + distsl.min();
      
      self.length += np.linalg.norm(t1);
      
      t1 = t0;
    
    c = center_line[0,:]; # + center_line[n2-2-i,:]) / 2;
    distsl = cdist([c], left_line);
    distsr = cdist([c], right_line);
    self.width[0] = distsr.min() + distsl.min();
    
    self.theta = np.mod(self.theta + np.pi, 2 * np.pi) - np.pi;
  
  
  def from_center_line(self, center_line, width = None):
    """Initialize worm model from center line and width
    
    Arguments:
      center_line (nx2 array): points of center line
      width (array or None): width of worm at reference points, if None use initial guess
    """
    
    self.npoints = center_line.shape[0];
    if self.npoints % 2 != 1:
      raise RuntimeWarning('WormShape: number of line points  %d expected to be odd, adding a point!' % self.npoints)
      self.npoints += 1; 
    n = self.npoints;
    n2 = self.center_index();
    
    #print n, center_line.shape
    #print center_line
    #plt.figure(101); plt.clf();
    #plt.plot(center_line[:,0], center_line[:,1]);
    
    center_line = resample_curve(center_line, n);
    
    #position
    self.position = center_line[n2,:];
    
    # orientation
    t0 = np.array([0,1], dtype = float); #vertical reference
    t1 = center_line[n2,:]-center_line[n2-1,:];
    self.orientation = np.arctan2(t0[0], t0[1]) - np.arctan2(t1[0], t1[1])
    
    # angles
    ts = center_line[1:,:] - center_line[:-1,:];
    t0 = ts[:-1, :];
    t1 = ts[1:,:];
    self.theta = np.arctan2(t0[:,0], t0[:,1]) - np.arctan2(t1[:,0], t1[:,1]);
    self.theta = np.mod(self.theta + np.pi, 2 * np.pi) - np.pi;
    
    #length
    self.length = np.sum(np.linalg.norm(ts, axis = 1));
    
    #width
    if width is None:
      self.width = self.guess_width();
    else:
      self.width = width.copy();
    
  
  def from_image(self, image, sigma = 1, absolute_threshold = None, threshold_factor = 0.95, 
                       npoints_contour = 100, delta = 0.3, smooth = 1.0,
                       npoints = 21, 
                       verbose = False, save = None):
    
    shape = shape_from_image(image, sigma = sigma, absolute_threshold = absolute_threshold,
                             threshold_factor = threshold_factor, npoints_contour = npoints_contour, 
                             delta = delta, smooth = smooth,
                             npoints = npoints, 
                             verbose = verbose, save = save);
    
    if shape[0]:
      #self.from_lines(shape[1], shape[2], shape[3]);
      self.from_center_line(shape[1], shape[-1]);
    else:
      self.npoints = shape[1].shape[0];
      self.valid = False; #curled ->  for now treat as invalid
  
  def to_array(self):
    if self.valid:
      cl = self.center_line();
      wd = self.width;
      return np.hstack([cl[:,0], cl[:,1], wd]);
    else:
      return -1 * np.ones(self.npoints * 3);
    
  def from_array(self, array):
    n = len(array)/3;
    cl = np.vstack([array[0:n], array[n:2*n]]).T;
    wd = array[2*n:];
    if wd[0] == -1:
      self.valid = False;
      self.npoints = len(wd);
    else:
      self.from_center_line(cl, width = wd);
      self.valid = True;
  
  ############################################################################
  ### Shape Properties 
  
  def center_index(self):
    """Returns index of centeral point"""
    return (self.npoints-1)//2;
    
  def center_line(self, npoints = all):
    """Returns center line of the worm
    
    Arguments:
      npoints (int or all): number of sample points
      
    Returns:
      array (nx2): points along center line
    """
    n = self.npoints;
    n2 = self.center_index();
    l = self.length / (n-1);
    
    xym = np.zeros((n, 2));
    xym[n2,:] = self.position;
    
    cos = np.cos(self.orientation);
    sin = np.sin(self.orientation);
    n0 = np.array([-sin, cos]);
    xym[n2-1,:] = xym[n2,:] - n0 * l;
        
    cos = np.cos(self.theta);
    sin = np.sin(self.theta);
    n1 = n0.copy();
    for i in range(n2):
      n1 = np.array([cos[n2-1+i] * n1[0] - sin[n2-1+i] * n1[1], sin[n2-1+i] * n1[0] + cos[n2-1+i] * n1[1]]);
      xym[i+n2+1,:] = xym[i+n2,:] + n1 * l;
    
    for i in range(n2-1):
      n0 = np.array([cos[n2-2-i] * n0[0] + sin[n2-2-i] * n0[1], - sin[n2-2-i] * n0[0] + cos[n2-2-i] * n0[1]]);
      xym[n2-2-i,:] = xym[n2-1-i,:] - n0 * l;
    
    if npoints is not all and npoints != n:
      xym = resample_curve(xym, npoints);
    
    return xym;
    
  
  def sides(self, npoints = all):
    """Returns left and right side and center line of the worm
    
    Arguments:
      npoints (int or all): number of sample points
      
    Returns:
      array (nx2): points along left side
      array (nx2): points along right side
      array (nx2): points along center line
    """
    
    center_line = self.center_line();
    width = self.width;
    
    # resample  points
    if npoints is not all and npoints != self.npoints:
      if npoints % 2 != 1:
        raise RuntimeWarning('WormModel: number of line points expected to be odd, adding a point!')
        npoints += 1;    
      center_line = resample_curve(center_line, npoints);
      width = resample_data(width, npoints);
      ts = center_line[1:,:] - center_line[:-1,:];
      t0 = ts[:-1, :];
      t1 = ts[1:,:];
      theta = np.arctan2(t0[:,0], t0[:,1]) - np.arctan2(t1[:,0], t1[:,1]);
      theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi;
    else:
      npoints = self.npoints;
      theta = self.theta;
    
    n  = npoints;
    n2 = (npoints-1)/2;
    
    xyl = np.zeros((n, 2));
    xyr = np.zeros((n, 2));
    
    cos = np.cos(self.orientation + (np.pi + theta[n2-1])/2.0);
    sin = np.sin(self.orientation + (np.pi + theta[n2-1])/2.0);
    t0 = np.array([-sin, cos]);    
    c = center_line[n2,:];
    xyl[n2,:] = c - t0 * width[n2] / 2.0;
    xyr[n2,:] = c + t0 * width[n2] / 2.0;
    t1 = t0.copy();
    
    thetaw = (np.hstack([0, theta]) + np.hstack([theta, 0]))/2.0;    
    cosw = np.cos(thetaw);  
    sinw = np.sin(thetaw);
        
    for i in range(n2):
      k = n2 + i; k1 = k + 1;
      t0 = np.array([cosw[k] * t0[0] - sinw[k] * t0[1], sinw[k] * t0[0] + cosw[k] * t0[1]]);
      c = center_line[k1,:];
      xyl[k1,:] = c - t0 * width[k1] / 2.0;
      xyr[k1,:] = c + t0 * width[k1] / 2.0;
    
    
    t1 = np.array([cosw[n2-1] * t1[0] + sinw[n2-1] * t1[1], - sinw[n2-1] * t1[0] + cosw[n2-1] * t1[1]]);
    c = center_line[n2-1,:];
    xyl[n2-1,:] = c - t1 * width[n2-1] / 2.0;
    xyr[n2-1,:] = c + t1 * width[n2-1] / 2.0;
        
    for i in range(n2-1):
      k = n2-2-i;
      t1 = np.array([cosw[k] * t1[0] + sinw[k] * t1[1], - sinw[k] * t1[0] + cosw[k] * t1[1]]);
      c = center_line[k,:]; # + xym[n2-1-i,:])/2.0;
      xyl[k,:] = c - t1 * width[k] / 2.0;
      xyr[k,:] = c + t1 * width[k] / 2.0;
    
    return xyl,xyr,center_line
   
  
  def polygon(self, npoints = all):
    """Returns polygon for the worm outline
    
    Arguments:
      npoints (int or all): number of points along one side of the worm
    
    Returns:
      array (2xm): reference points on the polygon
    """
    
    sides = self.sides();
    poly = np.vstack([sides[0], sides[1][::-1,:]]);
    
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
    xyl, xyr, xym = self.sides();
    
    for i in range(self.npoints-1):
      poly = np.array([xyl[i,:], xyr[i,:], xyr[i+1,:], xyl[i+1,:]], dtype = np.int32)
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
    cl = self.center_line();
    return cl[0,:];
  
  def tail(self):
    cl = self.center_line();
    return cl[-1,:];
    
  def head_tail(self):
    cl = self.center_line();
    return cl[[0,-1],:];
  
  def measure(self):
    """Measure some properties at the same time"""
   
    if self.valid:
      cl = self.center_line();
      n2 = self.center_index();
      
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
    self.position += np.array(xy);
    
  
  def rotate(self, angle):
    """Rotate worm around center point
    
    Arguments:
      ang (tuple): rotation angle
    """
    self.orientation += angle;
  
  
  def rotate_around_point(self, angle, point):
    """Rotate worm around a specified point
    
    Arguments:
      ang (tuple): rotation angle
      point (tuple): center point for rotation
    """
    cline = self.center_line();
    point = np.array(point);
    cline = np.dot(cline-point,[[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])+point;
    self.from_center_line(cline, self.width);
  
  
  def move_forward(self, distance, smooth = 1.0, straight = True):
    """Move worm peristaltically forward
    
    Arguments:
      distance (number): distance to move forward
      smooth (number): smoothness of the interpolation
      straight (bool): if True extrapolated points move straight
      
    Note:
      The head is first point in center line and postive distances will move the
      worm in this direction.
    """
    
    length = self.length;
    cline = self.center_line();
 
    cinterp, u = splprep(cline.T, u = None, s = smooth, per = 0)
    us = u - distance / length;
    x, y = splev(us, cinterp, der = 0); 
    cline2 = np.array([x,y]).T;   
    if straight:
       if distance > 0:
         idx = np.nonzero(us < 0)[0];
         if len(idx) > 0:
           d = cline[0,:] - cline[1,:];
           #l = np.linalg.norm(d);
           l = self.length / (self.npoints -1);
           d = d / l;
           m = idx.max();
           for i in idx:
             cline2[i,:] = cline[0,:] + distance * d * (m + 1.0 - i)/(m + 1.0);
       elif distance < 0:
         idx = np.nonzero(us > 1)[0];
         if len(idx) > 0:
           d = cline[-1,:] - cline[-2,:];
           #l = np.linalg.norm(d);
           l = self.length / (self.npoints -1);
           d = d / l;
           m = idx.min(); mx = idx.max();
           for i in idx:
             cline2[i,:] = cline[-1,:] - distance * d * (i - m + 1.0)/(mx + 1.0 - m);
    
    self.from_center_line(cline2, self.width);
    self.stretch(length / self.length);
  
     
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
    self.width *= factor;
    
  def widen(self, factor):
    """Change wodth of the worm
    
    Arguments:
      factor (number): factor by which to scale the worm width
    """
    self.width *= factor;
    
  
  def curve(self, mode_amplitudes):
    """Change curvature properties of the worm
    
    Arguments:
      mode_amplitudes (number or array): additional power in the first fourier modes of the worms angles
    """
    #changes curvature by the mode amplitudes;
    #print self.theta.shape
    #cos = np.cos(self.theta); -> ok to use theta directly 
    t = np.fft.rfft(self.theta);
    #print t.shape;
    mode_amplitudes = np.array(mode_amplitudes);
    t[:mode_amplitudes.shape[0]] += mode_amplitudes;
    #print t.shape;
    self.theta = np.fft.irfft(t, n = self.npoints-2);
    #print self.theta.shape
  
  
  def bend(self, bend, exponent = 4, head = True,):
    """Change curvature properties of the worm
    
    Arguments:
      bend (number): bending amplitude
      exponent (number): expoential modulation of the bending
      head (bool): if True bend head side otherwise tail side
    """
    #head tail bend profile
    n2 = self.center_index();
    
    if head:
      self.theta[:n2-1] += bend * np.exp(-exponent * np.linspace(0,1,n2-1));
    else:
      self.theta[n2:] += bend * np.exp(-exponent * np.linspace(1,0,n2-1));
  
  
  def invert(self):
    """Invert head and tail"""
    
    width = self.width;
    cl = self.center_line();
    self.from_center_line(cl[::-1,:], width = width[::-1]);

  ############################################################################
  ### Error estimation and fitting

  def error(self, image, phi = None, epsilon = 2, delta_phi = 9, curvature = None, 
                  means = [0, 98.5], out_vs_in = 10, border = None, border_epsilon = 10):
    """Error between worm shape and image
    
    Arguments:
      image (array): gray scale image of worm
      phi (None or array): implicit worm contour, if none calculated from actual shape
      delta_phi (number): range in phi to consider for error estimate
      epsilon (number): refularization parameter for the step functions
      curvature (bool): error due to curvature on phi
    """
    #means = [51, 98.5],    
    
    if phi is None:
      phi = self.phi(size = image.shape);
    
    
    # interior and exterior means
    #upts = np.flatnonzero(phi <= 0)  # interior points
    #vpts = np.flatnonzero(phi > 0)  # exterior points
    #u = np.sum(image.flat[upts]) / (len(upts) + eps)  # interior mean
    #v = np.sum(image.flat[vpts]) / (len(vpts) + eps)  # exterior mean

    theta_out = 0.5 * (np.tanh(phi / epsilon) + 1);
    theta_in = 1 - theta_out;
    
    if means is None:
      img_in = image * theta_in;
      img_out = image * theta_out;
    
      mean_in  = np.sum(img_in ) / (np.sum(theta_in ) + eps); 
      mean_out = np.sum(img_out) / (np.sum(theta_out) + eps); 
      #print mean_in, mean_out
    else:
      mean_in = means[0]; 
      mean_out = means[1];
    
    #plt.figure(11); plt.clf();
    #ii = np.zeros(image.shape);
    #plt.subplot(1,2,1);
    #plt.imshow(theta_in);
    #plt.subplot(1,2,2);
    #plt.imshow(theta_out);

    # error from image information
    error =  np.sum((theta_in * (image - mean_in ))**2);
    #error += out_vs_in * np.sum((theta_out *(image - mean_out))**2) / np.sum(theta_out);
    
    #could add some error for head and tail positions here
    
    # error from curvature constraints
    if curvature is not None:
      # regularized cruve information
      idx = np.flatnonzero(np.logical_and(phi <= delta_phi, phi >= -delta_phi))
    
      # disp
      #import matplotlib.pyplot as plt
      #plt.figure(10); plt.clf();
      #ii = np.zeros(image.shape);
      #ii.flat[idx] = phi.flat[idx];
      #plt.imshow(ii);
      #plt.show();

      if len(idx) != 0:
        c = curvature_from_phi(phi, idx);
        error += curvature * np.sum(np.abs(c));
      
    if border is not None:
      sx = nd.sobel(image.astype(float), axis=0, mode='reflect');
      sy = nd.sobel(image.astype(float), axis=1, mode='reflect');
      grad = np.hypot(sx, sy);
      
      b =  np.exp(-phi * phi / border_epsilon);
      print error
      error -= border * np.sum(b * grad) / np.sum(b);
      print error
      print '---'
      
      #plt.figure(15); plt.clf();
      #plt.subplot(1,3,1);
      #plt.imshow(grad);
      #plt.subplot(1,3,2);
      #plt.imshow(np.exp(-np.abs(phi) / 0.1))
      #plt.subplot(1,3,3);
      #plt.imshow(np.exp(-np.abs(phi) / 0.1) * grad)
    
    return error;
    
    
    
  def error_contour(self, phi, epsilon = 2.0, delta_phi = None):
    """Error between worm shape and contour from image given implicitly by phi
    
    Arguments:
      phi (array): implicit worm contour, if none calculated from actual shape
      delta_phi (number): range of phi values to consider for error estimate
      epsilon (number): refularization parameter for the step functions
      curvature (bool): error due to curvature on phi
    """
  
    phi2 = self.phi(size = phi.shape);
    
    if delta_phi is None:
      error =  np.sum((np.tanh(phi / epsilon)-np.tanh(phi2 / epsilon))**2);
    else:
      idx = np.flatnonzero(np.logical_and(phi2 <= delta_phi, phi2 >= -delta_phi))
      error =np.sum((np.tanh(phi[idx] / epsilon)-np.tanh(phi2[idx] / epsilon))**2);
    
    return error;
  
  
  def error_center_line(self, image, npoints = all, order = 1, overlap = True):
    """Error between worm center line and image
    
    Arguments:
      image (array): gray scale image of worm
    """
    import shapely.geometry as geo
    
    if overlap:
      xyl, xyr, cl = self.sides(npoints = npoints);
      w = np.ones(npoints);
      #plt.figure(141); plt.clf();
      worm = geo.Polygon();
      for i in range(npoints-1):
        poly = geo.Polygon(np.array([xyl[i,:], xyr[i,:], xyr[i+1,:], xyl[i+1,:]]));
        ovl = worm.intersection(poly).area;
        tot = poly.area;
        w[i+1] = 1 - ovl / tot;
        worm = worm.union(poly);
        
      #print w
      return np.sum(w * nd.map_coordinates(image.T, cl.T, order = order))
    else:
      cl = self.center_line(npoints = npoints);
      return np.sum(nd.map_coordinates(image.T, cl.T, order = order))
  
  
  def optimize2(self, image, maxiter = 30, debug = True, swarmsize = 100, nmodes = 3):
    #import scipy.optimize as opt;
    import pyswarm as pso;
    
    #ws = WormShape(nseg = self.nseg, l = self.l.copy(), theta = self.theta.copy(),
    #               width = self.width.copy(), position = self.position.copy(), orientation = self.orientation);
    ws = copy.deepcopy(self);
    
    def fun(x):
      #print '------'
      ws2 = copy.deepcopy(ws);
      #print ws2.l.shape, ws2.theta.shape, ws2.width.shape

      ws2.move_forward(x[0] * 10);
      ws2.bend(x[1], head = True, exponent = 5);
      ws2.bend(x[2], head = False, exponent = 5);

      #print ws2.l.shape, ws2.theta.shape, ws2.width.shape
      
      ws2.curve(x[3:-3]);
      #print ws2.l.shape, ws2.theta.shape, ws2.width.shape
      #ws2.translate(x[-3:-1] * 10);
      #print ws2.l.shape, ws2.theta.shape, ws2.width.shape
      #ws2.rotate(x[-1]);      
      return ws2.error(image = image, border = 1000000, epsilon = 10);
    
    #x0 = np.hstack([self.l, self.theta, self.width, self.position]);
    #x0 = np.zeros(3 + nmodes + 0*3);
    #lb = np.hstack([[-1, -1, -0.5], [-3 for i in range(nmodes)], [-4, -np.pi/6]]);
    #ub = np.hstack([[1, 1, 0.5], [3 for i in range(nmodes)], [4, np.pi/6]]);
    
    lb = np.hstack([[-1, -1, -1], [-3 for i in range(nmodes)]]);
    ub = np.hstack([[ 1,  1,  1], [ 3 for i in range(nmodes)]])    
    
    par, fopt = pso.pso(fun, lb, ub, debug = debug, maxiter = maxiter, swarmsize = swarmsize)
    
    #import scipy.optimize as opt;
    #def p(x):
    #  print x;
    #method = 'BFGS'; options = {'gtol': 1e-12, 'disp': True, 'eps' : 0.1}
    #x0 = np.zeros(3 + nmodes + 0*3);
    #res = opt.minimize(fun, x0, jac = False, method = method, options = options, callback = p)
    #par = res['x'];    

    self.move_forward(par[0] * 10.0);    
    self.bend(par[1], exponent = 5, head = True);
    self.bend(par[2], exponent = 5, head = False);
    self.curve(par[3:-3]);
    #self.translate(par[-3:-1]*10);
    #self.rotate(par[-1]);
    
    return par;   
  
  
  def optimize(self, image, maxiter = 30, debug = True, swarmsize = 100, nmodes = 3):
    
    import pyswarm as pso;
    
    ws = copy.deepcopy(self);
    
    def fun(par):
      ws2 = copy.deepcopy(ws);
      ws2.move_forward(par[0] * 10);
      ws2.bend(par[1]/10., head = True, exponent = 4);
      ws2.bend(par[2]/10., head = False, exponent = 4);
      ws2.curve(par[3:-3]);
      ws2.translate(par[-3:-1] * 10);
      ws2.rotate(par[-1]/10.);      
      return ws2.error_center_line(image = image, npoints = 2*self.npoints -1);
    

    lb = np.hstack([[-2, -2, -2], [-2 for i in range(nmodes)], [-1, -2]]);
    ub = np.hstack([[2, 2, 2], [2 for i in range(nmodes)], [1, 2]]);
     #lb = np.hstack([[-1, -1, -1], [-3 for i in range(nmodes)]]);
    #ub = np.hstack([[ 1,  1,  1], [ 3 for i in range(nmodes)]])    
    
    par, fopt = pso.pso(fun, lb, ub, debug = debug, maxiter = maxiter, swarmsize = swarmsize)
    
    #import scipy.optimize as opt;
    #def p(x):
    #  print x;
    #method = 'BFGS'; options = {'gtol': 1e-12, 'disp': True, 'eps' : 0.1}
    #x0 = np.zeros(3 + nmodes + 0*3);
    #res = opt.minimize(fun, x0, jac = False, method = method, options = options, callback = p)
    #par = res['x'];    

    self.move_forward(par[0] * 10);
    self.bend(par[1]/10., head = True, exponent = 4);
    self.bend(par[2]/10., head = False, exponent = 4);
    self.curve(par[3:-3]);
    self.translate(par[-3:-1] * 10);
    self.rotate(par[-1]/10.);  
    
    return par;   
    
  
  
  
  def normals(self):
    """Returns the normal vectors at each reference point
    """
    
    n = self.npoints;
    n2 = self.center_index();
    
    ts = np.zeros((n, 2));
    
    cos = np.cos(self.orientation + (np.pi + self.theta[n2-1])/2.0);
    sin = np.sin(self.orientation + (np.pi + self.theta[n2-1])/2.0);
    t0 = np.array([-sin, cos]);   
    ts[n2,:] = np.array([-sin, cos]);    
    t1 = t0.copy();
    
    thetaw = (np.hstack([0, self.theta]) + np.hstack([self.theta, 0]))/2.0;    
    cosw = np.cos(thetaw);  
    sinw = np.sin(thetaw);
        
    for i in range(n2):
      k = n2 + i; k1 = k + 1;
      t0 = np.array([cosw[k] * t0[0] - sinw[k] * t0[1], sinw[k] * t0[0] + cosw[k] * t0[1]]);
      ts[k1,:] = t0;
    
    t1 = np.array([cosw[n2-1] * t1[0] + sinw[n2-1] * t1[1], - sinw[n2-1] * t1[0] + cosw[n2-1] * t1[1]]);
    ts[n2-1,:] = t0;
    
    for i in range(n2-1):
      k = n2-2-i;
      t1 = np.array([cosw[k] * t1[0] + sinw[k] * t1[1], - sinw[k] * t1[0] + cosw[k] * t1[1]]);
      ts[k,:] = t1;
    
    return ts
   
    
  
  def optimize_center_line(self, image, deviation = 5, samples = 20):
    """Optimize center points separately
    
    """
    pass
    #cl = self.center_line();
    #ts = self.normals();    
    #for i in range(self.npoints):
    #  errors = np.zeros(samples);
    #  nd.map_coordinates(z, np.vstack((x,y)))
    
    
  
  def optimize_old(self, image, method = 'BFGS', options = {'gtol': 1e-12, 'disp': True, 'eps' : 0.1}, nmodes = 3):
    import scipy.optimize as opt;
    
    #ws = WormShape(nseg = self.nseg, l = self.l.copy(), theta = self.theta.copy(),
    #               width = self.width.copy(), position = self.position.copy(), orientation = self.orientation);
    ws = copy.deepcopy(self);
    
    def fun(x):
      #print '------'
      ws2 = copy.deepcopy(ws);
      #print ws2.l.shape, ws2.theta.shape, ws2.width.shape

      ws2.bend(x[0], front = True, exponent = 10);
      ws2.bend(x[1], front = False, exponent = 10);
      ws2.forward(x[2]/10);
      #print ws2.l.shape, ws2.theta.shape, ws2.width.shape
      
      ws2.curve(x[3:-3]);
      #print ws2.l.shape, ws2.theta.shape, ws2.width.shape
      #ws2.translate(x[-3:-1] * 1);
      #print ws2.l.shape, ws2.theta.shape, ws2.width.shape
      #ws2.rotate(x[-1]);      
      return ws2.error(image = image);
    
    #x0 = np.hstack([self.l, self.theta, self.width, self.position]);
    x0 = np.zeros(3 + nmodes + 0*3);
    
    #x0 = np.zeros(1);
    #res = opt.minimize(fun, x0, jac = False, hess = False, tol = 10);
    bnds = [(None, None) for i in x0];
    bnds[0] = (-0.2, 0.2);
    bnds = tuple(bnds);
    bnds = None;
    def p(x):
      print x;
    res = opt.minimize(fun, x0, jac = False, method = method, options = options, bounds = bnds, callback = p)
    
    par = res['x'];
    self.bend(par[0], exponent = 10, front = True);
    self.bend(par[1], exponent = 10, front = False);
    self.forward(par[2] / 10.0);
    self.curve(par[3:-3]);
    #self.translate(par[-3:-1]);
    #self.rotate(par[-1]);
    
    return res; 
  

  def optimize_parameter(self, image, method = 'BFGS', options = {'gtol': 1e-6, 'disp': True}):
    import scipy.optimize as opt;
    
    ws = WormModel(npoints = self.npoints, length = self.length, theta = self.theta,
                   width = self.width, position = self.position);
    
    def fun(x):
      ws.l     = x[:self.nseg];
      ws.theta = x[self.nseg:2*self.nseg];
      #ws.width = x[2*self.nseg:3*self.nseg];
      ws.position = x[-2:];
      return ws.error(image = image);
    
    #x0 = np.hstack([self.l, self.theta, self.width, self.position]);
    x0 = np.hstack([self.l, self.theta, self.position]);
    #res = opt.minimize(fun, x0, jac = False, hess = False, tol = 10);
    res = opt.minimize(fun, x0, jac = False, method = method, options = options)
    
    par = res['x'];
    self.l = par[:self.nseg];
    self.theta = par[self.nseg:2*self.nseg];
    #self.width = par[2*self.nseg:3*self.nseg];
    self.position = par[-2:];    
    
    return res;

  ############################################################################
  ### Visualization
  
  def plot(self, image = None, npoints = all, ccolor = 'black', lcolor = 'green', rcolor = 'red', ax = None):
    xyl, xyr, xym = self.sides(npoints = npoints);
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

def test_spline():
  import numpy as np
  import matplotlib.pyplot as plt
  import worm.model_spline as wm;
  reload(wm);

  s = wm.Spline(ncoefficients = 10, nsamples = 141, degree = 3);
  
  x = np.sin(10* s.s) + np.sin(2 * s.s);
  p = s.coefficients(x)  
  tck = splrep(s.s, x, t = s.knots, task = -1, k = s.degree);
  xpp = splev(s.s, tck);
  xp = s.values(p);
  
  plt.figure(1); plt.clf();
  plt.plot(x);
  plt.plot(xp);
  plt.plot(xpp);
  
  plt.figure(1); plt.clf();
  for i in range(s.ncoefficients):
    pp = np.zeros(s.ncoefficients);
    pp[i] = 1;
    xp = s.values(pp);
    plt.plot(s.s, xp);



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
