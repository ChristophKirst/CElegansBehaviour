# -*- coding: utf-8 -*-
"""
Worm Geometry

collection of routines to detect, convert and manipulate worm shapes features

See also: 
  :mod:`model`
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'


import numpy as np
import matplotlib.pyplot as plt

import shapely.geometry as geom
from skimage.filters import threshold_otsu

import scipy.ndimage.filters as filters
from scipy.interpolate import splev, splprep, UnivariateSpline
#from scipy.interpolate import splrep, dfitpack
from scipy.spatial import Voronoi

from interpolation.spline import Spline
from interpolation.resampling import resample as resample_curve

from signalprocessing.peak_detection import find_peaks

from imageprocessing.contours import detect_contour, sort_points_to_line, inside_polygon;



##############################################################################
### Center Line Detection

def center_from_sides_vonroi(left, right, npoints = all, nsamples = all, resample = False, with_width = False, smooth = 0.1):
  """Finds mid line between the two side lines using Vonroi tesselation
  
  Arguments:
    left, right (nx2 array): vertices of the left and right curves
    npoints (int or all): number of points of the center line
    nsamples (int or all): number of sample points to construct center line
    resample (bool): force resampling of the side curves (e..g in case not uniformly sampled)
    with_width (bool): if True also return estimated width
    smooth (float): smoothing factor for final sampling
  
  Returns:
    nx2 array of midline vertices
  """
  # resample to same size
  nl = left.shape[0];
  nr = right.shape[0];
  if nsamples is all:
    nsamples = max(nl,nr);
  if npoints is all:
    npoints = max(nl,nr);
  
  if nsamples != nl or nsamples != nr or resample:
    leftcurve  = resample_curve(left, nsamples);
    rightcurve = resample_curve(right, nsamples);
  else:
    leftcurve = left;
    rightcurve = right;
  
  # vonroi tesselate 
  polygon = np.vstack([leftcurve, rightcurve[::-1]])
  vor = Voronoi(polygon)
  #voronoi_plot_2d(vor)
  center = vor.vertices;
  
  # detect inside points and connect to line
  ins = inside_polygon(polygon, center);
  center = np.vstack([leftcurve[0], center[ins], leftcurve[-1]]);
  center = sort_points_to_line(center);
  
  plt.figure(10); plt.clf();
  plt.plot(vor.vertices[:,0], vor.vertices[:,1], '.b');
  plt.plot(center[:,0], center[:,1], '.r');
  plt.plot(leftcurve[:,0], leftcurve[:,1]);
  plt.plot(rightcurve[:,0], rightcurve[:,1]);
  
  #from scipy.spatial import voronoi_plot_2d
  #voronoi_plot_2d(vor)
  
  center = resample_curve(center, npoints, smooth = smooth);
  
  if not with_width:
    return center;
  
  # calculate normals along midline and intersection to left/right curves
  width = np.zeros(npoints);
  
  rightline = geom.LineString(rightcurve);
  leftline  = geom.LineString(leftcurve);
  
  for i in range(npoints):
    mid_point = geom.Point(center[i,0], center[i,1]);
    right_point = rightline.interpolate(rightline.project(mid_point));
    left_point  =  leftline.interpolate( leftline.project(mid_point));
    width[i] = np.linalg.norm(np.array([left_point.x - right_point.x, left_point.y - right_point.y]));

  return center, width
  

def center_from_sides_projection(left, right, npoints = all, nsamples = all, resample = False, with_width = False, nneighbours = all, smooth = 0):
  """Finds middle line between the two side curves using projection method
  
  Arguments:
    left, right (nx2 array): vertices of the left and right curves
    npoints (int or all): number of points of the mid line
    nsamples (int or all): number of sample points to contruct midline
    with_width (bool): if True also return estimated width
    nneighbours (int or all): number of neighbouring points to include for projection
  
  Returns:
    nx2 array of midline vertices
  """
  # resample to same size
  nl = left.shape[0];
  nr = right.shape[0];
  if nsamples is all:
    nsamples = max(nl,nr);
  if npoints is all:
    npoints = max(nl,nr);
  
  if nl != nsamples or nr != nsamples or resample:
    leftcurve  = resample_curve(left, nsamples);
    rightcurve = resample_curve(right, nsamples);
  else:
    leftcurve = left;
    rightcurve = right;
  
  # calculate center
  center = np.zeros((nsamples,2));
    
  if nneighbours is all:
    rightline = geom.LineString(rightcurve);
    leftline  = geom.LineString(leftcurve);
  
    for i in range(nsamples):
      right_point = geom.Point(rightcurve[i,0], rightcurve[i,1]);
      left_point  = geom.Point(leftcurve[i,0], leftcurve[i,1]);
        
      right_left_point =  leftline.interpolate( leftline.project(right_point));
      left_right_point = rightline.interpolate(rightline.project( left_point));
      
      center[i] = 0.25 * np.array([right_point.x + left_point.x + right_left_point.x + left_right_point.x,
                                   right_point.y + left_point.y + right_left_point.y + left_right_point.y]);
      
    center = resample_curve(center, npoints, smooth = smooth);
  
    if not with_width:
      return center;
    
    # calculate normals along midline and intersection to left/right curves
    width = np.zeros(npoints);
    
    rightline = geom.LineString(rightcurve);
    leftline  = geom.LineString(leftcurve);
    
    for i in range(npoints):
      mid_point = geom.Point(center[i,0], center[i,1]);
      right_point = rightline.interpolate(rightline.project(mid_point));
      left_point  =  leftline.interpolate( leftline.project(mid_point));
      width[i] = np.linalg.norm(np.array([left_point.x - right_point.x, left_point.y - right_point.y]));
  
    return center, width

  else: # only consider a certain subset of neighbours in projection (useful when worm is highly bend)
    nneighbours2 = int(np.ceil(nneighbours/2.0));
    for i in range(nsamples):
      il = max(0, i-nneighbours2);
      ir = min(i+nneighbours2, nsamples);
      
      rightline = geom.LineString(rightcurve[il:ir]);
      leftline  = geom.LineString( leftcurve[il:ir]);
      
      right_point = geom.Point(rightcurve[i,0], rightcurve[i,1]);
      left_point  = geom.Point(leftcurve[i,0], leftcurve[i,1]);
        
      right_left_point =  leftline.interpolate( leftline.project(right_point));
      left_right_point = rightline.interpolate(rightline.project( left_point));
      
      center[i] = 0.25 * np.array([right_point.x + left_point.x + right_left_point.x + left_right_point.x,
                                   right_point.y + left_point.y + right_left_point.y + left_right_point.y]);
    
    center = resample_curve(center, npoints, smooth = smooth);

    if not with_width:
      return center;
    
    # calculate normals along midline and intersection to left/right curves
    width = np.zeros(npoints);
    
    for i in range(nsamples):
      il = max(0, i-nneighbours2);
      ir = min(i+nneighbours2, nsamples);
      
      rightline = geom.LineString(rightcurve[il:ir]);
      leftline  = geom.LineString( leftcurve[il:ir]);
      
      mid_point = geom.Point(center[i,0], center[i,1]);
      right_point = rightline.interpolate(rightline.project(mid_point));
      left_point  =  leftline.interpolate( leftline.project(mid_point));
      width[i] = np.linalg.norm(np.array([left_point.x - right_point.x, left_point.y - right_point.y]));
  
    return center, width



def center_from_sides_mean(left, right, npoints = all, nsamples = all, resample = False, with_width = False):
  """Finds middle line between the two side curves by simply taking the mean
  
  Arguments:
    left, right (nx2 array): vertices of the left and right curves
    npoints (int or all): number of points of the mid line
    nsamples (int or all): number of sample points to contruct midline
    with_width (bool): if True also return estimated width
  
  Returns:
    nx2 array of midline vertices
  """

  # resample to same size
  nl = left.shape[0];
  nr = right.shape[0];
  if nsamples is all:
    nsamples = max(nl,nr);
  if npoints is all:
    npoints = max(nl,nr);
  
  if nl != nsamples or nr != nsamples or resample:
    leftcurve  = resample_curve(left, nsamples);
    rightcurve = resample_curve(right, nsamples);
  else:
    leftcurve = left;
    rightcurve = right;

  center = (leftcurve + rightcurve) / 2;
  
  if npoints != nsamples:
    center = resample_curve(center, npoints);
  
  if with_width:
    width  = np.linalg.norm(leftcurve - rightcurve, axis = 0);
    return center, width
  else:
    return center;


def center_from_sides_erosion(left, right, erode = 0.1, maxiter = 100, delta = 0.3, ncontour = 100, smooth = 1.0, npoints = all, nsamples = all, resample = False, with_width = False, verbose = False):
  """Finds middle line between the two side curves by simply taking the mean
  
  Arguments:
    left, right (nx2 array): vertices of the left and right curves
    npoints (int or all): number of points of the mid line
    nsamples (int or all): number of sample points to contruct midline
    with_width (bool): if True also return estimated width
  
  Returns:
    nx2 array of center line vertices
  """

  # resample to same size
  nl = left.shape[0];
  nr = right.shape[0];
  if nsamples is all:
    nsamples = max(nl,nr);
  if npoints is all:
    npoints = max(nl,nr);
  
  if nl != nsamples or nr != nsamples or resample:
    leftcurve  = resample_curve(left, nsamples);
    rightcurve = resample_curve(right, nsamples);
  else:
    leftcurve = left;
    rightcurve = right;

  #construct polygon and erode
  poly = geom.polygon.Polygon(np.vstack([leftcurve, rightcurve[::-1,:]]));
  
  last = poly;
  w = 0;
  for i in range(maxiter):
    new = last.buffer(-erode);
    if not isinstance(new.boundary, geom.linestring.LineString):
      new = last;
      break;
    last = new;
    w += erode;
    #x,y = new.boundary.xy
    #plt.plot(x, y, color='r', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
  
  x,y = last.boundary.xy;
  pts = np.vstack([x[:-1],y[:-1]]).T;
  
  if ncontour is all:
    ncontour = pts.shape[0];

  #plt.plot(pts[:,0], pts[:,1], 'm')
  #print pts
  
  #detect high curvatuerpoints along the contour 
  cinterp, u = splprep(pts.T, u = None, s = smooth, per = 1) 
  us = np.linspace(u.min(), u.max(), ncontour)
  x, y = splev(us[:-1], cinterp, der = 0)

  ### curvature along the points
  dx, dy = splev(us[:-1], cinterp, der = 1)
  d2x, d2y = splev(us[:-1], cinterp, der = 2)
  k = (dx * d2y - dy * d2x)/np.power(dx**2 + dy**2, 1.5);
  
  ### find tail / head via peak detection
  #pad k to detect peaks on both sides
  nextra = 20;
  kk = -k; # negative curvature peaks are heads/tails
  kk = np.hstack([kk,kk[:nextra]]);
  peaks = find_peaks(kk, delta = delta);

  if peaks.shape[0] > 0:  
    peaks = peaks[peaks[:,0] < k.shape[0],:];
  else:
    peaks = np.zeros((0,2));
  
  #print peaks
  if peaks.shape[0] < 2:
    if peaks.shape[0] == 1:
      # best guess is half way along the contour
      if verbose:
        print "Could only detect single curvature peak in contour, proceeding with opposite point as tail!"
      imax = np.sort(np.asarray(np.mod(peaks[0,0] + np.array([0,ncontour//2]), ncontour), dtype = int));
    else:
      if verbose:
        print "Could not detect any peaks in contour, proceeding with 0% and 50% of contour as head and tail!"
      imax = np.asarray(np.round([0, ncontour//2]), dtype = int);
  else:
    imax = np.sort(np.asarray(peaks[np.argsort(peaks[:,1])[-2:],0], dtype = int))
  #print imax  
  
  ### calcualte sides and midline
  u1 = np.linspace(us[imax[0]], us[imax[1]], npoints-2)
  x1, y1 =  splev(u1, cinterp, der = 0);
  left = np.vstack([x1,y1]).T;
  
  u2 = np.linspace(us[imax[0]], us[imax[1]]-1, npoints-2);
  u2 = np.mod(u2,1);
  x2, y2 = splev(u2, cinterp, der = 0);
  right = np.vstack([x2,y2]).T;
  
  center = (left + right) / 2;
  center = np.vstack([leftcurve[0], center, leftcurve[-1]]);
  
  #plt.plot(center[:,0], center[:,1], 'k')
  
  if with_width:
    width = np.vstack([0, np.linalg.norm(left - right, axis = 0) + w, 0]);
    cw = np.hstack([center, width[:,np.newaxis]]);
    cw = resample_curve(cw, npoints);
    center = cw[:,:2]; width = cw[:,-1];
    return center, width
  else:
    return resample_curve(center, npoints);


def center_from_sides(left, right, npoints = all, nsamples = all, resample = False, with_width = False, smooth = 0,  nneighbours = all, method = 'projection'):
  """Finds middle line between the two side curves
  
  Arguments:
    left, right (nx2 array): vertices of the left and right curves
    npoints (int or all): number of points of the mid line
    nsamples (int or all): number of sample points to contruct midline
    with_width (bool): if True also return estimated width
    nneighbours (int or all): number of neighbouring points to include for projection
    method ({'projection', 'vonroi', 'mean'}): the method to calculate midline
  
  Returns:
    nx2 array of midline vertices
  """
  
  if method == 'projection':
    return center_from_sides_projection(left, right, npoints = npoints, nsamples = nsamples, resample = resample, with_width = with_width, nneighbours = nneighbours, smooth = smooth);
  elif method == 'vonroi':
    return center_from_sides_vonroi(left, right, npoints = npoints, nsamples = nsamples, resample = resample, with_width = with_width, smooth = smooth);
  else:
    return center_from_sides_mean(left, right, npoints = npoints, nsamples = nsamples, resample = resample, with_width = with_width);


  
##############################################################################
### Center Line Bending



def lift_cirular(phi):
  """Lifts circular/angular valued curve to avoid discontinuities, assumes phi in range [-pi,pi]"""
  phip = phi >= 0;
  phipi = np.where(phip)[0] - 1;
  if len(phipi) > 0 and phipi[0] < 0:
    phipi = phipi[1:];
  
  phin = phi < 0;
  phini = np.where(phin)[0] - 1;
  if len(phini) > 0 and phini[0] < 0:
    phini = phini[1:];  
    
  cp = np.where(phip[phini])[0];
  cn = np.where(phin[phipi])[0];
  cp = phini[cp];
  cn = phipi[cn];

  lift = np.zeros(phi.shape);
  for i in cp:
    lift[i+1:] += 2 * np.pi;
  for i in cn:
    lift[i+1:] -= 2 * np.pi;
    
  return phi + lift;


  
def theta_from_center_discrete(center, npoints = all, nsamples = all, resample = False, smooth = 0):
  """Calculates bending angle theta along the center line using discrete mesh
  
  Arguments:
    center (nx2 array): center line
    npoints (int or all): number of final sample points for the center line
    nsamples (int) or all): number of sample points to construct theta
    resample (bool): forces uniform resampling if True
    smooth (float): smoothing factor for final sampling
  
  Returns:
    array: uniformly sampled bending angle along the center line
    float: absolute orientation with respect to vertical reference [1,0] at center of the worm
  
  Note:
    There are npoints - 2 angles along the center line, so the returned array 
    will be of length npoints-2.
    The returned points are samples of the deivative of the angle of the tangent 
    :math:`\\theta`: along the curve.
    The theta values are thus obtained via :math:`\that(s) \\approx = \\Delta \theta / \\Delta s`  
    and thus will be rescaled by the inverse 1/(npoints-2).
    The rescaling ensures that the spline represenation and integration return
    theta and phi.
  """
  
  nc = center.shape[0];
  
  if npoints is all:
    npoints = nc;
  if nsamples is all:
    nsamples = nc;
  n2 = (nsamples-1)//2;
  n2a= (nsamples-2)//2;
  
  # resample center lines 
  if nsamples != nc or resample:
    centercurve = resample_curve(center, nsamples);
  else:
    centercurve = center;

  #vectors along center line
  centervec = np.diff(centercurve, axis = 0);
  
  # orientation
  t0 = np.array([1,0], dtype = float); #vertical reference
  t1 = (centervec[n2] + centervec[n2a])/2.0;
  orientation = np.mod(np.arctan2(t0[0], t0[1]) - np.arctan2(t1[0], t1[1]) + np.pi, 2 * np.pi) - np.pi;
  
  # xy
  xy = centercurve[n2];
  
  # thetas
  theta = np.arctan2(centervec[:-1,0], centervec[:-1,1]) - np.arctan2(centervec[1:,0], centervec[1:,1]);
  theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi;
  theta *= (nsamples-1);
  
  #length
  length = np.linalg.norm(centervec, axis = 1).sum();

  if npoints != nsamples or resample:
    theta = resample_curve(theta, npoints - 2, smooth = smooth);
  
  return theta, orientation, xy, length



def theta_from_center_spline(center, npoints = all, nsamples = all, resample = False, smooth = 0):
  """Calculates bending angle theta along the center line using splines
  
  Arguments:
    center (nx2 array): center line
    npoints (int or all): number of final sample points for the center line
    nsamples (int) or all): number of sample points to construct theta
    resample (bool): forces uniform resampling if True
    smooth (float): smoothing factor for final sampling
  
  Returns:
    array: uniformly sampled bending angle along the center line
    float: absolute orientation with respect to vertical reference [1,0] at center of the worm
  
  Note:
    There are npoints - 2 angles along the center line, so the returned array will be of length npoints-2
  """
  
  nc = center.shape[0];
  
  if npoints is all:
    npoints = nc;
  if nsamples is all:
    nsamples = nc;
  
  #if nsamples % 2 != 1:
  #  raise RuntimeWarning('number of sample points expected to be odd adding a sample point!')
  #  nsamples += 1;
  n2 = (nsamples-1)//2;
  
  # resample center lines 
  if nsamples != nc or resample:
    centercurve = resample_curve(center, nsamples);
  else:
    centercurve = center;

  #vectors along center line
  centervec = np.diff(centercurve, axis = 0);
    
  # xy
  xy = centercurve[n2];
  
  #length
  length = np.linalg.norm(centervec, axis = 1).sum();
  
  # thetas
  phi = np.arctan2(centervec[:-1,0], centervec[:-1,1]);
  phi = lift_cirular(phi);
  phisp = Spline(phi);
  
  theta = phisp.derivative();
  
  orientation = float(theta(0.5));
  
  points = np.linspace(0,1,npoints-2);
  theta = theta(points);
  
  return theta, orientation, xy, length



def center_from_theta_discrete(theta, orientation = 0, xy = [0,0], length = 1, npoints = all, nsamples = all, resample = False, smooth = 0, with_normals = False):
  """Constructs center line from theta on discrete mesh
  
  Arguments:
    theta (nx2 array): angles along center line
    orientation (float): absolute orientation of the center line
    xy (2 array): absolute position of the center line
    length (float): length of center line
    npoints (int or all): number of final sample points along center line
    nsamples (int or all): number of smaple points to construct center line
    resample (bool): for resampling if True
    smooth (float): smoothing factor for final sampling

  Returns
    npointsx2 array: the sample points along the center line
  
  Note:
    The theta sample length is 2 less than the curve sample length.
    The absolute angle is integral of theta so that for discrete integration 
    we multiply this curve with the discretized smaple length of 1/
  """

  nt = theta.shape[0];
  
  if npoints is all:
    npoints = nt + 2;
  if nsamples is all:
    nsamples = nt + 2;
  n2 = (nsamples-1)//2;
  n2a= (nsamples-2)//2;
  
  # resample center lines 
  if nsamples != nt + 2 or resample:
    itheta = resample_curve(theta, nsamples - 2);
  else:
    itheta = theta;
  
  #cos/sin
  delta = 1.0 / (nsamples-1); # Delta s for discrete integration nsample points -> nsample-1 segments
  itheta = np.cumsum(np.hstack([0, itheta])) * delta;
  itheta += orientation - (itheta[n2]+itheta[n2a])/2.0;
  cos = np.cos(itheta);
  sin = np.sin(itheta);
  
  x = np.cumsum(cos);
  y = np.cumsum(sin);
  center = np.vstack([x,y]).T;
  center = np.vstack([[0,0], center]);
  center = float(length) * delta * center;
  center += xy - (center[n2] + center[n2a+1])/2.0;
  
  if npoints != nsamples or resample:
    center = resample_curve(center, npoints, smooth = smooth);
  
  if with_normals:
    if npoints != nsamples:
      itheta = resample_curve(itheta, npoints - 1);
    dtheta = np.diff(itheta);
    itheta += np.pi/2;
    itheta = np.hstack([itheta, itheta[-1]]);
    itheta[1:-1] -= dtheta / 2;
    return center, np.vstack([np.cos(itheta), np.sin(itheta)]).T;
  else:
    return center;


def center_from_theta_spline(theta, orientation = 0, xy = [0,0], length = 1, npoints = all, nsamples = all, resample = False, smooth = 0, with_normals = False):
  """Constructs center line from theta via spline integration
  
  Arguments:
    theta (nx2 array or Spline): angles along center line
    orientation (float): absolute orientation of the center line
    xy (2 array): absolute position of the center line
    length (float): length of center line
    npoints (int or all): number of final sample points along center line
    nsamples (int or all): number of smaple points to construct center line
    resample (bool): for resampling if True
    smooth (float): smoothing factor for final sampling

  Returns
    npointsx2 array: the sample points along the center line
  """
  
  if isinstance(theta, np.ndarray):
    nt = theta.shape[0];  
    if npoints is all:
      npoints = nt + 2;
    theta = np.hstack([theta]);
    theta = Spline(theta);
  
  if nsamples is all:
    nsamples = npoints; 
  
  s = np.linspace(0,1,nsamples)
  istheta = theta.integral();
  ithetac = float(istheta(0.5));
  itheta = istheta(s); 
  #itheta = np.hstack([0, itheta]);
  itheta += orientation - ithetac;
  
  cos = np.cos(itheta);
  sin = np.sin(itheta);
  
  # integrate
  #s = np.linspace(0,1,nsamples-1);
  x = UnivariateSpline(s, cos).antiderivative();
  y = UnivariateSpline(s, sin).antiderivative();
  
  #s = np.linspace(0,1,nsamples);
  center = np.vstack([x(s),y(s)]).T;
  #center = np.vstack([[0,0], center]);
  center = float(length) * center;
  center += xy - np.array([x(0.5), y(0.5)]);
  
  if npoints != nsamples or resample:
    center = resample_curve(center, npoints, smooth = smooth);
  
  if with_normals:
    if npoints != nsamples:
      itheta = istheta(np.linspace(0,1,npoints));
      itheta += orientation - ithetac;
    #dtheta = np.diff(itheta);
    #itheta += np.pi/2;
    #itheta[1:-1] -= dtheta / 2;
    return center, np.vstack([np.cos(itheta), np.sin(itheta)]).T;
  else:
    return center;

#center_from_theta = center_from_theta_discrete;
#theta_from_center = theta_from_center_discrete;

center_from_theta = center_from_theta_discrete;
theta_from_center = theta_from_center_discrete;



def normals_from_theta_discrete(theta, orientation = 0, npoints = all, nsamples = all, resample = False):
  """Constructs normals from theta on discrete mesh
  
  Arguments:
    theta (nx2 array): angles along center line
    orientation (float): absolute orientation of the center line
    xy (2 array): absolute position of the center line
    length (float): length of center line
    npoints (int or all): number of final sample points along center line
    nsamples (int or all): number of smaple points to construct center line
    resample (bool): for resampling if True

  Returns
    npointsx2 array: the sample points along the center line
  
  Note:
    The theta sample length is 2 less than the curve sample length.
    The absolute angle is integral of theta so that for discrete integration 
    we multiply this curve with the discretized smaple length of 1/
  """

  nt = theta.shape[0];
  
  if npoints is all:
    npoints = nt + 2;
  if nsamples is all:
    nsamples = nt + 2;
  n2 = (nsamples-1)//2;
  n2a= (nsamples-2)//2;
  
  # resample theta 
  if nsamples != nt + 2 or resample:
    itheta = resample_curve(theta, nsamples - 2);
  else:
    itheta = theta;
  
  #cos/sin
  delta = 1.0 / (nsamples-1); # Delta s for discrete integration nsample points -> nsample-1 segments
  itheta = np.cumsum(np.hstack([0, itheta])) * delta;
  itheta += orientation - (itheta[n2]+itheta[n2a])/2.0;
  
  if npoints != nsamples:
    itheta = resample_curve(itheta, npoints - 1);
  
  dtheta = np.diff(itheta);
  itheta += np.pi/2;
  itheta = np.hstack([itheta, itheta[-1]]);
  itheta[1:-1] -= dtheta / 2;
  
  return np.vstack([np.cos(itheta), np.sin(itheta)]).T;


normals_from_theta = normals_from_theta_discrete;


def normals_from_center_discrete(center):
  """Construct normals along center line
  
  Arguments:
    center (nx2 array): center line
  
  Returns:
    array (nx2): normals along center points
  """
  
  theta, orientation, xy, length = theta_from_center_discrete(center);
  return normals_from_theta_discrete(theta, orientation);
  

def shape_from_theta_discrete(theta, width, orientation = 0, xy = [0,0], length = 1, npoints = all, nsamples = all, resample = False, smooth = 0, with_normals = False):
  """Construct center line from theta on discrete mesh
  
  Arguments:
    theta (nx2 array): angles along center line
    width (n array): width profile
    length (float): length of center line
    npoints (int or all): number of final smaple points along center line
    nsamples (int or all): number of smaple points to construct center line
    resample (bool): for resampling if True
    smooth (float): smoothing factor for final sampling

  Returns
    array: the sample points along the center line
    array: the sample points along the left side
    array: the sample points along the right side
    array: the normal vector along the center points (if with_normals is True)
  
  Note:
    The theta sample length is 2 less than the curve sample length
    Returned normals
  """

  center, normals = center_from_theta_discrete(theta, orientation = orientation, xy = xy, length = length, 
                                               npoints = npoints, nsamples = nsamples, resample = resample, smooth = smooth, 
                                               with_normals = True)

  #w = np.hstack([0, width, 0]);  # assume zero width at head an tail
  if width.shape[0] != center.shape[0]:
    w = resample_curve(width, npoints = center.shape[0]);
  else:
    w = width;
  w = np.vstack([w,w]).T;
  left  = center + 0.5 * w * normals;
  right = center - 0.5 * w * normals;

  if with_normals:
    return center, left, right, normals
  else:
    return center, left, right
  


def shape_from_theta_spline(theta, width, orientation = 0, xy = [0,0], length = 1, npoints = all, nsamples = all, resample = False, smooth = 0, with_normals = False):
  """Constructs center line from theta on discrete mesh
  
  Arguments:
    theta (nx2 array): angles along center line
    width (n array): width profile
    length (float): length of center line
    npoints (int or all): number of final smaple points along center line
    nsamples (int or all): number of smaple points to construct center line
    resample (bool): for resampling if True
    smooth (float): smoothing factor for final sampling

  Returns
    array: the sample points along the center line
    array: the sample points along the left side
    array: the sample points along the right side
    array: the normal vector along the center points (if with_normals is True)
  
  Note:
    The theta sample length is 2 less than the curve sample length
    Returned normals
  """

  center, normals = center_from_theta_spline(theta, orientation = orientation, xy = xy, length = length, 
                                             npoints = npoints, nsamples = nsamples, resample = resample, smooth = smooth, 
                                             with_normals = True)

  #w = np.hstack([0, width, 0]);  # assume zero width at head an tail
  if width.shape[0] != center.shape[0]:
    w = resample_curve(width, npoints = center.shape[0]);
  else:
    w = width;
  w = np.vstack([w,w]).T;
  left  = center + 0.5 * w * normals;
  right = center - 0.5 * w * normals;

  if with_normals:
    return center, left, right, normals
  else:
    return center, left, right


shape_from_theta = shape_from_theta_discrete;
#shape_from_theta = shape_from_theta_spline;



def shape_from_center_discrete(center, width, normals = None):
  """Get side lines from center and width
  
  Arguments:
    center (nx2 array): center line
    width (n array): width profile
    normals (nx2 or None): normals, if None calculate
  
  Returns:
    tuple of arrays (nx2): left, right lines
  """
  
  if width.shape[0] != center.shape[0]:
    w = resample_curve(width, npoints = center.shape[0]);
  else:
    w = width;
  w = np.vstack([w,w]).T;
  
  if normals is None:
    normals = normals_from_center_discrete(center);
  
  left  = center + 0.5 * w * normals;
  right = center - 0.5 * w * normals;

  return left, right


def test_theta():
  import numpy as np
  import matplotlib.pyplot as plt;
  import worm.geometry as sh;
  reload(sh);
  
  #test lift
  phi = np.array([0,1,2,3,-3, -2, -1, -2, -3, 3, 2]);
  phil = sh.lift_cirular(phi);
  plt.figure(100); plt.clf();
  plt.plot(phi);
  plt.plot(phil);

  #test center from theta
  reload(sh);
  nn = 152;
  th = np.ones(nn) * np.pi;
  th2 = np.sin(np.linspace(0, 2, nn)) * np.pi;  
  c = sh.center_from_theta_discrete(th);
  c2 = sh.center_from_theta_discrete(th, orientation =-np.pi/2, xy = [1,1]);
  c3 = sh.center_from_theta_discrete(th2, orientation = -np.pi, xy = [-1,-1]);
  
  plt.figure(1); plt.clf();
  plt.plot(c[:,0], c[:,1]);
  plt.plot(c2[:,0], c2[:,1]);
  plt.plot(c3[:,0], c3[:,1]);
  
  # center form theta spline
  reload(sh);
  sc = sh.center_from_theta_spline(th);
  sc2 = sh.center_from_theta_spline(th, orientation = -np.pi/2, xy = [1,1]);
  sc3 = sh.center_from_theta_spline(th2, orientation = -np.pi, xy = [-1,-1]);
  
  #plt.figure(2); plt.clf();
  plt.plot( sc[:,0],  sc[:,1]);
  plt.plot(sc2[:,0], sc2[:,1]);
  plt.plot(sc3[:,0], sc3[:,1]);
  plt.axis('equal')
  plt.tight_layout();
  
  from utils.timer import timeit
  @timeit
  def ts():
    return sh.center_from_theta_spline(th, orientation = -np.pi, xy = [-1,-1]);
  @timeit
  def td():
    return sh.center_from_theta_discrete(th, orientation = -np.pi, xy = [-1,-1]);
  cs = ts();
  cd = td();
  np.allclose(cs, cd);
  

  #test inversions
  reload(sh);
  nn = 11;
  th = np.ones(nn) * np.pi;
  th = np.sin(np.linspace(0, 2, nn)) * np.pi;  
  c = sh.center_from_theta_discrete(th, orientation = -np.pi, xy = [-1,-1]); 
  thi, oi, xyi, li  = sh.theta_from_center_discrete(c)
  ci = sh.center_from_theta_discrete(thi, oi, xyi, li);
  np.allclose(th, thi);
  np.allclose(-np.pi, oi);
  np.allclose([-1,-1], xyi);
  np.allclose(1, li)
  np.allclose(c,ci)
  
  plt.figure(1); plt.clf();
  plt.plot(c[:,0], c[:,1]);
  plt.plot(ci[:,0], ci[:,1]);

  # test normals
  reload(sh);
  th = np.sin(np.linspace(0, 2, 8)) * 3;  
  c,n = sh.center_from_theta(th, orientation = -np.pi, xy = [-1,-1], length = 10, with_normals=True);
  plt.figure(2); plt.clf();
  plt.plot(c[:,0], c[:,1]);
  cp = c + n;
  cm = c - n;
  for i in range(n.shape[0]):
    plt.plot([cp[i,0], cm[i,0]], [cp[i,1], cm[i,1]], 'r');
  plt.axis('equal')
  
  # test shape from theta
  reload(sh);
  th = np.sin(np.linspace(0, 2, 8)) * 0.3;  
  width = 1-np.cos(np.linspace(0, 2*np.pi, 10));
  center, left, right = sh.shape_from_theta(th, width, orientation = -np.pi, xy = [-1,-1], length = 10, with_normals=False)
  
  plt.figure(13); plt.clf();
  plt.plot(left[:,0]  ,left[:,1]  , 'g', linewidth= 3)
  plt.plot(right[:,0] ,right[:,1] , 'y', linewidth= 3)
  plt.plot(center[:,0],center[:,1], 'b')
  plt.axis('equal')
  
  #how does center line recovery work
  c2 = sh.center_from_sides(left, right, nneighbours = 30, nsamples = 100);  
  c3 = sh.center_from_sides_mean(left, right);
  plt.plot(c3[:,0], c3[:,1]);
  plt.plot(c2[:,0], c2[:,1], 'r');



##############################################################################
### Shape Detection from Image 

def shape_from_image(image, sigma = 1, absolute_threshold = None, threshold_factor = 0.95, 
                     ncontour = 100, delta = 0.3, smooth = 1.0, nneighbours = 4,
                     npoints = 21, nsamples = all,
                     verbose = False, save = None):
  """Detect non self intersecting shapes of the the worm
  
  Arguments:
    image (array): the image to detect worm from
    sigma (float or None): width of Gaussian smoothing on image, if None use raw image
    absolute_threshold (float or None): if set use this as the threshold, if None the threshold is set via Otsu
    threshold_level (float): in case the threshold is determined by Otsu multiply by this factor
    ncontour (int): number of vertices in the contour
    delta (float): min height of peak in curvature to detect the head
    smooth (float): smoothing to use for the countour 
    nneighbours (int): number of neighbours to consider for midline detection
    npoints (int): number of vertices in the final center and side lines of the worm
    nsamples (int): number of vertices in for center line detection
    verbose (bool): plot results
    save (str or None): save result plot to this file
  
  Returns:
    success (bool): if True the shape was successfully extracted (False if worm is curled up or to many contours)
    arrays (npointsx2): center, left, right side lines of the worm
    
  Note:
    This is a fast way to detect the worm shape, fails for worms intersecting themselves
  """
  
  ### smooth image
  if sigma is not None:
    imgs = filters.gaussian_filter(np.asarray(image, float), sigma);
  else:
    imgs = image;
   
  ### get contours
  if absolute_threshold is not None:
    level = absolute_threshold;
  else:
    level = threshold_factor * threshold_otsu(imgs);
  
  pts = detect_contour(imgs, level);
  
  if len(pts) == 0:
    if verbose:
      print "Could not detect worm: No countours found!";
    return False, np.zeros((npoints,2)), np.zeros((npoints,2)), np.zeros((npoints,2))  
  elif len(pts) == 1:
    pts, pts_inner = pts[0], None;    
  elif len(pts) == 2:
    if verbose:
      print "Found two countours, worm most likely curled, proceeding with outer or largest contour!";

    if pts[0].shape[0] < pts[1].shape[0]:
      i,o = 0,1;
    else:
      i,o = 1,0;
    if inside_polygon(pts[i], pts[o][0,:]):
      i,o = o,i;
    
    pts, pts_inner = pts[o], pts[i];
  else:
    if verbose:
      print "Could not detect worm: Found %d countours!" % len(pts);
    return False, np.zeros((npoints,2)), np.zeros((npoints,2)), np.zeros((npoints,2))
  
  ### interpolate outer contour
  cinterp, u = splprep(pts.T, u = None, s = smooth, per = 1) 
  us = np.linspace(u.min(), u.max(), ncontour)
  x, y = splev(us[:-1], cinterp, der = 0)

  ### curvature along the points
  dx, dy = splev(us[:-1], cinterp, der = 1)
  d2x, d2y = splev(us[:-1], cinterp, der = 2)
  k = (dx * d2y - dy * d2x)/np.power(dx**2 + dy**2, 1.5);
  
  ### find tail / head via peak detection
  #pad k to detect peaks on both sides
  nextra = 20;
  kk = -k; # negative curvature peaks are heads/tails
  kk = np.hstack([kk,kk[:nextra]]);
  peaks = find_peaks(kk, delta = delta);
  #print peaks.shape
  if peaks.shape[0] > 0:  
    peaks = peaks[peaks[:,0] < k.shape[0],:];
  
  #if peaks.shape[0] < 2:
  #  peaks = find_peaks(np.abs(k), delta = 0.05);
  
  if peaks.shape[0] < 2:
    if peaks.shape[0] == 1:
      # best guess is half way along the contour
      if verbose:
        print "Could only detect on curvature peak in contour, proceeding with opposite point as tail!"
      imax = np.sort(np.asarray(np.mod(peaks[0,0] + np.array([0,ncontour//2]), ncontour), dtype = int));
    else:
      if verbose:
        print "Could not detect any peaks in contour, proceeding with 0% and 50% of contour as head and tail!"
      imax = np.asarray(np.round([0, ncontour//2]), dtype = int);
  else:
    imax = np.sort(np.asarray(peaks[np.argsort(peaks[:,1])[-2:],0], dtype = int))
  #print imax
  
  ### calcualte sides and midline
  u1 = np.linspace(us[imax[0]], us[imax[1]], npoints)
  x1, y1 =  splev(u1, cinterp, der = 0);
  left = np.vstack([x1,y1]).T;
  
  u2 = np.linspace(us[imax[0]], us[imax[1]]-1, npoints);
  u2 = np.mod(u2,1);
  x2, y2 = splev(u2, cinterp, der = 0);
  right = np.vstack([x2,y2]).T;
  
  # midline 
  #xm = (x1 + x2) / 2; ym = (y1 + y2) / 2; # simple
  center, width = center_from_sides_projection(left, right, nsamples = nsamples, with_width = True, nneighbours = nneighbours);
  
  
  # worm center
  #xymintp, u = splprep(xym.T, u = None, s = 1.0, per = 0);  
  #xc, yc = splev([0.5], xymintp, der = 0)
  #xc = xc[0]; yc = yc[0];
  
  ### plotting
  if verbose:
    #print 'max k at %s' % str(imax)
    
    plt.figure(11); plt.clf();
    plt.subplot(2,3,1)
    plt.imshow(image, interpolation ='nearest')
    plt.title('raw image');
    plt.subplot(2,3,2)
    plt.imshow(imgs)
    plt.title('smoothed image');
    plt.subplot(2,3,3)
    plt.imshow(imgs, cmap = 'gray')
    plt.contour(imgs, levels = [level])
    plt.title('contour dectection')
    
    #plot curvature
    plt.subplot(2,3,4)
    plt.plot(k)
    plt.scatter(imax, k[imax], c = 'r', s= 100);
    if peaks.shape[0] > 0:
      plt.scatter(peaks[:,0], -peaks[:,1], c = 'm', s= 40);
    plt.title('curvature')
    
    # shape detection
    plt.subplot(2,3,5)
    plt.imshow(image, cmap = 'gray', interpolation = 'nearest')
    plt.plot(left[:,0]  ,left[:,1]  , 'g', linewidth= 3)
    plt.plot(right[:,0] ,right[:,1] , 'y', linewidth= 3)
    plt.plot(center[:,0],center[:,1], 'b')
    # plot segments
    #for i in range(len(xm)):
    #    plt.plot([x1[i], x2[nu[i]]], [y1[i], y2[nu[i]]], 'm')
    #plot center
    n2 = (npoints-1)//2;
    plt.scatter(center[n2,0], center[n2,1], color = 'k')
    plt.scatter(x[imax], y[imax], s=150, color='r');
    plt.contour(imgs, levels = [level])
    plt.title('shape detection')
    
    #plot width profile    
    plt.subplot(2,3,6)
    plt.plot(width);
    plt.title('width')
    
    if isinstance(save, basestring):
      fig = plt.figure(11);
      fig.savefig(save);
  
  ### measure features
  # points
  #pos_head = np.array([x1[0], y1[0]])
  #pos_tail = np.array([x1[-1], y1[-1]])
  #pos_center = np.array([xc, yc]);
  
  # head tail distance:
  #dist_head_tail = np.linalg.norm(pos_head-pos_tail)
  #dist_head_center = np.linalg.norm(pos_head-pos_center)
  #dist_tail_center = np.linalg.norm(pos_tail-pos_center)
  
  #average curvature
  #dcx, dcy = splev(u, xymintp, der = 1)
  #d2cx, d2cy = splev(u, xymintp, der = 2)
  #ck = (dcx * d2cy - dcy * d2cx)/np.power(dcx**2 + dcy**2, 1.5);
  #curvature_mean = np.mean(ck);
  #curvature_variation = np.sum(np.abs(ck))
  
  ### returns
  success = pts_inner is None;
  return success, center, left, right, width





def contour_from_image(image, sigma = 1, absolute_threshold = None, threshold_factor = 0.95, 
                       verbose = False, save = None):
  """Detect the boundary contour(s) of the worm 
  
  Arguments:
    image (array): the image to detect worm from
    sigma (float or None): width of Gaussian smoothing on image, if None use raw image
    absolute_threshold (float or None): if set use this as the threshold, if None the threshold is set via Otsu
    threshold_level (float): in case the threshold is determined by Otsu multiply by this factor
    verbose (bool): plot results
    save (str or None): save result plot to this file
  
  Returns:
    tuple of arrays (npointsx2): outer and potentially inner contour line
  """
  
  ### smooth image
  if sigma is not None:
    imgs = filters.gaussian_filter(np.asarray(image, float), sigma);
  else:
    imgs = image;
   
  ### get contours
  if absolute_threshold is not None:
    level = absolute_threshold;
  else:
    level = threshold_factor * threshold_otsu(imgs);
  
  cts = detect_contour(imgs, level);
  
  if len(cts) == 0:
    if verbose:
      print "Could not detect worm: No countours found!";
    #outer, inner = None, None;
    cts = ();
  elif len(cts) == 1:
    #outer, inner = cts[0], None;    
    cts = cts;
  elif len(cts) == 2:
    if cts[0].shape[0] < cts[1].shape[0]:
      i,o = 0,1;
    else:
      i,o = 1,0;
    if inside_polygon(cts[i], cts[o][0,:]):
      i,o = o,i;
    #outer, inner = cts[o], cts[i];
    cts = (cts[o], cts[i]);
  else:
    if verbose:
      print "Found %d countours!" % len(cts);
    #sort by length as a guess for outer contour
    l = [len(c) for c in cts];
    cts = tuple([cts[i] for i in np.argsort(l)]);
  
  ### plotting
  if verbose:    
    plt.figure(11); plt.clf();
    plt.subplot(2,3,1)
    plt.imshow(image, interpolation ='nearest')
    plt.title('raw image');
    plt.subplot(2,3,2)
    plt.imshow(imgs)
    plt.title('smoothed image');
    plt.subplot(2,3,3)
    plt.imshow(imgs, cmap = 'gray') 
    for c in cts:
      plt.plot(c[:,0], c[:,1])
    plt.title('contour dectection')
        
    if isinstance(save, basestring):
      fig = plt.figure(11);
      fig.savefig(save);
  
  return cts


def head_tail_from_contour(contours, ncontour = all, delta = 0.3, smooth = 1.0, with_index = False,
                            verbose = False, save = None, image = None):
  """Detect non self intersecting shapes of the the worm
  
  Arguments:
    ncontour (int): number of vertices in the contour
    delta (float): min height of peak in curvature to detect the head
    smooth (float): smoothing to use for the countour before peak detection
    with_index (bool): return with indices of high curvature points
    verbose (bool): plot results
    save (str or None): save result plot to this file
    image (array): optional image for plotting
  
  Returns:
    nx2 arrays: potential positions for the head and tail
  """
  
  if len(contours)==0:
    if with_index:
      return np.zeros((0,2)), np.zeros(0, dtype = int);
    else:
      return np.zeros((0,2));
  
  # onlt detect heads on outer contour as ts very unlikely tofind it in inner one
  if ncontour is not all:
    ncontour = contours[0].shape[0];
  
  cinterp, u = splprep(contours[0].T, u = None, s = smooth, per = 1) 
  us = np.linspace(u.min(), u.max(), ncontour)
  x, y = splev(us[:-1], cinterp, der = 0)

  ### curvature along the points
  dx, dy = splev(us[:-1], cinterp, der = 1)
  d2x, d2y = splev(us[:-1], cinterp, der = 2)
  k = (dx * d2y - dy * d2x)/np.power(dx**2 + dy**2, 1.5);
  
  ### find tail / head via peak detection
  #pad k to detect peaks on both sides
  nextra = 20;
  kk = -k; # negative curvature peaks are heads/tails
  kk = np.hstack([kk,kk[:nextra]]);
  kpeaks = find_peaks(kk, delta = delta);
  #print peaks.shape
  if kpeaks.shape[0] > 0:  
    kpeaks = kpeaks[kpeaks[:,0] < k.shape[0],:];
    idx = np.asarray(kpeaks[:,0], dtype = int);
    peaks = np.vstack([x[idx], y[idx]]).T;
  else:
    peaks = np.zeros((0,2));
    idx = np.zeros(0,dtype = int);
    
  ### plotting
  if verbose:
    plt.figure(11); plt.clf();
    if image is not None:
      plt.subplot(1,2,1)
      plt.imshow(image, cmap = 'gray') 
      for c in contours:
        plt.plot(c[:,0], c[:,1])
      plt.plot(x,y);
      if len(peaks)> 0:
        plt.scatter(peaks[:,0], peaks[:,1], c = 'm', s = 20);
      plt.title('contour dectection')
      plt.subplot(1,2,2);
    
    plt.plot(k)
    #plt.scatter(imax, k[imax], c = 'r', s= 100);
    if kpeaks.shape[0] > 0:
      plt.scatter(kpeaks[:,0], -kpeaks[:,1], c = 'm', s= 40);
    plt.title('curvature')
    
    if isinstance(save, basestring):
      fig = plt.figure(11);
      fig.savefig(save);
  
  if with_index:
    return peaks, idx
  else:
    return peaks;


def normals_from_contour(contours):
  """Returns normal vectors along the contours
  
  Arguments:
    contours (tuple): tuple of contours
  
  Returns:
    tuple: tuple of normals corresponding to the contours
    
  Note:
    Assumes first contour to be outer contour and optional second contour to be inner contour
    Also assumes closed contour with contours[i][0]==contours[i][-1] for all i.
  """
  
  #ithetas = [];
  nrmls = [];
  for cts in contours:
    #vectors along contour line
    centervec = np.diff(cts, axis = 0);
    
    # absolute orientation
    t0 = np.array([1,0], dtype = float); #vertical reference
    t1 = centervec[0];
    orientation = np.mod(np.arctan2(t0[0], t0[1]) - np.arctan2(t1[0], t1[1]) + np.pi, 2 * np.pi) - np.pi;
      
    # discrete thetas (no rescaling)
    theta = np.arctan2(centervec[:-1,0], centervec[:-1,1]) - np.arctan2(centervec[1:,0], centervec[1:,1]);
    theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi;
    theta = np.hstack([0, theta]);
    
    # integrate and rotate by pi/2 / half angle at point
    itheta = np.cumsum(theta);
    itheta += np.pi/2 + orientation;
    itheta -= theta / 2;
    itheta[0] -= theta[-1] / 2;
    
    #ithetas.append(itheta);
    nrmls.append(np.vstack([np.cos(itheta), np.sin(itheta)]).T);
  
  #nrmls = [];
  #if len(ithetas) > 0:
  #  nrmls.append(np.vstack([np.cos(ithetas[0]), np.sin(ithetas[0])]).T);
  #if len(ithetas) > 1:
  #  nrmls.append(np.vstack([np.cos(ithetas[1] + np.pi), np.sin(ithetas[1] + np.pi)]).T);
  
  return tuple(nrmls);



def contour_from_shape(left, right):
  """Convert the worm shape to a contour
  
  Arguments:
    left, right (nx2 arrays): left and right side of the worm
    
  Returns
    tuple: contours of the worm
  """
  
  poly = geom.Polygon(np.vstack([left, right[::-1,:]]));
  poly = poly.buffer(0)
  bdr = poly.boundary;
  if isinstance(bdr, geom.multilinestring.MultiLineString):
    cts = [];
    for b in bdr:
      x,y = b.xy;
      cts.append(np.vstack([x,y]).T);
  else: # no self intersections
    x,y = bdr.xy;
    cts = np.vstack([x,y]).T;
  
  return tuple(cts)  
  

def self_occlusions_from_center_discrete(center, width, margin = 0.01, shape = True, normals = None, left = None, right = None, as_index = True):
  """Returns points on the center line whose boundary points that are occulded by the worm itself
  
  Arguments:
    center (nx2 array): center line    
    width (n array): width of the worm at the given center point
    normals (nx2 array): normals along the center line
    left,right (nx2 array): sides of the worm
    margin (float): margin by which to reduce width in order to detect real insiders (small fraction of thw width)
    shape (bool): if true return indices of occluded boundary else the occluded center points
    as_index (bool): if true return the indices of the center points

    
  Returns
    array: coordinates or indices of the boundary / centre points that are occluded
  """

  if normals is None:
    normals = normals_from_center_discrete(center = center);
  
  if left is None or right is None:
    left,right = shape_from_center_discrete(center = center, width = width, normals = normals);
  
  poly = geom.Polygon(np.vstack([left, right[::-1,:]]));
  poly = poly.buffer(-margin);
  
  inleft = [poly.contains(geom.Point(xy)) for xy in left];
  inright = [poly.contains(geom.Point(xy)) for xy in right]; 
  if shape:
    if as_index:
      return (np.where(inleft)[0], np.where(inright)[0]);
    else:
      return left[inleft], right[inright];
  else:
    incenter = np.logical_or(inleft, inright);
    if as_index:
      return np.where(incenter)[0];
    else:
      return center[incenter];


def phi_spline_from_contour_spline(contour):
  """


def normals_from_contour_points(contour, points):
  """Find normals on the spline curve contour at locations of the parameterization index points
  
  
  


def match_shape_to_contour(left, right, normals, contour, sigmas = None, min_alignment = 0, search_radius = None, match_head_tail = True):
  """Match points of the shape points n the contour



##############################################################################
### Tests

def test():
  import numpy as np
  import matplotlib.pyplot as plt
  import worm.geometry as wgeo
  from interpolation.resampling import resample as resample_curve
  reload(wgeo)
  
  ## Center form sides
  t = np.linspace(0,10,50);
  aline = np.vstack([t, np.sin(t)+0.5]).T;
  aline[0] = [0,0];
  bline = np.vstack([t, np.sin(t)]).T;
  aline[0] = bline[0];  
  aline[-1] = bline[-1];
  aline = resample_curve(aline, npoints = 50);
  bline = resample_curve(bline, npoints = 50);
  
  cline = wgeo.center_from_sides(aline, bline, nsamples = 50, npoints = 50, smooth = 0.1, resample = True, method = 'projection');
  
  plt.figure(1); plt.clf();
  plt.plot(aline[:,0], aline[:,1]);
  plt.plot(bline[:,0], bline[:,1]);
  plt.plot(cline[:,0], cline[:,1]);

  reload(wgeo)
  cline = wgeo.center_from_sides_erosion(aline, bline, erode = 0.05, nsamples = 50, npoints = 50, smooth = 0.1, resample = True);
  
  plt.figure(2); plt.clf();
  plt.plot(aline[:,0], aline[:,1]);
  plt.plot(bline[:,0], bline[:,1]);
  plt.plot(cline[:,0], cline[:,1]);

  
  # image analysis  
  reload(wgeo)
  import analysis.experiment as exp;
  img = exp.load_img(t = 100000);
  wgeo.shape_from_image(img, npoints = 15, verbose = True)
  
  
  # contour detection
  reload(wgeo)
  import analysis.experiment as exp;
  i = 25620;
  i += 1;
  img = exp.load_img(t = 500000+i);
  #plt.figure(1); plt.clf();
  #plt.imshow(img);
  cts = wgeo.contour_from_image(img, verbose = True);
  
  reload(wgeo)
  cts = wgeo.contour_from_image(img, verbose = False);
  wgeo.head_tail_from_contours(cts, delta = 0.3, smooth = 1.0, verbose = True, image = img);
  
  reload(wgeo)
  cts = wgeo.contour_from_image(img, verbose = False);
  cts = tuple([resample_curve(c, npoints = 50) for c in cts]);
  nrmls = wgeo.normals_from_contours(cts);
  plt.figure(50); plt.clf();
  plt.imshow(img);
  for c,n in zip(cts,nrmls):
    plt.plot(c[:,0], c[:,1]);
    plt.scatter(c[:,0], c[:,1])
    for ci,ni in zip(c,n):
      cp = np.vstack([ci, ci+5*ni]);
      plt.plot(cp[:,0], cp[:,1], 'k');
  plt.axis('equal')
    
  
  #self occlusions
  import worm.model as wm;
  reload(wgeo)
  nn = 20;
  w = wm.WormModel(theta = np.hstack([np.linspace(0.1, 0.8, nn)*13, 13* np.linspace(0.9, 0.1, nn+1)]) , length = 150);
  left, right, center, normals, width = w.shape(with_center=True, with_normals=True, with_width=True);
  
  plt.figure(1); plt.clf();
  w.plot()
  
  ocl,ocr = wgeo.self_occlusions_from_center_discrete(center, width, margin = 0.01, normals = normals, left = left, right = right, shape = True, as_index = True);
  pts = left[ocl];
  plt.scatter(pts[:,0], pts[:,1], c ='m', s = 40);
  pts = right[ocr];
  plt.scatter(pts[:,0], pts[:,1], c ='m', s = 40);
  
  occ = wgeo.self_occlusions_from_center_discrete(center, width, margin = 0.01, normals = normals, left = left, right = right, shape = False, as_index = True);
  pts = center[occ];
  plt.scatter(pts[:,0], pts[:,1], c ='b', s = 40);
  plt.axis('equal')
  
  
if __name__ == "__main__":
  test_theta();
  test();