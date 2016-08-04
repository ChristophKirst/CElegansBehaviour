# -*- coding: utf-8 -*-
"""
Worm Shapes

Routines to detect and process worm shapes

See also: 
  :mod:`model`
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'


import numpy as np
import matplotlib.pyplot as plt

import shapely.geometry as geom

from signalprocessing.resampling import resample_curve, resample_data

def midline_from_sides(left, right, npoints = all, nsamples = all, with_width = False):
  """Finds middle line between the two side curves
  
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
    npoints = max(nr,nl);
  
  leftcurve  = resample_curve(left, nsamples);
  rightcurve = resample_curve(right, nsamples);
  
  #leftline = geom.LineString(leftcurve);
  rightline = geom.LineString(rightcurve);
  
  #plt.figure(10); plt.clf();
  #plt.plot(leftcurve[:,0], leftcurve[:,1])
  #plt.plot(rightcurve[:,0], rightcurve[:,1])
  
  mid_points = np.zeros((nsamples,2));
  if with_width:
    w = np.zeros(nsamples);
  
  for i in range(nsamples):
    left_point = geom.Point(leftcurve[i,0], leftcurve[i,1]);
    right_point = rightline.interpolate(rightline.project(left_point))
    mid_points[i,:] = 0.5 * np.array([left_point.x + right_point.x, left_point.y + right_point.y]);
    
    if with_width:
      w[i] = np.linalg.norm(np.array([left_point.x - right_point.x, left_point.y - right_point.y]));
    
    #plt.plot([left_point.x,right_point.x], [left_point.y, right_point.y]);
  #plt.plot(mid_points[:,0], mid_points[:,1], 'red');
  
  if with_width:
    return resample_curve(mid_points, npoints), resample_data(w, npoints)
  else:
    return resample_curve(mid_points, npoints)

    
# without shapely
#from scipy.spatial.distance import cdist
#
#def find_midline(xl,yl, xr, yr, lookahead = 10, offset = 5):
#  """Finds the optimal midline given the side lines of a worm"""
#  n = xl.shape[0];
#  xm = np.zeros(n);
#  ym = np.zeros(n);
#  nu = np.zeros(n, dtype = int);
#  
#  #find the mapping that minimizes the distance between left and right sides  
#  for i in range(offset):
#    xm[i] = (xl[i] + xr[i])/2;
#    ym[i] = (yl[i] + yr[i])/2;
#    nu[i] = i;
#    
#  j = offset; jm = min(j+lookahead, n);
#  for i in range(offset, n - offset):
#    #find closest point from acutal ref point
#    dists = cdist([[xl[i], yl[i]]], np.array([xr[j:jm], yr[j:jm]]).T);
#    k = dists.argmin();
#    
#    xm[i] = (xl[i] + xr[j+k])/2;
#    ym[i] = (yl[i] + yr[j+k])/2;
#    nu[i] = j+k;
#    
#    j = j + k; jm = min(j+lookahead, n);
#  
#  for i in range(n-offset, n):
#    xm[i] = (xl[i] + xr[i])/2;
#    ym[i] = (yl[i] + yr[i])/2;
#    nu[i] = i;
#  
#  return xm,ym,nu


from skimage.filters import threshold_otsu

import scipy.ndimage.filters as filters
from scipy.interpolate import splprep, splev

from signalprocessing.peak_detection import find_peaks

from imageprocessing.contours import detect_contour, inside_polygon

def shape_from_image(image, sigma = 1, absolute_threshold = None, threshold_factor = 0.95, 
                     npoints_contour = 100, delta = 0.3, smooth = 1.0,
                     npoints = 21, 
                     verbose = False, save = None):
  """Detect non-overflapping shapes of the the worm
  
  Arguments:
    image (array): the image to detect worm from
    sigma (float or None): width of Gaussian smoothing on image, if None use raw image
    absolute_threshold (float or None): if set use this as the threshold, if None the threshold is set via Otsu
    threshold_level (float): in case the threshold is determined by Otsu multiply by this factor
    npoints_contour (int): number of vertices in the contour
    delta (float): min height of peak in curvature to detect the head
    smooth (float): smoothing to use for the countour 
    npoints (int): number of vertices in the final mid and side lines of the worm
    verbose (bool): plot results
    save (str or None): save result plot to this file
  
  Returns:
    success (bool): if True the shape was successfully extracted (False if worm is curled up or to many contours)
    arrays (npointsx2): mid, left, right side lines of the worm
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
  us = np.linspace(u.min(), u.max(), npoints_contour)
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
      imax = np.sort(np.asarray(np.mod(peaks[0,0] + np.array([0,npoints_contour/2]), npoints_contour), dtype = int));
    else:
      if verbose:
        print "Could not detect any peaks in contour, proceeding with 0% and 50% of contour as head and tail!"
      imax = np.asarray(np.round([0, npoints_contour/2]), dtype = int);
  else:
    imax = np.sort(np.asarray(peaks[np.argsort(peaks[:,1])[-2:],0], dtype = int))
  #print imax
  
  ### calcualte sides and midline
  u1 = np.linspace(us[imax[0]], us[imax[1]], npoints)
  x1, y1 =  splev(u1, cinterp, der = 0);
  
  u2 = np.linspace(us[imax[0]], us[imax[1]]-1, npoints);
  u2 = np.mod(u2,1);
  x2, y2 = splev(u2, cinterp, der = 0);
  
  # midline 
  #xm = (x1 + x2) / 2; ym = (y1 + y2) / 2; # simple
  xym, w = midline_from_sides(np.vstack([x1,y1]).T, np.vstack([x2,y2]).T, nsamples = 2 * npoints, with_width = True);
  xymintp, u = splprep(xym.T, u = None, s = 1.0, per = 0);
  
  # worm center
  xc, yc = splev([0.5], xymintp, der = 0)
  xc = xc[0]; yc = yc[0];
  
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
    plt.plot(x1,y1, 'g', linewidth= 4)
    plt.plot(x2,y2, 'y', linewidth= 4)
    plt.plot(xym[:,0],xym[:,1],'b')
    # plot segments
    #for i in range(len(xm)):
    #    plt.plot([x1[i], x2[nu[i]]], [y1[i], y2[nu[i]]], 'm')
    #plot center
    plt.scatter(xc, yc, color = 'k')
    plt.scatter(x[imax], y[imax], s=150, color='r');
    plt.title('shape detection')


    #plot width profile    
    plt.subplot(2,3,6)
    plt.plot(w);
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
  line_center = xym;
  line_left = np.vstack([x1, y1]).T;
  line_right = np.vstack([x2, y2]).T;
  width = w;
  success = pts_inner is None;
  return success, line_center, line_left, line_right, width


def test():
  import numpy as np
  import matplotlib.pyplot as plt
  import worm.shape as shp
  reload(shp)
  
  t = np.linspace(0,10,20);
  aline = np.vstack([t, np.sin(t)+1]).T;
  bline = np.vstack([t, np.sin(t)]).T;
  
  mline = shp.midline_from_sides(aline, bline, nsamples = 100);
  
  plt.figure(1); plt.clf();
  plt.plot(aline[:,0], aline[:,1]);
  plt.plot(bline[:,0], bline[:,1]);
  plt.plot(mline[:,0], mline[:,1]);
  
  import analysis.experiment as exp;
  
  img = exp.load_img(t = 100000);
  shp.shape_from_image(img, verbose = True)  