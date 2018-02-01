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
from scipy.interpolate import splev, splprep #, UnivariateSpline


import cv2

#from interpolation.spline import Spline
#from interpolation.curve import Curve

from interpolation.resampling import resample as resample_curve, resample_nd

#from interpolation.intersections import curve_intersections_discrete;

from signalprocessing.peak_detection import find_peaks

from imageprocessing.contours import detect_contour


import worm.geometry as wgeo

##############################################################################
### Worm width profile

def default_width(npoints = 21):
  """Default width profile for a adult worm
  
  Arguments:
    npoints (int): number of sample points
    
  Returns:
    array: width profile
  
  Note: 
    This might need to be adjusted to data / age etc or directly detected from 
    images (see :func:`shape_from_image`).
  """
  def w(x):
    a = 9.56 * 0.5;
    b = 0.351;
    return a * np.power(x,b)*np.power(1-x, b) * np.power(0.5, -2*b);
  
  if npoints is None:
    return w;
  else:
    return w(np.linspace(0, 1, npoints));





def center_from_sides_min_projection(left, right, npoints = all, nsamples = all, resample = False, with_width = False, smooth = 0, center_offset = 2, iterations = 3, verbose = False):
  """Finds middle line between the two side curves using projection method with minimal  advancements
  
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
    
  #plt.scatter(*leftcurve.T, c = 'w', s = 80);
  #plt.scatter(*rightcurve.T, c = 'w', s = 80);
  #plt.scatter(*rightcurve[0], c = 'm', s = 80);
  #plt.scatter(*leftcurve[0], c = 'b', s = 80);
  #print leftcurve.shape
  #print rightcurve.shape
  #return;
  
  #head tail always include as first last data point
  center_offset += 1;  
  
  # calculate center
  full_left = [leftcurve[i] for i in range(center_offset)];
  full_right = [rightcurve[i] for i in range(center_offset)];
  
  il = center_offset-1; ir = center_offset-1;
  leftline  = geom.LineString( leftcurve[ir:ir+2]); 
  rightline = geom.LineString(rightcurve[il:il+2]);
  
  il = center_offset; ir = center_offset;
  left_point  = geom.Point(leftcurve[il,0], leftcurve[il,1]);
  right_point = geom.Point(rightcurve[ir,0], rightcurve[ir,1]); 
  
  while il < nsamples-center_offset and ir < nsamples-center_offset:
    #print il,ir
    ul = leftline.project(right_point, normalized = True);
    ur = rightline.project(left_point, normalized = True);
    #print ul,ur
    #print left_point, right_point
    #print (il, ir, ul, ur)
    #print left_point.coords[0], leftcurve[il-1],  leftcurve[il]
    if ul == ur:
      l0 = leftcurve[il - 1]; l1 = leftcurve[il];
      r0 = rightcurve[ir - 1]; r1 = rightcurve[ir];
      q = np.argmin(np.linalg.norm([r0-l1, r1-l0, r1-l1], axis = 1));
      #print q, np.linalg.norm([r0-l1, r1-l0, r1-l1], axis = 1)
      
      if q == 0: #progress on left
        full_left.append(l1);
        full_right.append(r0);
        leftline = geom.LineString(leftcurve[il:il+2]);
        il+=1;
        left_point = geom.Point(leftcurve[il,0], leftcurve[il,1]);
      elif q ==1: # progress on right
        full_left.append(l0);
        full_right.append(r1);
        rightline = geom.LineString(rightcurve[ir:ir+2]);
        ir+=1;
        right_point = geom.Point(rightcurve[ir,0], rightcurve[ir,1]);
      else: # progress both
        full_left.append(left_point.coords[0]);
        full_right.append(right_point.coords[0]);
        leftline = geom.LineString(leftcurve[il:il+2]);
        rightline = geom.LineString(rightcurve[ir:ir+2]);
        il+=1;
        ir+=1;
        left_point = geom.Point(leftcurve[il,0], leftcurve[il,1]);
        right_point = geom.Point(rightcurve[ir,0], rightcurve[ir,1]);
    elif ul < ur: # add center from right
      full_left.append(leftline.interpolate(ul, normalized = True).coords[0]);
      full_right.append(right_point.coords[0]);
      rightline = geom.LineString(rightcurve[ir:ir+2]);
      ir+=1;
      right_point = geom.Point(rightcurve[ir,0], rightcurve[ir,1]);
    else:
      full_left.append(left_point.coords[0]);
      full_right.append(rightline.interpolate(ur, normalized = True).coords[0]);
      leftline = geom.LineString(leftcurve[il:il+2]);
      il+=1;
      left_point = geom.Point(leftcurve[il,0], leftcurve[il,1]);
  
  if il < ir:
    for i in range(il, nsamples-center_offset):
      full_left.append(leftcurve[i]);
      full_right.append(rightcurve[ir]);
  elif ir < il:
    for i in range(ir, nsamples-center_offset):
      full_left.append(leftcurve[il]);
      full_right.append(rightcurve[i]);
  
  
  full_left.extend([leftcurve[i] for i in range(-center_offset,0)]);
  full_right.extend([rightcurve[i] for i in range(-center_offset,0)]);
  
  full_left = np.array(full_left);
  full_right = np.array(full_right);
  
  #print full_left
  #print full_right
  center = (full_left + full_right)/2.0;
  center = resample_nd(center, npoints, smooth = smooth, iterations = 1);
  center = resample_nd(center, npoints, smooth = 0, iterations = 2); # homogenize distances between points
  
  #print center
  if verbose:
    plt.plot(*leftcurve.T, c = 'r');
    plt.plot(*rightcurve.T, c = 'b');
    for xy1,xy2 in zip(full_left, full_right):
      plt.plot([xy1[0], xy2[0]], [xy1[1], xy2[1]], c = 'g');
    #plt.scatter(*full_left[0], c = 'm', s = 50);
    plt.scatter(*leftcurve.T, c = np.arange(len(leftcurve)), s=  50);
    plt.scatter(*rightcurve.T, c = np.arange(len(leftcurve)), s = 50);

    
  
  if not with_width:
    return center;
  else:
    width = np.linalg.norm(full_left-full_right, axis = 1);
    width = 0.5 * resample_curve(width, npoints, smooth = 0);
    
    #lc = np.asarray(leftcurve, dtype = 'float32');
    #rc = np.asarray(rightcurve, dtype = 'float32');
    #cnt = np.vstack(lc, rc[::-1]);
    #width_l = np.array([np.min(np.linalg.norm(leftcurve - c, axis = 1)) for c in center])  
    #width_r = np.array([np.min(np.linalg.norm(rightcurve - c, axis = 1)) for c in center])  
    
    #width_l = np.array([cv2.pointPolygonTest(lc,(c[0],c[1]),True) for c in center]);
    #width_r = np.array([cv2.pointPolygonTest(rc,(c[0],c[1]),True) for c in center]);
    #width = (width_l + width_r); 
    #width = 2* np.min([width_l, width_r], axis = 0);
    #width[[0,-1]] = 0.0;
  
    return center, width


##############################################################################
### Shape Detection from Image 


def status(s):
  r = '';
  if s < 0:
    r += 'failed ';
  else:
    r += 'succes '
  
  cont_dict = {0 : 'no contour', 1 : 'single contour', 2 : 'single outer contour', 3 : 'multiple outer contours (contour hint)', 4 : 'multiple outer contours (size hint)', 5 : 'multiple outer contours (center)'}
  r += cont_dict[s % 10] + ' ';
  
  peak_dict = {0 : 'no peaks', 1 : 'multiple peaks (head tail hint)', 2 : 'multiple peaks (max)', 3: 'single peak (head tail hint)', 4 : 'single peak (half way)', 5 : 'no peaks (head tail hint)', 6: 'no peaks (0,n/2)'};
  r += peak_dict[s/100 % 10];
  
  if s/1000 > 0:
    r += ' (reduced peak detection)';
  if s/10000 > 0:
    r += ' (reduced threshold)';
  
  return r;
  


def shape_from_image(image, sigma = 1, absolute_threshold = None, threshold_factor = 0.95, 
                     ncontour = 100, delta = 0.3, smooth_head_tail = 1.0, smooth_left_right = 1.0, smooth_center = 10,
                     npoints = 21, center_offset = 3, 
                     threshold_reduce = 0.9, contour_hint = None, size_hint = None, min_size = 20, 
                     delta_reduce = 0.5, head_tail_hint = None,
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
    nsamples (int): number of vertices for center line detection
    verbose (bool): plot results
    save (str or None): save result plot to this file
  
  Returns:
    status (bool): 0 the shape was successfully extracted, otherwise id of what method was used or which failure
    arrays (npointsx2): center, left, right side lines of the worm
    
  Note:
    This is a fast way to detect the worm shape, fails for worms intersecting themselves
  """
  
  ### smooth image
  if sigma is not None:
    imgs = cv2.GaussianBlur(np.asarray(image, dtpye = float), ksize = (sigma, sigma), sigmaX = 0);
  else:
    imgs = image;
   
  ### get contours
  if absolute_threshold is not None:
    level = absolute_threshold;
  else:
    level = threshold_factor * threshold_otsu(imgs);
  
  pts, hrchy = detect_contour(imgs, level, with_hierarchy = True);
  if verbose:
    print("Found %d countours!" % len(pts));
    plt.subplot(2,3,3)
    plt.imshow(imgs, cmap = 'gray')
    for p in pts:
      plt.plot(p[:,0], p[:,1], c = 'red');
    
    plt.contour(imgs, levels = [level])
    plt.title('contour dectection')       
  
  status = 0;   
  if len(pts) == 0:
    if threshold_reduce is not None:
      pts, hrchy = detect_contour(imgs, threshold_reduce * level, with_hierarchy = True);
      status += 10000; # indicate we reduced the threshold !
    
    if len(pts) == 0: # we cannot find the worm and give up...      
      return -1-status, np.zeros((npoints,2)), np.zeros((npoints,2), np.zeros((npoints,2)), np.zeros(npoints))

  
  if len(pts) == 1:
    pts = pts[0];
    status += 1; # one contour only
  
  else:   # length is  >= 2
    # remove all contours that are children of others

    outer = np.where( np.logical_not(np.any(hrchy, axis = 0)) )[0];
    areas = np.array([cv2.contourArea(pts[o]) for o in outer]);
    outer = outer[areas > 0];
    #print outer
    
    if len(outer) == 0: # we cannot find the worm and give up...      
      return -2-status, np.zeros((npoints,2)), np.zeros((npoints,2), np.zeros((npoints,2)), np.zeros(npoints))
    
    elif len(outer) == 1:
      pts = pts[outer[0]]; # only one outer contour (worm mostlikely curled)
      status += 2;
      status += 10; # indicate there is an inner contour
      
    else:
      # is there contour with similar centroid and size to previous one
      moments = [cv2.moments(pts[o]) for o in outer];
      centroids = np.array([[(m["m10"] / m["m00"]), (m["m01"] / m["m00"])] for m in moments]);
      
      if contour_hint is not None:
        dist = [cv2.matchShapes(pts[o], contour_hint, 2,0) for o in outer];
        imin = np.argmin(dist);
        pts = pts[outer[imin]]; 
        status += 3;
        
      elif size_hint is not None:
        dist = np.array([cv2.contourArea(pts[o]) for o in outer]);
        dist = np.abs(dist - size_hint);
        imin = np.argmin(dist);
        status += 4;
        pts = pts[outer[imin]]; 
      
      
      else:
        #take most central one
        dist = np.linalg.norm(centroids - np.array(image.shape)/2, axis = 1);
        area = np.array([cv2.contourArea(pts[o]) for o in outer]);
        
        iarea = np.where(area > min_size)[0];
        dmin = np.argmin(dist[iarea]);
        imin = iarea[dmin]
        
        status += 5;
        pts = pts[outer[imin]];
        
      #check if contour has children
      #if np.sum(hrchy[:,-1] == outer[imin]) > 0:
      if hrchy[outer[imin]].sum() > 0:
        status += 10;
        
  print status, len(pts)
  
  ### interpolate outer contour
  nextra = min(len(pts)-1, 20); # pts[0]==pts[-1] !!
  #print pts[0], pts[-1]
  #ptsa = np.vstack([pts[-nextra:], pts, pts[:nextra]]);
  #cinterp, u = splprep(ptsa.T, u = None, s = smooth_head_tail, per = 0, k = 4) 
  #print pts
  cinterp, u = splprep(pts.T, u = None, s = smooth_head_tail, per = 1, k = 5) 
  #u0 = u[nextra]; u1 = u[-nextra-1];
  u0 = 0; u1 = 1;
  #print splev([0,1], cinterp, der = 2);
  #return
  
  us = np.linspace(u0, u1, ncontour+1)[:-1];
  x, y = splev(us, cinterp, der = 0)
  dx, dy = splev(us, cinterp, der = 1)
  d2x, d2y = splev(us, cinterp, der = 2)
  k = (dx * d2y - dy * d2x)/np.power(dx**2 + dy**2, 1.5);
  kk = np.hstack([k[-nextra:], k, k[:nextra]]);
  
  #plt.figure(5); plt.clf();
  #plt.plot(x,y);
  #plt.figure(19);
  
  peak_ids, peak_values = find_peaks(kk, delta = delta);
  
  if len(peak_ids) > 0:
    peak_ids -= nextra;
    peak_values = peak_values[peak_ids>= 0]; peak_ids = peak_ids[peak_ids>= 0];  
    peak_values = peak_values[peak_ids < ncontour]; peak_ids = peak_ids[peak_ids < ncontour]; 
  
  if len(peak_ids) < 2 and delta_reduce is not None:
    peak_ids, peak_values = find_peaks(kk, delta = delta * delta_reduce);
    status += 1000; # indicated we reduced peak strength

    if len(peak_ids) > 0:
      peak_ids -= nextra;
      peak_values = peak_values[peak_ids>= 0]; peak_ids = peak_ids[peak_ids>= 0];  
      peak_values = peak_values[peak_ids < ncontour]; peak_ids = peak_ids[peak_ids < ncontour]; 

  if verbose:
    print('Found %d peaks' % len(peak_ids));

  # find head and tail
  if len(peak_ids) >= 2:
    if head_tail_hint is not None:
      xy = np.array([x[peak_ids], y[peak_ids]]).T;
      dist_h = np.linalg.norm(xy - head_tail_hint[0], axis = 1);
      dist_t = np.linalg.norm(xy - head_tail_hint[1], axis = 1);
      
      i_h = np.argmin(dist_h);
      i_t = np.argmin(dist_t);      
      if i_h == i_t:
        dist_t[i_t] = np.inf;
        i_t = np.argmin(dist_t);        
      
      imax = [peak_ids[i_h], peak_ids[i_t]];   
      status += 100;
      
    else:
      # best guess are the two highest ones
      imax = np.sort(np.asarray(peak_ids[np.argsort(peak_values)[-2:]], dtype = int))
      status += 200;
  
  elif len(peak_ids) == 1:
    if head_tail_hint is not None:
      xy = np.array([x[peak_ids[0]], y[peak_ids[0]]]).T;
      dist_h = np.linalg.norm(xy - head_tail_hint[0], axis = 1);
      dist_t = np.linalg.norm(xy - head_tail_hint[1], axis = 1);
      
      #closest point on contour to previous missing head/tail:
      xy = np.array([x, y]).T;   
      if dist_h <= dist_t:
        dist = np.linalg.norm(xy - head_tail_hint[1], axis = 1);          
        i_h = peak_ids[0]; 
        i_t = np.argmin(dist);
        if i_h == i_t:
          dist[i_t] = np.inf;
          i_t = np.argmin(dist);        
        imax = [i_h, i_t];  
      else:
        dist = np.linalg.norm(xy - head_tail_hint[0], axis = 1);
        i_t = peak_ids[0]; 
        i_h = np.argmin(dist);
        if i_h == i_t:
          dist[i_h] = np.inf;
          i_h = np.argmin(dist);        
        imax = [i_h, i_t];  
      status += 300;
  
    else:
      # best guess is half way along the contour
      imax = np.sort(np.asarray(np.mod(peak_ids[0] + np.array([0,ncontour//2]), ncontour), dtype = int));
      status += 400
  
  else: #peaks.shape[0] == 0
    if head_tail_hint is not None:
      xy = np.array([x, y]).T;      
      dist_h = np.linalg.norm(xy - head_tail_hint[0], axis = 1);
      dist_t = np.linalg.norm(xy - head_tail_hint[1], axis = 1);
      i_h = np.argmin(dist_h);
      i_t = np.argmin(dist_t);
      if i_h == i_t:
        dist_t[i_t] = np.inf;
        i_t = np.argmin(dist_t);        
      imax = [i_h, i_t];  
      status += 500;
    else:
      imax = np.asarray(np.round([0, ncontour//2]), dtype = int);
      status += 600;
  #print imax, status
  
  
  ### calcualte sides and midline
  if smooth_left_right is not None and smooth_head_tail != smooth_left_right:
    #cinterp, u = splprep(ptsa.T, u = None, s = smooth_left_right, per = 0, k = 4) 
    cinterp, u = splprep(pts.T, u = None, s = smooth_left_right, per = 1, k = 5) 
    #u0 = u[nextra]; u1 = u[-nextra-1];
    #us = np.linspace(u0, u1, ncontour);
  
  
  du = u1-u0;
  if imax[0] > imax[1]:
    ul = np.linspace(us[imax[0]], us[imax[1]]+du, ncontour);
    ul[ul >= u1] -= du;
  else:
    ul = np.linspace(us[imax[0]], us[imax[1]], ncontour);
  x1, y1 =  splev(ul, cinterp, der = 0);
  left = np.vstack([x1,y1]).T;
  
  if imax[0] > imax[1]:
    ur = np.linspace(us[imax[1]], us[imax[0]], ncontour);
  else:
    ur = np.linspace(us[imax[0]], us[imax[1]]-du, ncontour);
    ur[ur < u0] += du;
  x2, y2 = splev(ur, cinterp, der = 0);
  right = np.vstack([x2,y2]).T;
  
  #print u0,u1,ul,ur
  #print left[[0,-1]];
  #print right[[0,-1]];
  #return
  # center and midline 
  center, width = center_from_sides_min_projection(left, right, npoints = npoints, nsamples = ncontour, with_width = True, smooth = smooth_center, center_offset = center_offset);
  
  ### plotting
  if verbose:
    #print 'max k at %s' % str(imax)
    #plt.figure(11); plt.clf();
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
    if len(peak_ids) > 0:
      plt.scatter(peak_ids, peak_values, c = 'm', s= 40);
    plt.title('curvature')
    
    # shape detection
    plt.subplot(2,3,5)
    plt.imshow(image, cmap = 'gray', interpolation = 'nearest')
    
    left1, right1 = wgeo.shape_from_center_discrete(center, width);
    plt.plot(left1[:,0]  , left1[:,1]  , 'r', linewidth= 2)
    plt.plot(right1[:,0] , right1[:,1] , 'r', linewidth= 2)    
    
    
    plt.plot(left[:,0]  , left[:,1]  , 'g', linewidth= 1)
    plt.plot(right[:,0] , right[:,1] , 'y', linewidth= 1)
    plt.plot(center[:,0], center[:,1], 'b')
    
    if smooth_left_right is not None and smooth_head_tail != smooth_left_right:
      #  x, y = splev(us, cinterp, der = 0);
      plt.plot(x,y, 'm', linewidth = 1);
    
    
    # plot segments
    #for i in range(len(xm)):
    #    plt.plot([x1[i], x2[nu[i]]], [y1[i], y2[nu[i]]], 'm')
    #plot center
    n2 = (npoints-1)//2;
    plt.scatter(center[n2,0], center[n2,1], color = 'k', s = 150)
    #plt.scatter(x[imax], y[imax], s=150, color='r');
    plt.contour(imgs, levels = [level])
    plt.title('shape detection')
    
    #plot width profile    
    plt.subplot(2,3,6)
    plt.plot(width);
    plt.title('width')
    
    if isinstance(save, basestring):
      fig = plt.gcf();
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
  #success = pts_inner is None;
  return status, left, right, center, width

