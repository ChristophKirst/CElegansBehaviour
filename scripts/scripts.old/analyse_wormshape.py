# -*- coding: utf-8 -*-
"""
Worm Shapes

Long Term Behaviour Analysis of C-elegans

Matlab to Numpy Data Conversion Routines for Worm Images
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'


import numpy as np
import matplotlib.pyplot as plt


from numpy import ma
import matplotlib._contour as _contour


def detect_contour(img, level):
  
  #parameter
  mask = None;
  corner_mask = True;
  nchunk = 0;

  #prepare image data
  z = ma.asarray(img, dtype=np.float64); 
  ny, nx = z.shape;
  x, y = np.meshgrid(np.arange(nx), np.arange(ny));

  #find contour
  contour_generator = _contour.QuadContourGenerator(x, y, z.filled(), mask, corner_mask, nchunk)
  vertices = contour_generator.create_contour(level);
  
  return vertices;


from matplotlib.path import Path

def inside_polygon(vertices, point):
  p = Path(vertices);
  return p.contains_point(point);




from scipy.spatial.distance import cdist

def find_midline(xl,yl, xr, yr, lookahead = 10, offset = 5):
  """Finds the optimal midline given the side lines of a worm"""
  n = xl.shape[0];
  xm = np.zeros(n);
  ym = np.zeros(n);
  nu = np.zeros(n, dtype = int);
  
  #find the mapping that minimizes the distance between left and right sides  
  for i in range(offset):
    xm[i] = (xl[i] + xr[i])/2;
    ym[i] = (yl[i] + yr[i])/2;
    nu[i] = i;
    
  j = offset; jm = min(j+lookahead, n);
  for i in range(offset, n - offset):
    #find closest point from acutal ref point
    dists = cdist([[xl[i], yl[i]]], np.array([xr[j:jm], yr[j:jm]]).T);
    k = dists.argmin();
    
    xm[i] = (xl[i] + xr[j+k])/2;
    ym[i] = (yl[i] + yr[j+k])/2;
    nu[i] = j+k;
    
    j = j + k; jm = min(j+lookahead, n);
  
  for i in range(n-offset, n):
    xm[i] = (xl[i] + xr[i])/2;
    ym[i] = (yl[i] + yr[i])/2;
    nu[i] = i;
  
  return xm,ym,nu



from skimage.filters import threshold_otsu

import scipy.ndimage.filters as filters
from scipy.interpolate import splprep, splev

from signalprocessing.peak_detection import find_peaks


def analyse_shape(img, sigma = 1, threshold_level = 0.95, npts_contour = 100, npts_sides = 21, smooth = 1.0, verbose = False, save = None):
  """Detect and ,measure shape features"""
  
  ### smooth image
  imgs = filters.gaussian_filter(np.asarray(img, float), sigma);
   
  ### get contours
  level = threshold_level * threshold_otsu(imgs);
  pts = detect_contour(imgs, level);
  
  if len(pts) == 0 or len(pts) > 2:
    raise RuntimeError('found %d contours, expected 1 or 2!' % len(pts));
    #return np.zeros(2), np.zeros(2), np.zeros(2), 0, 0, np.zeros((2, npts_sides)), np.zeros((2, npts_sides)), np.zeros((2, npts_sides))
    
  if len(pts) == 1:
    pts, pts_inner = pts[0], None;
  
  else: #len(pts) == 2
    if pts[0].shape[0] < pts[1].shape[0]:
      i,o = 0,1;
    else:
      i,o = 1,0;
    
    if inside_polygon(pts[i], pts[o][0,:]):
      i,o = o,i;
    
    pts, pts_inner = pts[o], pts[i];
  
  ### interpolate outer contour
  cinterp, u = splprep(pts.T, u = None, s = smooth, per = 1) 
  us = np.linspace(u.min(), u.max(), npts_contour)
  x, y = splev(us[:-1], cinterp, der = 0)

  ### curvature along the points
  dx, dy = splev(us[:-1], cinterp, der = 1)
  d2x, d2y = splev(us[:-1], cinterp, der = 2)
  k = (dx * d2y - dy * d2x)/np.power(dx**2 + dy**2, 1.5);


  ### find tail and head via peak detection

  #pad k to detect peaks on both sides
  nextra = 20;
  kk = -k; # negative curvature peaks are heads/tails
  kk = np.hstack([kk,kk[:nextra]]);
  peaks = find_peaks(kk, delta = 0.3);
  peaks = peaks[peaks[:,0] < k.shape[0],:];
  
  #plt.figure(15); plt.clf();
  #plt.plot(kk);
  #plt.scatter(peaks[:,0], peaks[:,1], c = 'r');

  
  #if peaks.shape[0] < 2:
  #  peaks = find_peaks(np.abs(k), delta = 0.05);
  print peaks.shape  
  if peaks.shape[0] < 2:
    if peaks.shape[0] == 1:
      # best guess is half way along the contour
      imax = np.sort(np.asarray(np.mod(peaks[0,0] + np.array([0,npts_contour/2]), npts_contour), dtype = int));
    else:
      imax = np.array([0,50]);
  else:
    imax = np.sort(np.asarray(peaks[np.argsort(peaks[:,1])[-2:],0], dtype = int))
  
  print imax
  
  # calcualte paths along both sides
  
  u1 = np.linspace(us[imax[0]], us[imax[1]], npts_sides)
  x1, y1 =  splev(u1, cinterp, der = 0);
  
  u2 = np.linspace(us[imax[0]], us[imax[1]]-1, npts_sides);
  u2 = np.mod(u2,1);
  x2, y2 = splev(u2, cinterp, der = 0);
  
  # midline (simple)
  #xm = (x1 + x2) / 2; ym = (y1 + y2) / 2;
  xm,ym,nu = find_midline(x1,y1,x2,y2, offset = 3);
  xym = np.vstack([xm,ym]);
  xymintp, u = splprep(xym, u = None, s = 1.0, per = 0);
  
  # worm center
  xc, yc = splev([0.5], xymintp, der = 0)
  xc = xc[0]; yc = yc[0];

  if verbose:
    print 'max k at %s' % str(imax)
    
    plt.figure(11); plt.clf();
    plt.subplot(2,3,1)
    plt.imshow(img, interpolation ='nearest')
    plt.subplot(2,3,2)
    plt.imshow(imgs)
    plt.subplot(2,3,3)
    plt.imshow(img, cmap = 'gray')
    plt.contour(imgs, levels = [threshold_level * threshold_otsu(imgs)])
    #plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
    #plt.scatter(pts[:,0], pts[:,1])
    #plt.plot(x, y, 'b--')


    #plot curvature
    plt.subplot(2,3,4)
    plt.plot(k)
    plt.scatter(imax, k[imax], c = 'r', s= 100);
    plt.scatter(peaks[:,0], -peaks[:,1], c = 'm', s= 40);

    
    # shape detection
    plt.subplot(2,3,5)
    plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
    
    #plot lines and midline
    plt.plot(x1,y1, 'g', linewidth= 4)
    plt.plot(x2,y2, 'y', linewidth= 4)
    plt.plot(xm,ym,'b')

    # plot segments
    for i in range(len(xm)):
        plt.plot([x1[i], x2[nu[i]]], [y1[i], y2[nu[i]]], 'm')

    #plot center
    plt.scatter(xc, yc, color = 'k')
    plt.scatter(x[imax], y[imax], s=150, color='r')
    
    
    #plot width profile    
    plt.subplot(2,3,6)
    
    w = np.zeros(len(xm));
    for i in range(len(xm)):
      w[i] = np.linalg.norm([x2[nu[i]]-x1[i], y2[nu[i]]- y1[i]]);
    plt.plot(w);
    
    if isinstance(save, basestring):
      fig = plt.figure(11);
      fig.savefig(save);


  ### measure features
  
  # points
  pos_head = np.array([x1[0], y1[0]])
  pos_tail = np.array([x1[-1], y1[-1]])
  pos_center = np.array([xc, yc]);
  
  # head tail distance:
  #dist_head_tail = np.linalg.norm(pos_head-pos_tail)
  #dist_head_center = np.linalg.norm(pos_head-pos_center)
  #dist_tail_center = np.linalg.norm(pos_tail-pos_center)
  
  #average curvature
  dcx, dcy = splev(u, xymintp, der = 1)
  d2cx, d2cy = splev(u, xymintp, der = 2)
  ck = (dcx * d2cy - dcy * d2cx)/np.power(dcx**2 + dcy**2, 1.5);
  curvature_mean = np.mean(ck);
  curvature_variation = np.sum(np.abs(ck))
  
  #cruves
  line_center = xym.T;
  line_left = np.vstack([x1, y1]).T;
  line_right = np.vstack([x2, y2]).T;
  
  return pos_center, pos_head, pos_tail, curvature_mean, curvature_variation, line_center, line_left, line_right


