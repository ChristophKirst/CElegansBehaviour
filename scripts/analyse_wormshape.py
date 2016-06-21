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


from skimage.filters import threshold_otsu
import scipy.ndimage.filters as filters

from scipy.interpolate import splprep, splev


from signalprocessing.peak_detection import find_peaks


def analyse_shape(img, sigma = 1, threshold_level = 0.95, npts_contour = 100, npts_sides = 50, smooth = 1.0, verbose = False, save = None):
  """Detect and ,measure shape features"""

  #plt.figure(10); plt.clf();
  #plt.imshow(img, cmap = 'gray', clim = (0,256),  interpolation="nearest");
  #plt.figure(11); plt.clf();
  #plt.hist(img.flatten(), bins = 256);
 
  ### smooth image -> contour from plt
  imgs = filters.gaussian_filter(np.asarray(img, float), sigma);
   
  ### get shape
  cs = plt.contour(imgs, levels = [threshold_level * threshold_otsu(imgs)]);
  pts = cs.collections[0].get_paths()[0].vertices

  # interpolation 
  cinterp, u = splprep(pts.T, u = None, s = smooth, per = 1) 
  us = np.linspace(u.min(), u.max(), npts_contour)
  x, y = splev(us[:-1], cinterp, der = 0)

  # curvature along the points
  dx, dy = splev(us[:-1], cinterp, der = 1)
  d2x, d2y = splev(us[:-1], cinterp, der = 2)
  k = (dx * d2y - dy * d2x)/np.power(dx**2 + dy**2, 1.5);


  # find tail and head via peak detection
  #pad k to detect peaks on both sides
  nextra = 20;
  kk = np.abs(k);
  kk = np.hstack([kk,kk[:nextra]]);
  peaks = find_peaks(kk, delta = 0.15);
  peaks = peaks[peaks[:,0] < k.shape[0],:];
  
  #plt.figure(15); plt.clf();
  #plt.plot(kk);
  #plt.scatter(peaks[:,0], peaks[:,1], c = 'r');
  
  if peaks.shape[0] < 2:
    peaks = find_peaks(np.abs(k), delta = 0.05);
    
  if peaks.shape[0] < 2:
    if peaks.shape[0] == 1:
      # best guess is half way along the contour
      imax = np.sort(np.asarray(np.mod(peaks[0,0] + np.array([0,npts_contour/2]), 100)));
    else:
      imax = np.array([0,50]);
  else:
    imax = np.sort(np.asarray(peaks[np.argsort(peaks[:,1])[-2:],0], dtype = int))
  
  # calcualte paths along both sides
  
  u1 = np.linspace(us[imax[0]], us[imax[1]], npts_sides)
  x1, y1 =  splev(u1, cinterp, der = 0);
  
  u2 = np.linspace(us[imax[0]], us[imax[1]]-1, npts_sides);
  u2 = np.mod(u2,1);
  x2, y2 = splev(u2, cinterp, der = 0);
  
  # midline (simple)
  xm = (x1 + x2) / 2; ym = (y1 + y2) / 2;
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
    cs = plt.contour(imgs, levels = [threshold_level * threshold_otsu(imgs)])

    #plot contour detection
    plt.subplot(2,3,4)
    plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
    plt.scatter(pts[:,0], pts[:,1])
    plt.plot(x, y, 'b--')
    
    
    # shape detection
    plt.subplot(2,3,5)
    plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
    
    #plot lines and midline
    plt.plot(x1,y1, 'g', linewidth= 4)
    plt.plot(x2,y2, 'y', linewidth= 4)
    plt.plot(xm,ym,'b')

    # plot some segments for fun
    for i in range(len(xm)):
        plt.plot([x1[i], x2[i]], [y1[i], y2[i]], 'm')

    #plot center
    plt.scatter(xc, yc, color = 'k')
    plt.scatter(x[imax], y[imax], s=150, color='r')
    
    
    #plot curvature    
    plt.subplot(2,3,6)
    plt.plot(np.abs(k))
    plt.scatter(imax, kk[imax], c = 'r', s= 100);
    plt.scatter(peaks[:,0], peaks[:,1], c = 'm', s= 40);

    
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
  line_center = xym;
  line_left = np.vstack([x1, y1]);
  line_right = np.vstack([x2, y2]);
  
  return pos_center, pos_head, pos_tail, curvature_mean, curvature_variation, line_center, line_left, line_right
    
