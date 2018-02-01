# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 00:59:07 2018

@author: ckirst
"""

import numpy as np

import matplotlib.pyplot as plt

import cv2


import worm.geometry as wgeo
reload(wgeo)



def detect_endpoint(image, level, smooth = 20, n_contour = 80, head_guess = None, delta = 0.3, verbose = False):
  cnts, hrchy = wgeo.detect_contour(image, level, with_hierarchy=True);

  n_cnts =  len(cnts);
  print('Number of contours: %d' % n_cnts);

  # check if inner outer contour
  # find children of each contour
  children = [];
  for cid in range(n_cnts):
    children.append(np.where(hrchy[:,-1]==cid)[0]);
  # outer vs inner 
  nc = np.array([len(c) for c in children]);
  if n_cnts > 1 and np.all(nc == 0): # choose larges conotour
    oid = np.argmax([len(c) for c in cnts]);
    #iid = oid;
  else:
     oid = np.argmax(nc);
     #iid = np.argmin(nc);
  #if verbose:
  #  plt.imshow(blur[fid], interpolation = 'none');
  #  for c,co in zip([cnts[i] for i in [oid, iid]], ['red', 'blue']):
  #    plt.plot(c[:,0], c[:,1], color = co);
  
  # max curvature of outer contour
  pts = cnts[oid];
  
  nextra = min(len(pts)-1, 20); # pts[0]==pts[-1] !!
  ptsa = np.vstack([pts[-nextra-1:-1], pts, pts[1:nextra+1]]);
  cinterp, u = wgeo.splprep(ptsa.T, u = None, s = smooth, per = 0, k = 4) 
  u0 = u[nextra];
  u1 = u[-nextra-1];
  us = np.linspace(u0, u1, n_contour+1)
  dx, dy = wgeo.splev(us, cinterp, der = 1)
  d2x, d2y = wgeo.splev(us, cinterp, der = 2) 
  k = - (dx * d2y - dy * d2x)/np.power(dx**2 + dy**2, 1.5);
  
  x,y = wgeo.splev(us, cinterp, der = 0);
  
  if head_guess is not None:
    kk = np.hstack([k[-nextra-1:-1], k, k[1:nextra+1]]);
    peaks = wgeo.find_peaks(kk, delta = delta);  
    #print peaks
    if len(peaks) > 0:
      peaks[:,0] -= nextra;
      peaks = peaks[peaks[:,0] >= 0,:];
      peaks = peaks[peaks[:,0] < n_contour,:];
    if len(peaks) > 1:
      peaks = peaks[np.argsort(peaks[:,1])][:2];
      pi = np.asarray(peaks[:,0], dtype = int);
      pos = np.array([x[pi], y[pi]]).T;
      imax = np.argmin(np.linalg.norm(pos- head_guess, axis = 1));
      imax = int(peaks[imax,0]);
    else:
      imax = np.argmax(k);
  else:
    imax = np.argmax(k);
  
  if verbose:
    plt.figure(100); plt.clf();
    plt.subplot(1,2,1);
    plt.plot(k)
    #plt.scatter(imax, k[imax], c = 'r', s= 100)
    plt.scatter(imax, k[imax], c = 'm', s= 40);
    plt.subplot(1,2,2);
    plt.imshow(image);
    plt.plot(pts[:,0], pts[:,1]);
    plt.scatter(x[imax], y[imax], c = 'r', s= 40);
    plt.scatter(y[[0]], x[[0]], c = 'b', s= 60);

  return np.array([x[imax], y[imax]]);
  

import scipy.ndimage as ndi

def gaussian(shape, sigma):

    delta = np.zeros(shape);
    delta[shape[0]//2, shape[1]//2] = 1;
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return ndi.filters.gaussian_filter(delta, sigma)



def angle(v1, v2):
  #v1 = (v1.T/np.linalg.norm(v1, axis = 1)).T;
  #v2 = (v2.T/np.linalg.norm(v2, axis = 1)).T;
  return np.arctan2(v1[:,1], v1[:,0]) - np.arctan2(v2[:,1], v2[:,0]);
  
  

def unroll_center(image, level, center_guess = None, radius = 2.15, n_points = 45, search_angle = 90, blur = (5,5), 
                  delta_angle = 1, smooth = 20, n_contour = 80, remove = 128, remove_factor = 1.2, prior_kappa  = 2, 
                  orient = 1, verbose = False, writer = None):
  import scipy.ndimage as ndi
  
  if center_guess is not None:
    head_guess = center_guess[0];
  else:
    head_guess = None;
  
  ref = detect_endpoint(image, level, smooth = smooth, n_contour = n_contour, head_guess = head_guess, verbose =  verbose);

  if center_guess is not None:
    n_points = center_guess.shape[0];
    ht_guess = center_guess[[0,-1]];
    ht_i = np.argmin(np.linalg.norm(ref -ht_guess, axis = 1));
    if ht_i == 1:
      center_ref = center_guess[::-1];
      #orient = 1;
    else:
      center_ref = center_guess;
      #orient = 1;
    
    p1 = center_ref[1:]; p0 = center_ref[:-1];
    if radius is None:    
      radius = np.mean(np.linalg.norm(p1-p0, axis = 1));
  
  print radius;
  
  n_search = int(360.0 / delta_angle);
  delta_angle = 360.0 / n_search;
  delta_phi   = 2 * np.pi / n_search;
  t = np.linspace(0, 2 * np.pi, n_search+1)[:n_search];
  sin = radius*np.sin(t);
  cos = radius*np.cos(t);
  dd = int(search_angle / delta_angle);
  if center_guess is None:
    dr = np.arange(0,n_search);
  else:
    dp = center_ref[1] - center_ref[0];
    d0 = int(np.arctan2(dp[1], dp[0]) / delta_phi);
    dr = np.arange(d0-dd, d0+dd) % n_search;
    
    dp = center_ref[1:] - center_ref[:-1];
    alpha = angle(dp[1:], dp[:-1]);
    prior = np.exp(prior_kappa * np.cos(t));
  
  if blur is not None:
    blur = cv2.GaussianBlur(image, ksize = (5,5), sigmaX = 0);
  else:
    blur = image;
  
  center = np.zeros_like(center_ref);
  center[0] = ref;
  
  if remove is not None:
    shape = np.array(image.shape);
    gauss = gaussian(shape * 2 + 1, radius * 1.1);
    gauss = gauss / gauss.max();
  
  for k in range(n_points-1):
    x, y = cos[dr] + ref[0],  sin[dr] + ref[1];
    
    #max intensity
    zi = ndi.map_coordinates(blur, np.vstack((y,x)), order = 5)
    
    #bias towards previous angle
    if center_guess is not None:
      if k > 0:
        dalpha = - int(alpha[k-1] / delta_phi);
        print alpha[k-1], dalpha
        pr = np.arange(dalpha-dd, dalpha+dd) % n_search;
        pr = prior[pr];
        zi *= pr;
    
    zmax = np.argmax(zi);
    from_side = np.where(zi > np.abs(orient))[0];
    if orient != 0 and len(from_side) > 0:
      if orient > 0:
        zmax2 = from_side[0];
        if zmax > zmax2:
          zmax = zmax2;
      elif orient < 0:
        zmax2 = from_side[-1];
        if zmax < zmax2:
          zmax = zmax2;

    
    #new search region
    d0 = int(np.arctan2(y[zmax]-ref[1],  x[zmax]- ref[0]) / np.pi * 180);
    dr = np.arange(d0-dd, d0+dd) % n_search;
    
    #remove image   
    if remove is not None and k > 0:
      iref = np.asarray(center[k-1], dtype = int);
      blur = blur - remove_factor * max(0, (blur[iref[1], iref[0]]- 130)) * gauss[shape[0] - iref[1]: 2*shape[0] - iref[1], shape[1] - iref[0]: 2*shape[1] - iref[0]];
      
      #blur = cv2.circle(blur, tuple(np.asarray(center[k-1], dtype = int)), int(radius*remove_factor), remove, -1);
    
    #update ref and centers
    ref = [x[zmax], y[zmax]];
    center[k+1] = ref;  
    
    if verbose:
      if k == 0:
        plt.figure(23); 
        plt.clf(); 
        #plt.subplot(2,2,1);
        im = plt.imshow(blur);
      im.set_array(blur);
      #plt.subplot(2,2,1);
      plt.scatter(x,y, color = 'white', s = 2);
      plt.scatter(x[zmax], y[zmax], color = 'red', s = 2);
      plt.scatter(x[0], y[0], color = 'blue', s = 2);
      #plt.subplot(2,2,2);
      #plt.plot(zi);
      #plt.scatter(zmax, zi[zmax], color = 'red', s = 2);
      #if k > 0:
      #  plt.subplot(2,2,3);
      #  plt.plot(pr);
      #  plt.subplot(2,2,4);
      #  plt.plot(zi/pr);
      plt.tight_layout();

      plt.draw(); plt.pause(0.01);
      if writer is not None:
        writer.grab_frame();
      #if k ==5:
      #  break;
      
    # determine width: TODO
    
  return center

