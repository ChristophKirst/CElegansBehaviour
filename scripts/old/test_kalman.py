# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:33:41 2016

@author: ckirst
"""
from functools import partial
import numpy as np


import signalprocessing.kalman as kalman
import worm.model as wmod;
import worm.geometry as wgeo;


####  Redefine residual and mean calculations
# as orientation variable is 2 pi periodic

def residual(parameter1, parameter2, angle_ids = -4):
    diff = parameter1 - parameter2;
    diff[diff[angle_ids] > np.pi]   -= 2*np.pi;
    diff[diff[angle_ids] <= -np.pi] += 2*np.pi;
    return diff;

def state_mean(sigmas, Wm, angle_ids = -4):
    n = sigmas[0].shape[0];
    x = np.zeros(n);
    ids = np.ones(n, dtype = bool);
    ids[angle_ids] = False;
    nangles = n - ids.sum();
    sum_sin = np.zeros(nangles);
    sum_cos = np.zeros(nangles);
    for i in range(len(sigmas)):
        s = sigmas[i]
        x[ids] += s[ids] * Wm[i]
        sum_sin += np.sin(s[angle_ids])*Wm[i]
        sum_cos += np.cos(s[angle_ids])*Wm[i]
    x[angle_ids] = np.mod(np.arctan2(sum_sin, sum_cos)+np.pi, 2*np.pi)-np.pi;
    return x


def move_worm(x, dt, worm):
    """Dynamic Model
    
    Note: 
      The parameter vector is:
        x[:nparameter_spline] = theta spline parameter
        x[[-4,-3]] = xy
        x[-2] = orientation
        x[-1] = speed
      See the WormModel class for details
    """
    worm.set_parameter(x);

    #shift the worm
    shift = worm.speed * dt;
    worm.move_forward(shift, with_xy = True, with_orientation = True);
    
    return worm.get_parameter();
  
    
def observe_worm(x, worm, contour, valid, search_radius=[5,20], min_alignment = 0, match_head_tail = None, verbose = False):
    """Observation Model, the distances of the worm to the contour"""
    #calculate distances
    worm.set_parameter(x);
    d = worm.distance_to_contour(contour, search_radius=search_radius, min_alignment = min_alignment, match_head_tail = match_head_tail, verbose = verbose);
    #nans are set to a large value
    print 'observation', np.isnan(d)
    d[np.isnan(d)] = np.max(search_radius);
    d = d[valid]; #only return valid points
    return d;


if __name__ == "__main__":

  ### Initialize 
  import numpy as np
  import matplotlib.pyplot as plt
  import scipy.ndimage.filters as filters  
  
  from interpolation.curve import Curve
  from interpolation.resampling import resample as resample_curve  
  import analysis.experiment as exp
  
  reload(wgeo); reload(wmod);
  
  ### Initialize Worm from Image
  npoints = 11;
  nobs = npoints*2-2; #d differences from the sides / head tail only once  
  worm = wmod.WormModel(npoints = npoints);
  nparameter = worm.nparameter; #full number of parameter
  
  t0 = 500000;
  threshold_factor = 0.9;
  absolute_threshold = None;
  
  img = exp.load_img(wid = 80, t = t0);  
  imgs = filters.gaussian_filter(np.asarray(img, float), 1.0); 
  worm.from_image(img, absolute_threshold = absolute_threshold, threshold_factor = threshold_factor, verbose = True);
  worm0 = worm.copy();
  
  plt.figure(1); 
  plt.subplot(2,3,2)
  worm0.plot(image = imgs)
  worm0.plot()
  
  
  ### Initialize Contour to match the worm to from another image
  img2 = exp.load_img(wid = 80, t = t0+10);
  
  plt.figure(2); plt.clf();
  cntrs = wgeo.contour_from_image(img2, sigma = 1, absolute_threshold = absolute_threshold, threshold_factor = threshold_factor, 
                                  verbose = True, save = None);
  cntr = resample_curve(cntrs[0], 100); #assume outer contour is first, only match to this
  contour = Curve(cntr, nparameter = 50);
  
  plt.figure(3); plt.clf();
  worm0.plot(image = img2);
  contour.plot();
                        
  ### Detect Head/Tail candidates
  plt.figure(3); plt.clf();
  match_head_tail = wgeo.head_tail_from_contour(cntrs, ncontour = all, delta = 0.3, smooth = 1.0, with_index = False,
                                                verbose = True, save = None, image = img2);
  
  
  #plt.figure(21); plt.clf();
  #plt.subplot(1,2,1)
  #dist = worm0.distance_to_contour(contour, verbose = True);
  #worm0.plot(color = 'k')
  #plt.subplot(1,2,2);
  #plt.plot(dist[:20])
  #plt.plot(dist[20:])
  
  ### Detect invalid points
  
  #potential self intersections
  valid_left, valid_right = worm.self_occlusions(margin = 0.01, with_bools = True);
  distance_args = dict(search_radius = [5,20], min_alignment = 0, match_head_tail = match_head_tail, verbose = True);
  plt.figure(15); plt.clf();
  distance = worm0.distance_to_contour(contour, **distance_args);
  valid_contour = np.logical_not(np.isnan(distance));
  valid = np.logical_and(valid_contour, np.hstack([valid_left[:-1], valid_right[1:]]));
  distance_args['verbose'] = False;
                                             
  ### Initialize the motion and observation models
                                                
  fx = partial(move_worm, worm = worm);
  hx = partial(observe_worm, worm = worm, contour = contour, **distance_args); 
  
  x_state_mean = partial(state_mean, angle_ids = -4);
  x_residual = partial(residual, angle_ids = -4);                           
  
  sigma_points = kalman.sigma_points.JulierSigmaPoints(nparameter, kappa =0.01);
  
  
  ### Initialize the Kalman filter
  ukf = kalman.UnscentedKalmanFilter(dim_x = nparameter, dim_z = nobs, dt=0.1, 
                                     hx = hx, fx = fx, points = sigma_points, 
                                     x_mean_fn = x_state_mean, x_residual = x_residual);
   
  
  ukf.x = worm0.get_parameter();
  ukf.P = 0.1 * np.diag(np.hstack([[1]*(npoints-2), [2*np.pi], [10, 10], 0.05]));  #  intial state correlations
  ukf.Q = 0.1 * np.diag(np.hstack([[1]*(npoints-2), [0.1 * 2*np.pi], [10, 10], 0.05]));  # state noise covariance
  ukf.R *= 1.0; # measurement noise convariance, distances should all have similar noise levels
  
  # test motion model
  #x1 = fx(ukf.x, 0.1);
  #x2 = fx(x1, 0.1);
  #plt.figure(9); plt.clf();
  #worm.set_parameter(x1);
  #worm.plot(image = img, color = 'k');
  #worm.set_parameter(x2);
  #worm.plot(color = 'g');
  #worm0.plot(color = 'b');
  #plt.tight_layout();
  
  # test single step
  observation = np.zeros(nobs); # the observation is a perfect fit of distances
  ukf.predict();
  x_pred = ukf.x; 
  ukf.update(observation, valid = valid);  
  
  
  plt.figure(20); plt.clf();
  #plt.subplot(1,2,1)
  worm.set_parameter(ukf.x);
  worm.plot(image = img2, color = 'g');
  worm0.plot(color = 'g');
  plt.axis('equal');
  #plt.subplot(1,2,2);
  worm.set_parameter(x_pred);
  worm.plot(color = 'r');
  worm0.plot(color = 'k');
  
  ### Run the filter on same contour
  nsteps = 3;

  worm_pars_predicted = np.zeros((nsteps, nparameter));
  worm_pars_updated = np.zeros((nsteps, nparameter));
  observation = np.zeros(nobs); # the observation is a perfect fit of distances
  for i in range(nsteps):
    ukf.predict();
    worm_pars_predicted[i] = ukf.x[:]
    
    ukf.update(observation, valid = valid);
    worm_pars_updated[i] = ukf.x[:];
  
  verbose = True;
  if verbose:
      plt.figure(99); plt.clf();
      for i in range(nsteps):
        plt.subplot(2,nsteps,i+1);
        worm.set_parameter(worm_pars_predicted[i]);
        worm.plot(image = img2, color = 'r');
        plt.title('prediction %d' % i);
        plt.axis('equal');
        plt.subplot(2,nsteps,(i+1)+nsteps);
        worm.set_parameter(worm_pars_updated[i]);
        worm.plot(image = img2, color = 'r');
        plt.title('update %d' % i);
        plt.axis('equal');
      plt.tight_layout();

