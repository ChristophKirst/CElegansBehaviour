# -*- coding: utf-8 -*-
"""
Detect plate boundaries
"""

import os
import numpy as np;

import matplotlib.pyplot as plt

import analysis.experiment as exp;
import scripts.preprocessing.file_order as fo;

dir_names = fo.directory_names;

strain = 'n2';
wid = 0;

nworms = len(fo.directory_names)

### Load Plate Images

import scipy.io
import glob;

def find_plate(strain = 'n2', wid = 0, verbose = True, save = None, thresholds = [160, 175]):
  
  fns = np.sort(np.unique(glob.glob(os.path.join(dir_names[wid], 'Traj*.mat'))))
  img_plate = scipy.io.loadmat(fns[-1])['Trace'];
  xy = exp.load(strain = 'n2', wid = wid, dtype = 'xy')
  
  ### Detect circles
  
  import cv2
   
  gray = 2**8 - np.array(np.sum(img_plate, axis = 2) / 3, dtype = 'uint8');
  
  th1 = thresholds[0]; th2 = thresholds[1];
  gray[gray < th1] = th1;
  gray[gray > th2] = th2;
  gray = np.array(gray, dtype = 'uint8');
  
  #plt.figure(8); plt.clf();
  #plt.subplot(1,2,1);
  #plt.imshow(gray, cmap = plt.cm.gray)
  #nn = 1500;
  #plt.scatter(np.round(xy[-nn:,0]-1), np.round(xy[-nn:,1]-1))
  #plt.subplot(1,2,2);
  #plt.imshow(gray > 200)
  #cimg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
  
  circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp = 2, minDist = 100, minRadius = 485, maxRadius = 500, param1 = 40, param2 = 50)
  
  # detect the correct dish  
  center = circles[0, :,:2]
  radius = circles[0, :, 2]
  
  xy = exp.load(strain = strain, wid = wid, dtype = 'xy')
  xyv = xy[np.logical_not(np.isnan(xy[:,0]))];
  pt = xyv[-1]
  
  if verbose:
    if circles is None:
      print 'worm %d / %d: no cirlces found!' % (wid, nworms)
    else:
      print 'worm %d / %d: %d circles found !' % (wid, nworms, len(circles[0]))
    
    fig = plt.figure(8); plt.clf();
    plt.imshow(gray, cmap = plt.cm.gray)
    ax = plt.gcf().gca();
    for (x, y, r) in circles[0]:
      # draw the outer circle
      ax.add_artist(plt.Circle((x,y),r, color = 'green',fill = False, linewidth = 2))
      # draw the center of the circle
      ax.add_artist(plt.Circle((x,y),2, color = 'green', fill = True))
  
    plt.title('Worm %s, Experiment: %s' % (wid, fo.experiment_names[wid]) )
    plt.show();

  
  plate = np.where(np.linalg.norm(center - pt, axis = 1) < radius)[0];
  
  if len(plate) != 1:
    print 'could not find plate for worm %d, %s!' % (wid, strain);
    assert False;
      
  plate = plate[0];
  x = center[plate,0]; y = center[plate,1];
  r = radius[plate]
  
  if  verbose:
    ax.add_artist(plt.Circle((x,y),r, color = 'red',fill = False, linewidth = 2))
    ax.add_artist(plt.Circle((x,y),2, color = 'red', fill = True))
    plt.pause(0.01)
    if save is not None:
      fig.savefig(save);
    
  return [x,y,r];


#find_plate(strain = 'n2', wid = 2, verbose = True, save = None, thresholds = [165, 195])


roi = [find_plate(strain = 'n2', wid = w, verbose = True, save = None, thresholds = [160, 195]) for w in range(nworms)]


roi = np.array(roi);

fn = os.path.join(exp.data_directory, 'n2_roi.npy')
np.save(fn, roi)

print 'result written to %s' % fn

#for wid in range(nworms):
#  fn = os.path.join(exp.data_directory, 'n2_roi_w=%d_s=all.npy' % wid);
#  np.save(fn, roi[wid]);














#
#
#### Clipboard
#
#### Detect plates
#
#from skimage import color
#from skimage.feature import canny
#from skimage.transform import hough_ellipse
#from skimage.draw import ellipse_perimeter
#
## Load picture, convert to grayscale and detect edges
#image_rgb = img_plate;
#image_gray = color.rgb2gray(image_rgb)
#edges = canny(image_gray, sigma=1.0,
#              low_threshold=None, high_threshold=None)
#
#plt.figure(1); plt.clf();
#for i,p in enumerate([image_gray, image_rgb, edges]):
#  plt.subplot(3,1,i+1);
#  plt.imshow(p)
#
#
## Perform a Hough Transform
## The accuracy corresponds to the bin size of a major axis.
## The value is chosen in order to get a single high accumulator.
## The threshold eliminates low accumulators
#result = hough_ellipse(edges, accuracy=20, threshold=250,
#                       min_size=100, max_size=120)
#result.sort(order='accumulator')
#
## Estimated parameters for the ellipse
#best = list(result[-1])
#yc, xc, a, b = [int(round(x)) for x in best[1:5]]
#orientation = best[5]
#
## Draw the ellipse on the original image
#cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
#image_rgb[cy, cx] = (0, 0, 255)
## Draw the edge (white) and the resulting ellipse (red)
#edges = color.gray2rgb(edges)
#edges[cy, cx] = (250, 0, 0)
#
#fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
#
#ax1.set_title('Original picture')
#ax1.imshow(image_rgb)
#
#ax2.set_title('Edge (white) and result (red)')
#ax2.imshow(edges)
#
#plt.show()

