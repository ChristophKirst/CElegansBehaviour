# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 17:16:09 2016

@author: ckirst
"""

### Test skeletonization of intersecting worm
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters  

import worm.model as wmod;
import worm.geometry as wgeo;

import analysis.experiment as exp

### Initialize Worm from Image
npoints = 21;
nobs = npoints*2-2; #d differences from the sides / head tail only once  
worm = wmod.WormModel(npoints = npoints);
nparameter = worm.nparameter; #full number of parameter

t0 = 500000;
t0 = 529202;


threshold_factor = 0.9;
absolute_threshold = None;

img = exp.load_img(wid = 80, t = t0);  

plt.figure(1); plt.clf();
worm.from_image(img, absolute_threshold = absolute_threshold, threshold_factor = threshold_factor, verbose = True);
worm0 = worm.copy();


plt.subplot(2,3,2)
worm0.plot(image = img)
worm0.plot()



### Test Skeletonization

from skimage.morphology import skeletonize
from scipy.ndimage.morphology import distance_transform_edt

t1 = -15;
t1 +=1;
img = exp.load_img(wid = 80, t = t0+10+t1);  
imgs = filters.gaussian_filter(np.asarray(img, float), 1.0);
imgth = imgs < 70;
imgsk = skeletonize(imgth);


imgd = distance_transform_edt(imgth);


plt.figure(2); plt.clf();
plt.subplot(1,3,1);
plt.imshow(img, interpolation = 'none');
plt.subplot(1,3,2);
plt.imshow(1.0 * imgth+imgsk, interpolation = 'none');
plt.subplot(1,3,3);
plt.imshow(imgd, interpolation = 'none');





## get coneter line from skeleton

#imgth = imgs < level;  
skel = skeletonize(imgth);



def skeleton_to_line(skeleton):
  """Converts a binary skeleton to a line if possible
  
  Arguments:
    skeleton (array): 2d binary skeleton image
    
  Returns:
    line (nx2 array): ordered points of the skeleton pixel
  """
  
  def get_neighbourhood(img,x,y):
    nhood = np.zeros((x.shape[0],9), dtype = bool);
    for xx in range(3):
      for yy in range(3):
          w = 3 * xx + yy;
          idx = x+xx-1; idy = y+yy-1;
          nhood[:,w]=img[idx, idy];
    nhood[:, 1, 1] = 0;
    return nhood;  
  
  x,y = np.where(skeleton);
  nh = get_neighbourhood_2d(skeleton, x,y);

  #end points
  nn = nh.sum(axis = 1); 
  
  ep = np.where(nn==1)[0];
  if len(ep) != 2: # not a line
    
    
  
      nhood.shape = (nhood.shape[0], 3, 3);

 
  adj = {}; 
  for i,pos in enumerate(ids):
    posnh = np.where(nh[i]);
    adj[tuple(pos)] = [tuple(p + pos - 1) for p in np.transpose(posnh)]
  return nx.from_dict_of_lists(adj);





  


