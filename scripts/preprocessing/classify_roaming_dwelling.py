# -*- coding: utf-8 -*-
"""
Classify Behaviour into Roaming and Dwelling

"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import os
import glob
import matplotlib.pyplot as plt;
import scipy.io
import numpy as np

import analysis.experiment as exp

### Convert Coordinates in Folder

base_directory = '/run/media/ckirst/My Book'

fig_directory = '/home/ckirst/Science/Projects/CElegansBehaviour/Analysis/WormBehaviour/Figures/2017_01_20_TimeAnalysis/'

worm_directories = sorted(glob.glob(os.path.join(base_directory, 'Results*/*/')))

exp.experiment_directory = '/home/ckirst/Science/Projects/CElegansBehaviour/Experiment/ImageData'

nworms = len(worm_directories);


### Roaming Dwelling 

# Average speed and Rotation

rate = 3; # Hz sample rate

time_bin = 1*60; # s;

nbins = rate * time_bin; # number of time bins to calculate mean

v_data = [];
r_data = [];

for wid in range(nworms):
  print 'processing %d / %d' % (wid, nworms);
  xy = exp.load(strain = 'N2', dtype = 'xy', wid = wid, memmap=None);
  rt = exp.load(strain = 'N2', dtype = 'phi',wid = wid, memmap=None);
  
  # invalid positions
  inv = xy[:,0] == 1.0;
  xy[inv,:] = np.nan;
  
  # calculate speed
  dxy = np.diff(xy, axis = 0);
  v = np.sqrt(dxy[:,0]**2 + dxy[:,1]**2) * rate;
  nv = min(len(v), len(rt));
  
  # bin speed
  nvm = nv/nbins * nbins;
  v_mean = v[:nvm];
  v_mean = v_mean.reshape([nvm/nbins, nbins]);
  v_mean = np.nanmean(v_mean, axis = 1);
  v_mean = np.hstack([v_mean, [np.nanmean(v[nvm:])]]);
  
  # bin rot
  r_mean = rt[:nvm];
  r_mean = r_mean.reshape([nvm/nbins, nbins]);
  r_mean = np.nanmean(r_mean, axis = 1);
  r_mean = np.hstack([r_mean, [np.nanmean(rt[nvm:])]]);
  
  # time bin
  #ts = int(exp_times[wid] / time_bin);
  #v_24hr[wid, ts:ts+len(v_mean)] = v_mean.copy();
  #r_24hr[wid, ts:ts+len(r_mean)] = r_mean.copy();
  
  v_data = np.hstack([v_data, v_mean.copy()]);
  r_data = np.hstack([r_data, r_mean.copy()]);
  

valid_data = np.logical_not(np.logical_or(np.isnan(v_data), np.isnan(r_data)));

v_th = np.percentile(v_data[valid_data], [97])[0];
#v_data_th = v_data[valid_data].copy();
#v_data_th[v_data_th > v_th] = th;

r_th = np.percentile(r_data[valid_data], [99.9])[0];
#r_data_th = r_data[valid_data].copy();
#r_data_th[r_data_th > th] = th;


nbins = 80
plt.figure(10); plt.clf();
plt.subplot(1, 2, 1);
res = plt.hist2d(r_data,v_data,  bins = nbins, range = [[0, r_th], [0, v_th]]);

plt.subplot(1, 2, 2);
res = plt.hist2d(r_data, np.log(v_data),  bins = nbins,  range = [[0, r_th], [np.nanmin(np.log(v_data)), np.log(v_th)]])


### Classify into Roaming and Dwelling ?

plt.figure(11); plt.clf();
plt.subplot(1,2,1);
plt.hist(r_data, bins = 128, range = [0,r_th])
plt.subplot(1,2,2);
plt.hist(v_data, bins = 128, range = [0,v_th])


for wid in range(nworms):
  rt = exp.load(strain = 'N2', dtype = 'phi',wid = wid, memmap=None);
  roam = rt < 1.42;

  exp.experiment_directory = '/home/ckirst/Science/Projects/CElegansBehaviour/Experiment/ImageData'
  fn_out = exp.filename(strain = 'N2', wid  = wid, dtype = 'roam');

  np.save(fn_out, roam);  
  