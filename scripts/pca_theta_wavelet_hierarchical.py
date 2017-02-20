# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 03:26:39 2016

@author: ckirst
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sig

import analysis.experiment as exp;

### Load Data


## write speed data to npy for memmap use
#shape_speed = data.speed.shape;
#fspeed = np.memmap('%s_speed.npy' % strain, dtype='float32', mode='w+', shape = shape_speed);
#fspeed[:] = data.speed[:];
#del fspeed

### Create numpy file for results
widths = np.array([2**i for i in range(17)]);

wt_shape = (data.nworms, widths.shape[0], shape_speed[1]);
fwt = np.memmap('%s_wavelets_hierarchical.npy' % strain, dtype = 'float32', mode = 'w+', shape = wt_shape);
del fwt;


### Run Parallel

def cwt(i):
  print 'processing worm %d' % i
  widths = np.array([2**ii for ii in range(17)]);
  fspeed = np.memmap('%s_speed.npy' % strain, dtype='float32', mode='r', shape = shape_speed);
  signal = fspeed[i,:];
  cwtm = sig.cwt(signal, sig.ricker, widths);  
  fwt = np.memmap('%s_wavelets_hierarchical.npy' % strain, dtype = 'float32', mode = 'r+', shape = wt_shape);
  fwt[i,:,:] = cwtm;
  del fwt;
  

from multiprocessing import Pool, cpu_count;

pool = Pool(processes = cpu_count()-6);

pool.map(cwt, range(data.nworms));



### Plot

fwt = np.memmap('Scripts/%s_wavelets_hierarchical.npy' % strain, dtype = 'float32', mode = 'r', shape = wt_shape);

plt.figure(1); plt.clf();
nplt = 10; #fwt.shape[0];
vmin = fwt[:nplt,:,:].min();
vmax = fwt[:nplt,:,:].max();

vmin = max(vmin,-vmax);
vmax = min(vmax, -vmin);

for i in range(nplt):
  plt.subplot(nplt, 1,i+1);
  cwt = fwt[i].copy();
  for k in range(-200,200,1):
      cwt[:,data.stage_switch[i] + k] = vmax;
  plt.imshow(cwt,aspect='auto', cmap='PRGn',# interpolation = 'none',
             vmin=vmin, vmax=vmax);
plt.tight_layout();



import plot as fplt

import experiment as exp

wd = exp.scales_to_array(fwt, worms_first=False);

fplt.plot_array(wd)


s = [11,12,13,14,15,16];
plt.figure(10); plt.clf();
nplt = len(s);
for i,s in enumerate(s):
  plt.subplot(nplt,1,i+1);
  cwt = fwt[:,s,:].copy();
  #for k in range(-200,200,1):
  #    cwt[:,data.stage_switch[i] + k] = vmax;
  vmin = cwt.min(); vmax = cwt.max();
  vmin = max(vmin,-vmax); vmax = min(vmax, -vmin);
  red = 0.5;
  vmin *= red; vmax *= red;
  plt.imshow(cwt,aspect='auto', cmap='PRGn', # interpolation = 'none',
             vmin=vmin, vmax=vmax);
plt.tight_layout();


s = [1,2,3,4,5,6];
plt.figure(10); plt.clf();
nplt = len(s);
for i,s in enumerate(s):
  plt.subplot(nplt,1,i+1);
  cwt = fwt[:,s,:].copy();
  #for k in range(-200,200,1):
  #    cwt[:,data.stage_switch[i] + k] = vmax;
  vmin = cwt.min(); vmax = cwt.max();
  vmin = max(vmin,-vmax); vmax = min(vmax, -vmin);
  red = 0.5;
  vmin *= red; vmax *= red;
  plt.imshow(cwt,aspect='auto', cmap='PRGn', # interpolation = 'none',
             vmin=vmin, vmax=vmax);
plt.tight_layout();



plt.figure(2); plt.clf();
nplt = 10; #fwt.shape[1];

for i in range(nplt):
  plt.subplot(nplt, 1,i+1);
  cwt = fwt[:,i,:].copy();
  #for k in range(-200,200,1):
  #    cwt[:,data.stage_switch[i] + k] = vmax;
  vmin = cwt.min(); vmax = cwt.max();
  vmin = max(vmin,-vmax); vmax = min(vmax, -vmin);
  plt.imshow(cwt,aspect='auto', cmap='PRGn', # interpolation = 'none',
             vmin=vmin, vmax=vmax);
plt.tight_layout();