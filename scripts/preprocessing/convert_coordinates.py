# -*- coding: utf-8 -*-
"""
Behavioural Development

Long Term Behaviour Analysis of C-elegans

Preprocessing Matlab to Numpy Data Conversion Routines for Worm Data
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


#%% Strain

import scripts.preprocessing.filenames as f;
strain = 'n2' # 'tph1', 'npr1'
strain = 'tph1'
strain = 'npr1'
strain = 'daf7'
nworms, exp_names, dir_names = f.filenames(strain = strain);


#%% Convert Coordinates

for wid, wdir in enumerate(dir_names):
  print 'processing worm %d/%d' % (wid, nworms);
  
  file_pattern = os.path.join(wdir, 'corrd*.mat');
  fns = np.sort(np.unique(np.array(glob.glob(file_pattern))));

  coords = [];
  for i,fn in enumerate(fns):
    #print 'processing %d / %d' % (i, len(fns));
    data = scipy.io.loadmat(fn)
    xy_part = data['x_y_coor'];
    coords.append(xy_part);
      
  xy = np.vstack(coords);
  
  # (1,1) is invalid coordinates
  inv = np.logical_and(xy[:,0] == 1.0, xy[:,1] == 1.0);
  xy[inv,:] = np.nan;
  
  print 'worm %d: len = %d' % (wid, xy.shape[0])
  fn_out = exp.filename(strain = strain, wid  = wid);
  np.save(fn_out, xy);


#%% Precalculate Speed

import analysis.features as feat;

delta = 3;


for wid in range(nworms):
  print 'processing worm %d/%d' % (wid, nworms);
  
  xy = exp.load(strain = strain, wid = wid, memmap = None);

  v = feat.speed(xy, delta = delta);
  fn_out = exp.filename(strain = strain, wid  = wid, dtype = 'speed');
  np.save(fn_out, v);
  
  r = feat.rotation(xy, delta = delta);
  fn_out = exp.filename(strain = strain, wid  = wid, dtype = 'rotation');
  np.save(fn_out, r);



