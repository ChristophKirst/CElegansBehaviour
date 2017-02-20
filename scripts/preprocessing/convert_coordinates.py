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

import scripts.preprocessing.file_order as fo

data_directories = fo.directory_names;
print '%d data directories!' % len(data_directories)

nworms = len(data_directories);

#exp.experiment_directory = '/home/ckirst/Science/Projects/CElegansBehaviour/Experiment/Data'

### Convert Coordinates

for wid, wdir in enumerate(data_directories):
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
  fn_out = exp.filename(strain = 'n2', wid  = wid);
  np.save(fn_out, xy);


### Precalculate Speed

import scripts.preprocessing.features as feat;

nworms = len(fo.experiment_names);
delta = 3;

exp.experiment_directory = '/home/ckirst/Science/Projects/CElegansBehaviour/Experiment/ImageData'

for wid in range(nworms):
  print 'processing worm %d/%d' % (wid, nworms);
  
  xy = exp.load(strain = 'n2', wid = wid, memmap = None);

  v = feat.speed(xy, delta = delta);
  fn_out = exp.filename(strain = 'n2', wid  = wid, dtype = 'speed');
  np.save(fn_out, v);
  
  r = feat.rotation(xy, delta = delta);
  fn_out = exp.filename(strain = 'n2', wid  = wid, dtype = 'rotation');
  np.save(fn_out, r);



