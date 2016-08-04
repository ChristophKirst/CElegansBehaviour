# -*- coding: utf-8 -*-
"""
Behavioural Development

Long Term Behaviour Analysis of C-elegans

Matlab to Numpy Data Conversion Routines
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import os
import numpy as np
import scipy.io as io


### Convert X_Y mat files to numpy arrays

files = ['n2_xy_n=109', 'cat-2_xy_n=48', 'tph-1_xy_n=48']
for f in files:
  basedir = '/home/ckirst/Science/Projects/CElegansBehaviour/';
  filename = os.path.join(basedir, 'Experiment/%s.mat' % f)
  f = f.split('_')[0];
  
  xyfile = os.path.join(basedir, 'Experiment/Data/%s_xy_w=%s_s=all.npy' % (f, '%d'));
  stagefile = os.path.join(basedir, 'Experiment/Data/%s_stage_w=%s_s=all.npy' % (f, '%d'));
  
  print 'converting %s to %s' % (filename, xyfile)

  data = io.loadmat(filename);
  XYdata = data['individual_X_Y'][0];
    
  # write out individual worm data sets
  for i,d in enumerate(XYdata):
    # replace invalid coordinates with nan's
    iids = np.logical_or(XYdata[i][:, 0] == 1, XYdata[i][:, 1] == 1);
    XYdata[i][iids,0] = np.nan;
    XYdata[i][iids,1] = np.nan;
    
    #write data
    np.save(xyfile % i, XYdata[i][:,:-1]);
    np.save(stagefile % i, XYdata[i][:,-1]);


