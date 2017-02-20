# -*- coding: utf-8 -*-
"""
Worm Trajectory Feature Analysis
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


exp.experiment_directory = '/home/ckirst/Science/Projects/CElegansBehaviour/Experiment/ImageData'


xy = exp.load(strain = 'N2', dtype = 'xy', wid = 0)

### Create features from xy positions

# xy.shape = (ntimes, 2)

def dxy(xy, delta = 1):
  return (xy[delta:] - xy[:-delta]);

def speed(xy, delta = 1, dt = 1):  #

  
def distance(xy, delta = 1):
  d = speed(xy, delta, dt = 1);
  

def area(xy, delta = 10):
  

def elongation(xy, delta = 10):
  








# Rotation 

di = 3;

for wid in range(nworms):
  xy = exp.load(strain = 'N2', wid = wid, memmap = None);
  inv = xy[:,0] == 1.0;
  xy[inv,:] = np.nan;
  
  dxy = xy[di:,:] - xy[:-di,:];
  nrm = np.linalg.norm(dxy, axis = 1)  
  
  dot = np.sum(dxy[1:,:] * dxy[:-1,:], axis = 1);
  nrm1 = nrm[1:];
  nrm0 = nrm[:-1];
  
  inv = np.logical_or(nrm1 == 0, nrm0 == 0);
  nrm1[inv] = np.nan;
  nrm0[inv] = np.nan;
  
  rot = np.arccos(dot / (nrm1 * nrm0));
  
  exp.experiment_directory = '/home/ckirst/Science/Projects/CElegansBehaviour/Experiment/ImageData'
  fn_out = exp.filename(strain = 'N2', wid  = wid, dtype = 'phi');

  np.save(fn_out, rot);
  
  
plt.figure(6); plt.clf();
plt.plot(rot)



